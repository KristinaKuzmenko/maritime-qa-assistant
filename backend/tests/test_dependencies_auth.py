"""
Tests for authentication, rate limiting, and access validation dependencies.

Tests for core/dependencies.py new features:
- JWT authentication (User, create_access_token, verify_token, get_current_user)
- Rate limiting (check_rate_limit)
- Document access validation (validate_document_access)

Run with: pytest test_dependencies_auth.py -v
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import jwt
import time

from core.dependencies import (
    User,
    create_access_token,
    verify_token,
    get_current_user,
    get_current_user_optional,
    check_rate_limit,
    validate_document_access,
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    _rate_limit_store,
    RATE_LIMITS
)
from core.exceptions import (
    UnauthorizedError,
    ForbiddenError,
    RateLimitExceededError,
    DocumentNotFoundError
)


# =============================================================================
# JWT AUTHENTICATION TESTS
# =============================================================================

class TestJWTAuthentication:
    """Tests for JWT token creation and verification."""
    
    def test_create_access_token(self):
        """Test JWT token creation."""
        token = create_access_token(
            user_id="user_123",
            role="user",
            username="john_doe"
        )
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decode and verify payload
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert payload["user_id"] == "user_123"
        assert payload["role"] == "user"
        assert payload["username"] == "john_doe"
        assert "exp" in payload
    
    def test_verify_token_valid(self):
        """Test verifying valid token."""
        token = create_access_token(
            user_id="user_456",
            role="admin",
            username="admin_user"
        )
        
        payload = verify_token(token)
        
        assert payload is not None
        assert payload["user_id"] == "user_456"
        assert payload["role"] == "admin"
        assert payload["username"] == "admin_user"
    
    def test_verify_token_invalid(self):
        """Test verifying invalid token."""
        invalid_token = "invalid.jwt.token"
        
        payload = verify_token(invalid_token)
        
        assert payload is None
    
    def test_verify_token_expired(self):
        """Test verifying expired token."""
        # Create token that expires immediately
        expire = datetime.utcnow() - timedelta(hours=1)
        payload = {
            "user_id": "user_789",
            "role": "user",
            "exp": expire
        }
        expired_token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        result = verify_token(expired_token)
        
        assert result is None
    
    def test_get_current_user_valid_token(self):
        """Test get_current_user with valid token."""
        token = create_access_token(
            user_id="user_111",
            role="user",
            username="test_user"
        )
        
        user = get_current_user(authorization=f"Bearer {token}")
        
        assert isinstance(user, User)
        assert user.id == "user_111"
        assert user.role == "user"
        assert user.username == "test_user"
    
    def test_get_current_user_no_header(self):
        """Test get_current_user without authorization header."""
        with pytest.raises(UnauthorizedError) as exc_info:
            get_current_user(authorization=None)
        
        assert exc_info.value.status_code == 401
        assert "missing" in exc_info.value.message.lower()
    
    def test_get_current_user_invalid_scheme(self):
        """Test get_current_user with invalid auth scheme."""
        with pytest.raises(UnauthorizedError) as exc_info:
            get_current_user(authorization="Basic token123")
        
        assert exc_info.value.status_code == 401
        assert "scheme" in exc_info.value.message.lower()
    
    def test_get_current_user_invalid_format(self):
        """Test get_current_user with invalid header format."""
        with pytest.raises(UnauthorizedError) as exc_info:
            get_current_user(authorization="InvalidFormat")
        
        assert exc_info.value.status_code == 401
    
    def test_get_current_user_expired_token(self):
        """Test get_current_user with expired token."""
        expire = datetime.utcnow() - timedelta(hours=1)
        payload = {"user_id": "user_exp", "role": "user", "exp": expire}
        expired_token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        with pytest.raises(UnauthorizedError) as exc_info:
            get_current_user(authorization=f"Bearer {expired_token}")
        
        assert exc_info.value.status_code == 401
    
    def test_get_current_user_optional_valid(self):
        """Test get_current_user_optional with valid token."""
        token = create_access_token(user_id="user_opt", role="user")
        
        user = get_current_user_optional(authorization=f"Bearer {token}")
        
        assert user is not None
        assert user.id == "user_opt"
    
    def test_get_current_user_optional_no_token(self):
        """Test get_current_user_optional without token."""
        user = get_current_user_optional(authorization=None)
        
        assert user is None
    
    def test_get_current_user_optional_invalid_token(self):
        """Test get_current_user_optional with invalid token."""
        user = get_current_user_optional(authorization="Bearer invalid")
        
        assert user is None


# =============================================================================
# RATE LIMITING TESTS
# =============================================================================

class TestRateLimiting:
    """Tests for rate limiting functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Clear rate limit store before and after each test."""
        _rate_limit_store.clear()
        yield
        _rate_limit_store.clear()
    
    def test_check_rate_limit_guest_allowed(self):
        """Test rate limit for guest - first request allowed."""
        # Guest without token
        user_id = check_rate_limit(user=None)
        
        assert user_id == "anonymous"
    
    def test_check_rate_limit_guest_exceeded(self):
        """Test rate limit for guest - exceeds limit."""
        # Guest limit is 2 requests per hour
        check_rate_limit(user=None)  # 1st request
        check_rate_limit(user=None)  # 2nd request
        
        # 3rd request should fail
        with pytest.raises(RateLimitExceededError) as exc_info:
            check_rate_limit(user=None)
        
        assert exc_info.value.status_code == 429
        assert "retry_after" in exc_info.value.details
    
    def test_check_rate_limit_user_allowed(self):
        """Test rate limit for authenticated user."""
        user = User(id="user_123", role="user", username="test")
        
        # User limit is 20 requests per hour
        for _ in range(20):
            user_id = check_rate_limit(user=user)
            assert user_id == "user_123"
    
    def test_check_rate_limit_user_exceeded(self):
        """Test rate limit for user - exceeds limit."""
        user = User(id="user_456", role="user", username="test")
        
        # Make 20 requests (limit)
        for _ in range(20):
            check_rate_limit(user=user)
        
        # 21st request should fail
        with pytest.raises(RateLimitExceededError) as exc_info:
            check_rate_limit(user=user)
        
        assert exc_info.value.status_code == 429
        assert "user" in exc_info.value.message.lower()
    
    def test_check_rate_limit_admin_high_limit(self):
        """Test rate limit for admin - higher limit."""
        admin = User(id="admin_1", role="admin", username="admin")
        
        # Admin limit is 100 requests per hour
        for _ in range(100):
            user_id = check_rate_limit(user=admin)
            assert user_id == "admin_1"
    
    def test_check_rate_limit_different_users(self):
        """Test rate limits are independent per user."""
        user1 = User(id="user_1", role="user", username="user1")
        user2 = User(id="user_2", role="user", username="user2")
        
        # Each user has their own limit
        for _ in range(20):
            check_rate_limit(user=user1)
            check_rate_limit(user=user2)
        
        # Both should be at limit, next request fails
        with pytest.raises(RateLimitExceededError):
            check_rate_limit(user=user1)
        
        with pytest.raises(RateLimitExceededError):
            check_rate_limit(user=user2)
    
    def test_rate_limit_reset_after_time(self):
        """Test rate limit resets after time window."""
        user = User(id="user_reset", role="guest", username="guest")
        
        # Use up limit (2 for guest)
        check_rate_limit(user=user)
        check_rate_limit(user=user)
        
        # Should fail
        with pytest.raises(RateLimitExceededError):
            check_rate_limit(user=user)
        
        # Manually advance time by resetting store
        key = f"guest:user_reset"
        _rate_limit_store[key]["reset_at"] = time.time() - 1
        _rate_limit_store[key]["count"] = 0
        
        # Should work again
        user_id = check_rate_limit(user=user)
        assert user_id == "user_reset"


# =============================================================================
# DOCUMENT ACCESS VALIDATION TESTS
# =============================================================================

class TestDocumentAccessValidation:
    """Tests for document access validation."""
    
    @pytest.mark.asyncio
    async def test_validate_document_access_admin(self):
        """Test admin has access to all documents."""
        mock_graph = AsyncMock()
        mock_graph.get_document_metadata = AsyncMock(return_value={
            "id": "doc_1",
            "title": "Test Doc",
            "owner": "other_user"
        })
        
        admin = User(id="admin_1", role="admin", username="admin")
        
        doc = await validate_document_access(
            doc_id="doc_1",
            user=admin,
            graph=mock_graph
        )
        
        assert doc["id"] == "doc_1"
        assert doc["owner"] == "other_user"
    
    @pytest.mark.asyncio
    async def test_validate_document_access_owner(self):
        """Test user can access own documents."""
        mock_graph = AsyncMock()
        mock_graph.get_document_metadata = AsyncMock(return_value={
            "id": "doc_2",
            "title": "My Doc",
            "owner": "user_123"
        })
        
        user = User(id="user_123", role="user", username="john")
        
        doc = await validate_document_access(
            doc_id="doc_2",
            user=user,
            graph=mock_graph
        )
        
        assert doc["id"] == "doc_2"
        assert doc["owner"] == "user_123"
    
    @pytest.mark.asyncio
    async def test_validate_document_access_global(self):
        """Test user can access global documents."""
        mock_graph = AsyncMock()
        mock_graph.get_document_metadata = AsyncMock(return_value={
            "id": "doc_global",
            "title": "Global Doc",
            "owner": "global"
        })
        
        user = User(id="user_456", role="user", username="jane")
        
        doc = await validate_document_access(
            doc_id="doc_global",
            user=user,
            graph=mock_graph
        )
        
        assert doc["owner"] == "global"
    
    @pytest.mark.asyncio
    async def test_validate_document_access_forbidden(self):
        """Test user cannot access other user's documents."""
        mock_graph = AsyncMock()
        mock_graph.get_document_metadata = AsyncMock(return_value={
            "id": "doc_3",
            "title": "Private Doc",
            "owner": "other_user"
        })
        
        user = User(id="user_789", role="user", username="bob")
        
        with pytest.raises(ForbiddenError) as exc_info:
            await validate_document_access(
                doc_id="doc_3",
                user=user,
                graph=mock_graph
            )
        
        assert exc_info.value.status_code == 403
        assert "permission" in exc_info.value.message.lower()
        assert exc_info.value.details["resource"] == "document:doc_3"
    
    @pytest.mark.asyncio
    async def test_validate_document_access_guest_denied(self):
        """Test guest cannot access any documents."""
        mock_graph = AsyncMock()
        mock_graph.get_document_metadata = AsyncMock(return_value={
            "id": "doc_4",
            "title": "Any Doc",
            "owner": "global"
        })
        
        guest = User(id="guest_1", role="guest", username="guest")
        
        with pytest.raises(ForbiddenError) as exc_info:
            await validate_document_access(
                doc_id="doc_4",
                user=guest,
                graph=mock_graph
            )
        
        assert exc_info.value.status_code == 403
        assert "guest" in exc_info.value.message.lower()
        assert exc_info.value.details["required_role"] == "user"
    
    @pytest.mark.asyncio
    async def test_validate_document_access_not_found(self):
        """Test validation fails if document doesn't exist."""
        mock_graph = AsyncMock()
        mock_graph.get_document_metadata = AsyncMock(return_value=None)
        
        user = User(id="user_999", role="user", username="test")
        
        with pytest.raises(DocumentNotFoundError) as exc_info:
            await validate_document_access(
                doc_id="nonexistent",
                user=user,
                graph=mock_graph
            )
        
        assert exc_info.value.status_code == 404
        assert "nonexistent" in exc_info.value.details["doc_id"]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestDependenciesIntegration:
    """Integration tests with FastAPI app."""
    
    def test_protected_endpoint_with_valid_token(self):
        """Test accessing protected endpoint with valid token."""
        app = FastAPI()
        
        @app.get("/protected")
        def protected_route(user: User = Depends(get_current_user)):
            return {"user_id": user.id, "role": user.role}
        
        client = TestClient(app)
        token = create_access_token(user_id="test_user", role="user")
        
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        assert response.json()["user_id"] == "test_user"
    
    def test_protected_endpoint_without_token(self):
        """Test accessing protected endpoint without token."""
        app = FastAPI()
        
        @app.get("/protected")
        def protected_route(user: User = Depends(get_current_user)):
            return {"user_id": user.id}
        
        # Add exception handler
        from fastapi.responses import JSONResponse
        from core.exceptions import MaritimeQAException
        
        @app.exception_handler(MaritimeQAException)
        async def handler(request, exc):
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.__class__.__name__, "message": exc.message}
            )
        
        client = TestClient(app)
        response = client.get("/protected")
        
        assert response.status_code == 401
        assert "error" in response.json()
    
    def test_rate_limited_endpoint(self):
        """Test rate limiting on endpoint."""
        app = FastAPI()
        
        # Clear rate limit store
        _rate_limit_store.clear()
        
        @app.post("/query")
        def query_route(
            user_id: str = Depends(check_rate_limit)
        ):
            return {"user_id": user_id}
        
        # Add exception handler
        from fastapi.responses import JSONResponse
        from core.exceptions import MaritimeQAException
        
        @app.exception_handler(MaritimeQAException)
        async def handler(request, exc):
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.__class__.__name__, "message": exc.message}
            )
        
        client = TestClient(app)
        
        # Guest can make 2 requests
        resp1 = client.post("/query")
        assert resp1.status_code == 200
        
        resp2 = client.post("/query")
        assert resp2.status_code == 200
        
        # 3rd request should fail
        resp3 = client.post("/query")
        assert resp3.status_code == 429
        assert "RateLimitExceededError" in resp3.json()["error"]
        
        # Clear for next test
        _rate_limit_store.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
