"""
Example routes demonstrating authentication, rate limiting, and access validation.

These examples show how to use the new dependency injection features:
- JWT authentication (get_current_user)
- Rate limiting (check_rate_limit)
- Document access validation (validate_document_access)
"""

from fastapi import APIRouter, Depends, Body
from pydantic import BaseModel
from typing import Optional

from core.dependencies import (
    User,
    get_current_user,
    get_current_user_optional,
    check_rate_limit,
    validate_document_access,
    QueryServices,
    create_access_token
)
from core.exceptions import ForbiddenError

router = APIRouter()


# ============================================================================
# Authentication Examples
# ============================================================================

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    role: str


@router.post("/auth/login", response_model=TokenResponse)
async def login(credentials: LoginRequest):
    """
    Login endpoint - generates JWT token.
    
    In production, validate credentials against database.
    """
    # TODO: Validate credentials against database
    # For demo purposes, accept any credentials
    
    # Determine role (in production, get from database)
    role = "user" if credentials.username != "admin" else "admin"
    user_id = f"user_{credentials.username}"
    
    # Generate JWT token
    token = create_access_token(
        user_id=user_id,
        role=role,
        username=credentials.username
    )
    
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user_id=user_id,
        role=role
    )


@router.get("/auth/me")
async def get_current_user_info(user: User = Depends(get_current_user)):
    """
    Get current authenticated user info.
    
    Requires: Authorization: Bearer <token>
    """
    return {
        "user_id": user.id,
        "username": user.username,
        "role": user.role,
        "email": user.email
    }


@router.get("/auth/optional")
async def optional_auth_example(user: Optional[User] = Depends(get_current_user_optional)):
    """
    Example of optional authentication.
    
    Works with or without token.
    """
    if user:
        return {
            "message": f"Hello, {user.username}!",
            "authenticated": True,
            "role": user.role
        }
    else:
        return {
            "message": "Hello, anonymous!",
            "authenticated": False
        }


# ============================================================================
# Rate Limiting Examples
# ============================================================================

class QueryRequest(BaseModel):
    question: str


@router.post("/example/query-with-rate-limit")
async def query_with_rate_limit(
    request: QueryRequest,
    user_id: str = Depends(check_rate_limit),  # Automatic rate limiting
    services: QueryServices = Depends()
):
    """
    Example query endpoint with automatic rate limiting.
    
    Rate limits by role:
    - admin: 100 requests/hour
    - user: 20 requests/hour
    - guest: 2 requests/hour
    
    Returns 429 if limit exceeded with retry_after header.
    """
    return {
        "question": request.question,
        "user_id": user_id,
        "message": "Query processed successfully"
    }


# ============================================================================
# Document Access Validation Examples
# ============================================================================

@router.get("/example/documents/{doc_id}/secure")
async def get_secure_document(
    doc_id: str,
    doc: dict = Depends(validate_document_access)  # Automatic validation
):
    """
    Example of secure document access.
    
    Automatically validates:
    - User is authenticated (401 if not)
    - Document exists (404 if not)
    - User has access (403 if not)
    
    Access rules:
    - admin: access to all documents
    - user: access to own documents + global
    - guest: no access
    """
    return {
        "doc_id": doc["id"],
        "title": doc["title"],
        "owner": doc["owner"],
        "status": doc.get("status"),
        "message": "You have access to this document"
    }


@router.delete("/example/documents/{doc_id}/secure")
async def delete_secure_document(
    doc_id: str,
    doc: dict = Depends(validate_document_access),
    user: User = Depends(get_current_user)
):
    """
    Example of secure document deletion.
    
    Combines multiple checks:
    - Authentication required
    - Document must exist
    - User must have access
    """
    # In production, actually delete the document
    # await services.graph.delete_document(doc_id)
    # await services.vector.delete_document_vectors(doc_id)
    
    return {
        "message": f"Document {doc_id} deleted by {user.username}",
        "doc_id": doc_id,
        "deleted_by": user.id
    }


# ============================================================================
# Combined Dependencies Example
# ============================================================================

@router.post("/example/protected-query")
async def protected_query(
    request: QueryRequest,
    user: User = Depends(get_current_user),  # 1. Check authentication
    user_id: str = Depends(check_rate_limit),  # 2. Check rate limit
    services: QueryServices = Depends()  # 3. Check services available
):
    """
    Example combining all three dependency types.
    
    All checks happen automatically before the handler runs:
    1. JWT token validation
    2. Rate limiting
    3. Service availability
    
    If any check fails, appropriate error is returned:
    - 401: Invalid/missing token
    - 429: Rate limit exceeded
    - 503: Service unavailable
    """
    return {
        "question": request.question,
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role
        },
        "message": "All checks passed, query processed successfully"
    }


# ============================================================================
# Role-Based Access Example
# ============================================================================

def require_admin(user: User = Depends(get_current_user)) -> User:
    """Custom dependency - require admin role."""
    if user.role != "admin":
        raise ForbiddenError(
            message="Admin role required for this operation",
            required_role="admin"
        )
    return user


@router.post("/example/admin-only")
async def admin_only_endpoint(
    data: dict = Body(...),
    admin: User = Depends(require_admin)  # Only admins can access
):
    """
    Example of admin-only endpoint.
    
    Returns 403 for non-admin users.
    """
    return {
        "message": "Admin operation completed",
        "admin_id": admin.id,
        "data": data
    }


# ============================================================================
# Usage Instructions
# ============================================================================

"""
To use these examples:

1. Login to get token:
   POST /auth/login
   {"username": "john", "password": "secret"}
   
   Response: {"access_token": "eyJ...", "token_type": "bearer", ...}

2. Use token in subsequent requests:
   Authorization: Bearer eyJ...
   
3. Try different endpoints:
   - GET /auth/me - get user info
   - GET /auth/optional - works with/without token
   - POST /example/query-with-rate-limit - rate limited
   - GET /example/documents/{doc_id}/secure - access validation
   - POST /example/protected-query - all checks combined
   - POST /example/admin-only - admin only

4. Test error cases:
   - No token: 401 Unauthorized
   - Invalid token: 401 Unauthorized
   - Rate limit exceeded: 429 Too Many Requests
   - No access to document: 403 Forbidden
   - Non-admin accessing admin endpoint: 403 Forbidden
"""
