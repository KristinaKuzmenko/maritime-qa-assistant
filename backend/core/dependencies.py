"""
FastAPI dependencies for Maritime QA Assistant.

Provides reusable dependencies for dependency injection in routes.
"""

from typing import Optional, Any
from dataclasses import dataclass
from fastapi import Depends, HTTPException, Header
import logging
import jwt
import os
from datetime import datetime, timedelta

from services.graph_service import Neo4jClient
from services.vector_service import VectorService
from services.embedding_service import EmbeddingService
from services.storage_service import StorageService
from services.document_processor import DocumentProcessor
from core.exceptions import (
    ServiceUnavailableError,
    Neo4jConnectionError,
    QdrantConnectionError,
    UnauthorizedError,
    ForbiddenError,
    RateLimitExceededError,
    DocumentNotFoundError
)

logger = logging.getLogger(__name__)


# ============================================================================
# User Model & JWT Configuration
# ============================================================================

@dataclass
class User:
    """Authenticated user model."""
    id: str
    role: str  # 'admin', 'user', 'guest'
    username: Optional[str] = None
    email: Optional[str] = None


# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


def create_access_token(user_id: str, role: str, username: Optional[str] = None) -> str:
    """Create JWT access token."""
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        "user_id": user_id,
        "role": role,
        "username": username,
        "exp": expire
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> Optional[dict]:
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ============================================================================
# Service Dependencies
# ============================================================================

# Global service instances (set by main.py during startup)
_graph_client: Optional[Neo4jClient] = None
_vector_service: Optional[VectorService] = None
_embedding_service: Optional[EmbeddingService] = None
_storage_service: Optional[StorageService] = None
_document_processor: Optional[DocumentProcessor] = None
_qa_graph = None


def set_services(
    graph_client: Optional[Neo4jClient] = None,
    vector_service: Optional[VectorService] = None,
    embedding_service: Optional[EmbeddingService] = None,
    storage_service: Optional[StorageService] = None,
    document_processor: Optional[DocumentProcessor] = None,
    qa_graph = None,
):
    """
    Set global service instances (called from main.py during startup).
    
    This allows routes to use Depends() pattern while keeping single instances.
    """
    global _graph_client, _vector_service, _embedding_service, _storage_service
    global _document_processor, _qa_graph
    
    _graph_client = graph_client
    _vector_service = vector_service
    _embedding_service = embedding_service
    _storage_service = storage_service
    _document_processor = document_processor
    _qa_graph = qa_graph


# ============================================================================
# Required Service Dependencies
# ============================================================================

def get_graph_client() -> Neo4jClient:
    """
    Get Neo4j graph client.
    
    Raises:
        Neo4jConnectionError: If Neo4j is not connected
    """
    if _graph_client is None:
        raise Neo4jConnectionError(
            "Neo4j service is not available. Check database connection."
        )
    return _graph_client


def get_vector_service() -> VectorService:
    """
    Get Qdrant vector service.
    
    Raises:
        QdrantConnectionError: If Qdrant is not connected
    """
    if _vector_service is None:
        raise QdrantConnectionError(
            "Qdrant service is not available. Check vector database connection."
        )
    return _vector_service


def get_embedding_service() -> EmbeddingService:
    """
    Get embedding service (OpenAI).
    
    Raises:
        ServiceUnavailableError: If embedding service is not available
    """
    if _embedding_service is None:
        raise ServiceUnavailableError(
            "Embedding",
            "Embedding service is not available. Check OpenAI API configuration."
        )
    return _embedding_service


def get_storage_service() -> StorageService:
    """
    Get storage service (S3 or local).
    
    Raises:
        ServiceUnavailableError: If storage service is not available
    """
    if _storage_service is None:
        raise ServiceUnavailableError(
            "Storage",
            "Storage service is not available. Check storage configuration."
        )
    return _storage_service


def get_document_processor() -> DocumentProcessor:
    """
    Get document processor.
    
    Raises:
        ServiceUnavailableError: If document processor is not available
    """
    if _document_processor is None:
        raise ServiceUnavailableError(
            "DocumentProcessor",
            "Document processor is not available. Check service initialization."
        )
    return _document_processor


def get_qa_graph():
    """
    Get Q&A workflow graph.
    
    Raises:
        ServiceUnavailableError: If Q&A graph is not available
    """
    if _qa_graph is None:
        raise ServiceUnavailableError(
            "Q&A Workflow",
            "Q&A service is not available. Check Neo4j and Qdrant connections."
        )
    return _qa_graph


# ============================================================================
# Optional Service Dependencies
# ============================================================================

def get_graph_client_optional() -> Optional[Neo4jClient]:
    """Get Neo4j client if available, None otherwise."""
    return _graph_client


def get_vector_service_optional() -> Optional[VectorService]:
    """Get Qdrant service if available, None otherwise."""
    return _vector_service


def get_storage_service_optional() -> Optional[StorageService]:
    """Get storage service if available, None otherwise."""
    return _storage_service


def get_qa_graph_optional():
    """Get Q&A workflow graph if available, None otherwise."""
    return _qa_graph


# ============================================================================
# Combined Dependencies for Common Use Cases
# ============================================================================

class DocumentServices:
    """Combined services for document operations."""
    
    def __init__(
        self,
        processor: DocumentProcessor = Depends(get_document_processor),
        storage: StorageService = Depends(get_storage_service),
        graph: Neo4jClient = Depends(get_graph_client),
        vector: VectorService = Depends(get_vector_service),
    ):
        self.processor = processor
        self.storage = storage
        self.graph = graph
        self.vector = vector


class QueryServices:
    """Combined services for Q&A operations."""
    
    def __init__(
        self,
        qa_graph = Depends(get_qa_graph),
        graph: Neo4jClient = Depends(get_graph_client),
        vector: VectorService = Depends(get_vector_service),
        storage: Optional[StorageService] = Depends(get_storage_service),
    ):
        self.qa_graph = qa_graph
        self.graph = graph
        self.vector = vector
        self.storage = storage


# ============================================================================
# Health Check Dependencies
# ============================================================================

class HealthStatus:
    """Health check status for all services."""
    
    def __init__(
        self,
        graph: Optional[Neo4jClient] = Depends(get_graph_client_optional),
        vector: Optional[VectorService] = Depends(get_vector_service_optional),
        storage: Optional[StorageService] = Depends(get_storage_service_optional),
        qa = Depends(get_qa_graph_optional),
    ):
        self.graph = graph
        self.vector = vector
        self.storage = storage
        self.qa = qa
        self.graph_available = graph is not None
        self.vector_available = vector is not None
        self.services_healthy = self.graph_available and self.vector_available


# ============================================================================
# Authentication Dependencies
# ============================================================================

def get_current_user(authorization: Optional[str] = Header(None)) -> User:
    """
    Get current authenticated user from JWT token.
    
    Usage:
        @router.post("/api/query")
        async def ask_question(
            user: User = Depends(get_current_user),
            services: QueryServices = Depends()
        ):
            # user.id, user.role, user.username are available
            logger.info(f"Query from {user.username} ({user.role})")
    """
    if not authorization:
        raise UnauthorizedError("Authorization header missing")
    
    # Extract token from "Bearer <token>"
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise UnauthorizedError("Invalid authentication scheme")
    except ValueError:
        raise UnauthorizedError("Invalid authorization header format")
    
    # Verify token
    payload = verify_token(token)
    if not payload:
        raise UnauthorizedError("Invalid or expired token")
    
    return User(
        id=payload.get("user_id"),
        role=payload.get("role", "guest"),
        username=payload.get("username"),
        email=payload.get("email")
    )


def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[User]:
    """Optional authentication - returns None if no valid token."""
    try:
        return get_current_user(authorization)
    except UnauthorizedError:
        return None


# ============================================================================
# Rate Limiting Dependencies
# ============================================================================

# Simple in-memory rate limiter
from collections import defaultdict
from time import time

_rate_limit_store: dict = defaultdict(lambda: {"count": 0, "reset_at": 0})

RATE_LIMITS = {
    "admin": 100,  # requests per hour
    "user": 20,
    "guest": 2
}


def check_rate_limit(
    user: Optional[User] = Depends(get_current_user_optional)
) -> str:
    """
    Check rate limit for current user.
    
    Usage:
        @router.post("/api/query")
        async def ask_question(
            user_id: str = Depends(check_rate_limit),
            services: QueryServices = Depends()
        ):
            # Rate limit is already checked
    """
    # Get user info
    user_id = user.id if user else "anonymous"
    role = user.role if user else "guest"
    
    # Get rate limit for role
    limit = RATE_LIMITS.get(role, 2)
    
    # Check current usage
    now = time()
    key = f"{role}:{user_id}"
    store = _rate_limit_store[key]
    
    # Reset if hour passed
    if now >= store["reset_at"]:
        store["count"] = 0
        store["reset_at"] = now + 3600  # 1 hour
    
    # Check limit
    if store["count"] >= limit:
        retry_after = int(store["reset_at"] - now)
        raise RateLimitExceededError(
            message=f"Rate limit exceeded for {role} role",
            retry_after=retry_after
        )
    
    # Increment counter
    store["count"] += 1
    
    return user_id


# ============================================================================
# Document Access Validation Dependencies
# ============================================================================

async def validate_document_access(
    doc_id: str,
    user: User = Depends(get_current_user),
    graph: Neo4jClient = Depends(get_graph_client)
) -> dict:
    """
    Validate document exists and user has access to it.
    
    Usage:
        @router.get("/api/documents/{doc_id}")
        async def get_document(
            doc_id: str,
            doc: dict = Depends(validate_document_access)
        ):
            # Document is validated and accessible
            return doc
    
    Rules:
    - Document must exist
    - Admin role: access to all documents
    - User role: access only to own documents
    - Guest role: no access
    """
    # Get document metadata
    doc = await graph.get_document_metadata(doc_id)
    
    if not doc:
        raise DocumentNotFoundError(doc_id)
    
    # Check permissions
    doc_owner = doc.get("owner", "global")
    
    # Admin has access to everything
    if user.role == "admin":
        return doc
    
    # Guest has no access
    if user.role == "guest":
        raise ForbiddenError(
            message="Guest users cannot access documents",
            resource=f"document:{doc_id}",
            required_role="user"
        )
    
    # User can only access own documents or global ones
    if doc_owner != user.id and doc_owner != "global":
        raise ForbiddenError(
            message="You don't have permission to access this document",
            resource=f"document:{doc_id}",
            required_role="owner"
        )
    
    return doc

