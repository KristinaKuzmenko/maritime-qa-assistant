"""
Custom exceptions for Maritime QA Assistant.

Provides typed exceptions with automatic HTTP status code mapping.
"""

from typing import Optional, Dict, Any


# ============================================================================
# Base Exceptions
# ============================================================================

class MaritimeQAException(Exception):
    """Base exception for all Maritime QA errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


# ============================================================================
# Service Exceptions (500-level errors)
# ============================================================================

class ServiceUnavailableError(MaritimeQAException):
    """Service not initialized or unavailable."""
    
    def __init__(self, service_name: str, message: Optional[str] = None):
        msg = message or f"{service_name} service is not available"
        super().__init__(
            message=msg,
            status_code=503,
            details={"service": service_name}
        )


class Neo4jConnectionError(ServiceUnavailableError):
    """Neo4j connection failure."""
    
    def __init__(self, message: Optional[str] = None):
        super().__init__(
            service_name="Neo4j",
            message=message or "Neo4j graph database is not connected"
        )


class QdrantConnectionError(ServiceUnavailableError):
    """Qdrant connection failure."""
    
    def __init__(self, message: Optional[str] = None):
        super().__init__(
            service_name="Qdrant",
            message=message or "Qdrant vector database is not connected"
        )


class ProcessingError(MaritimeQAException):
    """Document or query processing failed."""
    
    def __init__(self, message: str, doc_id: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=500,
            details={"doc_id": doc_id} if doc_id else {}
        )


# ============================================================================
# Client Exceptions (400-level errors)
# ============================================================================

class ValidationError(MaritimeQAException):
    """Request validation failed."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=400,
            details={"field": field} if field else {}
        )


class PromptInjectionError(MaritimeQAException):
    """Prompt injection detected."""
    
    def __init__(
        self,
        message: str,
        risk_level: str,
        detected_patterns: list,
        query_preview: str
    ):
        super().__init__(
            message=message,
            status_code=400,
            details={
                "risk_level": risk_level,
                "detected_patterns": detected_patterns,
                "query_preview": query_preview[:100]
            }
        )


class DocumentNotFoundError(MaritimeQAException):
    """Document not found in database."""
    
    def __init__(self, doc_id: str):
        super().__init__(
            message=f"Document not found: {doc_id}",
            status_code=404,
            details={"doc_id": doc_id}
        )


class FileUploadError(MaritimeQAException):
    """File upload or validation failed."""
    
    def __init__(self, message: str, filename: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=400,
            details={"filename": filename} if filename else {}
        )


# ============================================================================
# Authentication Exceptions
# ============================================================================

class UnauthorizedError(MaritimeQAException):
    """Authentication failed or token invalid."""
    
    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            status_code=401,
            details={}
        )


class ForbiddenError(MaritimeQAException):
    """Access denied - insufficient permissions."""
    
    def __init__(
        self,
        message: str = "Access denied",
        resource: Optional[str] = None,
        required_role: Optional[str] = None
    ):
        details = {}
        if resource:
            details["resource"] = resource
        if required_role:
            details["required_role"] = required_role
        
        super().__init__(
            message=message,
            status_code=403,
            details=details
        )


# ============================================================================
# Rate Limiting Exceptions
# ============================================================================

class RateLimitExceededError(MaritimeQAException):
    """Rate limit exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None
    ):
        super().__init__(
            message=message,
            status_code=429,
            details={"retry_after": retry_after} if retry_after else {}
        )
