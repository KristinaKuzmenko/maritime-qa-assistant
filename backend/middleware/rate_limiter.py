"""
Rate limiting middleware for FastAPI.
"""

from fastapi import Request, HTTPException, status
from typing import Callable, Dict, Optional
import time
from collections import defaultdict
import threading
from functools import wraps


# Exceptions

class RateLimitExceeded(HTTPException):
    """Custom exception for rate limit exceeded."""
    def __init__(self, retry_after: int):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )


# Memory-based Rate Limiter

class MemoryRateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests: Dict[str, list] = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(
        self, 
        key: str, 
        max_requests: int, 
        window_seconds: int
    ) -> tuple[bool, Optional[int]]:
        """Check if request is allowed."""
        with self.lock:
            now = time.time()
            cutoff = now - window_seconds
            
            # Clean old requests
            self.requests[key] = [
                req_time for req_time in self.requests[key] 
                if req_time > cutoff
            ]
            
            # Check limit
            if len(self.requests[key]) >= max_requests:
                oldest_request = min(self.requests[key])
                retry_after = int(oldest_request + window_seconds - now) + 1
                return False, retry_after
            
            # Allow request
            self.requests[key].append(now)
            return True, None
    
    def reset(self, key: str):
        """Reset counter for a key."""
        with self.lock:
            if key in self.requests:
                del self.requests[key]
    
    def get_remaining(self, key: str, max_requests: int, window_seconds: int) -> int:
        """Get remaining requests for a key."""
        with self.lock:
            now = time.time()
            cutoff = now - window_seconds
            
            # Clean old requests
            self.requests[key] = [
                req_time for req_time in self.requests[key] 
                if req_time > cutoff
            ]
            
            return max(0, max_requests - len(self.requests[key]))


# Global limiter instance
memory_limiter = MemoryRateLimiter()


# Rate Limit Configurations

RATE_LIMITS = {
    "admin": {
        "upload": (10, 3600),      # 10 uploads per hour
        "qa": (100, 3600),          # 100 questions per hour
    },
    "user": {
        "upload": (3, 3600),       # 3 uploads per hour
        "qa": (20, 3600),           # 20 questions per hour
    },
    "guest": {
        "upload": (0, 3600),        # No uploads
        "qa": (2, 3600),           # 2 questions per hour
    }
}


# Decorators

def rate_limit(max_requests: int, window_seconds: int):
    """
    Simple rate limit decorator.
    
    Example:
        @rate_limit(max_requests=10, window_seconds=60)
        async def my_endpoint(request: Request):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Use IP address as key
            key = f"ip:{request.client.host}:{func.__name__}"
            
            # Check rate limit
            is_allowed, retry_after = memory_limiter.is_allowed(
                key, max_requests, window_seconds
            )
            
            if not is_allowed:
                raise RateLimitExceeded(retry_after)
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator


def role_rate_limit(action: str):
    """
    Role-based rate limit decorator.
    
    Example:
        @role_rate_limit("upload")
        async def upload_document(request: Request):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Get user info from request state (set by auth middleware)
            user_role = getattr(request.state, "user_role", "guest")
            user_id = getattr(request.state, "user_id", request.client.host)
            
            # Check rate limit
            is_allowed, retry_after = check_role_rate_limit(
                user_role, action, user_id
            )
            
            if not is_allowed:
                if retry_after:
                    raise RateLimitExceeded(retry_after)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Action '{action}' not allowed for {user_role} role."
                    )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator



# Helper Functions

def check_role_rate_limit(
    user_role: str, 
    action: str, 
    user_id: str
) -> tuple[bool, Optional[int]]:
    """Check rate limit based on user role."""
    limits = RATE_LIMITS.get(user_role, RATE_LIMITS["guest"])
    max_requests, window_seconds = limits.get(action, (0, 3600))
    
    if max_requests == 0:
        return False, None
    
    key = f"role:{user_role}:{user_id}:{action}"
    return memory_limiter.is_allowed(key, max_requests, window_seconds)


def get_rate_limit_info(user_role: str, action: str, user_id: str) -> dict:
    """Get rate limit information for user."""
    limits = RATE_LIMITS.get(user_role, RATE_LIMITS["guest"])
    max_requests, window_seconds = limits.get(action, (0, 3600))
    
    key = f"role:{user_role}:{user_id}:{action}"
    remaining = memory_limiter.get_remaining(key, max_requests, window_seconds)
    
    return {
        "limit": max_requests,
        "window_seconds": window_seconds,
        "remaining": remaining,
        "reset_after": window_seconds
    }