"""Rate limiting middleware using slowapi"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# Initialize limiter with key function based on client IP
limiter = Limiter(key_func=get_remote_address)


def rate_limit_error_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors"""
    logger.warning(f"Rate limit exceeded: {request.client.host} - {request.url.path}")
    return JSONResponse(
        status_code=429, content={"detail": "Too many requests. Please try again later."}
    )


def get_limiter():
    """Get the rate limiter instance"""
    return limiter
