"""Global error handler middleware"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


async def error_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions"""
    # Log the error with full context
    logger.error(
        f"Unhandled exception: {type(exc).__name__}",
        exc_info=True,
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown",
        },
    )

    # Return safe error response (never expose stack trace in production)
    if isinstance(exc, StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    # Generic 500 for unknown errors
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error. Please check server logs."}
    )


async def validation_exception_handler(request: Request, exc: Exception):
    """Handle validation errors with user-friendly messages"""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(status_code=422, content={"detail": str(exc)})
