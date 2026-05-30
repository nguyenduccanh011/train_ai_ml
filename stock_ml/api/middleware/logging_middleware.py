"""Request/response logging middleware"""
import logging
import time
from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all HTTP requests with duration and status code"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Get request info
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"

        try:
            # Process request
            response = await call_next(request)
            duration = time.time() - start_time

            # Log successful request
            logger.info(
                f"{method} {path} {response.status_code}",
                extra={
                    "method": method,
                    "path": path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2),
                    "client_ip": client_ip
                }
            )

            return response

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"{method} {path} ERROR",
                exc_info=True,
                extra={
                    "method": method,
                    "path": path,
                    "duration_ms": round(duration * 1000, 2),
                    "client_ip": client_ip,
                    "error": str(e)
                }
            )
            raise
