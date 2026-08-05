"""
Request Middleware

Middleware for request ID generation, logging, and timing.
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to generate/extract and propagate request IDs."""

    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract or generate request ID
        request_id = request.headers.get(self.header_name) or str(uuid.uuid4())[:12]

        # Store in request state for access in handlers
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add to response headers
        response.headers[self.header_name] = request_id

        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging."""

    def __init__(self, app: ASGIApp, log_requests: bool = True):
        super().__init__(app)
        self.log_requests = log_requests

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.log_requests:
            return await call_next(request)

        request_id = getattr(request.state, "request_id", "unknown")
        start_time = time.perf_counter()

        # Log request
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            client_host=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        try:
            response = await call_next(request)
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Request failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Log response
        logger.info(
            "Request completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        burst: int = 10,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self._requests: dict[str, list[float]] = {}

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use X-Forwarded-For if behind proxy, else client host
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup_old_requests(self, client_id: str, now: float) -> None:
        """Remove requests older than 1 minute."""
        minute_ago = now - 60
        if client_id in self._requests:
            self._requests[client_id] = [
                ts for ts in self._requests[client_id] if ts > minute_ago
            ]

    def _is_rate_limiting_enabled(self, request: Request) -> bool:
        """Check if rate limiting should be applied."""
        # Disable rate limiting in test environment
        import os
        return os.environ.get("ENVIRONMENT") != "test"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/ready", "/metrics"):
            return await call_next(request)

        # Skip rate limiting in test environment
        if not self._is_rate_limiting_enabled(request):
            return await call_next(request)

        client_id = self._get_client_id(request)
        now = time.time()

        self._cleanup_old_requests(client_id, now)

        # Check rate limit
        request_count = len(self._requests.get(client_id, []))

        if request_count >= self.requests_per_minute:
            from backend.app.exceptions import RateLimitError
            from backend.app.exception_handlers import herbal_ai_exception_handler

            retry_after = 60 - int(now - self._requests[client_id][0]) if self._requests.get(client_id) else 60
            exc = RateLimitError(
                limit=self.requests_per_minute,
                window_seconds=60,
                retry_after=max(1, retry_after),
            )
            return await herbal_ai_exception_handler(request, exc)

        # Check burst limit
        recent_requests = [
            ts for ts in self._requests.get(client_id, [])
            if ts > now - 10  # Last 10 seconds
        ]
        if len(recent_requests) >= self.burst:
            from backend.app.exceptions import RateLimitError
            from backend.app.exception_handlers import herbal_ai_exception_handler

            exc = RateLimitError(
                limit=self.burst,
                window_seconds=10,
                retry_after=10,
            )
            return await herbal_ai_exception_handler(request, exc)

        # Record request
        if client_id not in self._requests:
            self._requests[client_id] = []
        self._requests[client_id].append(now)

        return await call_next(request)