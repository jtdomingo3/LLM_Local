"""
Request logging and performance tracking middleware.

Assigns a unique request ID to each incoming request, measures latency,
and logs structured metadata for observability.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("app.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that records telemetry on each HTTP request:
    - Injects unique X-Request-ID if not present
    - Measures wall-clock execution time
    - Logs method, path, response status, and duration
    - Sets X-Process-Time-Ms on outgoing responses
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"

        logger.info(
            "--> %s %s [%s] from %s",
            request.method,
            request.url.path,
            request_id[:8],
            client_ip,
        )

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

            logger.info(
                "<-- %s %s [%s] status=%d duration=%.2fms",
                request.method,
                request.url.path,
                request_id[:8],
                response.status_code,
                duration_ms,
            )
            return response
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(
                "<-- ERROR %s %s [%s] duration=%.2fms: %s",
                request.method,
                request.url.path,
                request_id[:8],
                duration_ms,
                exc,
            )
            raise
