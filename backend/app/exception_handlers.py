"""
Exception Handlers

Global exception handlers for FastAPI to ensure consistent
error responses across all endpoints.
"""

import logging
import traceback
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.exceptions import (
    HerbalAIError,
    ValidationError,
    FileValidationError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ModelError,
    LLMError,
    ExternalServiceError,
    RateLimitError,
    ConfigurationError,
)

logger = logging.getLogger(__name__)


def get_request_id(request: Request) -> str:
    """Extract or generate request ID."""
    return getattr(request.state, "request_id", "unknown")


async def herbal_ai_exception_handler(request: Request, exc: HerbalAIError) -> JSONResponse:
    """Handle all HerbalAIError subclasses."""
    request_id = get_request_id(request)

    log_data = {
        "request_id": request_id,
        "error_code": exc.error_code,
        "error_message": exc.message,
        "details": exc.details,
        "path": str(request.url),
        "method": request.method,
    }

    if exc.status_code >= 500:
        logger.error("Herbal-AI error", extra=log_data)
    else:
        logger.warning("Herbal-AI client error", extra=log_data)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "request_id": request_id,
            },
        },
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic/FastAPI validation errors."""
    request_id = get_request_id(request)

    # Extract field-level errors
    errors = []
    for error in exc.errors():
        field_path = ".".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"],
        })

    log_data = {
        "request_id": request_id,
        "error_code": "VALIDATION_ERROR",
        "errors": errors,
        "path": str(request.url),
        "method": request.method,
    }
    logger.warning("Request validation failed", extra=log_data)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {"errors": errors},
                "request_id": request_id,
            },
        },
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle Starlette HTTP exceptions."""
    request_id = get_request_id(request)

    log_data = {
        "request_id": request_id,
        "error_code": "HTTP_ERROR",
        "status_code": exc.status_code,
        "detail": exc.detail,
        "path": str(request.url),
        "method": request.method,
    }
    logger.warning("HTTP exception", extra=log_data)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": "HTTP_ERROR",
                "message": str(exc.detail),
                "details": {"status_code": exc.status_code},
                "request_id": request_id,
            },
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all unhandled exceptions."""
    request_id = get_request_id(request)

    log_data = {
        "request_id": request_id,
        "error_code": "INTERNAL_ERROR",
        "exception_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "path": str(request.url),
        "method": request.method,
    }
    logger.error("Unhandled exception", extra=log_data)

    # Don't expose internal details in production
    from backend.app.config import settings
    if settings.environment == "production":
        message = "An internal server error occurred"
        details = {}
    else:
        message = str(exc)
        details = {"exception_type": type(exc).__name__}

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": message,
                "details": details,
                "request_id": request_id,
            },
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the FastAPI app."""
    app.add_exception_handler(HerbalAIError, herbal_ai_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)