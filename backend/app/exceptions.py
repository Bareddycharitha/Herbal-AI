"""
Custom Exceptions

Structured exception hierarchy for Herbal-AI with error codes
for consistent error handling and client-side error management.
"""

from typing import Any, Optional


class HerbalAIError(Exception):
    """Base exception for all Herbal-AI errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[dict[str, Any]] = None,
        status_code: int = 500,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# ==========================================================
# Validation Errors (4xx)
# ==========================================================


class ValidationError(HerbalAIError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        field: Optional[str] = None,
    ):
        error_details = details or {}
        if field:
            error_details["field"] = field
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=error_details,
            status_code=400,
        )


class FileValidationError(ValidationError):
    """Raised when uploaded file fails validation."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            details=details,
        )
        self.error_code = "FILE_VALIDATION_ERROR"


class FileSizeError(FileValidationError):
    """Raised when uploaded file exceeds size limit."""

    def __init__(self, max_size_mb: int, actual_size_mb: float):
        super().__init__(
            message=f"File size ({actual_size_mb:.1f} MB) exceeds maximum allowed ({max_size_mb} MB)",
            details={
                "max_size_mb": max_size_mb,
                "actual_size_mb": round(actual_size_mb, 1),
            },
        )
        self.error_code = "FILE_SIZE_EXCEEDED"


class FileTypeError(FileValidationError):
    """Raised when uploaded file type is not allowed."""

    def __init__(self, allowed_types: list[str], detected_type: str):
        super().__init__(
            message=f"File type '{detected_type}' is not allowed. Allowed types: {', '.join(allowed_types)}",
            details={
                "allowed_types": allowed_types,
                "detected_type": detected_type,
            },
        )
        self.error_code = "FILE_TYPE_NOT_ALLOWED"


# ==========================================================
# Authentication & Authorization Errors (4xx)
# ==========================================================


class AuthenticationError(HerbalAIError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
        )


class AuthorizationError(HerbalAIError):
    """Raised when authorization fails."""

    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
        )


# ==========================================================
# Not Found Errors (4xx)
# ==========================================================


class NotFoundError(HerbalAIError):
    """Raised when a resource is not found."""

    def __init__(
        self,
        resource: str,
        identifier: Optional[str] = None,
    ):
        message = f"{resource} not found"
        if identifier:
            message += f": {identifier}"
        super().__init__(
            message=message,
            error_code="NOT_FOUND",
            details={"resource": resource, "identifier": identifier} if identifier else {"resource": resource},
            status_code=404,
        )


# ==========================================================
# Model & Inference Errors (5xx)
# ==========================================================


class ModelError(HerbalAIError):
    """Raised when model inference fails."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        error_details = details or {}
        if model_name:
            error_details["model"] = model_name
        super().__init__(
            message=message,
            error_code="MODEL_ERROR",
            details=error_details,
            status_code=500,
        )


class ModelLoadError(ModelError):
    """Raised when model checkpoint fails to load."""

    def __init__(
        self,
        model_path: str,
        reason: str,
    ):
        super().__init__(
            message=f"Failed to load model from {model_path}: {reason}",
            details={"model_path": model_path, "reason": reason},
        )
        self.error_code = "MODEL_LOAD_ERROR"


class ModelInferenceError(ModelError):
    """Raised when model inference fails."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            model_name=model_name,
        )
        self.error_code = "MODEL_INFERENCE_ERROR"


class ModelArchitectureMismatchError(ModelError):
    """Raised when model architecture doesn't match checkpoint."""

    def __init__(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ):
        super().__init__(
            message="Model architecture mismatch between code and checkpoint",
            details={
                "expected": expected,
                "actual": actual,
            },
        )
        self.error_code = "MODEL_ARCHITECTURE_MISMATCH"


# ==========================================================
# LLM Errors (5xx)
# ==========================================================


class LLMError(HerbalAIError):
    """Raised when LLM (Ollama) operations fail."""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        retryable: bool = True,
    ):
        error_details = details or {}
        if model:
            error_details["model"] = model
        error_details["retryable"] = retryable
        super().__init__(
            message=message,
            error_code="LLM_ERROR",
            details=error_details,
            status_code=503 if retryable else 500,
        )


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""

    def __init__(self, timeout_seconds: int):
        super().__init__(
            message=f"LLM request timed out after {timeout_seconds} seconds",
            details={"timeout_seconds": timeout_seconds},
            retryable=True,
        )
        self.error_code = "LLM_TIMEOUT"


class LLMUnavailableError(LLMError):
    """Raised when LLM service is unavailable."""

    def __init__(self, host: str):
        super().__init__(
            message=f"LLM service unavailable at {host}",
            details={"host": host},
            retryable=True,
        )
        self.error_code = "LLM_UNAVAILABLE"


# ==========================================================
# External Service Errors (5xx)
# ==========================================================


class ExternalServiceError(HerbalAIError):
    """Raised when an external service call fails."""

    def __init__(
        self,
        service: str,
        message: str,
        details: Optional[dict[str, Any]] = None,
        retryable: bool = True,
    ):
        error_details = details or {}
        error_details["service"] = service
        error_details["retryable"] = retryable
        super().__init__(
            message=f"{service} error: {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            details=error_details,
            status_code=503 if retryable else 502,
        )


# ==========================================================
# Rate Limiting Errors (4xx)
# ==========================================================


class RateLimitError(HerbalAIError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        limit: int,
        window_seconds: int,
        retry_after: Optional[int] = None,
    ):
        details = {"limit": limit, "window_seconds": window_seconds}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window_seconds} seconds",
            error_code="RATE_LIMIT_EXCEEDED",
            details=details,
            status_code=429,
        )


# ==========================================================
# Configuration Errors (5xx)
# ==========================================================


class ConfigurationError(HerbalAIError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str,
        setting: Optional[str] = None,
    ):
        details = {}
        if setting:
            details["setting"] = setting
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details,
            status_code=500,
        )