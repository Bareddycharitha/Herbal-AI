"""
Tests for Custom Exceptions
"""

import pytest
from backend.app.exceptions import (
    HerbalAIError,
    ValidationError,
    FileValidationError,
    FileSizeError,
    FileTypeError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ModelError,
    ModelLoadError,
    ModelInferenceError,
    ModelArchitectureMismatchError,
    LLMError,
    LLMTimeoutError,
    LLMUnavailableError,
    ExternalServiceError,
    RateLimitError,
    ConfigurationError,
)


class TestHerbalAIError:
    """Tests for base HerbalAIError."""

    def test_basic_creation(self):
        """Test basic exception creation."""
        exc = HerbalAIError("Test message", error_code="TEST_ERROR", status_code=400)
        assert exc.message == "Test message"
        assert exc.error_code == "TEST_ERROR"
        assert exc.status_code == 400
        assert exc.details == {}

    def test_with_details(self):
        """Test exception with details."""
        exc = HerbalAIError("Test", details={"key": "value", "count": 5})
        assert exc.details == {"key": "value", "count": 5}

    def test_to_dict(self):
        """Test dictionary conversion."""
        exc = HerbalAIError("Test message", error_code="TEST_ERROR", details={"foo": "bar"})
        d = exc.to_dict()
        assert d == {
            "error_code": "TEST_ERROR",
            "message": "Test message",
            "details": {"foo": "bar"},
        }


class TestValidationErrors:
    """Tests for validation-related exceptions."""

    def test_validation_error(self):
        """Test ValidationError."""
        exc = ValidationError("Invalid input", field="email")
        assert exc.error_code == "VALIDATION_ERROR"
        assert exc.status_code == 400
        assert exc.details["field"] == "email"

    def test_file_validation_error(self):
        """Test FileValidationError."""
        exc = FileValidationError("Bad file", details={"reason": "corrupt"})
        assert exc.error_code == "FILE_VALIDATION_ERROR"
        assert exc.status_code == 400

    def test_file_size_error(self):
        """Test FileSizeError."""
        exc = FileSizeError(max_size_mb=10, actual_size_mb=15.5)
        assert exc.error_code == "FILE_SIZE_EXCEEDED"
        assert exc.details["max_size_mb"] == 10
        assert exc.details["actual_size_mb"] == 15.5
        assert "15.5" in exc.message

    def test_file_type_error(self):
        """Test FileTypeError."""
        exc = FileTypeError(allowed_types=["image/jpeg", "image/png"], detected_type="image/gif")
        assert exc.error_code == "FILE_TYPE_NOT_ALLOWED"
        assert exc.details["allowed_types"] == ["image/jpeg", "image/png"]
        assert exc.details["detected_type"] == "image/gif"


class TestAuthErrors:
    """Tests for authentication/authorization errors."""

    def test_authentication_error(self):
        """Test AuthenticationError."""
        exc = AuthenticationError("Token expired")
        assert exc.error_code == "AUTHENTICATION_ERROR"
        assert exc.status_code == 401
        assert exc.message == "Token expired"

    def test_authorization_error(self):
        """Test AuthorizationError."""
        exc = AuthorizationError("Admin required")
        assert exc.error_code == "AUTHORIZATION_ERROR"
        assert exc.status_code == 403

    def test_default_messages(self):
        """Test default messages."""
        auth_exc = AuthenticationError()
        assert auth_exc.message == "Authentication required"
        authz_exc = AuthorizationError()
        assert authz_exc.message == "Insufficient permissions"


class TestNotFoundError:
    """Tests for NotFoundError."""

    def test_with_identifier(self):
        """Test NotFoundError with identifier."""
        exc = NotFoundError("User", "123")
        assert exc.error_code == "NOT_FOUND"
        assert exc.status_code == 404
        assert "123" in exc.message
        assert exc.details["resource"] == "User"
        assert exc.details["identifier"] == "123"

    def test_without_identifier(self):
        """Test NotFoundError without identifier."""
        exc = NotFoundError("Model")
        assert "Model" in exc.message
        assert exc.details == {"resource": "Model"}


class TestModelErrors:
    """Tests for model-related errors."""

    def test_model_error(self):
        """Test ModelError."""
        exc = ModelError("Inference failed", model_name="skin_classifier")
        assert exc.error_code == "MODEL_ERROR"
        assert exc.status_code == 500
        assert exc.details["model"] == "skin_classifier"

    def test_model_load_error(self):
        """Test ModelLoadError."""
        exc = ModelLoadError("/path/model.pth", "architecture mismatch")
        assert exc.error_code == "MODEL_LOAD_ERROR"
        assert exc.details["model_path"] == "/path/model.pth"
        assert exc.details["reason"] == "architecture mismatch"

    def test_model_inference_error(self):
        """Test ModelInferenceError."""
        exc = ModelInferenceError("CUDA out of memory", model_name="herb_classifier")
        assert exc.error_code == "MODEL_INFERENCE_ERROR"
        assert exc.details["model"] == "herb_classifier"

    def test_model_architecture_mismatch(self):
        """Test ModelArchitectureMismatchError."""
        expected = {"arch": "efficientnetv2_s", "classes": 22}
        actual = {"arch": "efficientnetv2_m", "classes": 1000}
        exc = ModelArchitectureMismatchError(expected, actual)
        assert exc.error_code == "MODEL_ARCHITECTURE_MISMATCH"
        assert exc.details["expected"] == expected
        assert exc.details["actual"] == actual


class TestLLMErrors:
    """Tests for LLM-related errors."""

    def test_llm_error(self):
        """Test LLMError."""
        exc = LLMError("Generation failed", model="llama3.2:3b", retryable=True)
        assert exc.error_code == "LLM_ERROR"
        assert exc.status_code == 503  # retryable -> 503
        assert exc.details["model"] == "llama3.2:3b"
        assert exc.details["retryable"] is True

    def test_llm_error_non_retryable(self):
        """Test LLMError non-retryable."""
        exc = LLMError("Invalid model", retryable=False)
        assert exc.status_code == 500  # non-retryable -> 500

    def test_llm_timeout_error(self):
        """Test LLMTimeoutError."""
        exc = LLMTimeoutError(timeout_seconds=120)
        assert exc.error_code == "LLM_TIMEOUT"
        assert exc.details["timeout_seconds"] == 120
        assert exc.details["retryable"] is True

    def test_llm_unavailable_error(self):
        """Test LLMUnavailableError."""
        exc = LLMUnavailableError("http://localhost:11434")
        assert exc.error_code == "LLM_UNAVAILABLE"
        assert exc.details["host"] == "http://localhost:11434"


class TestExternalServiceError:
    """Tests for external service errors."""

    def test_external_service_error(self):
        """Test ExternalServiceError."""
        exc = ExternalServiceError("Ollama", "Connection refused", retryable=True)
        assert exc.error_code == "EXTERNAL_SERVICE_ERROR"
        assert exc.status_code == 503
        assert exc.details["service"] == "Ollama"
        assert exc.details["retryable"] is True

    def test_external_service_error_non_retryable(self):
        """Test ExternalServiceError non-retryable."""
        exc = ExternalServiceError("PaymentAPI", "Invalid API key", retryable=False)
        assert exc.status_code == 502


class TestRateLimitError:
    """Tests for rate limiting errors."""

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        exc = RateLimitError(limit=60, window_seconds=60, retry_after=30)
        assert exc.error_code == "RATE_LIMIT_EXCEEDED"
        assert exc.status_code == 429
        assert exc.details["limit"] == 60
        assert exc.details["window_seconds"] == 60
        assert exc.details["retry_after_seconds"] == 30

    def test_rate_limit_error_no_retry_after(self):
        """Test RateLimitError without retry_after."""
        exc = RateLimitError(limit=10, window_seconds=10)
        assert "retry_after_seconds" not in exc.details


class TestConfigurationError:
    """Tests for configuration errors."""

    def test_configuration_error(self):
        """Test ConfigurationError."""
        exc = ConfigurationError("Missing API key", setting="OLLAMA_HOST")
        assert exc.error_code == "CONFIGURATION_ERROR"
        assert exc.status_code == 500
        assert exc.details["setting"] == "OLLAMA_HOST"