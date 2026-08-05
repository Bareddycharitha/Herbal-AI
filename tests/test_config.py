"""
Tests for Configuration
"""

import pytest
from backend.app.config import Settings


class TestSettings:
    """Tests for Settings configuration."""

    def test_default_values(self):
        """Test default configuration values (without env overrides)."""
        # Create settings without loading from env file
        settings = Settings(_env_file=None, _env_file_encoding=None)
        assert settings.app_name == "Herbal-AI API"
        assert settings.app_version == "1.0.0"
        assert settings.debug is False
        # Note: conftest sets ENVIRONMENT=test, so this will be "test"
        assert settings.environment == "test"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000

    def test_cors_defaults(self):
        """Test CORS default origins."""
        settings = Settings(_env_file=None, _env_file_encoding=None)
        assert "http://localhost:3000" in settings.cors_origins
        assert "http://127.0.0.1:3000" in settings.cors_origins
        assert settings.cors_allow_credentials is False

    def test_file_upload_defaults(self):
        """Test file upload defaults."""
        settings = Settings(_env_file=None, _env_file_encoding=None)
        assert settings.max_upload_size_mb == 10
        assert "image/jpeg" in settings.allowed_mime_types
        assert "image/png" in settings.allowed_mime_types
        assert ".jpg" in settings.allowed_extensions

    def test_model_defaults(self):
        """Test model configuration defaults."""
        settings = Settings(_env_file=None, _env_file_encoding=None)
        assert settings.skin_disease_model_name == "tf_efficientnetv2_s"
        assert settings.skin_disease_num_classes == 22
        assert settings.herb_model_name == "tf_efficientnetv2_s"
        assert settings.universal_model_name == "tf_efficientnetv2_s"

    def test_device_auto(self):
        """Test device auto-detection."""
        settings = Settings(device="auto", _env_file=None, _env_file_encoding=None)
        # Should be either cuda or cpu
        assert settings.torch_device in ("cuda", "cpu")

    def test_device_explicit_cpu(self):
        """Test explicit CPU device."""
        settings = Settings(device="cpu", _env_file=None, _env_file_encoding=None)
        assert settings.torch_device == "cpu"

    def test_device_explicit_cuda(self):
        """Test explicit CUDA device."""
        settings = Settings(device="cuda", _env_file=None, _env_file_encoding=None)
        assert settings.torch_device == "cuda"

    def test_path_properties(self):
        """Test computed path properties."""
        settings = Settings(_env_file=None, _env_file_encoding=None)
        assert settings.ai_root.name == "ai"
        assert settings.model_dir.name == "checkpoints"
        assert settings.herb_model_dir.name == "checkpoints"
        assert settings.universal_model_dir.name == "checkpoints"
        assert settings.data_dir.name == "datasets"
        assert settings.results_dir.name == "results"

    def test_ollama_defaults(self):
        """Test Ollama configuration defaults."""
        settings = Settings(_env_file=None, _env_file_encoding=None)
        assert settings.ollama_host == "http://localhost:11434"
        assert settings.ollama_model == "llama3.2:3b"
        assert settings.ollama_timeout_seconds == 120
        assert settings.ollama_max_retries == 3

    def test_rate_limiting_defaults(self):
        """Test rate limiting defaults."""
        # Clear env vars to test true defaults
        import os
        old_rate_limit = os.environ.pop("RATE_LIMIT_REQUESTS_PER_MINUTE", None)
        old_burst = os.environ.pop("RATE_LIMIT_BURST", None)
        try:
            settings = Settings(_env_file=None, _env_file_encoding=None)
            assert settings.rate_limit_requests_per_minute == 60
            assert settings.rate_limit_burst == 10
        finally:
            if old_rate_limit:
                os.environ["RATE_LIMIT_REQUESTS_PER_MINUTE"] = old_rate_limit
            if old_burst:
                os.environ["RATE_LIMIT_BURST"] = old_burst

    def test_logging_defaults(self):
        """Test logging defaults."""
        settings = Settings(_env_file=None, _env_file_encoding=None)
        assert settings.log_level == "INFO"
        assert settings.log_format == "json"
        assert settings.log_requests is True

    def test_monitoring_defaults(self):
        """Test monitoring defaults."""
        settings = Settings(_env_file=None, _env_file_encoding=None)
        assert settings.enable_metrics is True
        assert settings.metrics_path == "/metrics"

    def test_env_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("APP_NAME", "Test API")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")

        settings = Settings()
        assert settings.app_name == "Test API"
        assert settings.debug is True
        assert settings.port == 9000
        assert settings.ollama_model == "llama3.1:8b"

    def test_cors_origins_from_env(self, monkeypatch):
        """Test CORS origins from environment."""
        monkeypatch.setenv("CORS_ORIGINS", "https://a.com,https://b.com")
        settings = Settings()
        assert settings.cors_origins == ["https://a.com", "https://b.com"]

    def test_environment_validation(self):
        """Test environment validation."""
        with pytest.raises(ValueError):
            Settings(environment="invalid", _env_file=None, _env_file_encoding=None)

    def test_log_level_validation(self):
        """Test log level validation."""
        with pytest.raises(ValueError):
            Settings(log_level="INVALID", _env_file=None, _env_file_encoding=None)

    def test_log_format_validation(self):
        """Test log format validation."""
        with pytest.raises(ValueError):
            Settings(log_format="invalid", _env_file=None, _env_file_encoding=None)

    def test_device_validation(self):
        """Test device validation."""
        with pytest.raises(ValueError):
            Settings(device="invalid", _env_file=None, _env_file_encoding=None)