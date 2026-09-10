"""
Tests for Configuration
"""

import pytest
from backend.app.config import Settings

# Check if CUDA is available (torch may not be importable in all test envs)
try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False


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

    @pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available on this machine")
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

    def test_openrouter_defaults(self):
        """Test OpenRouter configuration defaults."""
        settings = Settings(_env_file=None, _env_file_encoding=None)
        assert settings.openrouter_model == "google/gemini-flash-1.5"
        assert settings.openrouter_chat_model == "google/gemini-flash-1.5"  # conftest env var
        assert settings.openrouter_timeout_seconds == 15
        assert settings.openrouter_max_retries == 1

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
        monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-pro")
        monkeypatch.setenv("OPENROUTER_CHAT_MODEL", "nvidia/nemotron-4:free")

        settings = Settings()
        assert settings.app_name == "Test API"
        assert settings.debug is True
        assert settings.port == 9000
        assert settings.openrouter_model == "google/gemini-pro"
        assert settings.openrouter_chat_model == "nvidia/nemotron-4:free"

    def test_cors_origins_from_env(self, monkeypatch):
        """Test CORS origins from environment."""
        monkeypatch.setenv("CORS_ORIGINS", "https://a.com,https://b.com")
        settings = Settings()
        assert settings.cors_origins == ["https://a.com", "https://b.com"]

    def test_supabase_url_env(self, monkeypatch):
        """Test Supabase URL from environment."""
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        settings = Settings()
        assert settings.supabase_url == "https://test.supabase.co"

    def test_supabase_anon_key_env(self, monkeypatch):
        """Test Supabase anon key from environment."""
        monkeypatch.setenv("SUPABASE_ANON_KEY", "test-anon-key")
        settings = Settings()
        assert settings.supabase_anon_key == "test-anon-key"

    def test_supabase_service_role_key_env(self, monkeypatch):
        """Test Supabase service role key from environment."""
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-key")
        settings = Settings()
        assert settings.supabase_service_role_key == "test-service-key"

    def test_clerk_publishable_key_env(self, monkeypatch):
        """Test Clerk publishable key from environment."""
        monkeypatch.setenv("CLERK_PUBLISHABLE_KEY", "pk_test_123")
        settings = Settings()
        assert settings.clerk_publishable_key == "pk_test_123"

    def test_clerk_secret_key_env(self, monkeypatch):
        """Test Clerk secret key from environment."""
        monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_456")
        settings = Settings()
        assert settings.clerk_secret_key == "sk_test_456"

    def test_clerk_jwks_url_env(self, monkeypatch):
        """Test Clerk JWKS URL from environment."""
        monkeypatch.setenv("CLERK_JWKS_URL", "https://test.clerk.accounts.dev/.well-known/jwks.json")
        settings = Settings()
        assert settings.clerk_jwks_url == "https://test.clerk.accounts.dev/.well-known/jwks.json"

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