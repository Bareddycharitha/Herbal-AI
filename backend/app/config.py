"""
Backend Configuration

Centralized configuration management using Pydantic BaseSettings.
All settings can be overridden via environment variables.
"""

from pathlib import Path
from typing import Union

import logging
import warnings

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================
    # Application
    # ==========================================================
    app_name: str = "Herbal-AI API"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = Field(default="development", pattern="^(development|staging|production|test)$")

    # Secret key for JWT signing, sessions, CSRF, etc.
    # In production, this MUST be set via environment variable — no insecure default.
    secret_key: str = Field(
        default="dev-insecure-secret-key-change-in-production",
        min_length=1,
    )

    # ==========================================================
    # Server
    # ==========================================================
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # ==========================================================
    # CORS
    # ==========================================================
    cors_origins: Union[list[str], str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    cors_allow_credentials: bool = False
    cors_allow_methods: Union[list[str], str] = Field(default_factory=lambda: ["GET", "POST"])
    cors_allow_headers: Union[list[str], str] = Field(default_factory=lambda: ["*"])

    @field_validator("cors_origins", "cors_allow_methods", "cors_allow_headers", mode="before")
    @classmethod
    def parse_comma_separated_list(cls, v: Union[list[str], str]) -> list[str]:
        """Parse comma-separated string from environment variable into list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("allowed_mime_types", "allowed_extensions", mode="before")
    @classmethod
    def parse_comma_separated_list_v2(cls, v: Union[list[str], str]) -> list[str]:
        """Parse comma-separated string from environment variable into list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    # ==========================================================
    # File Upload
    # ==========================================================
    max_upload_size_mb: int = 10
    allowed_mime_types: list[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/jpg"]
    )
    allowed_extensions: list[str] = Field(default_factory=lambda: [".jpg", ".jpeg", ".png"])

    # ==========================================================
    # Paths (resolved from project root)
    # ==========================================================
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )

    @property
    def ai_root(self) -> Path:
        return self.project_root / "ai"

    @property
    def model_dir(self) -> Path:
        return self.ai_root / "checkpoints"

    @property
    def herb_model_dir(self) -> Path:
        return self.ai_root / "herb" / "checkpoints"

    @property
    def universal_model_dir(self) -> Path:
        return self.ai_root / "image_classifier" / "checkpoints"

    @property
    def data_dir(self) -> Path:
        return self.ai_root / "datasets"

    @property
    def results_dir(self) -> Path:
        return self.ai_root / "results"

    @property
    def disease_kb_path(self) -> Path:
        return self.data_dir / "knowledge_base" / "disease_knowledge_base.json"

    @property
    def herbal_kb_path(self) -> Path:
        return self.data_dir / "knowledge_base" / "herbal_knowledge_base.json"

    # ==========================================================
    # Model Settings
    # ==========================================================
    skin_disease_model_name: str = "tf_efficientnetv2_s"
    skin_disease_num_classes: int = 22
    skin_disease_image_size: int = 256
    skin_disease_high_confidence: float = 70.0
    skin_disease_medium_confidence: float = 40.0

    herb_model_name: str = "tf_efficientnetv2_s"
    herb_image_size: int = 256
    herb_confidence_threshold: float = 0.5
    herb_top_k: int = 3

    universal_model_name: str = "tf_efficientnetv2_s"
    universal_num_classes: int = 3
    universal_image_size: int = 224

    # ==========================================================
    # Device & Performance
    # ==========================================================
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda)$")
    num_workers: int = 2
    pin_memory: bool = True
    use_amp: bool = True

    @property
    def torch_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    # ==========================================================
    # Ollama LLM
    # ==========================================================
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: int = 120
    ollama_max_retries: int = 3
    ollama_retry_backoff: float = 1.0

    # ==========================================================
    # Rate Limiting
    # ==========================================================
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst: int = 10

    # ==========================================================
    # Logging
    # ==========================================================
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_format: str = Field(default="json", pattern="^(json|text)$")
    log_requests: bool = True

    # ==========================================================
    # Monitoring
    # ==========================================================
    enable_metrics: bool = True
    metrics_path: str = "/metrics"

    # ==========================================================
    # PDF Reports
    # ==========================================================
    reports_dir: str = "backend/app/services/pdf/reports"
    logo_path: str = "backend/app/services/pdf/assets/logo.png"

    # ==========================================================
    # Supabase (database only - no Supabase Auth)
    # ==========================================================
    supabase_url: str = Field(default="")
    supabase_anon_key: str = Field(default="")
    supabase_service_role_key: str = Field(default="")

    # ==========================================================
    # Clerk Authentication
    # ==========================================================
    clerk_publishable_key: str = Field(default="")
    clerk_secret_key: str = Field(default="")
    clerk_jwks_url: str = Field(default="")

    # ==========================================================
    # Production Validation
    # ==========================================================

    @model_validator(mode="after")
    def validate_production_settings(self):
        """Validate settings for production environment."""

        if self.environment == "production":
            # 1. Secret key must be non-empty and not the development default
            dev_default = "dev-insecure-secret-key-change-in-production"
            if not self.secret_key or self.secret_key == dev_default:
                raise ValueError(
                    "SECRET_KEY must be set to a non-default, non-empty value in production. "
                    "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
                )
            if len(self.secret_key) < 16:
                raise ValueError(
                    "SECRET_KEY must be at least 16 characters in production."
                )

            # 2. Debug must never be enabled in production
            if self.debug:
                raise ValueError(
                    "DEBUG mode must not be enabled when ENVIRONMENT=production."
                )

            # 3. CORS: warn if only localhost origins are configured in production
            localhost_origins = {
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            }
            origins = set(self.cors_origins)
            non_localhost = origins - localhost_origins
            if not non_localhost:
                logger.warning(
                    "Production environment has only localhost CORS origins configured. "
                    "Frontend clients will be unable to connect. Set CORS_ORIGINS to your "
                    "production domain(s)."
                )

            # 4. Log level should not be DEBUG in production
            if self.log_level == "DEBUG":
                logger.warning(
                    "DEBUG log level is enabled in production. "
                    "Consider using INFO or higher for security."
                )

        return self


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Dependency injection for FastAPI."""
    return settings