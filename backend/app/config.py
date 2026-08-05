"""
Backend Configuration

Centralized configuration management using Pydantic BaseSettings.
All settings can be overridden via environment variables.
"""

from pathlib import Path
from typing import Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    # Authentication
    # ==========================================================
    secret_key: str = Field(
        default_factory=lambda: (_ for _ in ()).throw(
            RuntimeError(
                "SECRET_KEY must be set via the SECRET_KEY environment variable. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )
        ),
        min_length=32,
    )
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    cookie_secure: bool = False  # Set to True in production with HTTPS
    cookie_samesite: str = "lax"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Dependency injection for FastAPI."""
    return settings