"""
Herbal-AI FastAPI Application

Main application entry point with all middleware, routes, and configuration.
"""

import os
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings, get_settings
from backend.app.middleware import RequestIDMiddleware, LoggingMiddleware, RateLimitMiddleware
from backend.app.exception_handlers import register_exception_handlers
from backend.app.services.universal_classifier import init_classifier, shutdown_classifier
from backend.app.utils.logging import setup_logging, get_logger
from ai.llm.ollama_client import init_ollama_client, shutdown_ollama_client
from backend.app.api.herb import router as herb_router
from backend.app.api.prediction import router as prediction_router
from backend.app.api.summary import router as summary_router
from backend.app.api.chat import router as chat_router
from backend.app.api.report import router as report_router
from backend.app.api.auth import router as auth_router
from backend.app.database.database import init_db, shutdown_db

from ai.config import RESULTS_DIR


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    # Startup
    logger.info("Starting Herbal-AI API", version=settings.app_version, environment=settings.environment)

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Initialize classifier
    init_classifier(settings)
    logger.info("Universal classifier initialized")

    # Initialize Ollama client
    await init_ollama_client()
    logger.info("Ollama client initialized")

    # Log configuration (non-sensitive)
    logger.info(
        "Configuration loaded",
        host=settings.host,
        port=settings.port,
        device=settings.torch_device,
        ollama_host=settings.ollama_host,
        ollama_model=settings.ollama_model,
    )

    yield

    # Shutdown
    await shutdown_ollama_client()
    shutdown_classifier()
    await shutdown_db()
    logger.info("Shutting down Herbal-AI API")


# ==========================================================
# FastAPI App
# ==========================================================

app = FastAPI(
    title=settings.app_name,
    description="AI Powered Herbal Recommendation System",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    openapi_url="/openapi.json" if settings.environment != "production" else None,
)

# ==========================================================
# Middleware (order matters - first added = outermost)
# ==========================================================

# 1. Request ID (outermost - needs to wrap everything)
app.add_middleware(RequestIDMiddleware)

# 2. Logging
app.add_middleware(LoggingMiddleware, log_requests=settings.log_requests)

# 3. Rate Limiting
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=settings.rate_limit_requests_per_minute,
    burst=settings.rate_limit_burst,
)

# 4. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# ==========================================================
# Exception Handlers
# ==========================================================
register_exception_handlers(app)

# ==========================================================
# Static Files (Grad-CAM Images)
# ==========================================================
app.mount(
    "/results",
    StaticFiles(directory=str(RESULTS_DIR)),
    name="results",
)

# ==========================================================
# API Routes (versioned)
# ==========================================================
API_PREFIX = "/api/v1"

app.include_router(prediction_router, prefix=API_PREFIX, tags=["Prediction"])
app.include_router(summary_router, prefix=API_PREFIX, tags=["LLM Summary"])
app.include_router(chat_router, prefix=API_PREFIX, tags=["AI Chat"])
app.include_router(report_router, prefix=API_PREFIX, tags=["PDF Report"])
app.include_router(herb_router, prefix=API_PREFIX, tags=["Herb Identification"])
app.include_router(auth_router, prefix=API_PREFIX, tags=["Authentication"])

# ==========================================================
# Health & Readiness Endpoints
# ==========================================================

@app.get("/health", tags=["Health"], include_in_schema=False)
async def health_check():
    """
    Liveness probe - returns 200 if process is alive.

    Used by load balancers to detect if the process should be restarted.
    """
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/ready", tags=["Health"], include_in_schema=False)
async def readiness_check():
    """
    Readiness probe - checks if service can handle requests.

    Verifies:
    - Model checkpoints exist
    - Ollama is reachable
    - Disk space available
    """
    from backend.app.config import settings
    import httpx

    checks = {}
    all_healthy = True

    # Check model files
    model_checks = {
        "skin_disease_model": settings.model_dir / "best_model.pth",
        "herb_model": settings.herb_model_dir / "best_model.pth",
        "universal_model": settings.universal_model_dir / "best_model.pth",
    }

    for name, path in model_checks.items():
        exists = path.exists()
        checks[name] = {
            "status": "ok" if exists else "missing",
            "path": str(path),
        }
        if not exists:
            all_healthy = False

    # Check Ollama connectivity
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_host}/api/tags")
            ollama_ok = response.status_code == 200
            checks["ollama"] = {
                "status": "ok" if ollama_ok else "unreachable",
                "host": settings.ollama_host,
                "model": settings.ollama_model,
            }
            if not ollama_ok:
                all_healthy = False
    except Exception as e:
        checks["ollama"] = {
            "status": "error",
            "host": settings.ollama_host,
            "error": str(e),
        }
        all_healthy = False

    # Check disk space (warn if < 1GB free)
    try:
        import shutil
        total, used, free = shutil.disk_usage(settings.project_root)
        free_gb = free / (1024**3)
        checks["disk"] = {
            "status": "ok" if free_gb > 1 else "low",
            "free_gb": round(free_gb, 2),
        }
        if free_gb <= 1:
            all_healthy = False
    except Exception as e:
        checks["disk"] = {
            "status": "error",
            "error": str(e),
        }

    # Check results directory writable
    try:
        test_file = RESULTS_DIR / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        checks["results_dir"] = {"status": "writable"}
    except Exception as e:
        checks["results_dir"] = {"status": "error", "error": str(e)}
        all_healthy = False

    status_code = 200 if all_healthy else 503

    response_data = {
        "status": "ready" if all_healthy else "not_ready",
        "service": settings.app_name,
        "version": settings.app_version,
        "checks": checks,
    }

    return Response(
        content=json.dumps(response_data),
        media_type="application/json",
        status_code=status_code,
    )


# ==========================================================
# Metrics Endpoint (Prometheus)
# ==========================================================

if settings.enable_metrics:
    from prometheus_client import make_asgi_app

    metrics_app = make_asgi_app()
    app.mount(settings.metrics_path, metrics_app)


# ==========================================================
# Root
# ==========================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs" if settings.environment != "production" else "disabled",
        "health": "/health",
        "ready": "/ready",
        "api": API_PREFIX,
    }


# ==========================================================
# CLI Entry Point
# ==========================================================

if __name__ == "__main__":
    import uvicorn

    # Setup logging before starting
    setup_logging(settings.log_level, settings.log_format)

    uvicorn.run(
        "backend.app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.debug,
        log_config=None,  # We use structlog
    )