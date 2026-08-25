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
from backend.app.services.universal_classifier import init_classifier, shutdown_classifier, get_classifier
from backend.app.utils.logging import setup_logging, get_logger
from ai.llm.ollama_client import init_ollama_client, shutdown_ollama_client, get_ollama_client
from ai.utils.model_loader import check_checkpoint_status
from backend.app.api.herb import router as herb_router
from backend.app.api.prediction import router as prediction_router
from backend.app.api.summary import router as summary_router
from backend.app.api.chat import router as chat_router
from backend.app.api.report import router as report_router
from backend.app.api.auth import router as auth_router

from ai.config import RESULTS_DIR


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    # Startup
    logger.info("Starting Herbal-AI API", version=settings.app_version, environment=settings.environment)

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize classifier (universal model)
    init_classifier(settings)
    logger.info("Universal classifier initialized")

    # Eagerly initialize skin disease and herb models for readiness checks
    try:
        from ai.training.inference import get_inference
        get_inference()
        logger.info("Skin disease model loaded successfully")
    except Exception as e:
        logger.error("Skin disease model failed to load at startup", error=str(e))

    try:
        from ai.herb.inference import get_herb_predictor
        get_herb_predictor()
        logger.info("Herb model loaded successfully")
    except Exception as e:
        logger.error("Herb model failed to load at startup", error=str(e))

    # Initialize Ollama client (optional — does not block startup)
    try:
        await init_ollama_client()
        logger.info("Ollama client initialized")
    except Exception as e:
        logger.warning("Ollama client initialization failed (optional service)", error=str(e))

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
    - Universal model loaded successfully at startup
    - Skin disease model loaded successfully at startup
    - Herb model loaded successfully at startup
    - Required checkpoints available on disk
    - Ollama connectivity (optional — degradation only, does not block readiness)
    - Disk space available
    - Results directory writable
    """
    from backend.app.config import settings
    from ai.utils.model_loader import check_checkpoint_status
    import httpx

    checks = {}
    all_ready = True

    # ==========================================================
    # Required Model Load Status (actual runtime load)
    # ==========================================================

    # Universal model (loaded at startup via init_classifier)
    universal_loaded = False
    universal_info = {}
    try:
        classifier = get_classifier(settings)
        universal_loaded = classifier.model_load_status.is_ready
        universal_info = {
            "model_name": classifier.model_load_status.model_name,
            "checkpoint_path": classifier.model_load_status.checkpoint_path,
        }
    except Exception as e:
        universal_info = {"error": str(e)}

    universal_info["status"] = "ok" if universal_loaded else "not_loaded"
    universal_info["model_loaded"] = universal_loaded
    checks["universal_model"] = universal_info
    if not universal_loaded:
        all_ready = False

    # Skin disease model (eagerly loaded at startup via get_inference)
    skin_loaded = False
    skin_checkpoint = settings.model_dir / "best_model.pth"
    try:
        from ai.training.inference import get_inference
        get_inference()  # Eagerly load if not already loaded
        skin_loaded = True
    except Exception as e:
        # Check if checkpoint is at least loadable (file check)
        skin_loaded = False
        logger.warning("Skin disease model not loaded", error=str(e))

    # Also verify checkpoint file is loadable
    skin_status = check_checkpoint_status(skin_checkpoint, "skin_disease_classifier")
    checks["skin_disease_model"] = {
        "status": "ok" if (skin_loaded or skin_status.is_ready) else "not_ready",
        "checkpoint_path": str(skin_checkpoint),
        "file_exists": skin_checkpoint.exists(),
        "model_loaded": skin_loaded,
    }
    if not (skin_loaded or skin_status.is_ready):
        all_ready = False

    # Herb model (eagerly loaded at startup via get_herb_predictor)
    herb_loaded = False
    herb_checkpoint = settings.herb_model_dir / "best_model.pth"
    try:
        from ai.herb.inference import get_herb_predictor
        get_herb_predictor()  # Eagerly load if not already loaded
        herb_loaded = True
    except Exception as e:
        herb_loaded = False
        logger.warning("Herb model not loaded", error=str(e))

    # Also verify checkpoint file is loadable
    herb_status = check_checkpoint_status(herb_checkpoint, "herb_classifier")
    checks["herb_model"] = {
        "status": "ok" if (herb_loaded or herb_status.is_ready) else "not_ready",
        "checkpoint_path": str(herb_checkpoint),
        "file_exists": herb_checkpoint.exists(),
        "model_loaded": herb_loaded,
    }
    if not (herb_loaded or herb_status.is_ready):
        all_ready = False

    # ==========================================================
    # Ollama (optional — does not block readiness)
    # ==========================================================

    ollama_status = "unavailable"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_host}/api/tags")
            ollama_ok = response.status_code == 200
            ollama_status = "ok" if ollama_ok else "unreachable"
    except Exception as e:
        ollama_status = "error"

    checks["ollama"] = {
        "status": ollama_status,
        "host": settings.ollama_host,
        "model": settings.ollama_model,
        "required": False,
        "impact": "AI summaries will use fallback templates. ML predictions remain fully functional.",
    }

    # ==========================================================
    # Disk Space
    # ==========================================================

    try:
        import shutil
        total, used, free = shutil.disk_usage(settings.project_root)
        free_gb = free / (1024**3)
        checks["disk"] = {
            "status": "ok" if free_gb > 1 else "low",
            "free_gb": round(free_gb, 2),
        }
        if free_gb <= 1:
            all_ready = False
    except Exception as e:
        checks["disk"] = {
            "status": "error",
            "error": str(e),
        }

    # ==========================================================
    # Results Directory Writable
    # ==========================================================

    try:
        test_file = RESULTS_DIR / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        checks["results_dir"] = {"status": "writable"}
    except Exception as e:
        checks["results_dir"] = {"status": "error", "error": str(e)}
        all_ready = False

    status_code = 200 if all_ready else 503

    response_data = {
        "status": "ready" if all_ready else "not_ready",
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