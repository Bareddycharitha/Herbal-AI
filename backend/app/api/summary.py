"""
LLM Summary API Endpoint

Generates AI summaries for skin disease predictions.

This endpoint is intentionally decoupled from POST /api/v1/predict/ so the
prediction response can return as soon as inference is complete without
waiting on the OpenRouter network call. The frontend calls this endpoint
separately after rendering the prediction.
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from backend.app.config import get_settings, Settings
from backend.app.dependencies import get_optional_user
from backend.app.utils.logging import get_logger
from backend.app.exceptions import LLMError, ModelError

from ai.llm.summary_engine import SummaryEngine
from ai.llm.prompt_builder import build_summary_prompt

router = APIRouter(
    prefix="/summary",
    tags=["LLM Summary"]
)

_summary_engine = None


def get_summary_engine() -> SummaryEngine:
    """Get or create summary engine (lazy initialization for testing)."""
    global _summary_engine
    if _summary_engine is None:
        _summary_engine = SummaryEngine()
    return _summary_engine


# Hard upper bound for the AI summary request. The OpenRouter client
# already has its own timeout + retry + circuit breaker, but we cap the
# *whole* call here so a misbehaving upstream can never hang the
# request thread for long. Five seconds is enough for the free-tier
# model to return a useful response in the happy path; anything that
# takes longer than that should fall back to the deterministic
# FALLBACK_SUMMARY below rather than block the user.
SUMMARY_TIMEOUT_SECONDS = 30.0

FALLBACK_SUMMARY = (
    "AI medical summary is temporarily unavailable. "
    "The prediction above is still accurate and based on the model's analysis "
    "of your image. For medical advice, please consult a qualified healthcare "
    "professional."
)


class SummaryRequest(BaseModel):
    prediction: str
    confidence: float
    disease_information: dict
    herbs: list


@router.post("/")
async def generate_summary(
    request: SummaryRequest,
    settings: Settings = Depends(get_settings),
    engine: SummaryEngine = Depends(get_summary_engine),
    current_user = Depends(get_optional_user),
):
    """
    Generate AI summary for a skin disease prediction.

    This is a protected endpoint — the caller must present a valid Clerk JWT
    via the ``Authorization: Bearer <token>`` header. Authentication is
    enforced by ``get_current_active_user``; do not remove it.

    The endpoint never raises on OpenRouter failure: it returns HTTP 200 with
    ``success=False`` and a friendly fallback ``summary`` so the client can
    still render the prediction.
    """

    logger = get_logger(__name__)

    # Healthy-skin path: deterministic, no LLM call, always succeeds.
    if request.prediction == "Healthy Skin":
        return {
            "success": True,
            "summary": (
                "No visible skin disease was detected in the uploaded image. "
                "Maintain a healthy skincare routine by cleansing regularly, "
                "using sunscreen daily, moisturizing when needed, staying "
                "hydrated, and eating a balanced diet. If you experience "
                "itching, pain, redness, or any unusual skin changes that are "
                "not visible in the image, consult a qualified dermatologist."
            ),
        }

    # If OpenRouter is not configured, return the fallback immediately
    # rather than burning the request budget on a guaranteed failure.
    if not settings.openrouter_api_key:
        logger.warning("OpenRouter API key not configured; returning fallback summary")
        return {
            "success": False,
            "summary": FALLBACK_SUMMARY,
            "error": "AI service not configured",
        }

    prompt = build_summary_prompt(
        request.prediction,
        request.confidence,
        request.disease_information,
        request.herbs,
    )

    # Run the async LLM call in a worker thread so the event loop stays
    # responsive, and cap the wait with a hard timeout.
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(
                _run_summary_sync,
                engine,
                prompt,
            ),
            timeout=SUMMARY_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Summary generation exceeded hard timeout",
            timeout_seconds=SUMMARY_TIMEOUT_SECONDS,
        )
        return {
            "success": False,
            "summary": FALLBACK_SUMMARY,
            "error": "timeout",
        }
    except Exception as e:
        logger.error(
            "Summary generation failed",
            err=str(e),
            error_type=type(e).__name__,
        )
        return {
            "success": False,
            "summary": FALLBACK_SUMMARY,
            "error": type(e).__name__,
        }

    if result.get("success"):
        return {
            "success": True,
            "summary": result.get("response", ""),
        }

    # LLM call returned without raising but reported failure (e.g. circuit
    # breaker open, missing key inside the client, or a fallback response).
    logger.warning(
        "Summary engine returned failure",
        error=result.get("error"),
    )
    return {
        "success": False,
        "summary": FALLBACK_SUMMARY,
        "error": result.get("error") or "service_unavailable",
    }


def _run_summary_sync(engine: SummaryEngine, prompt: str) -> dict:
    """
    Drive the async summary generation from a worker thread.

    ``SummaryEngine._generate_async`` is an ``async`` coroutine; we run it on
    a fresh event loop inside the worker thread so the request handler's
    event loop is never blocked.
    """
    import concurrent.futures

    async def _runner() -> dict:
        return await engine._generate_async(prompt, temperature=0.3, max_tokens=250)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Should be unreachable: run_in_threadpool runs the callable in a
        # thread without an active loop, but guard anyway.
        future = asyncio.run_coroutine_threadsafe(_runner(), loop)
        return future.result(timeout=SUMMARY_TIMEOUT_SECONDS)

    return asyncio.run(_runner())