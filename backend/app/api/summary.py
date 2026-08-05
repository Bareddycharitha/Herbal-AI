"""
LLM Summary API Endpoint

Generates AI summaries for skin disease predictions.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app.config import get_settings, Settings
from backend.app.dependencies import get_current_active_user
from backend.app.utils.logging import get_logger
from backend.app.exceptions import LLMError, ModelError

from ai.llm.summary_engine import SummaryEngine

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
    current_user = Depends(get_current_active_user),
):
    """Generate AI summary for skin disease prediction."""

    try:
        summary = engine.generate_summary(
            prediction=request.prediction,
            confidence=request.confidence,
            disease_information=request.disease_information,
            herbs=request.herbs,
        )

        return {
            "success": True,
            "summary": summary,
        }

    except Exception as e:
        logger = get_logger(__name__)
        logger.error("Summary generation failed", err=str(e), error_type=type(e).__name__)

        # Check if it's an Ollama-related error
        if "ollama" in str(e).lower() or "connection" in str(e).lower():
            raise LLMError(
                message=f"Failed to generate summary: {e}",
                model=settings.ollama_model,
                retryable=True,
            )

        raise ModelError(
            message=f"Summary generation failed: {e}",
        )