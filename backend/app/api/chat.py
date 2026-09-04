"""
AI Chat API Endpoint

Context-aware chatbot for skin disease and herb queries.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app.config import get_settings, Settings
from backend.app.dependencies import get_optional_user
from backend.app.utils.logging import get_logger
from backend.app.exceptions import LLMError, ModelError

from ai.llm.chatbot_engine import ChatbotEngine

router = APIRouter(
    prefix="/chat",
    tags=["AI Chat"]
)

_chatbot = None


def get_chatbot() -> ChatbotEngine:
    """Get or create chatbot engine (lazy initialization for testing)."""
    global _chatbot
    if _chatbot is None:
        _chatbot = ChatbotEngine()
    return _chatbot


class ChatRequest(BaseModel):
    prediction: str
    confidence: float
    disease_information: dict
    herbs: list
    question: str


@router.post("/")
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    engine: ChatbotEngine = Depends(get_chatbot),
    current_user = Depends(get_optional_user),
):
    """Chat with AI assistant about skin disease or herb identification."""

    try:
        answer = engine.ask(
            prediction=request.prediction,
            confidence=request.confidence,
            disease_information=request.disease_information,
            herbs=request.herbs,
            question=request.question,
        )

        return {
            "success": True,
            "answer": answer,
        }

    except Exception as e:
        logger = get_logger(__name__)
        logger.error("Chat failed", err=str(e), error_type=type(e).__name__)

        # Check if it's an LLM-related error
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            raise LLMError(
                message=f"Failed to generate response: {e}",
                model=settings.openrouter_model,
                retryable=True,
            )

        raise ModelError(
            message=f"Chat failed: {e}",
        )