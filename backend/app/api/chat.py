"""
AI Chat API Endpoint

Context-aware chatbot for skin disease and herb queries.
"""

import asyncio

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
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


# Hard upper bound for the AI chat request. Mirrors the summary endpoint
# (backend/app/api/summary.py). The OpenRouter client already has its own
# timeout + retry + circuit breaker, but we cap the *whole* call here so a
# misbehaving upstream can never hang the request thread for long.
CHAT_TIMEOUT_SECONDS = 60.0


@router.post("/")
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    engine: ChatbotEngine = Depends(get_chatbot),
    current_user = Depends(get_optional_user),
):
    """Chat with AI assistant about skin disease or herb identification.

    Runs the synchronous ``engine.ask`` on a worker thread so the
    request handler's event loop stays responsive. The previous
    implementation called ``engine.ask`` directly inside the event loop,
    which caused a 60-second hang: ``engine.ask`` tries to use
    ``asyncio.run_coroutine_threadsafe`` on the *current* loop (which
    is not allowed — that function only works for cross-thread
    dispatch) and ends up timing out, surfacing as a 500 with an
    empty ``Chat failed:`` message.
    """
    logger = get_logger(__name__)

    try:
        answer = await asyncio.wait_for(
            run_in_threadpool(
                engine.ask,
                prediction=request.prediction,
                confidence=request.confidence,
                disease_information=request.disease_information,
                herbs=request.herbs,
                question=request.question,
            ),
            timeout=CHAT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Chat generation exceeded hard timeout",
            timeout_seconds=CHAT_TIMEOUT_SECONDS,
        )
        raise LLMError(
            message="Chat request timed out. Please try again.",
            model=settings.openrouter_chat_model,
            retryable=True,
        )
    except Exception as e:
        logger.error(
            "Chat failed",
            err=str(e),
            error_type=type(e).__name__,
        )

        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            raise LLMError(
                message=f"Failed to generate response: {e}",
                model=settings.openrouter_chat_model,
                retryable=True,
            )

        raise ModelError(
            message=f"Chat failed: {e}",
        )

    return {
        "success": True,
        "answer": answer,
    }