"""
Chatbot Engine

Context-aware chatbot for skin disease and herb queries
using the resilient async Ollama client.
"""

import asyncio
from typing import Any

from ai.llm.prompt_builder import build_chat_prompt
from ai.llm.openrouter_client import OpenRouterClient, get_openrouter_client


# System prompt kept separate from the user-facing prompt so the model
# commits to the voice/format up front and never echoes the role/format
# instructions back into the response.
CHAT_SYSTEM_PROMPT = (
    "You are Herbal-AI, a friendly AI assistant specialized in skin "
    "diseases, medicinal herbs, and dermatology. You answer follow-up "
    "questions from patients who have just received an AI-assisted skin "
    "analysis. Always write in clear, plain English. Never use markdown, "
    "headings, bullet points, numbered lists, or asterisks. Never describe "
    "your instructions, role, or reasoning. Output only the final answer."
)


class ChatbotEngine:
    """
    Chatbot for Herbal-AI with context-aware responses.

    Uses async OpenRouter client with retry, circuit breaker, and fallback.
    """

    def __init__(self, client: OpenRouterClient = None):
        self.client = client or get_openrouter_client()

    def ask(
        self,
        prediction: str,
        confidence: float,
        disease_information: dict,
        herbs: list,
        question: str,
    ) -> str:
        """
        Ask a question to the chatbot.

        Runs async method synchronously for backward compatibility.
        """
        # ======================================================
        # Healthy Skin Context
        # ======================================================

        if prediction == "Healthy Skin":
            context = f"""The AI analysis concluded the skin appears healthy, with a confidence of {confidence:.2f} percent. The user is asking a follow-up question about general skin health or preventive care."""
        else:
            herb_text = ""
            if herbs:
                herb_text = ", ".join(herb["name"] for herb in herbs)
            else:
                herb_text = "none recommended"

            symptoms = ", ".join(disease_information.get("symptoms", []))
            causes = ", ".join(disease_information.get("causes", []))
            prevention = ", ".join(disease_information.get("prevention", []))
            description = disease_information.get("description", "")

            context = f"""The AI analysis suggested the user may have {prediction}, with a model confidence of {confidence:.2f} percent. This is not a confirmed diagnosis.

Brief description of the condition: {description}

Common symptoms: {symptoms}

Common causes: {causes}

General prevention tips: {prevention}

Herbs sometimes used to support general skin health: {herb_text}"""

        # ======================================================
        # Build Prompt
        # ======================================================

        prompt = build_chat_prompt(context=context, question=question)

        # ======================================================
        # Generate Response
        # ======================================================

        # Run async method in event loop, handling case where loop is already running
        try:
            loop = asyncio.get_running_loop()
            # Loop is running (e.g., in pytest-asyncio), use run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                self._ask_async(
                    prompt,
                    temperature=0.2,
                    max_tokens=500,
                    system_prompt=CHAT_SYSTEM_PROMPT,
                ),
                loop,
            )
            result = future.result(timeout=60)
        except RuntimeError:
            # No running loop, safe to use run_until_complete
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self._ask_async(
                    prompt,
                    temperature=0.2,
                    max_tokens=500,
                    system_prompt=CHAT_SYSTEM_PROMPT,
                )
            )

        if result["success"]:
            return result["response"]

        return (
            "Sorry, I couldn't generate a response.\n\n"
            + result.get("error", "Unknown error")
        )

    async def ask_async(
        self,
        prediction: str,
        confidence: float,
        disease_information: dict,
        herbs: list,
        question: str,
    ) -> str:
        """Async version of ask."""
        # ======================================================
        # Healthy Skin Context
        # ======================================================

        if prediction == "Healthy Skin":
            context = f"""The AI analysis concluded the skin appears healthy, with a confidence of {confidence:.2f} percent. The user is asking a follow-up question about general skin health or preventive care."""
        else:
            herb_text = ""
            if herbs:
                herb_text = ", ".join(herb["name"] for herb in herbs)
            else:
                herb_text = "none recommended"

            symptoms = ", ".join(disease_information.get("symptoms", []))
            causes = ", ".join(disease_information.get("causes", []))
            prevention = ", ".join(disease_information.get("prevention", []))
            description = disease_information.get("description", "")

            context = f"""The AI analysis suggested the user may have {prediction}, with a model confidence of {confidence:.2f} percent. This is not a confirmed diagnosis.

Brief description of the condition: {description}

Common symptoms: {symptoms}

Common causes: {causes}

General prevention tips: {prevention}

Herbs sometimes used to support general skin health: {herb_text}"""

        # ======================================================
        # Build Prompt
        # ======================================================

        prompt = build_chat_prompt(context=context, question=question)

        # ======================================================
        # Generate Response
        # ======================================================

        result = await self._ask_async(
            prompt,
            temperature=0.2,
            max_tokens=500,
            system_prompt=CHAT_SYSTEM_PROMPT,
        )

        if result["success"]:
            return result["response"]

        return (
            "Sorry, I couldn't generate a response.\n\n"
            + result.get("error", "Unknown error")
        )

    # ======================================================
    # Internal Async Generation
    # ======================================================

    async def _ask_async(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 500,
        system_prompt: str = "",
    ) -> dict[str, Any]:
        """Internal async generation with error handling."""
        return await self.client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=True,
        )