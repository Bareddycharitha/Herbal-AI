"""
Chatbot Engine

Context-aware chatbot for skin disease and herb queries
using the resilient async Ollama client.
"""

import asyncio
from typing import Any

from ai.llm.prompt_builder import build_chat_prompt
from ai.llm.openrouter_client import OpenRouterClient, get_openrouter_client


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
            context = f"""
Prediction:
Healthy Skin

Confidence:
{confidence:.2f}%

The uploaded image appears to show healthy skin.

Provide only preventive skincare advice.
Do not recommend disease treatments unless the user specifically asks general educational questions.
"""

        else:
            herb_text = ""

            if herbs:
                herb_text = "\n".join(
                    f"- {herb['name']}"
                    for herb in herbs
                )
            else:
                herb_text = "None"

            context = f"""
Prediction:
{prediction}

Confidence:
{confidence:.2f}%

Description:
{disease_information.get('description', '')}

Symptoms:
{', '.join(disease_information.get('symptoms', []))}

Causes:
{', '.join(disease_information.get('causes', []))}

Prevention:
{', '.join(disease_information.get('prevention', []))}

Recommended Herbs:

{herb_text}
"""

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
                self._ask_async(prompt, temperature=0.2, max_tokens=350), loop
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
                self._ask_async(prompt, temperature=0.2, max_tokens=350)
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
            context = f"""
Prediction:
Healthy Skin

Confidence:
{confidence:.2f}%

The uploaded image appears to show healthy skin.

Provide only preventive skincare advice.
Do not recommend disease treatments unless the user specifically asks general educational questions.
"""

        else:
            herb_text = ""

            if herbs:
                herb_text = "\n".join(
                    f"- {herb['name']}"
                    for herb in herbs
                )
            else:
                herb_text = "None"

            context = f"""
Prediction:
{prediction}

Confidence:
{confidence:.2f}%

Description:
{disease_information.get('description', '')}

Symptoms:
{', '.join(disease_information.get('symptoms', []))}

Causes:
{', '.join(disease_information.get('causes', []))}

Prevention:
{', '.join(disease_information.get('prevention', []))}

Recommended Herbs:

{herb_text}
"""

        # ======================================================
        # Build Prompt
        # ======================================================

        prompt = build_chat_prompt(context=context, question=question)

        # ======================================================
        # Generate Response
        # ======================================================

        result = await self._ask_async(prompt, temperature=0.2, max_tokens=350)

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
        max_tokens: int = 350,
    ) -> dict[str, Any]:
        """Internal async generation with error handling."""
        return await self.client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=True,
        )