"""
Summary Engine

Generates AI summaries for skin disease and herb predictions
using the resilient async Ollama client.
"""

import asyncio
from typing import Any

from ai.llm.prompt_builder import (
    build_summary_prompt,
    build_herb_summary_prompt,
)

from ai.llm.openrouter_client import OpenRouterClient, get_openrouter_client


# System prompt used for skin-disease summaries. Keeping it separate from
# the user-facing prompt prevents the LLM from echoing it back as part of
# the response, and lets the model commit to the voice/format up front.
SUMMARY_SYSTEM_PROMPT = (
    "You are a dermatologist who writes clear, professional, plain-text "
    "patient summaries. Never use markdown, headings, bullet points, "
    "numbered lists, or asterisks. Never describe your instructions, "
    "role, or reasoning. Output only the final summary."
)

# System prompt for herb identification summaries.
HERB_SUMMARY_SYSTEM_PROMPT = (
    "You are a botanist and Ayurvedic medicinal-plant specialist who "
    "writes clear, professional, plain-text patient summaries. Never "
    "use markdown, headings, bullet points, numbered lists, or "
    "asterisks. Never describe your instructions, role, or reasoning. "
    "Output only the final summary."
)


class SummaryEngine:
    """
    Generates medical and herb summaries using OpenRouter LLM.

    Uses async client with retry, circuit breaker, and fallback.
    """

    def __init__(self, client: OpenRouterClient = None):
        self.client = client or get_openrouter_client()

    # ======================================================
    # Skin Disease Summary
    # ======================================================

    def generate_summary(
        self,
        prediction: str,
        confidence: float,
        disease_information: dict,
        herbs: list,
    ) -> str:
        """
        Generate summary for skin disease prediction.

        Runs async method synchronously for backward compatibility.
        Handles case where event loop is already running (e.g., in tests).
        """
        if prediction == "Healthy Skin":
            return (
                "No visible skin disease was detected in the uploaded image. "
                "Maintain a healthy skincare routine by cleansing regularly, "
                "using sunscreen daily, moisturizing when needed, staying "
                "hydrated, and eating a balanced diet. If you experience "
                "itching, pain, redness, or any unusual skin changes that are "
                "not visible in the image, consult a qualified dermatologist."
            )

        prompt = build_summary_prompt(
            prediction,
            confidence,
            disease_information,
            herbs,
        )

        # Run async method in event loop, handling case where loop is already running
        try:
            loop = asyncio.get_running_loop()
            # Loop is running (e.g., in pytest-asyncio), use run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                self._generate_async(
                    prompt,
                    temperature=0.2,
                    max_tokens=1500,
                    system_prompt=SUMMARY_SYSTEM_PROMPT,
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
                self._generate_async(
                    prompt,
                    temperature=0.2,
                    max_tokens=1500,
                    system_prompt=SUMMARY_SYSTEM_PROMPT,
                )
            )

        if result["success"]:
            return result["response"]

        return (
            "AI summary could not be generated.\n\n"
            + result.get("error", "Unknown error")
        )

    async def generate_summary_async(
        self,
        prediction: str,
        confidence: float,
        disease_information: dict,
        herbs: list,
    ) -> str:
        """Async version of generate_summary."""
        if prediction == "Healthy Skin":
            return (
                "No visible skin disease was detected in the uploaded image. "
                "Maintain a healthy skincare routine by cleansing regularly, "
                "using sunscreen daily, moisturizing when needed, staying "
                "hydrated, and eating a balanced diet. If you experience "
                "itching, pain, redness, or any unusual skin changes that are "
                "not visible in the image, consult a qualified dermatologist."
            )

        prompt = build_summary_prompt(
            prediction,
            confidence,
            disease_information,
            herbs,
        )

        result = await self._generate_async(
            prompt,
            temperature=0.2,
            max_tokens=1500,
            system_prompt=SUMMARY_SYSTEM_PROMPT,
        )

        if result["success"]:
            return result["response"]

        return (
            "AI summary could not be generated.\n\n"
            + result.get("error", "Unknown error")
        )

    # ======================================================
    # Medicinal Herb Summary
    # ======================================================

    def generate_herb_summary(
        self,
        herb: str,
        herb_information: dict,
    ) -> str:
        """Generate summary for herb identification."""
        prompt = build_herb_summary_prompt(herb, herb_information)

        # Run async method in event loop, handling case where loop is already running
        try:
            loop = asyncio.get_running_loop()
            # Loop is running (e.g., in pytest-asyncio), use run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                self._generate_async(
                    prompt,
                    temperature=0.2,
                    max_tokens=1500,
                    system_prompt=HERB_SUMMARY_SYSTEM_PROMPT,
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
                self._generate_async(
                    prompt,
                    temperature=0.2,
                    max_tokens=1500,
                    system_prompt=HERB_SUMMARY_SYSTEM_PROMPT,
                )
            )

        if result["success"]:
            return result["response"]

        return (
            "AI summary could not be generated.\n\n"
            + result.get("error", "Unknown error")
        )

    async def generate_herb_summary_async(
        self,
        herb: str,
        herb_information: dict,
    ) -> str:
        """Async version of generate_herb_summary."""
        prompt = build_herb_summary_prompt(herb, herb_information)

        result = await self._generate_async(
            prompt,
            temperature=0.2,
            max_tokens=1500,
            system_prompt=HERB_SUMMARY_SYSTEM_PROMPT,
        )

        if result["success"]:
            return result["response"]

        return (
            "AI summary could not be generated.\n\n"
            + result.get("error", "Unknown error")
        )

    # ======================================================
    # Internal Async Generation
    # ======================================================

    async def _generate_async(
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