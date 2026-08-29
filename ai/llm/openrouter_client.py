"""
Resilient OpenRouter Client

Async HTTP client for OpenRouter with:
- Exponential backoff retry
- Circuit breaker pattern
- Configurable timeouts
- Fallback support
- Structured logging
- Response caching
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import httpx
import structlog

from backend.app.config import settings


logger = structlog.get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for external service calls."""

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 30.0

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        # Check if timeout has passed to transition from OPEN to HALF_OPEN
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.timeout_seconds:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info("Circuit breaker transitioned to HALF_OPEN")
        return self._state

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker CLOSED - service recovered")
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0  # Reset on success

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning("Circuit breaker OPEN - service still failing")
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker OPEN - threshold exceeded")

    def can_execute(self) -> bool:
        """Check if request can be executed."""
        return self.state != CircuitState.OPEN


class OpenRouterClient:
    """
    Async OpenRouter client with resilience patterns.

    Features:
    - Exponential backoff retry (configurable)
    - Circuit breaker
    - Request/response logging
    - Timeout handling
    - Health check
    """

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        timeout_seconds: int = None,
        max_retries: int = None,
        retry_backoff: float = None,
        http_referer: str = None,
        app_name: str = None,
    ):
        self.model = model or settings.openrouter_model
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.timeout_seconds = timeout_seconds or settings.openrouter_timeout_seconds
        self.max_retries = max_retries or settings.openrouter_max_retries
        self.retry_backoff = retry_backoff or settings.openrouter_retry_backoff
        self.http_referer = http_referer or settings.openrouter_http_referer
        self.app_name = app_name or settings.openrouter_app_name

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker()

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

        # Fallback cache for summaries
        self._summary_cache: dict[str, str] = {}
        self._cache_max_size = 100

        # Validate configuration
        if not self.api_key:
            logger.warning(
                "OpenRouter API key not configured. LLM features will use fallback responses."
            )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            if self.http_referer:
                headers["HTTP-Referer"] = self.http_referer
            if self.app_name:
                headers["X-Title"] = self.app_name

            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                headers=headers,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def health_check(self) -> bool:
        """Check if OpenRouter service is healthy."""
        if not self.api_key:
            logger.warning("OpenRouter API key not configured - skipping health check")
            return False

        try:
            client = await self._get_client()
            # OpenRouter doesn't have a specific health endpoint, so we'll make a minimal request
            # We'll check if we can reach the API with a simple request
            response = await client.get(
                f"{self.base_url}/models",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning("OpenRouter health check failed", error=str(e))
            return False

    def _get_cache_key(self, prompt: str, system_prompt: str = "") -> str:
        """Generate cache key for prompt."""
        import hashlib
        return hashlib.md5(f"{self.model}:{system_prompt}:{prompt}".encode()).hexdigest()[:16]

    def _get_cached_summary(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Get cached summary if available."""
        key = self._get_cache_key(prompt, system_prompt)
        return self._summary_cache.get(key)

    def _cache_summary(self, prompt: str, response: str, system_prompt: str = "") -> None:
        """Cache summary response."""
        if len(self._summary_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._summary_cache))
            del self._summary_cache[oldest_key]
        key = self._get_cache_key(prompt, system_prompt)
        self._summary_cache[key] = response

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 300,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """
        Generate response from OpenRouter with retry and circuit breaker.

        Returns:
            dict with keys: success, response, error (if failed), cached, fallback
        """
        # Check cache first
        if use_cache:
            cached = self._get_cached_summary(prompt, system_prompt)
            if cached:
                logger.info("Returning cached OpenRouter response")
                return {"success": True, "response": cached, "cached": True}

        # Check if API key is configured
        if not self.api_key:
            logger.warning("OpenRouter API key not configured - using fallback")
            return await self._fallback_response(prompt, system_prompt, "API key not configured")

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            logger.warning("Circuit breaker OPEN - using fallback")
            return await self._fallback_response(prompt, system_prompt, "Circuit breaker open")

        # Prepare request (OpenAI-compatible format)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                client = await self._get_client()

                logger.debug(
                    "Calling OpenRouter",
                    attempt=attempt + 1,
                    model=self.model,
                    prompt_length=len(prompt),
                )

                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout_seconds,
                )

                response.raise_for_status()
                result = response.json()

                # Success
                self.circuit_breaker.record_success()

                # Extract response text from OpenAI-compatible format
                response_text = ""
                if "choices" in result and len(result["choices"]) > 0:
                    message = result["choices"][0].get("message", {})
                    response_text = message.get("content", "").strip()

                # Cache successful response
                if use_cache and response_text:
                    self._cache_summary(prompt, response_text, system_prompt)

                logger.info(
                    "OpenRouter response received",
                    attempt=attempt + 1,
                    response_length=len(response_text),
                )

                return {
                    "success": True,
                    "response": response_text,
                    "cached": False,
                }

            except httpx.TimeoutException as e:
                last_error = f"Request timed out after {self.timeout_seconds}s"
                logger.warning("OpenRouter timeout", attempt=attempt + 1, error=str(e))

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text}"
                logger.warning("OpenRouter HTTP error", attempt=attempt + 1, error=last_error)

                # Don't retry on client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    break

            except httpx.RequestError as e:
                last_error = f"Request failed: {str(e)}"
                logger.warning("OpenRouter request error", attempt=attempt + 1, error=str(e))

            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error("OpenRouter unexpected error", attempt=attempt + 1, error=str(e))

            # Record failure for circuit breaker
            self.circuit_breaker.record_failure()

            # Exponential backoff before retry
            if attempt < self.max_retries:
                backoff = self.retry_backoff * (2 ** attempt)
                logger.info("Retrying OpenRouter request", backoff_seconds=backoff)
                await asyncio.sleep(backoff)

        # All retries exhausted
        logger.error("OpenRouter all retries exhausted", error=last_error)
        return await self._fallback_response(prompt, system_prompt, last_error)

    async def _fallback_response(
        self, prompt: str, system_prompt: str = "", error: str = None
    ) -> dict[str, Any]:
        """Provide fallback response when OpenRouter is unavailable."""
        # Try cache as last resort
        cached = self._get_cached_summary(prompt, system_prompt)
        if cached:
            logger.info("Using cached response as fallback")
            return {
                "success": True,
                "response": cached + "\n\n[Note: Generated from cache due to service unavailability]",
                "cached": True,
                "fallback": True,
            }

        # Return degraded response
        fallback_msg = (
            "AI service is temporarily unavailable. "
            "Please try again later or consult a healthcare professional for medical advice."
        )

        if error:
            fallback_msg += f"\n\nTechnical details: {error}"

        return {
            "success": False,
            "response": fallback_msg,
            "error": error or "Service unavailable",
            "fallback": True,
        }


# Global client instance
_openrouter_client: Optional[OpenRouterClient] = None


def get_openrouter_client() -> OpenRouterClient:
    """Get or create global OpenRouter client."""
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = OpenRouterClient()
    return _openrouter_client


async def init_openrouter_client() -> OpenRouterClient:
    """Initialize OpenRouter client at startup."""
    global _openrouter_client
    _openrouter_client = OpenRouterClient()
    # Warm up with health check (non-blocking)
    try:
        await _openrouter_client.health_check()
    except Exception as e:
        logger.warning("OpenRouter client health check failed during init", error=str(e))
    return _openrouter_client


async def shutdown_openrouter_client() -> None:
    """Shutdown OpenRouter client."""
    global _openrouter_client
    if _openrouter_client:
        await _openrouter_client.close()
        _openrouter_client = None