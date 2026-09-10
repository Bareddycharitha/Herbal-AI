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


# Matches a labeled phrase the model sometimes echoes back when leaking
# its planning structure into the response. Two flavors:
#   1. "1. **Analyze the Request:**" / "* **Role:** Medical writer."
#      — bold label, with or without a numeric/bullet prefix.
#   2. "* Overview:" / "* What to look for:" — plain bullet label.
# The frontend renders the summary as plain prose, so these would
# otherwise show up as raw markdown noise.
_LABEL_PATTERN = __import__("re").compile(
    r"\*\*[^*\n]{1,40}?\*\*\s*[:\-]?"
)
_PLAIN_LABEL_PATTERN = __import__("re").compile(
    r"\*\s+[A-Z][A-Za-z &\-]{1,30}?:\s*"
)


def _looks_like_planning_preamble(text: str) -> bool:
    """
    Heuristic: a "planning preamble" is text that contains two or more
    labeled annotations within the first ~1000 characters. A single
    stray label mid-response is not a preamble.
    """
    if not text:
        return False
    head = text[:1000]
    bold_count = len(_LABEL_PATTERN.findall(head))
    plain_count = len(_PLAIN_LABEL_PATTERN.findall(head))
    return (bold_count + plain_count) >= 2


def _strip_planning_preamble(text: str) -> str:
    """
    Detect a leading planning preamble (a chain of labeled annotations
    like "1. **Analyze the Request:** ... * **Role:** ... * **Task:**
    ...") and return the text that comes after the last such label.
    Plain text responses pass through unchanged.
    """
    if not text or not _looks_like_planning_preamble(text):
        return text
    head = text[:1500]
    bold_ends = [m.end() for m in _LABEL_PATTERN.finditer(head)]
    plain_ends = [m.end() for m in _PLAIN_LABEL_PATTERN.finditer(head)]
    all_ends = sorted(bold_ends + plain_ends)
    if not all_ends:
        return text
    cut = all_ends[-1]
    return text[cut:].lstrip()


def _strip_leading_label_lines(text: str) -> str:
    """
    Light pass over leading lines: drop any line that starts with a
    bolded label (e.g. "**Overview:**") or a numbered/bullet item, plus
    the literal "AI medical summary" / "Generated automatically" headers
    the frontend already renders. Stops at the first real content line.
    """
    if not text:
        return text
    lines = text.split("\n")
    cleaned: list[str] = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not started:
            if not stripped:
                continue
            if _LABEL_PATTERN.match(stripped) or _PLAIN_LABEL_PATTERN.match(stripped):
                continue
            lower = stripped.lower()
            if lower.startswith("ai medical summary"):
                continue
            if lower.startswith("generated automatically"):
                continue
            started = True
        cleaned.append(line)
    if cleaned:
        return "\n".join(cleaned).strip()
    return text.strip()


def _is_pure_reasoning_scratchpad(text: str) -> bool:
    """Check if the response consists solely of model internal monologue/prompt reasoning."""
    if not text:
        return False
    lower = text.lower().strip()
    reasoning_phrases = [
        "we need answer user request",
        "need produce two short paragraphs",
        "need summarize ai-assisted",
        "need produce",
        "need summarize",
        "user asks write",
        "the user asks",
        "task asks",
        "thinking process",
        "let's analyze",
        "let me think",
        "1. analyze the request",
    ]
    for phrase in reasoning_phrases:
        if phrase in lower:
            # If the text is dominated by reasoning and has no clear patient-facing opening
            if not ("based on the" in lower or "the ai model" in lower or "for daily care" in lower or "the image suggests" in lower):
                return True
    return False


def _extract_final_draft(text: str) -> str:
    """Extract final drafted paragraphs if model leaked its scratchpad."""
    if not text:
        return text
    lower = text.lower()
    
    # Check common draft delimiters
    for marker in [
        "here is the summary:",
        "here is the response:",
        "final response:",
        "drafted summary:",
        "i'll write:",
        "let me write:",
        "let me refine:",
        "let me draft:",
        "draft:",
        "paragraph 1:",
    ]:
        if marker in lower:
            idx = lower.rfind(marker)
            candidate = text[idx + len(marker):].strip()
            # Clean paragraph markers if present
            candidate = candidate.replace("Paragraph 2:", "\n\n").strip()
            # Remove word count checks at the end if present
            for wc_marker in ["total word count", "word count"]:
                if wc_marker in candidate.lower():
                    candidate = candidate[:candidate.lower().rfind(wc_marker)].strip()
            if len(candidate) > 40 and not _is_pure_reasoning_scratchpad(candidate):
                return candidate

    # Look for quoted text that looks like a drafted response
    if '"' in text:
        import re
        quotes = re.findall(r'"([^"]{60,})"', text)
        if quotes:
            return "\n\n".join(quotes)

    return text


def _strip_leading_annotations(text: str) -> str:
    """
    Multi-stage cleanup for the model's response:
    1. Extract final draft if reasoning scratchpad is present.
    2. Detect and remove a leading planning preamble if present.
    3. Drop any remaining stray labeled lines or meta prompt regurgitation at the start.
    """
    if not text:
        return text
    text = _extract_final_draft(text)
    text = _strip_planning_preamble(text)
    text = _strip_leading_label_lines(text)
    return text


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
        """Get cached summary if available and valid."""
        key = self._get_cache_key(prompt, system_prompt)
        cached = self._summary_cache.get(key)
        if cached:
            if _is_pure_reasoning_scratchpad(cached) or len(cached) < 30:
                del self._summary_cache[key]
                return None
        return cached

    def _cache_summary(self, prompt: str, response: str, system_prompt: str = "") -> None:
        """Cache summary response if valid."""
        if not response or _is_pure_reasoning_scratchpad(response) or len(response) < 30:
            return
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

        # List of models to try in order of preference
        fallback_models = [
            self.model,
            "inclusionai/ling-3.0-flash-sante:free",
            "google/gemma-4-31b-it:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen-2.5-72b-instruct:free",
            "nex-agi/nex-n2.5-pro:free",
        ]
        # De-duplicate while preserving order
        models_to_try = []
        for m in fallback_models:
            if m and m not in models_to_try:
                models_to_try.append(m)

        last_error = None

        for current_model in models_to_try:
            payload["model"] = current_model
            for attempt in range(self.max_retries + 1):
                try:
                    client = await self._get_client()

                    logger.debug(
                        "Calling OpenRouter",
                        attempt=attempt + 1,
                        model=current_model,
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

                    # Extract response text from OpenAI-compatible format (supporting reasoning models)
                    response_text = ""
                    if "choices" in result and len(result["choices"]) > 0:
                        message = result["choices"][0].get("message", {})
                        content_val = message.get("content")
                        if content_val and isinstance(content_val, str) and content_val.strip():
                            raw_text = content_val.strip()
                        else:
                            # Content was empty (e.g. reasoning model token budget).
                            # Extract actual answer from reasoning if possible.
                            reasoning_text = (message.get("reasoning") or "").strip()
                            if "let me draft" in reasoning_text.lower():
                                raw_text = reasoning_text.lower().split("let me draft")[-1]
                            elif "draft:" in reasoning_text.lower():
                                raw_text = reasoning_text.lower().split("draft:")[-1]
                            else:
                                raw_text = reasoning_text

                        # Filter out reasoning/thinking headers if present
                        if "thinking process" in raw_text.lower() or "<think>" in raw_text:
                            import re
                            raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
                            parts = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
                            # Pick the last paragraph containing summary content
                            for part in reversed(parts):
                                if "thinking process" not in part.lower() and "count words" not in part.lower() and len(part) > 30:
                                    response_text = part
                                    break
                            if not response_text and parts:
                                response_text = parts[-1]
                        else:
                            response_text = raw_text

                        # Strip any leading annotation lines or reasoning preamble
                        if response_text:
                            response_text = _strip_leading_annotations(response_text)

                        # If the output is still pure reasoning monologue without patient content, try next model
                        if _is_pure_reasoning_scratchpad(response_text) or len(response_text) < 30:
                            logger.warning(
                                "Model generated pure reasoning scratchpad, trying next fallback model",
                                model=current_model,
                            )
                            continue

                    # Cache successful response
                    if use_cache and response_text:
                        self._cache_summary(prompt, response_text, system_prompt)

                    logger.info(
                        "OpenRouter response received",
                        attempt=attempt + 1,
                        model=current_model,
                        response_length=len(response_text),
                    )

                    return {
                        "success": True,
                        "response": response_text,
                        "cached": False,
                    }

                except httpx.TimeoutException as e:
                    last_error = f"Request timed out after {self.timeout_seconds}s with {current_model}"
                    logger.warning("OpenRouter timeout", model=current_model, attempt=attempt + 1, error=str(e))

                except httpx.HTTPStatusError as e:
                    last_error = f"HTTP {e.response.status_code} on {current_model}: {e.response.text[:100]}"
                    logger.warning("OpenRouter HTTP error", model=current_model, attempt=attempt + 1, error=last_error)
                    # If 4xx (e.g. 429 rate limit or 400 invalid model), move immediately to next fallback model
                    break

                except httpx.RequestError as e:
                    last_error = f"Request failed on {current_model}: {str(e)}"
                    logger.warning("OpenRouter request error", model=current_model, attempt=attempt + 1, error=str(e))

                except Exception as e:
                    last_error = f"Unexpected error on {current_model}: {str(e)}"
                    logger.error("OpenRouter unexpected error", model=current_model, attempt=attempt + 1, error=str(e))

                # Exponential backoff before retry on the same model
                if attempt < self.max_retries:
                    backoff = self.retry_backoff * (2 ** attempt)
                    logger.info("Retrying OpenRouter request", model=current_model, backoff_seconds=backoff)
                    await asyncio.sleep(backoff)

        # Record failure for circuit breaker if all fallback models failed
        self.circuit_breaker.record_failure()

        # All retries and models exhausted
        logger.error("OpenRouter all models exhausted", error=last_error)
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