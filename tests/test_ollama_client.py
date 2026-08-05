"""
Tests for Ollama Client
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from ai.llm.ollama_client import OllamaClient, CircuitBreaker, CircuitState


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_closed(self):
        """Test initial state is closed."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_half_open_after_timeout(self):
        """Test circuit goes half-open after timeout."""
        cb = CircuitBreaker(failure_threshold=2, timeout_seconds=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # Wait for timeout
        import time
        time.sleep(0.2)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.can_execute() is True

    def test_closes_after_successes(self):
        """Test circuit closes after success threshold in half-open."""
        cb = CircuitBreaker(failure_threshold=2, success_threshold=2, timeout_seconds=0.1)
        cb.record_failure()
        cb.record_failure()
        import time
        time.sleep(0.2)  # Go to half-open
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_failure_in_half_open_reopens(self):
        """Test failure in half-open reopens circuit."""
        cb = CircuitBreaker(failure_threshold=2, timeout_seconds=0.1)
        cb.record_failure()
        cb.record_failure()
        import time
        time.sleep(0.2)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestOllamaClient:
    """Tests for OllamaClient."""

    @pytest.fixture
    def client(self):
        return OllamaClient(
            model="test-model",
            host="http://localhost:11434",
            timeout_seconds=5,
            max_retries=2,
            retry_backoff=0.1,
        )

    @pytest.mark.asyncio
    async def test_generate_success(self, client):
        """Test successful generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Generated text"}

        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await client.generate("Test prompt")

        assert result["success"] is True
        assert result["response"] == "Generated text"
        assert result["cached"] is False

    @pytest.mark.asyncio
    async def test_generate_timeout(self, client):
        """Test generation timeout."""
        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Timeout")
            mock_get_client.return_value = mock_client

            result = await client.generate("Test prompt")

        assert result["success"] is False
        assert "fallback" in result
        assert result["fallback"] is True

    @pytest.mark.asyncio
    async def test_generate_http_error_5xx(self, client):
        """Test generation with 5xx error (retryable)."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503", request=MagicMock(), response=mock_response
        )

        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await client.generate("Test prompt")

        assert result["success"] is False
        assert result["fallback"] is True

    @pytest.mark.asyncio
    async def test_generate_http_error_4xx(self, client):
        """Test generation with 4xx error (non-retryable)."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400", request=MagicMock(), response=mock_response
        )

        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await client.generate("Test prompt")

        assert result["success"] is False
        # Should not retry on 4xx
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_caches_response(self, client):
        """Test successful response is cached."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Cached response"}

        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            # First call
            result1 = await client.generate("Test prompt", use_cache=True)
            assert result1["cached"] is False

            # Second call should use cache
            result2 = await client.generate("Test prompt", use_cache=True)
            assert result2["cached"] is True
            assert result2["response"] == "Cached response"

    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        """Test health check success."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await client.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client):
        """Test health check failure."""
        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.RequestError("Connection failed")
            mock_get_client.return_value = mock_client

            result = await client.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks(self, client):
        """Test circuit breaker blocks requests when open."""
        # Force circuit open
        client.circuit_breaker._state = CircuitState.OPEN
        client.circuit_breaker._failure_count = 10

        result = await client.generate("Test prompt")
        assert result["success"] is False
        assert result["fallback"] is True