"""
OpenAI (and Ollama-compatible) LLM adapter.

Supports:
- Standard OpenAI API (GPT-4o, GPT-4o-mini, etc.)
- Local Ollama server via ``base_url`` override
- JSON mode via ``response_format={"type": "json_object"}``
- Automatic retries with exponential back-off (tenacity)
- Accurate token counting via tiktoken (falls back to char-heuristic)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import AsyncIterator, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from research_agent.llm.base import BaseLLM, LLMError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional tiktoken import
# ---------------------------------------------------------------------------
try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]
    _TIKTOKEN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional openai import
# ---------------------------------------------------------------------------
try:
    from openai import AsyncOpenAI, APIStatusError, APIConnectionError, RateLimitError
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    AsyncOpenAI = None  # type: ignore[assignment]
    APIStatusError = APIConnectionError = RateLimitError = Exception  # type: ignore[assignment]


class OpenAILLM(BaseLLM):
    """Async LLM adapter for OpenAI-compatible APIs.

    Can be pointed at the official OpenAI endpoint or at a local Ollama
    instance by setting ``base_url`` (e.g. ``http://localhost:11434/v1``).

    Args:
        model: Model identifier (default: ``"gpt-4o-mini"``).
        api_key: Provider API key.  Falls back to the ``OPENAI_API_KEY``
            environment variable when omitted.
        base_url: Override the API base URL for Ollama or other compatible
            servers.  Falls back to ``OLLAMA_BASE_URL`` env var when omitted
            and the provider is ``"ollama"``.
        provider: Label stored on the instance (``"openai"`` or ``"ollama"``).
        timeout: HTTP timeout in seconds (default: 60).
        max_retries: Maximum number of retry attempts (default: 3).
    """

    provider: str = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "openai",
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "The 'openai' package is required for OpenAILLM. "
                "Install it with: pip install openai"
            )
        self.model = model
        self.provider = provider
        self.timeout = timeout
        self.max_retries = max_retries

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        resolved_base = base_url or os.environ.get("OLLAMA_BASE_URL")

        client_kwargs: dict = {
            "api_key": resolved_key or "ollama",  # Ollama ignores the key
            "timeout": httpx.Timeout(timeout),
        }
        if resolved_base:
            # Ensure path ends with /v1 for Ollama compatibility
            if not resolved_base.rstrip("/").endswith("/v1"):
                resolved_base = resolved_base.rstrip("/") + "/v1"
            client_kwargs["base_url"] = resolved_base

        self._client = AsyncOpenAI(**client_kwargs)

        # Initialise tiktoken encoder once (best-effort)
        self._encoder = None
        if _TIKTOKEN_AVAILABLE:
            try:
                self._encoder = tiktoken.encoding_for_model(model)
            except KeyError:
                try:
                    self._encoder = tiktoken.get_encoding("cl100k_base")
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[dict] = None,
    ) -> str:
        """Request a single completion from the OpenAI API.

        Retries up to :attr:`max_retries` times on transient errors.

        Args:
            messages: Chat history in OpenAI format.
            temperature: Sampling temperature (0–2).
            max_tokens: Maximum tokens to generate.
            response_format: Optional format dict, e.g.
                ``{"type": "json_object"}`` to enable JSON mode.

        Returns:
            The assistant reply as a string.

        Raises:
            LLMError: After all retries are exhausted.
        """
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        last_exc: Optional[Exception] = None
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type((RateLimitError, APIConnectionError, asyncio.TimeoutError)),
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=30),
                reraise=False,
            ):
                with attempt:
                    start = time.monotonic()
                    response = await self._client.chat.completions.create(**kwargs)
                    elapsed = time.monotonic() - start
                    content = response.choices[0].message.content or ""
                    logger.debug(
                        "OpenAI completion: model=%s tokens_used=%s elapsed=%.2fs",
                        self.model,
                        response.usage.total_tokens if response.usage else "?",
                        elapsed,
                    )
                    return content
        except APIStatusError as exc:
            last_exc = exc
            raise LLMError(
                f"OpenAI API error {exc.status_code}: {exc.message}",
                provider=self.provider,
                original=exc,
            ) from exc
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

        raise LLMError(
            f"OpenAI request failed after {self.max_retries} retries: {last_exc}",
            provider=self.provider,
            original=last_exc,
        )

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Stream token chunks from the OpenAI API.

        Args:
            messages: Chat history in OpenAI format.
            temperature: Sampling temperature (0–2).

        Yields:
            String delta chunks as they arrive.

        Raises:
            LLMError: On API or network errors.
        """
        try:
            async with self._client.chat.completions.stream(
                model=self.model,
                messages=messages,
                temperature=temperature,
            ) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
        except APIStatusError as exc:
            raise LLMError(
                f"OpenAI stream error {exc.status_code}: {exc.message}",
                provider=self.provider,
                original=exc,
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"OpenAI stream failed: {exc}",
                provider=self.provider,
                original=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken when available.

        Falls back to the character-heuristic from :class:`BaseLLM` when
        tiktoken is not installed or the model encoding is unknown.

        Args:
            text: Text to measure.

        Returns:
            Token count as an integer.
        """
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:  # noqa: BLE001
                pass
        return super().count_tokens(text)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def for_ollama(
        cls,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        **kwargs,
    ) -> "OpenAILLM":
        """Convenience constructor for local Ollama servers.

        Args:
            model: Ollama model tag (e.g. ``"llama3"``, ``"mistral"``).
            base_url: Ollama HTTP base URL (default: ``http://localhost:11434``).
            **kwargs: Forwarded to :class:`OpenAILLM`.

        Returns:
            A configured :class:`OpenAILLM` instance pointed at Ollama.
        """
        return cls(model=model, base_url=base_url, provider="ollama", api_key="ollama", **kwargs)
