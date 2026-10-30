"""
Anthropic Claude LLM adapter.

Supports:
- claude-3-5-sonnet, claude-3-opus, claude-3-haiku, etc.
- System messages handled separately (Anthropic API requirement)
- JSON extraction from assistant replies (no native JSON mode on Anthropic)
- Automatic retries with exponential back-off (tenacity)
- Streaming via the Anthropic streaming API
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import AsyncIterator, Optional

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from research_agent.llm.base import BaseLLM, LLMError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional anthropic import
# ---------------------------------------------------------------------------
try:
    import anthropic
    from anthropic import AsyncAnthropic, APIStatusError, APIConnectionError, RateLimitError
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    anthropic = None  # type: ignore[assignment]
    AsyncAnthropic = None  # type: ignore[assignment]
    APIStatusError = APIConnectionError = RateLimitError = Exception  # type: ignore[assignment]


class AnthropicLLM(BaseLLM):
    """Async LLM adapter for Anthropic Claude models.

    The Anthropic Messages API separates the ``system`` prompt from the
    conversation turns.  This adapter extracts any ``{"role": "system", ...}``
    message from the provided list and passes it to the API's ``system``
    parameter automatically.

    Args:
        model: Claude model identifier (default: ``"claude-3-5-sonnet-20241022"``).
        api_key: Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY``
            environment variable when omitted.
        timeout: HTTP timeout in seconds (default: 60).
        max_retries: Maximum number of retry attempts on transient errors
            (default: 3).
        max_tokens: Default maximum tokens for completions (default: 4096).
    """

    provider: str = "anthropic"

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        max_tokens: int = 4096,
    ) -> None:
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicLLM. "
                "Install it with: pip install anthropic"
            )
        self.model = model
        self.provider = "anthropic"
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_max_tokens = max_tokens

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "An Anthropic API key is required. Pass api_key= or set ANTHROPIC_API_KEY."
            )

        self._client = AsyncAnthropic(api_key=resolved_key, timeout=timeout)

    # ------------------------------------------------------------------
    # Message normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_messages(
        messages: list[dict],
    ) -> tuple[Optional[str], list[dict]]:
        """Separate the system prompt from the conversation turns.

        The Anthropic API does not allow ``system`` role messages inside the
        ``messages`` list; they must be supplied via the top-level ``system``
        parameter.  This method extracts and concatenates all system messages
        (there should usually be only one) and returns the rest as the
        conversation turns.

        If the first human turn is missing (Anthropic requires the conversation
        to start with a human turn), a placeholder is injected.

        Args:
            messages: OpenAI-style message list, possibly containing system
                messages.

        Returns:
            A tuple of ``(system_prompt_or_None, anthropic_messages)``.
        """
        system_parts: list[str] = []
        conversation: list[dict] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role in {"user", "assistant"}:
                conversation.append({"role": role, "content": content})
            # Ignore unknown roles silently

        system_prompt: Optional[str] = "\n\n".join(system_parts) if system_parts else None

        # Anthropic requires conversation to start with a user turn
        if conversation and conversation[0]["role"] != "user":
            conversation.insert(0, {"role": "user", "content": "(conversation start)"})

        return system_prompt, conversation

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
        """Request a single completion from the Anthropic Messages API.

        When ``response_format={"type": "json_object"}`` is requested, this
        adapter appends a nudge to the last user message asking Claude to reply
        with valid JSON only.

        Args:
            messages: Chat history in OpenAI format (system messages allowed).
            temperature: Sampling temperature (0–1 for Anthropic).
            max_tokens: Maximum tokens to generate.
            response_format: Optional format hint; ``{"type": "json_object"}``
                triggers a JSON-mode nudge in the prompt.

        Returns:
            The assistant reply as a string.

        Raises:
            LLMError: After all retries are exhausted.
        """
        system_prompt, conversation = self._split_messages(messages)

        # JSON-mode nudge: Anthropic has no native JSON mode
        if response_format and response_format.get("type") == "json_object":
            if conversation and conversation[-1]["role"] == "user":
                conversation[-1]["content"] += (
                    "\n\nImportant: Reply with valid JSON only. "
                    "Do not include any prose, markdown fences, or explanation."
                )

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": min(temperature, 1.0),  # Anthropic clamps at 1.0
            "messages": conversation,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

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
                    response = await self._client.messages.create(**kwargs)
                    elapsed = time.monotonic() - start
                    content = response.content[0].text if response.content else ""
                    logger.debug(
                        "Anthropic completion: model=%s input_tokens=%s output_tokens=%s elapsed=%.2fs",
                        self.model,
                        response.usage.input_tokens,
                        response.usage.output_tokens,
                        elapsed,
                    )
                    return content
        except APIStatusError as exc:
            last_exc = exc
            raise LLMError(
                f"Anthropic API error {exc.status_code}: {exc.message}",
                provider=self.provider,
                original=exc,
            ) from exc
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

        raise LLMError(
            f"Anthropic request failed after {self.max_retries} retries: {last_exc}",
            provider=self.provider,
            original=last_exc,
        )

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Stream token chunks from the Anthropic Messages API.

        Args:
            messages: Chat history in OpenAI format.
            temperature: Sampling temperature (0–1).

        Yields:
            String delta chunks as they arrive.

        Raises:
            LLMError: On API or network errors.
        """
        system_prompt, conversation = self._split_messages(messages)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.default_max_tokens,
            "temperature": min(temperature, 1.0),
            "messages": conversation,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except APIStatusError as exc:
            raise LLMError(
                f"Anthropic stream error {exc.status_code}: {exc.message}",
                provider=self.provider,
                original=exc,
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"Anthropic stream failed: {exc}",
                provider=self.provider,
                original=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Estimate tokens using Anthropic's character-based approximation.

        Anthropic does not expose a public token counter; we use the same
        ~4 chars/token heuristic as the base class but apply a small
        correction factor that better matches Claude's BPE tokeniser for
        English text.

        Args:
            text: Text to measure.

        Returns:
            Estimated token count.
        """
        # Claude's tokeniser produces slightly fewer tokens than GPT's for
        # typical English prose; 3.8 chars/token is a reasonable approximation.
        return max(1, int(len(text) / 3.8))
