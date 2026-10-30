"""
Abstract base class for LLM provider adapters.

All concrete LLM adapters must inherit from :class:`BaseLLM` and implement
:meth:`complete` and :meth:`stream`.  The base class supplies a lightweight
token-counting fallback that does *not* require any external library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional


class BaseLLM(ABC):
    """Abstract interface for a language model provider.

    Concrete subclasses wrap a specific provider (OpenAI, Anthropic, Ollama,
    etc.) and normalise its API surface to the two methods below so that the
    rest of the pipeline remains provider-agnostic.

    Attributes:
        model: Identifier of the underlying model (e.g. ``"gpt-4o-mini"``).
        provider: Human-readable provider label (e.g. ``"openai"``).
    """

    model: str
    provider: str

    def __init__(self, model: str = "", provider: str = "") -> None:
        self.model = model
        self.provider = provider

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[dict] = None,
    ) -> str:
        """Request a single completion from the LLM.

        Args:
            messages: Conversation history in OpenAI chat format
                (``[{"role": ..., "content": ...}, ...]``).
            temperature: Sampling temperature; lower values produce more
                deterministic outputs.
            max_tokens: Upper bound on the number of tokens to generate.
            response_format: Optional format hint passed to the provider
                (e.g. ``{"type": "json_object"}`` for JSON mode).

        Returns:
            The assistant's reply as a plain string.

        Raises:
            LLMError: When the provider returns an unrecoverable error.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Stream a completion token-by-token from the LLM.

        Args:
            messages: Conversation history in OpenAI chat format.
            temperature: Sampling temperature.

        Yields:
            Successive string chunks as they arrive from the provider.

        Raises:
            LLMError: When the provider returns an unrecoverable error.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Estimate the number of tokens in *text*.

        The base implementation uses a character-based approximation
        (1 token ≈ 4 characters).  Subclasses should override this method
        with a provider-specific counter (e.g. tiktoken for OpenAI models).

        Args:
            text: The text to measure.

        Returns:
            Estimated token count as an integer.
        """
        return max(1, len(text) // 4)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider!r}, model={self.model!r})"


class LLMError(Exception):
    """Raised when an LLM provider returns an unrecoverable error."""

    def __init__(self, message: str, provider: str, original: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.original = original
