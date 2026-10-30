"""
Abstract base classes for research tools.

Every tool in the pipeline must inherit from :class:`BaseTool` and implement
:meth:`run` and the :attr:`schema` property.  The :class:`ToolResult` model
provides a uniform envelope for tool outputs so that the orchestrator can
handle success and failure uniformly without catching per-tool exceptions.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Uniform envelope for tool outputs.

    Attributes:
        success: ``True`` when the tool completed without error.
        data: The tool's payload — type varies by tool.  Callers should
            inspect this only when :attr:`success` is ``True``.
        error: Human-readable error description when :attr:`success` is
            ``False``; ``None`` otherwise.
        metadata: Optional key-value bag for extra diagnostics (elapsed time,
            result counts, source URLs, etc.).
    """

    success: bool = Field(..., description="Whether the tool completed successfully")
    data: Any = Field(default=None, description="Tool output payload")
    error: Optional[str] = Field(default=None, description="Error description on failure")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Supplementary diagnostics (timing, counts, etc.)",
    )

    @classmethod
    def ok(cls, data: Any, **meta: Any) -> "ToolResult":
        """Convenience constructor for a successful result.

        Args:
            data: The tool's output payload.
            **meta: Extra key-value pairs stored in :attr:`metadata`.

        Returns:
            A :class:`ToolResult` with ``success=True``.
        """
        return cls(success=True, data=data, metadata=dict(meta))

    @classmethod
    def fail(cls, error: str, **meta: Any) -> "ToolResult":
        """Convenience constructor for a failed result.

        Args:
            error: Human-readable error message.
            **meta: Extra key-value pairs stored in :attr:`metadata`.

        Returns:
            A :class:`ToolResult` with ``success=False``.
        """
        return cls(success=False, data=None, error=error, metadata=dict(meta))

    def __bool__(self) -> bool:
        """Allow ``if result:`` checks as shorthand for ``result.success``."""
        return self.success


class BaseTool(ABC):
    """Abstract base class for all research tools.

    Subclasses must define :attr:`name`, :attr:`description`, implement
    :meth:`run`, and expose an OpenAI-compatible JSON schema via the
    :attr:`schema` property.

    The :meth:`safe_run` helper wraps :meth:`run` with timing and exception
    catching so that the orchestrator always receives a :class:`ToolResult`.

    Class Attributes:
        name: Unique tool identifier used by the planner and executor.
        description: One-line description used in prompts and logs.
    """

    name: str
    description: str

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given arguments.

        Implementations should raise :class:`Exception` on unrecoverable
        errors; :meth:`safe_run` will catch them and return a failure
        :class:`ToolResult`.

        Args:
            **kwargs: Tool-specific keyword arguments as described in
                :attr:`schema`.

        Returns:
            A :class:`ToolResult` with the tool's output.
        """
        ...

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """OpenAI-compatible JSON schema for LLM function calling.

        The returned dict must conform to the OpenAI ``tools`` parameter
        format::

            {
                "type": "function",
                "function": {
                    "name": "<tool_name>",
                    "description": "<description>",
                    "parameters": {
                        "type": "object",
                        "properties": { ... },
                        "required": [ ... ],
                    },
                },
            }

        Returns:
            A dictionary describing the tool for function-calling prompts.
        """
        ...

    def _ok(self, data: Any, **meta: Any) -> ToolResult:
        """Helper to create a successful ToolResult."""
        return ToolResult.ok(data, **meta)

    def _fail(self, error: str, **meta: Any) -> ToolResult:
        """Helper to create a failed ToolResult."""
        return ToolResult.fail(error, **meta)

    def _err(self, error: str, **meta: Any) -> ToolResult:
        """Alias for _fail."""
        return ToolResult.fail(error, **meta)

    async def safe_run(self, **kwargs: Any) -> ToolResult:
        """Execute :meth:`run` with automatic timing and exception handling.

        Catches all exceptions and converts them to a failure
        :class:`ToolResult` so the orchestrator never has to guard individual
        tool calls.

        Args:
            **kwargs: Forwarded verbatim to :meth:`run`.

        Returns:
            The :class:`ToolResult` from :meth:`run` on success, or a failure
            result on exception.
        """
        start = time.monotonic()
        try:
            result = await self.run(**kwargs)
            elapsed_ms = (time.monotonic() - start) * 1000
            result.metadata.setdefault("elapsed_ms", round(elapsed_ms, 2))
            result.metadata.setdefault("tool_name", self.name)
            return result
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.monotonic() - start) * 1000
            return ToolResult.fail(
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=round(elapsed_ms, 2),
                tool_name=self.name,
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
