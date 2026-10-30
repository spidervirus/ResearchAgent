"""
ResearchContext — shared mutable state threaded through all pipeline stages.

The context acts as the single source of truth for a research session.  Every
pipeline stage (planner, searcher, fetcher, verifier, synthesiser) reads from
and writes to this object so that stages remain loosely coupled.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from research_agent.models import Evidence, ResearchAnswer, ResearchPlan, TokenUsage, ToolCall


class ResearchContext(BaseModel):
    """Shared mutable state for a single research session.

    Attributes:
        query: The original research question.
        session_id: Unique identifier for this research session.
        plan: The :class:`~research_agent.models.ResearchPlan` produced by the
            planner stage; ``None`` until planning completes.
        raw_evidence: All :class:`~research_agent.models.Evidence` collected
            before verification / filtering.
        verified_evidence: Evidence that survived the verification stage.
        tool_calls: Ordered log of every :class:`~research_agent.models.ToolCall`
            made during the session.
        messages: LLM message history in OpenAI chat-completion format
            (``{"role": ..., "content": ...}`` dicts).
        answer: The final :class:`~research_agent.models.ResearchAnswer`;
            ``None`` until synthesis completes.
        metadata: Arbitrary key-value store for inter-stage communication.
        created_at: UTC timestamp when the context was constructed.
    """

    query: str = Field(..., min_length=1, description="Original research question")
    session_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique session identifier",
    )
    plan: Optional[ResearchPlan] = Field(default=None)
    raw_evidence: list[Evidence] = Field(default_factory=list)
    verified_evidence: list[Evidence] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="LLM message history (OpenAI chat format)",
    )
    previous_context: Optional[str] = Field(
        default=None,
        description="Context summary from previous turns in a multi-turn session",
    )
    answer: Optional[ResearchAnswer] = Field(default=None)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    degradation_reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value store for pipeline stages",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    def touch(self) -> None:
        """Bump the updated_at timestamp."""
        self.updated_at = datetime.now(tz=timezone.utc)

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Evidence helpers
    # ------------------------------------------------------------------

    def add_evidence(self, evidence: Evidence, *, verified: bool = False) -> None:
        """Append a single evidence item to the appropriate collection.

        Args:
            evidence: The :class:`~research_agent.models.Evidence` to add.
            verified: When ``True``, also append to :attr:`verified_evidence`.
        """
        self.raw_evidence.append(evidence)
        if verified:
            self.verified_evidence.append(evidence)

    def add_evidence_batch(
        self, items: list[Evidence], *, verified: bool = False
    ) -> None:
        """Append multiple evidence items at once.

        Args:
            items: List of :class:`~research_agent.models.Evidence` objects.
            verified: Forwarded to :meth:`add_evidence` for each item.
        """
        for item in items:
            self.add_evidence(item, verified=verified)

    def set_verified_evidence(self, items: list[Evidence]) -> None:
        """Replace the entire verified evidence collection.

        Typically called by the verification stage after filtering
        :attr:`raw_evidence`.

        Args:
            items: The filtered evidence list that passed verification.
        """
        self.verified_evidence = list(items)

    # ------------------------------------------------------------------
    # Tool-call helpers
    # ------------------------------------------------------------------

    def add_tool_call(self, call: ToolCall) -> None:
        """Append a tool-call record to the session log.

        Args:
            call: The :class:`~research_agent.models.ToolCall` to record.
        """
        self.tool_calls.append(call)

    def get_tool_calls_for(self, tool_name: str) -> list[ToolCall]:
        """Return all recorded calls for a specific tool.

        Args:
            tool_name: The tool name to filter by.

        Returns:
            A list of matching :class:`~research_agent.models.ToolCall` objects.
        """
        return [tc for tc in self.tool_calls if tc.tool_name == tool_name]

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the LLM conversation history.

        Args:
            role: OpenAI role string — ``"system"``, ``"user"``, or
                ``"assistant"``.
            content: Text content of the message.
        """
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(
                f"Invalid role '{role}'. Must be one of: system, user, assistant, tool."
            )
        self.messages.append({"role": role, "content": content})

    def add_system_message(self, content: str) -> None:
        """Convenience wrapper: append a system message."""
        self.add_message("system", content)

    def add_user_message(self, content: str) -> None:
        """Convenience wrapper: append a user message."""
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Convenience wrapper: append an assistant message."""
        self.add_message("assistant", content)

    def clear_messages(self) -> None:
        """Erase all messages from the conversation history."""
        self.messages.clear()

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def set_meta(self, key: str, value: Any) -> None:
        """Write an arbitrary value into the metadata store.

        Args:
            key: Metadata key.
            value: Any JSON-serialisable value.
        """
        self.metadata[key] = value

    def get_meta(self, key: str, default: Any = None) -> Any:
        """Read a value from the metadata store.

        Args:
            key: Metadata key.
            default: Returned when the key is absent.

        Returns:
            The stored value or *default*.
        """
        return self.metadata.get(key, default)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the context to a plain Python dictionary.

        The output is safe for JSON serialisation (dates are ISO strings).

        Returns:
            A dictionary representation of the entire context.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchContext":
        """Reconstruct a :class:`ResearchContext` from a serialised dictionary.

        Args:
            data: A dictionary as produced by :meth:`to_dict`.

        Returns:
            A fully populated :class:`ResearchContext` instance.
        """
        return cls.model_validate(data)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def has_plan(self) -> bool:
        """``True`` when a plan has been assigned."""
        return self.plan is not None

    @property
    def has_answer(self) -> bool:
        """``True`` when synthesis has produced a final answer."""
        return self.answer is not None

    @property
    def evidence_count(self) -> int:
        """Total number of raw evidence items collected."""
        return len(self.raw_evidence)

    @property
    def verified_evidence_count(self) -> int:
        """Number of evidence items that passed verification."""
        return len(self.verified_evidence)

    def __repr__(self) -> str:
        return (
            f"ResearchContext("
            f"session_id={self.session_id!r}, "
            f"query={self.query!r}, "
            f"evidence={self.evidence_count}, "
            f"verified={self.verified_evidence_count}, "
            f"has_answer={self.has_answer}"
            f")"
        )
