"""
Pydantic v2 data models for the research agent pipeline.

All models use strict typing and field validation to ensure data integrity
throughout the research lifecycle.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class StepStatus(str, Enum):
    """Lifecycle status of a single plan step."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class Severity(str, Enum):
    """Severity level for a research limitation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Plan models
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    """A single, executable step within a research plan.

    Attributes:
        id: Unique step identifier (auto-generated UUID4 hex).
        description: Human-readable description of what this step does.
        tool_name: The name of the tool to invoke for this step.
        tool_args: Keyword arguments forwarded to the tool's ``run()`` method.
        status: Current execution status of this step.
        result: Output produced by the tool after successful execution.
        error: Error message if the step failed; ``None`` otherwise.
        started_at: Wall-clock time when the step began execution.
        finished_at: Wall-clock time when the step completed or failed.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    description: str = Field(..., min_length=1, description="What this step accomplishes")
    tool_name: str = Field(..., min_length=1, description="Registered tool name to invoke")
    tool_args: dict[str, Any] = Field(default_factory=dict)
    status: StepStatus = Field(default=StepStatus.PENDING)
    depends_on: list[str] = Field(
        default_factory=list,
        description="IDs or 1-based indices of prerequisite steps that must complete before this step runs",
    )
    result: Optional[Any] = Field(default=None)
    error: Optional[str] = Field(default=None)
    started_at: Optional[datetime] = Field(default=None)
    finished_at: Optional[datetime] = Field(default=None)

    model_config = {"use_enum_values": True}

    @field_validator("depends_on", mode="before")
    @classmethod
    def _coerce_depends_on(cls, v: Any) -> list[str]:
        if not v:
            return []
        if isinstance(v, (str, int)):
            return [str(v)]
        if isinstance(v, (list, tuple, set)):
            return [str(x) for x in v if x is not None]
        return []

    def mark_running(self) -> None:
        """Transition the step to RUNNING state."""
        self.status = StepStatus.RUNNING
        self.started_at = datetime.now(tz=timezone.utc)

    def mark_done(self, result: Any) -> None:
        """Transition the step to DONE state and record its result."""
        self.status = StepStatus.DONE
        self.result = result
        self.finished_at = datetime.now(tz=timezone.utc)

    def mark_failed(self, error: str) -> None:
        """Transition the step to FAILED state and record the error."""
        self.status = StepStatus.FAILED
        self.error = error
        self.finished_at = datetime.now(tz=timezone.utc)

    def mark_skipped(self, reason: str) -> None:
        """Transition the step to SKIPPED state and record reason."""
        self.status = StepStatus.SKIPPED
        self.error = reason
        self.finished_at = datetime.now(tz=timezone.utc)

    @property
    def duration_ms(self) -> Optional[float]:
        """Elapsed milliseconds between start and finish, if available."""
        if self.started_at and self.finished_at:
            delta = self.finished_at - self.started_at
            return delta.total_seconds() * 1000
        return None


class ResearchPlan(BaseModel):
    """An ordered list of steps that the agent will execute to answer a query.

    Attributes:
        id: Unique plan identifier.
        query: The original research question this plan addresses.
        steps: Ordered list of :class:`PlanStep` objects.
        created_at: UTC timestamp when the plan was created.
        rationale: Optional explanation of why this plan was chosen.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    query: str = Field(..., min_length=1)
    steps: list[PlanStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    rationale: Optional[str] = Field(default=None)

    @property
    def pending_steps(self) -> list[PlanStep]:
        """Return all steps not yet completed or failed."""
        return [s for s in self.steps if s.status == StepStatus.PENDING]

    @property
    def completed_steps(self) -> list[PlanStep]:
        """Return all successfully completed steps."""
        return [s for s in self.steps if s.status == StepStatus.DONE]

    @property
    def failed_steps(self) -> list[PlanStep]:
        """Return all failed steps."""
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    @property
    def skipped_steps(self) -> list[PlanStep]:
        """Return all skipped steps."""
        return [s for s in self.steps if s.status == StepStatus.SKIPPED]

    @property
    def is_complete(self) -> bool:
        """``True`` when every step is DONE, FAILED, or SKIPPED."""
        return all(
            s.status in (StepStatus.DONE, StepStatus.FAILED, StepStatus.SKIPPED)
            for s in self.steps
        )


# ---------------------------------------------------------------------------
# Evidence models
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """A single piece of sourced evidence retrieved during research.

    Attributes:
        id: Unique evidence identifier.
        source_url: Canonical URL of the source page.
        source_title: Title of the source document or page.
        excerpt: Verbatim or near-verbatim text extracted from the source.
        relevance_score: How relevant this evidence is to the query (0–1).
        quality_score: Estimated quality/credibility of the source (0–1).
        retrieved_at: UTC timestamp when the evidence was fetched.
        step_id: ID of the plan step that produced this evidence.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    source_url: str = Field(..., min_length=1)
    source_title: str = Field(default="")
    excerpt: str = Field(..., min_length=1)
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    step_id: Optional[str] = Field(default=None)

    @field_validator("relevance_score", "quality_score", mode="before")
    @classmethod
    def clamp_scores(cls, v: float) -> float:
        """Silently clamp score to [0, 1] range."""
        return max(0.0, min(1.0, float(v)))

    @property
    def composite_score(self) -> float:
        """Weighted combination of relevance and quality (60/40 split)."""
        return 0.6 * self.relevance_score + 0.4 * self.quality_score


# ---------------------------------------------------------------------------
# Answer-component models
# ---------------------------------------------------------------------------


class Limitation(BaseModel):
    """A known limitation or caveat of the research findings.

    Attributes:
        description: Human-readable description of the limitation.
        severity: How much this limitation affects confidence in the answer.
    """

    description: str = Field(..., min_length=1)
    severity: Severity = Field(default=Severity.MEDIUM)

    model_config = {"use_enum_values": True}


class FollowUpQuestion(BaseModel):
    """A suggested follow-up question to deepen the research.

    Attributes:
        question: The follow-up question text.
        rationale: Why this question would be valuable to explore next.
    """

    question: str = Field(..., min_length=1)
    rationale: str = Field(default="")


# ---------------------------------------------------------------------------
# Tool tracking models
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A record of a single tool invocation during the research session.

    Attributes:
        id: Unique call identifier.
        tool_name: Name of the tool that was called.
        args: Arguments passed to the tool.
        result: Serializable result returned by the tool; ``None`` on error.
        error: Error message if the call failed; ``None`` on success.
        duration_ms: How long the call took in milliseconds.
        called_at: UTC timestamp of the call.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tool_name: str = Field(..., min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = Field(default=None)
    error: Optional[str] = Field(default=None)
    duration_ms: Optional[float] = Field(default=None)
    called_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    @property
    def succeeded(self) -> bool:
        """``True`` when the tool call produced no error."""
        return self.error is None


# ---------------------------------------------------------------------------
# Verification model
# ---------------------------------------------------------------------------


class VerificationResult(BaseModel):
    """Output of the evidence verification / filtering stage.

    Attributes:
        passed: Whether the verification step concluded evidence is acceptable.
        issues: List of identified problems with the evidence set.
        filtered_evidence: Evidence that passed verification filters.
        notes: Free-form notes from the verifier for downstream inspection.
    """

    passed: bool = Field(..., description="Overall pass/fail of verification")
    issues: list[str] = Field(default_factory=list)
    filtered_evidence: list[Evidence] = Field(default_factory=list)
    notes: str = Field(default="")

    @property
    def has_issues(self) -> bool:
        """``True`` when at least one issue was identified."""
        return len(self.issues) > 0


# ---------------------------------------------------------------------------
# Confidence and Metrics models
# ---------------------------------------------------------------------------


class ConfidenceFactors(BaseModel):
    """Rigorous breakdown of factors driving the overall confidence score.

    Attributes:
        source_count_score: (0-1) Score based on volume of high-quality sources (>=3 sources = 1.0).
        agreement_score: (0-1) Level of consensus across independent sources vs contradictions.
        recency_score: (0-1) Freshness and timeliness of retrieved evidence relative to modern developments.
        constraint_coverage_score: (0-1) How completely the evidence covers every specific constraint in the query.
        weighted_score: Final calculated score from the weighted factors.
        justification: Brief explanation of the factor assessments.
    """

    source_count_score: float = Field(default=0.5, ge=0.0, le=1.0)
    agreement_score: float = Field(default=0.5, ge=0.0, le=1.0)
    recency_score: float = Field(default=0.5, ge=0.0, le=1.0)
    constraint_coverage_score: float = Field(default=0.5, ge=0.0, le=1.0)
    weighted_score: float = Field(default=0.5, ge=0.0, le=1.0)
    justification: str = Field(default="")

    @classmethod
    def calculate(
        cls,
        source_count: float,
        agreement: float,
        recency: float,
        constraint_coverage: float,
        justification: str = "",
    ) -> "ConfidenceFactors":
        # Weights: 30% source quality/count, 25% agreement, 15% recency, 30% constraint coverage
        w = (
            0.30 * max(0.0, min(1.0, float(source_count)))
            + 0.25 * max(0.0, min(1.0, float(agreement)))
            + 0.15 * max(0.0, min(1.0, float(recency)))
            + 0.30 * max(0.0, min(1.0, float(constraint_coverage)))
        )
        return cls(
            source_count_score=max(0.0, min(1.0, float(source_count))),
            agreement_score=max(0.0, min(1.0, float(agreement))),
            recency_score=max(0.0, min(1.0, float(recency))),
            constraint_coverage_score=max(0.0, min(1.0, float(constraint_coverage))),
            weighted_score=round(w, 2),
            justification=justification,
        )


class TokenUsage(BaseModel):
    """Tracks token consumption and estimated inference costs."""

    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    estimated_cost_usd: float = Field(default=0.0)

    def add(
        self,
        prompt: int,
        completion: int,
        cost_per_1k_prompt: float = 0.00015,
        cost_per_1k_comp: float = 0.0006,
    ) -> None:
        """Add consumed tokens and update cost estimate (default: gpt-4o-mini rates)."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        cost = (prompt / 1000.0) * cost_per_1k_prompt + (completion / 1000.0) * cost_per_1k_comp
        self.estimated_cost_usd = round(self.estimated_cost_usd + cost, 5)


# ---------------------------------------------------------------------------
# Final answer model
# ---------------------------------------------------------------------------


class ResearchAnswer(BaseModel):
    """The complete, structured answer produced by the research agent.

    Attributes:
        id: Unique answer identifier.
        query: The original research question.
        direct_answer: A concise, direct answer to the query.
        key_findings: Bullet-point findings that support the direct answer.
        evidence: Sourced evidence pieces backing the answer.
        confidence: Overall confidence in the answer (0–1).
        confidence_rationale: Explanation of how the confidence score was derived.
        limitations: Known limitations or caveats of the research.
        follow_up_questions: Suggested questions for deeper exploration.
        plan: The research plan that was executed.
        session_id: Optional session identifier for persistence.
        created_at: UTC timestamp when this answer was produced.
        model_used: Identifier of the LLM that synthesised this answer.
        search_provider_used: Identifier of the search provider used.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    query: str = Field(..., min_length=1)
    direct_answer: str = Field(..., min_length=1)
    key_findings: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_rationale: str = Field(default="")
    limitations: list[Limitation] = Field(default_factory=list)
    follow_up_questions: list[FollowUpQuestion] = Field(default_factory=list)
    plan: ResearchPlan
    session_id: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    model_used: Optional[str] = Field(default=None)
    search_provider_used: Optional[str] = Field(default=None)
    confidence_factors: Optional[ConfidenceFactors] = Field(default=None)
    token_usage: Optional[TokenUsage] = Field(default=None)
    is_degraded: bool = Field(default=False, description="True if answer generated under partial failure")
    degradation_reasons: list[str] = Field(default_factory=list)

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        """Silently clamp confidence to [0, 1]."""
        return max(0.0, min(1.0, float(v)))

    @property
    def high_quality_evidence(self) -> list[Evidence]:
        """Evidence items with a composite score >= 0.6."""
        return [e for e in self.evidence if e.composite_score >= 0.6]

    @property
    def has_high_severity_limitations(self) -> bool:
        """``True`` when any limitation has HIGH severity."""
        return any(lim.severity == Severity.HIGH for lim in self.limitations)

    def summary(self) -> str:
        """Return a brief human-readable summary of this answer."""
        lines = [
            f"Query: {self.query}",
            f"Confidence: {self.confidence:.0%}",
            f"Answer: {self.direct_answer}",
            f"Evidence pieces: {len(self.evidence)}",
            f"Limitations: {len(self.limitations)}",
        ]
        return "\n".join(lines)
