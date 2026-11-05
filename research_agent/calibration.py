"""
Stand-alone Confidence Calibration & Verification Layer.

Extracts the calibrated confidence scoring checklist, evidence verification,
and explicit limitations generation from ResearchAgent into a framework-agnostic,
zero-dependency component that can be used directly with Claude Code, OpenClaw,
Hermes, LangChain, LlamaIndex, or any custom LLM agent loop.

Usage:
------
```python
from research_agent.calibration import Calibrator, EvidenceItem

calibrator = Calibrator()

# Add evidence retrieved from any agent / tools
evidence = [
    EvidenceItem(
        url="https://qdrant.tech/benchmarks",
        title="Qdrant Benchmarks 2024",
        excerpt="Achieved 12,500 QPS with HNSW under 100M 768-dim vectors.",
        relevance=0.9,
        quality=0.85,
    ),
    EvidenceItem(
        url="https://milvus.io/docs/benchmark",
        title="Milvus Performance Report",
        excerpt="Cluster setup yielded 11,200 QPS with distributed query nodes.",
        relevance=0.85,
        quality=0.8,
    ),
]

# Calculate calibrated confidence matrix
factors = calibrator.calculate_factors(
    evidence=evidence,
    query="Compare Qdrant and Milvus QPS on 100M vectors with latency constraints",
    agreement_score=0.9,
    recency_score=0.95,
    constraint_coverage_score=0.85,
    justification="Multiple independent 2024 benchmarks validate QPS constraints.",
)

print(f"Calibrated Confidence: {factors.weighted_score * 100}%")
print(f"Breakdown: {factors.model_dump()}")
```
"""

from __future__ import annotations

import json
from typing import Any, List, Optional
from urllib.parse import urlparse
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Reputation Heuristics (Deterministic)
# ---------------------------------------------------------------------------

REPUTATION_BOOSTS: dict[str, float] = {
    "arxiv.org": 0.25,
    "github.com": 0.15,
    "docs.": 0.15,
    "readthedocs.io": 0.15,
    ".edu": 0.20,
    ".gov": 0.20,
    "nature.com": 0.25,
    "ieee.org": 0.25,
    "acm.org": 0.25,
    "wikipedia.org": 0.10,
}

REPUTATION_PENALTIES: dict[str, float] = {
    "reddit.com": -0.15,
    "quora.com": -0.20,
    "medium.com": -0.05,
    "pinterest.com": -0.30,
}


class EvidenceItem(BaseModel):
    """Lightweight evidence item compatible with any agent tool output."""

    url: str = Field(..., min_length=1)
    title: str = Field(default="")
    excerpt: str = Field(..., min_length=1)
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    quality: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("relevance", "quality", mode="before")
    @classmethod
    def clamp_scores(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @property
    def composite_score(self) -> float:
        """Weighted combination of relevance (60%) and quality (40%)."""
        return 0.6 * self.relevance + 0.4 * self.quality


class CalibrationFactors(BaseModel):
    """Rigorous 4-factor breakdown driving the overall confidence score."""

    source_count_score: float = Field(default=0.5, ge=0.0, le=1.0)
    agreement_score: float = Field(default=0.5, ge=0.0, le=1.0)
    recency_score: float = Field(default=0.5, ge=0.0, le=1.0)
    constraint_coverage_score: float = Field(default=0.5, ge=0.0, le=1.0)
    weighted_score: float = Field(default=0.5, ge=0.0, le=1.0)
    justification: str = Field(default="")


class CalibratedLimitation(BaseModel):
    """Explicit limitation with severity ranking."""

    description: str
    severity: str = Field(default="medium", description="'low', 'medium', or 'high'")


class CalibratedAssessment(BaseModel):
    """Comprehensive post-execution calibration report for any agent."""

    confidence: float
    confidence_factors: CalibrationFactors
    confidence_rationale: str
    limitations: list[CalibratedLimitation] = Field(default_factory=list)
    passed_verification: bool = True
    issues: list[str] = Field(default_factory=list)
    is_degraded: bool = False
    degradation_reasons: list[str] = Field(default_factory=list)


class Calibrator:
    """Reusable engine for source verification and confidence calibration.

    Can be injected into Claude Code workflows, OpenClaw, Hermes, or custom agents.
    """

    def __init__(self, min_relevance_score: float = 0.30) -> None:
        self.min_relevance_score = min_relevance_score

    def adjust_reputation(self, url: str, base_quality: float) -> float:
        """Adjust base quality score using deterministic domain reputation heuristics."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        score = base_quality

        for d, boost in REPUTATION_BOOSTS.items():
            if d in domain:
                score += boost
                break

        for d, penalty in REPUTATION_PENALTIES.items():
            if d in domain:
                score += penalty
                break

        return max(0.0, min(1.0, score))

    def filter_evidence(
        self, evidence: list[EvidenceItem]
    ) -> tuple[list[EvidenceItem], list[str]]:
        """Filter out low-relevance or spam sources and flag quality issues."""
        accepted: list[EvidenceItem] = []
        issues: list[str] = []

        for item in evidence:
            item.quality = self.adjust_reputation(item.url, item.quality)
            if item.relevance < self.min_relevance_score:
                issues.append(
                    f"Dropped '{item.title or item.url}' (relevance {item.relevance:.2f} < {self.min_relevance_score})"
                )
            else:
                accepted.append(item)

        if not accepted and evidence:
            issues.append("All retrieved evidence was filtered out due to low relevance.")

        return accepted, issues

    def score_source_volume(self, evidence: list[EvidenceItem]) -> float:
        """Score volume of high-quality sources (>=3 high quality = 1.0, 1-2 = 0.5-0.7, 0 = 0.1)."""
        high_quality = [e for e in evidence if e.composite_score >= 0.60]
        count = len(high_quality)
        if count >= 3:
            return 1.0
        elif count == 2:
            return 0.75
        elif count == 1:
            return 0.50
        elif len(evidence) > 0:
            return 0.30
        return 0.10

    def calculate_factors(
        self,
        evidence: list[EvidenceItem],
        agreement_score: float = 0.8,
        recency_score: float = 0.8,
        constraint_coverage_score: float = 0.8,
        justification: str = "",
        query: str = "",
    ) -> CalibrationFactors:
        """Compute the weighted 4-factor confidence matrix."""
        source_count_score = self.score_source_volume(evidence)

        # 30% Source count/quality, 25% Agreement, 15% Recency, 30% Constraint coverage
        weighted = (
            0.30 * source_count_score
            + 0.25 * max(0.0, min(1.0, float(agreement_score)))
            + 0.15 * max(0.0, min(1.0, float(recency_score)))
            + 0.30 * max(0.0, min(1.0, float(constraint_coverage_score)))
        )

        return CalibrationFactors(
            source_count_score=round(source_count_score, 2),
            agreement_score=round(agreement_score, 2),
            recency_score=round(recency_score, 2),
            constraint_coverage_score=round(constraint_coverage_score, 2),
            weighted_score=round(weighted, 2),
            justification=justification,
        )

    def assess(
        self,
        evidence: list[EvidenceItem],
        direct_answer: str,
        agreement_score: float = 0.8,
        recency_score: float = 0.8,
        constraint_coverage_score: float = 0.8,
        limitations: Optional[list[CalibratedLimitation]] = None,
        failed_tools: Optional[list[str]] = None,
        confidence_rationale: str = "",
    ) -> CalibratedAssessment:
        """Produce a complete calibration assessment with graceful degradation checks."""
        filtered, issues = self.filter_evidence(evidence)
        degraded = False
        degradation_reasons: list[str] = []

        if failed_tools:
            degraded = True
            degradation_reasons.append(f"Tools failed: {', '.join(failed_tools)}")

        if len(filtered) < 2:
            degraded = True
            degradation_reasons.append(
                f"Low evidence count ({len(filtered)} verified sources)"
            )

        factors = self.calculate_factors(
            evidence=filtered,
            agreement_score=agreement_score,
            recency_score=recency_score,
            constraint_coverage_score=constraint_coverage_score,
            justification="; ".join(degradation_reasons) if degraded else "Sufficient multi-source evidence",
        )

        final_conf = factors.weighted_score
        if degraded:
            final_conf = min(final_conf, 0.55)

        default_limitations = limitations or []
        if degraded and not any("degraded" in l.description.lower() for l in default_limitations):
            default_limitations.append(
                CalibratedLimitation(
                    description=f"Confidence degraded: {'; '.join(degradation_reasons)}",
                    severity="high",
                )
            )

        return CalibratedAssessment(
            confidence=round(final_conf, 2),
            confidence_factors=factors,
            confidence_rationale=confidence_rationale
            or f"Calculated from 4-factor matrix with {len(filtered)} verified sources.",
            limitations=default_limitations,
            passed_verification=len(issues) == 0,
            issues=issues,
            is_degraded=degraded,
            degradation_reasons=degradation_reasons,
        )

    @staticmethod
    def prompt_template() -> str:
        """Returns the standardized LLM prompt snippet to enforce 4-factor scoring in any agent."""
        return """\
### Confidence Calibration & Verification Checklist
Before producing your final response, evaluate your answer across these 4 factors:
1. source_count_score (0.0-1.0): (>=3 reputable sources = 1.0; 1-2 = 0.5-0.7; 0 = 0.1)
2. agreement_score (0.0-1.0): Consensus across independent sources (1.0 = unanimous, 0.4 = conflicting)
3. recency_score (0.0-1.0): Timeliness of data (1.0 = modern/fresh, 0.5 = outdated/undated)
4. constraint_coverage_score (0.0-1.0): Coverage of ALL constraints stated in the query (1.0 = 100%, 0.4 = partial)

Weighted Formula: (0.30 * sources) + (0.25 * agreement) + (0.15 * recency) + (0.30 * constraints).
Explicitly list at least 1-2 caveats or limitations with severity: "low", "medium", or "high".
"""

