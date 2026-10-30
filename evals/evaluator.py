"""
Rubric-based evaluation engine for ResearchAgent answers.

Scores research answers across 5 core pillars:
1. Direct Answer Decisiveness (20 points)
2. Constraint & Keyword Coverage (25 points)
3. Grounded Citation & Source Verification (25 points)
4. Confidence Calibration & Factor Breakdown (15 points)
5. Explicit Limitations & Severity Categorization (15 points)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from research_agent.models import ResearchAnswer


@dataclass
class EvalCheck:
    name: str
    passed: bool
    score: float
    max_score: float
    details: str


@dataclass
class EvalResult:
    question_id: str
    query: str
    total_score: float
    max_score: float = 100.0
    passed: bool = False
    checks: list[EvalCheck] = field(default_factory=list)

    @property
    def percentage(self) -> float:
        return (self.total_score / self.max_score) * 100.0


class AnswerEvaluator:
    """Evaluates a ResearchAnswer against curated question criteria."""

    def evaluate(self, spec: dict[str, Any], answer: ResearchAnswer) -> EvalResult:
        checks: list[EvalCheck] = []
        total = 0.0

        # Pillar 1: Direct Answer Decisiveness (20 pts)
        ans_text = answer.direct_answer.strip()
        ans_len = len(ans_text)
        if ans_len >= 80:
            score1 = 20.0
            p1 = True
            d1 = f"Decisive direct answer provided ({ans_len} characters)"
        elif ans_len >= 30:
            score1 = 10.0
            p1 = True
            d1 = f"Brief direct answer ({ans_len} characters)"
        else:
            score1 = 0.0
            p1 = False
            d1 = "Direct answer is missing or trivial"
        checks.append(EvalCheck("Direct Answer Decisiveness", p1, score1, 20.0, d1))
        total += score1

        # Pillar 2: Constraint Coverage (25 pts)
        req_constraints = spec.get("required_constraints", [])
        combined_text = (
            f"{answer.direct_answer} {' '.join(answer.key_findings)} "
            f"{' '.join(lim.description for lim in answer.limitations)}"
        ).lower()

        matched = [c for c in req_constraints if c.lower() in combined_text]
        coverage_ratio = len(matched) / len(req_constraints) if req_constraints else 1.0
        score2 = round(coverage_ratio * 25.0, 1)
        p2 = coverage_ratio >= 0.6
        d2 = f"Matched {len(matched)}/{len(req_constraints)} required constraints ({', '.join(matched)})"
        checks.append(EvalCheck("Constraint & Keyword Coverage", p2, score2, 25.0, d2))
        total += score2

        # Pillar 3: Grounded Citations (25 pts)
        has_sources = len(answer.evidence) >= spec.get("min_evidence_count", 1)
        citations_found = []
        for finding in answer.key_findings:
            links = re.findall(r"\[([^\]]+)\]\((https?://[^\)]+)\)", finding)
            citations_found.extend(links)

        if has_sources and len(citations_found) >= 1:
            score3 = 25.0
            p3 = True
            d3 = f"Found {len(citations_found)} markdown citations backed by {len(answer.evidence)} verified sources"
        elif has_sources:
            score3 = 15.0
            p3 = True
            d3 = f"Evidence items collected ({len(answer.evidence)}), but few explicit inline citations"
        elif len(citations_found) >= 1:
            score3 = 10.0
            p3 = False
            d3 = "Inline citations present but missing underlying evidence records"
        else:
            score3 = 0.0
            p3 = False
            d3 = "Zero citations or source evidence found"
        checks.append(EvalCheck("Grounded Citations", p3, score3, 25.0, d3))
        total += score3

        # Pillar 4: Confidence Calibration (15 pts)
        min_conf = spec.get("min_confidence", 0.60)
        has_factors = answer.confidence_factors is not None
        conf_ok = answer.confidence >= min_conf

        if conf_ok and has_factors:
            score4 = 15.0
            p4 = True
            d4 = f"Confidence {answer.confidence:.0%} meets threshold with rigorous 4-factor breakdown"
        elif conf_ok:
            score4 = 10.0
            p4 = True
            d4 = f"Confidence {answer.confidence:.0%} meets threshold without detailed factor breakdown"
        else:
            score4 = 5.0
            p4 = False
            d4 = f"Confidence {answer.confidence:.0%} below threshold {min_conf:.0%}"
        checks.append(EvalCheck("Confidence Calibration", p4, score4, 15.0, d4))
        total += score4

        # Pillar 5: Explicit Limitations (15 pts)
        lim_count = len(answer.limitations)
        if lim_count >= 2:
            score5 = 15.0
            p5 = True
            d5 = f"Found {lim_count} severity-rated limitations"
        elif lim_count == 1:
            score5 = 10.0
            p5 = True
            d5 = "Found 1 limitation"
        else:
            score5 = 0.0
            p5 = False
            d5 = "No explicit limitations or caveats stated"
        checks.append(EvalCheck("Explicit Limitations", p5, score5, 15.0, d5))
        total += score5

        overall_passed = total >= 70.0 and all(c.passed for c in checks[:3])
        return EvalResult(
            question_id=spec.get("id", "unknown"),
            query=spec.get("query", ""),
            total_score=total,
            passed=overall_passed,
            checks=checks,
        )

