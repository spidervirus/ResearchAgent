"""
Synthesizer — Stage 4 of the research pipeline.

Transforms verified evidence, plan execution logs, and critique notes into
a structured, source-backed ResearchAnswer with:
- Direct answer / recommendation
- Key findings with grounded citations
- Rigorous confidence calibration checklist (sources, agreement, recency, constraint coverage)
- Graceful degradation under partial tool or evidence failures
- Known limitations and caveats
- Suggested next questions / verification steps
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from research_agent.core.context import ResearchContext
from research_agent.llm.base import BaseLLM
from research_agent.models import (
    ConfidenceFactors,
    FollowUpQuestion,
    Limitation,
    ResearchAnswer,
    ResearchPlan,
    Severity,
)

logger = logging.getLogger(__name__)

_SYNTHESIZER_SYSTEM = """\
You are an elite research synthesizer. Your goal is to produce a rigorous,
highly practical, source-backed research answer.

Guidelines:
1. Direct Answer: Provide a crisp, decisive executive summary addressing the exact question.
2. Key Findings: Provide clear analytical points. Reference specific evidence using markdown links [Source Title](URL).
3. Confidence Calibration Checklist:
   You MUST score 4 explicit factors between 0.0 and 1.0 before assigning final confidence:
   - source_count_score: (>=3 strong sources = 1.0; 1-2 sources = 0.5; 0 sources = 0.1)
   - agreement_score: (do sources agree on numbers/conclusions? 1.0 = unanimous, 0.4 = conflicting)
   - recency_score: (is evidence fresh/modern? 1.0 = 2023-2024+, 0.5 = older/undated)
   - constraint_coverage_score: (does evidence cover EVERY explicit constraint in the query? 1.0 = full coverage, 0.4 = partially covered)
   Calculate weighted confidence = (0.30 * source_count) + (0.25 * agreement) + (0.15 * recency) + (0.30 * constraint_coverage).
4. Limitations: Explicitly call out assumptions, missing data, potential counter-arguments, or time-sensitive data. Specify severity: "low", "medium", or "high".
5. Follow-Up Questions: Suggest 2 to 4 concrete follow-up questions or benchmark tests to verify findings.

Return ONLY a valid JSON object with the exact keys:
{
  "direct_answer": "...",
  "key_findings": [
    "Qdrant delivers ~12k QPS under HNSW indexing with minimal memory overhead [Qdrant Benchmarks](https://example.com/bench)",
    "Milvus requires separate coordinator nodes, making ops heavier for small teams [Milvus Docs](https://example.com/milvus)"
  ],
  "confidence_factors": {
    "source_count_score": 0.9,
    "agreement_score": 0.85,
    "recency_score": 0.8,
    "constraint_coverage_score": 0.9,
    "justification": "Three independent 2024 benchmarks validate QPS and memory constraints."
  },
  "confidence": 0.87,
  "confidence_rationale": "High consistency across official documentation and third-party benchmarks; however real-world cluster pricing fluctuates.",
  "limitations": [
    {"description": "Benchmark numbers depend heavily on vector dimensions (e.g., 768 vs 1536)", "severity": "medium"}
  ],
  "follow_up_questions": [
    {"question": "What is the memory footprint difference when quantization is enabled?", "rationale": "Quantization cuts RAM cost by up to 4x."}
  ]
}
"""


class Synthesizer:
    """Produces a structured ResearchAnswer from verified evidence with calibrated confidence."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def synthesize(self, ctx: ResearchContext) -> ResearchAnswer:
        """Synthesize evidence into a final ResearchAnswer.

        Gracefully degrades when evidence is sparse or tools experienced partial failure.

        Args:
            ctx: Active research context containing evidence and execution logs.

        Returns:
            A populated :class:`~research_agent.models.ResearchAnswer`.
        """
        evidence_list = ctx.verified_evidence
        is_degraded = False
        degradation_reasons = list(ctx.degradation_reasons)

        # Check for tool failures in plan
        if ctx.plan:
            failed_steps = [s for s in ctx.plan.steps if s.error]
            if failed_steps:
                is_degraded = True
                for s in failed_steps:
                    degradation_reasons.append(f"Step '{s.description}' failed: {s.error}")

        # Check if verifier filtered out everything
        if not evidence_list:
            if ctx.raw_evidence:
                evidence_list = ctx.raw_evidence
                is_degraded = True
                degradation_reasons.append("All retrieved sources fell below the strict credibility threshold; using raw fallback data.")
            else:
                is_degraded = True
                degradation_reasons.append("Zero external evidence items could be retrieved by tools.")

        # Format evidence items for prompt
        ev_items = []
        for i, ev in enumerate(evidence_list[:15]):  # Up to 15 top sources
            ev_items.append(
                f"Source #{i+1}:\n"
                f"Title: {ev.source_title}\n"
                f"URL: {ev.source_url}\n"
                f"Excerpt: {ev.excerpt[:800]}\n"
                f"Credibility Score: {ev.composite_score:.2f}"
            )
        evidence_block = "\n\n".join(ev_items) if ev_items else "No external evidence retrieved."

        user_content = (
            f"Research Question: {ctx.query}\n\n"
            f"Gathered Evidence:\n{evidence_block}\n\n"
        )
        if ctx.metadata.get("verifier_issues"):
            user_content += f"Known Evidence Gaps/Issues:\n{ctx.metadata['verifier_issues']}\n\n"

        if is_degraded:
            user_content += (
                f"⚠️ SYSTEM ALERT - PARTIAL FAILURE / DEGRADED MODE:\n"
                f"The following execution issues occurred:\n- "
                + "\n- ".join(degradation_reasons)
                + "\nProvide the best possible analysis, but explicitly reflect high uncertainty: "
                "cap confidence at 0.45 or lower, justify the low score in confidence_factors, "
                "and list the missing data under Limitations.\n\n"
            )

        messages = [
            {"role": "system", "content": _SYNTHESIZER_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        # Track prompt tokens
        prompt_str = _SYNTHESIZER_SYSTEM + user_content
        prompt_tokens = self.llm.count_tokens(prompt_str)

        raw = await self.llm.complete(messages, temperature=0.2)
        comp_tokens = self.llm.count_tokens(raw)
        ctx.token_usage.add(prompt_tokens, comp_tokens)

        answer = self._parse_synthesis(raw, ctx, is_degraded, degradation_reasons)
        ctx.answer = answer
        return answer

    def _parse_synthesis(
        self,
        raw: str,
        ctx: ResearchContext,
        is_degraded: bool,
        degradation_reasons: list[str],
    ) -> ResearchAnswer:
        """Parse raw LLM output into a ResearchAnswer with calibrated confidence and fallback."""
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        plan = ctx.plan or ResearchPlan(query=ctx.query, steps=[])

        try:
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                else:
                    raise

            # Parse limitations
            parsed_limits = []
            for item in data.get("limitations", []):
                if isinstance(item, dict):
                    desc = item.get("description", "")
                    sev_str = str(item.get("severity", "medium")).lower()
                    sev = (
                        Severity.HIGH
                        if "high" in sev_str
                        else Severity.LOW
                        if "low" in sev_str
                        else Severity.MEDIUM
                    )
                    if desc:
                        parsed_limits.append(Limitation(description=desc, severity=sev))
                elif isinstance(item, str):
                    parsed_limits.append(Limitation(description=item, severity=Severity.MEDIUM))

            # Add degradation limitations if running degraded
            if is_degraded:
                for reason in degradation_reasons:
                    parsed_limits.insert(
                        0,
                        Limitation(
                            description=f"Degraded run: {reason}",
                            severity=Severity.HIGH,
                        ),
                    )

            # Parse follow up questions
            parsed_followups = []
            for item in data.get("follow_up_questions", []):
                if isinstance(item, dict):
                    q = item.get("question", "")
                    r = item.get("rationale", "")
                    if q:
                        parsed_followups.append(FollowUpQuestion(question=q, rationale=r))
                elif isinstance(item, str):
                    parsed_followups.append(FollowUpQuestion(question=item, rationale=""))

            # Parse and calibrate confidence factors
            cf_data = data.get("confidence_factors")
            confidence_factors: Optional[ConfidenceFactors] = None
            if isinstance(cf_data, dict):
                confidence_factors = ConfidenceFactors.calculate(
                    source_count=cf_data.get("source_count_score", 0.5),
                    agreement=cf_data.get("agreement_score", 0.5),
                    recency=cf_data.get("recency_score", 0.5),
                    constraint_coverage=cf_data.get("constraint_coverage_score", 0.5),
                    justification=cf_data.get("justification", ""),
                )
                final_conf = confidence_factors.weighted_score
            else:
                final_conf = float(data.get("confidence", 0.7 if not is_degraded else 0.4))

            if is_degraded:
                final_conf = min(0.45, final_conf)

            final_conf = max(0.0, min(1.0, final_conf))

            return ResearchAnswer(
                query=ctx.query,
                direct_answer=data.get("direct_answer", cleaned[:500]),
                key_findings=data.get("key_findings", []),
                evidence=ctx.verified_evidence or ctx.raw_evidence,
                confidence=final_conf,
                confidence_factors=confidence_factors,
                confidence_rationale=data.get(
                    "confidence_rationale",
                    confidence_factors.justification if confidence_factors else "Confidence based on synthesis.",
                ),
                limitations=parsed_limits,
                follow_up_questions=parsed_followups,
                plan=plan,
                session_id=ctx.session_id,
                model_used=getattr(self.llm, "model", None),
                token_usage=ctx.token_usage,
                is_degraded=is_degraded,
                degradation_reasons=degradation_reasons,
            )

        except Exception as exc:
            logger.warning("Failed to parse synthesis JSON (%s); constructing graceful degraded answer", exc)
            fallback_conf = 0.35 if is_degraded else 0.5
            return ResearchAnswer(
                query=ctx.query,
                direct_answer=cleaned[:1000] if cleaned else "Synthesis completed with partial fallback.",
                key_findings=["Direct synthesis completed with unstructured response."],
                evidence=ctx.verified_evidence or ctx.raw_evidence,
                confidence=fallback_conf,
                confidence_factors=ConfidenceFactors.calculate(0.3, 0.4, 0.4, 0.4, "Fallback parsing mode"),
                confidence_rationale="Fallback parsing used due to formatting mismatch in model output.",
                limitations=[
                    Limitation(
                        description="Model returned unstructured text rather than JSON.",
                        severity=Severity.LOW,
                    )
                ]
                + [Limitation(description=r, severity=Severity.HIGH) for r in degradation_reasons],
                follow_up_questions=[
                    FollowUpQuestion(
                        question="Would you like to rerun with a narrower search scope?",
                        rationale="A narrower query reduces search noise.",
                    )
                ],
                plan=plan,
                session_id=ctx.session_id,
                model_used=getattr(self.llm, "model", None),
                token_usage=ctx.token_usage,
                is_degraded=True,
                degradation_reasons=degradation_reasons or [str(exc)],
            )
