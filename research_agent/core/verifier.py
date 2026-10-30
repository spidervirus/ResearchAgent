"""
Verifier / Critic — Stage 3 of the research pipeline.

Evaluates raw evidence for:
1. Relevance to the original query
2. Source credibility and reputation
3. Consistency across sources and coverage of critical questions
4. Filters out noise and ranks high-value sources

Combines deterministic domain heuristics with LLM-based semantic critique.

Usage::

    verifier = Verifier(llm=llm, min_relevance_score=0.35)
    result = await verifier.verify(ctx)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional
from urllib.parse import urlparse

from research_agent.core.context import ResearchContext
from research_agent.llm.base import BaseLLM
from research_agent.models import Evidence, VerificationResult

logger = logging.getLogger(__name__)

# Domain reputation adjustments
_REPUTATION_BOOSTS = {
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

_REPUTATION_PENALTIES = {
    "reddit.com": -0.15,
    "quora.com": -0.20,
    "medium.com": -0.05,
    "pinterest.com": -0.30,
}


_VERIFIER_SYSTEM = """\
You are an expert research verifier and evidence critic.
Analyze the provided evidence items gathered for the research query.

For each item:
1. Score relevance to the query (0.0 to 1.0).
2. Score credibility and technical depth (0.0 to 1.0).
3. Identify any factual contradictions or significant gaps across the evidence.

Return ONLY a valid JSON object with the following schema:
{
  "evaluations": [
    {
      "index": 0,
      "relevance_score": 0.85,
      "quality_score": 0.80,
      "critique": "Directly benchmarks QPS and cost"
    }
  ],
  "issues": ["Issue 1: Contradiction on memory footprint between source 1 and 3"],
  "notes": "Good coverage of Qdrant and Milvus, limited data on Chroma high concurrency"
}
"""


class Verifier:
    """Validates, scores, and filters evidence gathered by the executor.

    Attributes:
        llm: Optional language model used for semantic analysis.
        min_relevance_score: Minimum relevance score required to retain an evidence item.
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        min_relevance_score: float = 0.3,
    ) -> None:
        self.llm = llm
        self.min_relevance_score = min_relevance_score

    async def verify(self, ctx: ResearchContext) -> VerificationResult:
        """Evaluate and filter all evidence in the context.

        Args:
            ctx: The active :class:`~research_agent.core.context.ResearchContext`.

        Returns:
            A populated :class:`~research_agent.models.VerificationResult`.
        """
        raw = ctx.raw_evidence
        if not raw:
            logger.warning("No raw evidence found to verify.")
            res = VerificationResult(
                passed=False,
                issues=["No evidence collected from tools"],
                filtered_evidence=[],
                notes="Execution produced zero evidence records.",
            )
            ctx.set_verified_evidence([])
            return res

        # 1. Apply heuristic adjustments first
        for ev in raw:
            self._apply_heuristics(ev)

        issues: list[str] = []
        notes = "Heuristic evaluation completed."

        # 2. LLM semantic critique if model is configured
        if self.llm and len(raw) > 0:
            llm_issues, llm_notes = await self._run_llm_critique(ctx, raw)
            if llm_issues:
                issues.extend(llm_issues)
            if llm_notes:
                notes = llm_notes

        # 3. Filter items meeting the threshold
        filtered = [ev for ev in raw if ev.relevance_score >= self.min_relevance_score]

        # Sort descending by composite score
        filtered.sort(key=lambda e: e.composite_score, reverse=True)

        passed = len(filtered) > 0
        if not passed:
            issues.append(f"All evidence fell below minimum relevance score of {self.min_relevance_score}")

        result = VerificationResult(
            passed=passed,
            issues=issues,
            filtered_evidence=filtered,
            notes=notes,
        )

        ctx.set_verified_evidence(filtered)
        return result

    def _apply_heuristics(self, ev: Evidence) -> None:
        """Adjust scores using domain reputation and excerpt quality."""
        url_lower = ev.source_url.lower()

        # Domain boosts
        for pattern, boost in _REPUTATION_BOOSTS.items():
            if pattern in url_lower:
                ev.quality_score = min(1.0, ev.quality_score + boost)
                break

        # Domain penalties
        for pattern, penalty in _REPUTATION_PENALTIES.items():
            if pattern in url_lower:
                ev.quality_score = max(0.1, ev.quality_score + penalty)
                break

        # Content length heuristic
        if len(ev.excerpt.strip()) < 50:
            ev.relevance_score = max(0.1, ev.relevance_score - 0.2)
            ev.quality_score = max(0.1, ev.quality_score - 0.2)

    async def _run_llm_critique(
        self, ctx: ResearchContext, evidence_list: list[Evidence]
    ) -> tuple[list[str], str]:
        """Ask LLM to evaluate evidence quality and identify inconsistencies."""
        summaries = []
        for i, ev in enumerate(evidence_list[:12]):  # Limit to 12 items to prevent context blowup
            snippet = ev.excerpt[:300].replace("\n", " ")
            summaries.append(f"[{i}] Title: {ev.source_title}\nURL: {ev.source_url}\nExcerpt: {snippet}")

        evidence_text = "\n\n".join(summaries)
        messages = [
            {"role": "system", "content": _VERIFIER_SYSTEM},
            {
                "role": "user",
                "content": f"Query: {ctx.query}\n\nEvidence items:\n{evidence_text}",
            },
        ]

        try:
            raw_response = await self.llm.complete(messages, temperature=0.1)
            cleaned = raw_response.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

            data = json.loads(cleaned)
            evals = data.get("evaluations", [])
            for item in evals:
                idx = item.get("index")
                if isinstance(idx, int) and 0 <= idx < len(evidence_list):
                    rel = item.get("relevance_score")
                    qual = item.get("quality_score")
                    if rel is not None:
                        evidence_list[idx].relevance_score = float(rel)
                    if qual is not None:
                        evidence_list[idx].quality_score = float(qual)

            issues = data.get("issues", [])
            notes = data.get("notes", "")
            return issues, notes
        except Exception as exc:
            logger.warning("LLM verification critique failed: %s; relying on heuristics", exc)
            return [], f"Semantic critique fallback: {exc}"

