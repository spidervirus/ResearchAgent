"""
Tests for Verifier stage in research_agent.core.verifier.
"""

import json
import pytest
from research_agent.core.context import ResearchContext
from research_agent.core.verifier import Verifier
from research_agent.models import Evidence
from tests.conftest import MockLLM


@pytest.mark.asyncio
async def test_verifier_filters_low_relevance(
    sample_context: ResearchContext,
    sample_evidence: list[Evidence],
) -> None:
    sample_context.raw_evidence = list(sample_evidence)
    # min_relevance_score is 0.5; third item in sample_evidence has 0.2
    verifier = Verifier(llm=None, min_relevance_score=0.5)

    res = await verifier.verify(sample_context)

    assert res.passed
    assert len(res.filtered_evidence) == 2
    assert all(e.relevance_score >= 0.5 for e in res.filtered_evidence)
    assert len(sample_context.verified_evidence) == 2


@pytest.mark.asyncio
async def test_verifier_boosts_academic_domains(sample_context: ResearchContext) -> None:
    academic_ev = Evidence(
        source_url="https://arxiv.org/abs/2401.99999",
        source_title="ArXiv Benchmark",
        excerpt="Detailed latency and recall trade-offs under high throughput.",
        relevance_score=0.7,
        quality_score=0.5,
    )
    sample_context.raw_evidence = [academic_ev]
    verifier = Verifier(llm=None)

    await verifier.verify(sample_context)

    # ArXiv domain receives a +0.25 quality boost
    assert academic_ev.quality_score > 0.5


@pytest.mark.asyncio
async def test_verifier_handles_empty_evidence(sample_context: ResearchContext) -> None:
    sample_context.raw_evidence = []
    verifier = Verifier(llm=None)

    res = await verifier.verify(sample_context)

    assert not res.passed
    assert len(res.filtered_evidence) == 0
    assert len(res.issues) > 0


@pytest.mark.asyncio
async def test_verifier_llm_critique(sample_context: ResearchContext) -> None:
    ev = Evidence(
        source_url="https://example.com/db",
        source_title="DB Guide",
        excerpt="A thorough benchmark of latency and costs for Qdrant.",
        relevance_score=0.5,
        quality_score=0.5,
    )
    sample_context.raw_evidence = [ev]

    canned_critique = {
        "evaluations": [
            {"index": 0, "relevance_score": 0.95, "quality_score": 0.9, "critique": "Strong technical content"}
        ],
        "issues": ["Minor: lacks memory quantization comparison"],
        "notes": "Good baseline source",
    }
    mock_llm = MockLLM(responses=[json.dumps(canned_critique)])
    verifier = Verifier(llm=mock_llm, min_relevance_score=0.5)

    res = await verifier.verify(sample_context)

    assert res.passed
    assert ev.relevance_score == 0.95
    assert "Minor: lacks memory quantization comparison" in res.issues

