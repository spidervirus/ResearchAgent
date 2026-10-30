"""
Tests for Synthesizer stage in research_agent.core.synthesizer.
"""

import json
import pytest
from research_agent.core.context import ResearchContext
from research_agent.core.synthesizer import Synthesizer
from research_agent.models import Evidence, PlanStep, ResearchPlan, Severity
from tests.conftest import MockLLM


@pytest.mark.asyncio
async def test_synthesizer_produces_answer(sample_context: ResearchContext) -> None:
    ev = Evidence(
        source_url="https://example.com/vector-bench",
        source_title="Vector Benchmark 2024",
        excerpt="Qdrant sustained 12k QPS on a 4-core node with 0.98 recall.",
        relevance_score=0.9,
        quality_score=0.9,
    )
    sample_context.verified_evidence = [ev]
    sample_context.plan = ResearchPlan(
        query=sample_context.query,
        steps=[PlanStep(description="Search", tool_name="web_search")],
    )

    canned_answer = {
        "direct_answer": "Qdrant is the recommended open-source vector database.",
        "key_findings": [
            "Qdrant maintains 12k QPS on modest hardware [Vector Benchmark 2024](https://example.com/vector-bench)"
        ],
        "confidence": 0.88,
        "confidence_rationale": "Empirical benchmark data directly validates throughput targets.",
        "limitations": [
            {"description": "Benchmark used 768-dim vectors; higher dims reduce QPS.", "severity": "medium"}
        ],
        "follow_up_questions": [
            {"question": "How does product quantization impact recall?", "rationale": "Cuts memory cost"}
        ],
    }

    mock_llm = MockLLM(responses=[json.dumps(canned_answer)])
    synthesizer = Synthesizer(llm=mock_llm)

    answer = await synthesizer.synthesize(sample_context)

    assert answer is not None
    assert "Qdrant" in answer.direct_answer
    assert answer.confidence == 0.88
    assert len(answer.key_findings) == 1
    assert len(answer.limitations) == 1
    assert answer.limitations[0].severity == Severity.MEDIUM
    assert sample_context.answer == answer


@pytest.mark.asyncio
async def test_synthesizer_handles_parse_error(sample_context: ResearchContext) -> None:
    # Model returns raw markdown instead of pure JSON
    raw_text = "Here is my answer: We recommend Qdrant due to its high efficiency and Rust backend."
    mock_llm = MockLLM(responses=[raw_text])
    synthesizer = Synthesizer(llm=mock_llm)

    answer = await synthesizer.synthesize(sample_context)

    # Fallback synthesizer logic should still return a valid ResearchAnswer
    assert answer is not None
    assert "Qdrant" in answer.direct_answer
    assert 0.0 <= answer.confidence <= 1.0
    assert len(answer.limitations) >= 1


@pytest.mark.asyncio
async def test_synthesizer_calibrates_confidence_factors(sample_context: ResearchContext) -> None:
    sample_context.verified_evidence = [
        Evidence(
            source_url="https://example.com/source",
            source_title="Source",
            excerpt="High quality benchmark evidence",
            relevance_score=0.9,
            quality_score=0.9,
        )
    ]
    canned_answer = {
        "direct_answer": "Recommendation backed by calibrated evidence.",
        "key_findings": ["Finding 1 [Link](https://example.com)"],
        "confidence_factors": {
            "source_count_score": 1.0,
            "agreement_score": 0.8,
            "recency_score": 0.9,
            "constraint_coverage_score": 0.9,
            "justification": "All criteria strongly validated",
        },
        "confidence": 0.9,
        "confidence_rationale": "High confidence",
        "limitations": [{"description": "Minor caveat", "severity": "low"}],
        "follow_up_questions": [],
    }
    mock_llm = MockLLM(responses=[json.dumps(canned_answer)])
    synthesizer = Synthesizer(llm=mock_llm)

    answer = await synthesizer.synthesize(sample_context)
    assert answer.confidence_factors is not None
    # 0.30*1.0 + 0.25*0.8 + 0.15*0.9 + 0.30*0.9 = 0.30 + 0.20 + 0.135 + 0.27 = 0.905 -> 0.9
    assert 0.88 <= answer.confidence <= 0.92
    assert answer.confidence_factors.source_count_score == 1.0


@pytest.mark.asyncio
async def test_synthesizer_graceful_degradation(sample_context: ResearchContext) -> None:
    # Context has failed plan steps and zero evidence
    sample_context.raw_evidence = []
    sample_context.verified_evidence = []
    step = PlanStep(description="Search", tool_name="web_search")
    step.mark_failed("Network unreachable")
    sample_context.plan = ResearchPlan(query=sample_context.query, steps=[step])

    canned = {
        "direct_answer": "Degraded answer due to network outage.",
        "key_findings": [],
        "confidence": 0.8,  # Attempting high confidence despite degradation
        "confidence_rationale": "High",
        "limitations": [],
    }
    mock_llm = MockLLM(responses=[json.dumps(canned)])
    synthesizer = Synthesizer(llm=mock_llm)

    answer = await synthesizer.synthesize(sample_context)
    assert answer.is_degraded
    # Confidence must be capped to <= 0.45 under degradation
    assert answer.confidence <= 0.45
    assert any("Network unreachable" in r for r in answer.degradation_reasons)

