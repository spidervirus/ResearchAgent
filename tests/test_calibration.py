"""
Tests for the standalone Confidence Calibration & Verification Layer.
"""

from research_agent.calibration import (
    CalibratedLimitation,
    Calibrator,
    EvidenceItem,
)


def test_reputation_heuristic_adjustments():
    calibrator = Calibrator()
    # arxiv gets boost
    arxiv_score = calibrator.adjust_reputation("https://arxiv.org/abs/2301.0001", 0.5)
    assert arxiv_score == 0.75

    # reddit gets penalty
    reddit_score = calibrator.adjust_reputation("https://reddit.com/r/MachineLearning", 0.5)
    assert reddit_score == 0.35


def test_filter_evidence_drops_low_relevance():
    calibrator = Calibrator(min_relevance_score=0.40)
    evidence = [
        EvidenceItem(url="https://docs.qdrant.tech", title="Docs", excerpt="Great QPS", relevance=0.8),
        EvidenceItem(url="https://spam.com", title="Spam", excerpt="Unrelated", relevance=0.2),
    ]
    accepted, issues = calibrator.filter_evidence(evidence)
    assert len(accepted) == 1
    assert accepted[0].title == "Docs"
    assert len(issues) == 1
    assert "Dropped 'Spam'" in issues[0]


def test_4_factor_calculation():
    calibrator = Calibrator()
    evidence = [
        EvidenceItem(url="https://arxiv.org/1", excerpt="Source 1", relevance=0.9, quality=0.8),
        EvidenceItem(url="https://docs.python.org", excerpt="Source 2", relevance=0.85, quality=0.8),
        EvidenceItem(url="https://nature.com/article", excerpt="Source 3", relevance=0.9, quality=0.9),
    ]

    factors = calibrator.calculate_factors(
        evidence=evidence,
        agreement_score=0.9,
        recency_score=0.8,
        constraint_coverage_score=1.0,
        justification="Strong alignment across verified domains",
    )

    assert factors.source_count_score == 1.0
    assert factors.agreement_score == 0.9
    assert factors.recency_score == 0.8
    assert factors.constraint_coverage_score == 1.0
    # Expected weighted: 0.30(1.0) + 0.25(0.9) + 0.15(0.8) + 0.30(1.0) = 0.30 + 0.225 + 0.12 + 0.30 = 0.945 -> 0.95
    assert factors.weighted_score == 0.95


def test_graceful_degradation_assessment():
    calibrator = Calibrator()
    # Only 1 source -> should trigger degradation
    evidence = [
        EvidenceItem(url="https://medium.com/post", excerpt="Single post", relevance=0.7, quality=0.5),
    ]

    assessment = calibrator.assess(
        evidence=evidence,
        direct_answer="Partial analysis",
        failed_tools=["page_fetcher"],
    )

    assert assessment.is_degraded is True
    assert assessment.confidence <= 0.55
    assert any("page_fetcher" in r for r in assessment.degradation_reasons)
    assert any(l.severity == "high" for l in assessment.limitations)
