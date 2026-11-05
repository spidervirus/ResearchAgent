"""
Integration Example: Using ResearchAgent's Reusable Calibration & Verification Layer
with Claude Code, OpenClaw, Hermes, or custom LLM agent pipelines.

Unlike black-box agents that assert answers with uncalibrated confidence, this
pattern takes raw outputs and evidence from ANY external agent or search tool and
runs it through:
1. Domain reputation scoring (deterministic URL heuristics)
2. Relevance filtering
3. 4-factor confidence matrix calculation:
   - Source volume (30%)
   - Inter-source agreement (25%)
   - Freshness / recency (15%)
   - Constraint coverage (30%)
4. Graceful degradation detection
5. Explicit limitations ranking
"""

from research_agent.calibration import (
    CalibratedLimitation,
    Calibrator,
    EvidenceItem,
)


def run_claude_or_openclaw_post_processing():
    print("=" * 70)
    print("🔬 Reusable Calibration Layer for Claude Code / OpenClaw / Hermes")
    print("=" * 70)

    # 1. Instantiate the standalone calibrator (no LLM keys or complex setup needed)
    calibrator = Calibrator(min_relevance_score=0.40)

    # 2. Simulated tool outputs from an external agent (e.g., Claude Code, Hermes, or custom tool loop)
    raw_agent_evidence = [
        EvidenceItem(
            url="https://arxiv.org/abs/2401.03456",
            title="Vector DB Scaling Benchmarks",
            excerpt="Qdrant sustained 12,400 QPS with p99 latency < 15ms on 50M vectors.",
            relevance=0.92,
            quality=0.85,
        ),
        EvidenceItem(
            url="https://docs.qdrant.tech/cloud/benchmarks",
            title="Official Qdrant Cloud Metrics",
            excerpt="HNSW indexing delivers 10k-15k QPS with quantized memory reduction.",
            relevance=0.88,
            quality=0.80,
        ),
        EvidenceItem(
            url="https://reddit.com/r/MachineLearning/comments/123",
            title="Reddit Discussion on Milvus vs Qdrant",
            excerpt="I tried Milvus on Kubernetes and setup was complicated, took 2 days.",
            relevance=0.45,
            quality=0.50,
        ),
        EvidenceItem(
            url="https://spammy-content-farm.com/best-vector-db",
            title="Top 10 Vector DBs",
            excerpt="Vector databases are good for artificial intelligence.",
            relevance=0.20,  # Below min_relevance_score -> filtered out
            quality=0.30,
        ),
    ]

    print("\n[1] Filtering evidence and applying reputation heuristics...")
    filtered_evidence, issues = calibrator.filter_evidence(raw_agent_evidence)

    for item in filtered_evidence:
        print(f"  ✓ Retained: {item.title} (Relevance: {item.relevance:.2f}, Adj. Quality: {item.quality:.2f})")
    for issue in issues:
        print(f"  ✗ Issue flagged: {issue}")

    # 3. Assess the response with calibrated confidence
    assessment = calibrator.assess(
        evidence=filtered_evidence,
        direct_answer="For high-throughput vector search on 50M vectors under strict latency constraints, Qdrant is recommended.",
        agreement_score=0.90,
        recency_score=0.95,
        constraint_coverage_score=0.85,
        limitations=[
            CalibratedLimitation(
                description="Cluster network latency was not benchmarked under cross-region replication.",
                severity="medium",
            )
        ],
    )

    print("\n[2] Calibrated Confidence Assessment:")
    print(f"  Confidence Score: {assessment.confidence * 100:.1f}%")
    print(f"  Is Degraded: {assessment.is_degraded}")
    print(f"  Rationale: {assessment.confidence_rationale}")

    print("\n[3] 4-Factor Calibration Checklist:")
    factors = assessment.confidence_factors
    print(f"  • Source Volume Score:     {factors.source_count_score:.2f} (Weight: 30%)")
    print(f"  • Agreement Score:         {factors.agreement_score:.2f} (Weight: 25%)")
    print(f"  • Recency Score:           {factors.recency_score:.2f} (Weight: 15%)")
    print(f"  • Constraint Coverage:     {factors.constraint_coverage_score:.2f} (Weight: 30%)")
    print(f"  ──────────────────────────────────────────")
    print(f"  • Weighted Score:          {factors.weighted_score:.2f}")

    print("\n[4] Explicit Limitations:")
    for lim in assessment.limitations:
        print(f"  • [{lim.severity.upper()}] {lim.description}")

    print("\n" + "=" * 70)
    print("Integration complete! The same layer can be added to any agent harness.")


if __name__ == "__main__":
    run_claude_or_openclaw_post_processing()
