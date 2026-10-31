"""
Tests for Pydantic data models in research_agent.models.
"""

import pytest
from research_agent.models import (
    Evidence,
    FollowUpQuestion,
    Limitation,
    PlanStep,
    ResearchAnswer,
    ResearchPlan,
    Severity,
    StepStatus,
    ToolCall,
    VerificationResult,
)


def test_plan_step_lifecycle() -> None:
    step = PlanStep(description="Search vector DBs", tool_name="web_search")
    assert step.status == StepStatus.PENDING
    assert step.result is None
    assert step.error is None

    step.mark_running()
    assert step.status == StepStatus.RUNNING
    assert step.started_at is not None

    step.mark_done({"results": [1, 2]})
    assert step.status == StepStatus.DONE
    assert step.result == {"results": [1, 2]}
    assert step.finished_at is not None
    assert step.duration_ms is not None and step.duration_ms >= 0


def test_plan_step_failure() -> None:
    step = PlanStep(description="Fetch broken URL", tool_name="fetch_page")
    step.mark_running()
    step.mark_failed("HTTP 404")
    assert step.status == StepStatus.FAILED
    assert step.error == "HTTP 404"


def test_research_plan_properties() -> None:
    step1 = PlanStep(description="Step 1", tool_name="web_search")
    step2 = PlanStep(description="Step 2", tool_name="fetch_page")
    plan = ResearchPlan(query="Test query", steps=[step1, step2])

    assert len(plan.pending_steps) == 2
    assert not plan.is_complete

    step1.mark_done("done")
    assert len(plan.completed_steps) == 1
    assert len(plan.pending_steps) == 1

    step2.mark_failed("err")
    assert len(plan.failed_steps) == 1
    assert plan.is_complete

    step3 = PlanStep(description="Step 3", tool_name="code_runner", depends_on=["1", 2])
    assert step3.depends_on == ["1", "2"]
    step3.mark_skipped("Prerequisite failed")
    assert step3.status == StepStatus.SKIPPED

    plan2 = ResearchPlan(query="Test 2", steps=[step3])
    assert len(plan2.skipped_steps) == 1
    assert plan2.is_complete


def test_evidence_score_clamping() -> None:
    ev = Evidence(
        source_url="https://example.com",
        source_title="Example",
        excerpt="Benchmark details",
        relevance_score=1.5,
        quality_score=-0.5,
    )
    assert ev.relevance_score == 1.0
    assert ev.quality_score == 0.0
    assert ev.composite_score == 0.6  # 0.6*1.0 + 0.4*0.0


def test_research_answer_serialization() -> None:
    step = PlanStep(description="Step 1", tool_name="web_search")
    plan = ResearchPlan(query="Vector DB query", steps=[step])
    ev = Evidence(
        source_url="https://example.com/paper",
        source_title="Vector Paper",
        excerpt="Findings here",
        relevance_score=0.9,
        quality_score=0.9,
    )

    answer = ResearchAnswer(
        query="Vector DB query",
        direct_answer="Qdrant is recommended for 10k QPS startup.",
        key_findings=["Qdrant handles HNSW well with Rust efficiency."],
        evidence=[ev],
        confidence=0.88,
        confidence_rationale="Solid benchmark backing.",
        limitations=[Limitation(description="Requires RAM tuning", severity=Severity.MEDIUM)],
        follow_up_questions=[FollowUpQuestion(question="What about scalar quantization?", rationale="Cuts cost")],
        plan=plan,
    )

    json_str = answer.model_dump_json()
    assert "Qdrant" in json_str
    reconstructed = ResearchAnswer.model_validate_json(json_str)
    assert reconstructed.query == answer.query
    assert reconstructed.confidence == 0.88
    assert len(reconstructed.evidence) == 1
    assert len(reconstructed.limitations) == 1
    assert reconstructed.summary() != ""

