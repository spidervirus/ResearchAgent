"""
Evaluation Runner for ResearchAgent.

Benchmarks ResearchAgent against the curated research questions in questions.json.

Usage::

    # Run quick mock evaluation (offline, zero API keys required)
    python evals/run_evals.py --mock

    # Run live evaluation with configured LLM & search tools
    python evals/run_evals.py --live --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure package is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evals.evaluator import AnswerEvaluator, EvalResult
from research_agent import ResearchAgent
from research_agent.models import (
    ConfidenceFactors,
    Evidence,
    FollowUpQuestion,
    Limitation,
    PlanStep,
    ResearchAnswer,
    ResearchPlan,
    Severity,
)

console = Console()


def _generate_mock_answer(spec: dict[str, Any]) -> ResearchAnswer:
    """Generate a realistic mock answer matching the question specification for testing."""
    qid = spec["id"]
    query = spec["query"]
    constraints = spec.get("required_constraints", [])

    # Sample mock responses grounded in the question's domain
    direct_ans = (
        f"Based on rigorous benchmark analysis for '{query}', the primary recommendation focuses on "
        f"evaluating {', '.join(constraints)}. Empirical tests show that single-node deployment achieves "
        f"the required throughput and cost targets while minimizing operational infrastructure overhead."
    )

    ev1 = Evidence(
        source_url="https://arxiv.org/abs/2401.00001",
        source_title=f"Empirical Benchmark Study for {constraints[0].capitalize()}",
        excerpt=f"Empirical tests validate {constraints[0]} and {constraints[1]} under sustained load.",
        relevance_score=0.92,
        quality_score=0.95,
    )
    ev2 = Evidence(
        source_url=f"https://docs.{constraints[0].lower()}.io/guides",
        source_title=f"Official {constraints[0].capitalize()} Documentation",
        excerpt=f"Architecture guide on memory sizing, indexing, and {constraints[2] if len(constraints) > 2 else 'scaling'}.",
        relevance_score=0.88,
        quality_score=0.90,
    )

    findings = [
        f"{constraints[0].capitalize()} directly meets constraints under production workloads [ArXiv Benchmark](https://arxiv.org/abs/2401.00001).",
        f"Official guides recommend quantization and tuning for {constraints[1] if len(constraints) > 1 else 'cost'} [Documentation](https://docs.{constraints[0].lower()}.io/guides).",
    ]

    cf = ConfidenceFactors.calculate(
        source_count=0.90,
        agreement=0.90,
        recency=0.85,
        constraint_coverage=0.95,
        justification="Multi-source empirical data confirms operational benchmarks.",
    )

    return ResearchAnswer(
        query=query,
        direct_answer=direct_ans,
        key_findings=findings,
        evidence=[ev1, ev2],
        confidence=cf.weighted_score,
        confidence_factors=cf,
        confidence_rationale="Grounded in empirical benchmarks and official architecture guides.",
        limitations=[
            Limitation(
                description=f"Performance depends on hardware instance type and network latency.",
                severity=Severity.MEDIUM,
            ),
            Limitation(
                description="Long-term operational costs require ongoing monitoring.",
                severity=Severity.LOW,
            ),
        ],
        follow_up_questions=[
            FollowUpQuestion(
                question="What is the p99 latency under 2x write bursts?",
                rationale="Validates high-concurrency resilience.",
            )
        ],
        plan=ResearchPlan(query=query, steps=[PlanStep(description="Execute benchmark", tool_name="web_search")]),
    )


async def run_evals(mock_mode: bool = True, limit: int = 10, category_filter: str = "") -> None:
    questions_file = Path(__file__).parent / "questions.json"
    with open(questions_file, "r", encoding="utf-8") as f:
        questions: list[dict[str, Any]] = json.load(f)

    if category_filter:
        questions = [q for q in questions if q.get("category") == category_filter]

    questions = questions[:limit]

    console.print(
        Panel(
            f"[bold cyan]ResearchAgent Evaluation Harness v0.1[/bold cyan]\n"
            f"Testing {len(questions)} research questions | Mode: {'MOCK' if mock_mode else 'LIVE'}",
            border_style="cyan",
        )
    )

    evaluator = AnswerEvaluator()
    results: list[EvalResult] = []

    agent: Any = None
    if not mock_mode:
        agent = ResearchAgent(human_gate=False, verbose=False)

    for i, spec in enumerate(questions, start=1):
        console.print(f"[bold dim][{i}/{len(questions)}][/bold dim] Evaluating [cyan]{spec['id']}[/cyan]...")

        if mock_mode:
            answer = _generate_mock_answer(spec)
        else:
            try:
                answer = await agent.research(spec["query"])
            except Exception as exc:
                console.print(f"  [red]Agent execution failed: {exc}[/red]")
                continue

        res = evaluator.evaluate(spec, answer)
        results.append(res)

    # Render Summary Table
    table = Table(title="📊 Evaluation Benchmark Report", header_style="bold magenta")
    table.add_column("Question ID", style="cyan")
    table.add_column("Score", style="bold yellow", width=8)
    table.add_column("Passed", width=8)
    table.add_column("Direct Ans", width=11)
    table.add_column("Constraints", width=12)
    table.add_column("Citations", width=10)
    table.add_column("Confidence", width=11)
    table.add_column("Limitations", width=12)

    total_score = 0.0
    passed_count = 0

    for r in results:
        total_score += r.total_score
        if r.passed:
            passed_count += 1
        pass_badge = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"

        c = {check.name: check.passed for check in r.checks}
        table.add_row(
            r.question_id,
            f"{r.total_score:.1f}",
            pass_badge,
            "✓" if c.get("Direct Answer Decisiveness") else "✗",
            "✓" if c.get("Constraint & Keyword Coverage") else "✗",
            "✓" if c.get("Grounded Citations") else "✗",
            "✓" if c.get("Confidence Calibration") else "✗",
            "✓" if c.get("Explicit Limitations") else "✗",
        )

    console.print()
    console.print(table)

    avg_score = total_score / len(results) if results else 0.0
    pass_pct = (passed_count / len(results)) * 100.0 if results else 0.0

    console.print(
        f"\n[bold]Summary:[/bold] {passed_count}/{len(results)} Passed ({pass_pct:.1f}%) "
        f"• [bold]Average Score:[/bold] {avg_score:.1f} / 100.0"
    )

    # Save JSON report
    report_file = Path(__file__).parent / "evals_report.json"
    report_data = {
        "total_questions": len(results),
        "passed": passed_count,
        "pass_percentage": round(pass_pct, 1),
        "average_score": round(avg_score, 1),
        "results": [
            {
                "id": r.question_id,
                "score": r.total_score,
                "passed": r.passed,
                "checks": [{"name": c.name, "score": c.score, "passed": c.passed, "details": c.details} for c in r.checks],
            }
            for r in results
        ],
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    console.print(f"[dim]Report saved to: {report_file}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchAgent Evaluation Harness")
    parser.add_argument("--mock", action="store_true", default=True, help="Run mock evaluation (default)")
    parser.add_argument("--live", action="store_true", help="Run live evaluation with external LLM and tools")
    parser.add_argument("--limit", type=int, default=10, help="Max questions to evaluate (default 10)")
    parser.add_argument("--filter", type=str, default="", help="Filter by category")

    args = parser.parse_args()
    mock_mode = not args.live

    asyncio.run(run_evals(mock_mode=mock_mode, limit=args.limit, category_filter=args.filter))


if __name__ == "__main__":
    main()
