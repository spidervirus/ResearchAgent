"""
Progressive Tutorial — Step 6: Structured Final Answer with Confidence + Limitations.

WHY THIS STEP EXISTS:
Unstructured text outputs from LLMs leave users wondering:
- How confident is the model, and why?
- What are the assumptions and limitations that could invalidate the answer?
- Where did each specific claim come from?

The Synthesizer solves this by enforcing a strictly modeled ResearchAnswer containing:
1. Direct Answer
2. Key Findings with Sourced Citations ([Source](URL))
3. Calibrated Confidence Score & Rationale
4. Limitations with Severity Ratings (LOW, MEDIUM, HIGH)
5. Actionable Follow-up Questions

Run:
    python examples/06_structured_answer_confidence.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from research_agent.core.context import ResearchContext
from research_agent.core.synthesizer import Synthesizer
from research_agent.models import (
    Evidence,
    FollowUpQuestion,
    Limitation,
    PlanStep,
    ResearchAnswer,
    ResearchPlan,
    Severity,
)

console = Console()


async def main() -> None:
    console.print(
        Panel(
            "[bold cyan]Step 6: Structured Synthesis with Confidence & Limitations[/bold cyan]\n"
            "[dim]Producing an auditable, citation-grounded research answer[/dim]",
            border_style="cyan",
        )
    )

    query = "Compare top 3 open-source vector databases for 10k QPS startup on a tight budget"
    ctx = ResearchContext(query=query)

    ev1 = Evidence(
        source_url="https://arxiv.org/abs/2401.00001",
        source_title="Vector Search Benchmarking Under 10k QPS",
        excerpt="Qdrant sustained 12k QPS on a 4-core instance with 98% recall. Milvus achieved 18k QPS on a 3-node cluster.",
        relevance_score=0.95,
        quality_score=0.9,
    )
    ev2 = Evidence(
        source_url="https://qdrant.tech/documentation/guides/quantization",
        source_title="Qdrant Quantization & RAM Reduction Guide",
        excerpt="Scalar quantization cuts vector RAM requirements by up to 4x while maintaining high throughput.",
        relevance_score=0.9,
        quality_score=0.85,
    )
    ctx.verified_evidence = [ev1, ev2]
    ctx.plan = ResearchPlan(query=query, steps=[PlanStep(description="Benchmark search", tool_name="web_search")])

    has_llm = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))

    if has_llm:
        from research_agent.agent import _resolve_llm

        synthesizer = Synthesizer(llm=_resolve_llm())
        console.print("[dim]Synthesizing structured answer via LLM...[/dim]")
        answer = await synthesizer.synthesize(ctx)
    else:
        # Self-contained educational representation
        answer = ResearchAnswer(
            query=query,
            direct_answer=(
                "Qdrant is the recommended open-source vector database for a startup targeting 10k QPS "
                "with tight budget constraints. Its single-binary Rust engine offers superior memory "
                "efficiency over distributed JVM setups, and scalar quantization enables running 10M+ "
                "vectors on affordable commodity hardware."
            ),
            key_findings=[
                "Qdrant sustains 12k QPS on modest 4-core hardware [Vector Benchmarks](https://arxiv.org/abs/2401.00001)",
                "Scalar quantization achieves ~4x RAM reduction with negligible throughput loss [Qdrant Docs](https://qdrant.tech/documentation/guides/quantization)",
                "Milvus delivers high peak throughput but requires multiple coordinated services, raising operational cost.",
            ],
            confidence=0.88,
            confidence_rationale="Empirical benchmarks and official documentation corroborate single-node throughput and RAM savings.",
            limitations=[
                Limitation(
                    description="Benchmark results assume 768-dim embeddings; 1536-dim vectors increase memory pressure by 2x.",
                    severity=Severity.MEDIUM,
                ),
                Limitation(
                    description="Pricing estimates depend on cloud provider compute spot rates.",
                    severity=Severity.LOW,
                ),
            ],
            follow_up_questions=[
                FollowUpQuestion(
                    question="What is the latency trade-off between In-Memory vs On-Disk HNSW graphs?",
                    rationale="On-disk graphs cut RAM costs further but can increase p99 latency under write load.",
                )
            ],
            plan=ctx.plan,
            evidence=ctx.verified_evidence,
        )

    # 1. Render Direct Answer
    console.print(Panel(answer.direct_answer, title="🎯 Direct Recommendation", border_style="bold green"))

    # 2. Confidence Bar
    conf_pct = int(answer.confidence * 100)
    bar = "█" * (conf_pct // 10) + "░" * (10 - (conf_pct // 10))
    console.print(f"  [bold]Confidence:[/bold] [green][{bar}] {conf_pct}%[/green]")
    console.print(f"  [dim]Rationale: {answer.confidence_rationale}[/dim]\n")

    # 3. Key Findings with Citations
    table = Table(title="📌 Key Findings & Grounded Citations", header_style="bold cyan")
    table.add_column("#", width=3)
    table.add_column("Finding")
    for i, finding in enumerate(answer.key_findings, 1):
        table.add_row(str(i), finding)
    console.print(table)
    console.print()

    # 4. Limitations
    lim_table = Table(title="⚠️ Known Limitations & Caveats", header_style="bold yellow")
    lim_table.add_column("Severity", width=10)
    lim_table.add_column("Description")
    for lim in answer.limitations:
        color = "red" if lim.severity == Severity.HIGH else "yellow" if lim.severity == Severity.MEDIUM else "green"
        lim_table.add_row(f"[{color}]{lim.severity.upper()}[/{color}]", lim.description)
    console.print(lim_table)

    console.print(
        "\n[bold yellow]Observations & Next Step:[/bold yellow]\n"
        "1. We now have a fully structured, auditable answer.\n"
        "2. But what if we want human approval before running costly tools or synthesis?\n"
        "3. In [bold green]Step 7 (examples/07_human_in_the_loop_gates.py)[/bold green], "
        "we insert interactive approval checkpoints into the pipeline."
    )


if __name__ == "__main__":
    asyncio.run(main())

