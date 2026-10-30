"""
Progressive Tutorial — Step 5: Source Verification and Filtering.

WHY THIS STEP EXISTS:
Raw search results are filled with noisy blog posts, marketing hype, or outdated discussions.
The Verifier stage acts as a critical filter:
1. Applies domain heuristics (+0.25 for academic/arXiv, +0.15 for docs/github, penalties for forums).
2. Computes a composite credibility score (60% relevance + 40% quality).
3. Evaluates excerpts for depth and cross-source contradictions.
4. Drops any evidence falling below the relevance threshold (e.g. < 0.4).

Run:
    python examples/05_source_verification.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from research_agent.core.context import ResearchContext
from research_agent.core.verifier import Verifier
from research_agent.models import Evidence

console = Console()


async def main() -> None:
    console.print(
        Panel(
            "[bold cyan]Step 5: Source Verification & Evidence Filtering[/bold cyan]\n"
            "[dim]Separating authoritative evidence from marketing noise and low-signal forums[/dim]",
            border_style="cyan",
        )
    )

    ctx = ResearchContext(query="Benchmark vector databases for 10k QPS startup")

    # Simulate a mixed bag of raw evidence collected by search tools
    ctx.raw_evidence = [
        Evidence(
            source_url="https://arxiv.org/abs/2401.00001",
            source_title="Benchmarking Distributed Vector Search Under 10k QPS Concurrency",
            excerpt="Empirical analysis measuring recall vs latency across Qdrant, Milvus, and Weaviate.",
            relevance_score=0.9,
            quality_score=0.7,
        ),
        Evidence(
            source_url="https://qdrant.tech/documentation/benchmarks",
            source_title="Official Qdrant Throughput Guide",
            excerpt="Detailed instructions on running HNSW index with mmap and scalar quantization.",
            relevance_score=0.85,
            quality_score=0.7,
        ),
        Evidence(
            source_url="https://www.reddit.com/r/programming/comments/123/vectordb_opinions",
            source_title="What is your favorite vector database?",
            excerpt="I tried Chroma on my laptop and it was super easy to install with pip.",
            relevance_score=0.3,
            quality_score=0.5,
        ),
        Evidence(
            source_url="https://random-marketing-blog.com/top-10-ai-trends-2024",
            source_title="Top 10 AI Tools You Must Know",
            excerpt="Vectors are the new oil of artificial intelligence.",
            relevance_score=0.1,
            quality_score=0.3,
        ),
    ]

    console.print(f"[bold]Input Raw Evidence Items:[/bold] {len(ctx.raw_evidence)}\n")

    # Run Verifier (deterministic heuristics + threshold filter)
    verifier = Verifier(llm=None, min_relevance_score=0.4)
    v_res = await verifier.verify(ctx)

    # Render filtered evidence table
    table = Table(title="🔍 Verified & Ranked Evidence", header_style="bold green")
    table.add_column("Source Title", style="white")
    table.add_column("Domain", style="cyan")
    table.add_column("Quality Score", style="magenta", width=14)
    table.add_column("Composite Score", style="bold yellow", width=16)
    table.add_column("Status", width=10)

    for ev in ctx.raw_evidence:
        domain = ev.source_url.split("/")[2]
        passed = ev in v_res.filtered_evidence
        status_text = "[green]KEPT[/green]" if passed else "[red]FILTERED[/red]"
        table.add_row(
            ev.source_title[:35] + "...",
            domain,
            f"{ev.quality_score:.2f}",
            f"{ev.composite_score:.2f}",
            status_text,
        )

    console.print(table)
    console.print(f"\n[bold]Surviving Verified Items:[/bold] {len(v_res.filtered_evidence)}")

    console.print(
        "\n[bold yellow]Observations & Next Step:[/bold yellow]\n"
        "1. Notice that arXiv and official docs gained quality score boosts.\n"
        "2. Reddit and marketing fluff were filtered out due to low relevance.\n"
        "3. In [bold green]Step 6 (examples/06_structured_answer_confidence.py)[/bold green], "
        "we feed this verified evidence to the Synthesizer to generate a structured answer."
    )


if __name__ == "__main__":
    asyncio.run(main())

