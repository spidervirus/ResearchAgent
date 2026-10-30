"""
Progressive Tutorial — Step 4: Multi-Step Execution Loop with Intermediate Results.

WHY THIS STEP EXISTS:
Once a plan exists, we need an Executor that:
1. Iterates through pending plan steps in sequence.
2. Dispatches arguments to the registered tool.
3. Automatically converts tool outputs (search results, page text) into structured Evidence objects.
4. Records an audit trail of ToolCall events with timings.

Run:
    python examples/04_multistep_loop.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from research_agent.core.context import ResearchContext
from research_agent.core.executor import Executor
from research_agent.models import PlanStep, ResearchPlan
from research_agent.tools.code_runner import CodeRunnerTool
from research_agent.tools.web_search import WebSearchTool

console = Console()


async def main() -> None:
    console.print(
        Panel(
            "[bold cyan]Step 4: Multi-Step Execution Loop[/bold cyan]\n"
            "[dim]Executing plan steps and aggregating raw evidence in ResearchContext[/dim]",
            border_style="cyan",
        )
    )

    query = "Vector database RAM calculation and QPS benchmark"
    ctx = ResearchContext(query=query)

    # Construct a sample 2-step plan: search + calculation
    step1 = PlanStep(
        description="Search vector database benchmarks",
        tool_name="web_search",
        tool_args={"query": "qdrant milvus benchmark 10k qps", "num_results": 3},
    )
    step2 = PlanStep(
        description="Calculate vector memory footprint in Python",
        tool_name="code_runner",
        tool_args={
            "code": (
                "num_vectors = 5_000_000\n"
                "dim = 768\n"
                "bytes_total = num_vectors * dim * 4\n"
                "gb = bytes_total / (1024**3)\n"
                "print(f'Uncompressed Float32 RAM: {gb:.2f} GB')\n"
                "print(f'With Int8 Quantization: {gb/4:.2f} GB')"
            ),
            "language": "python",
        },
    )
    ctx.plan = ResearchPlan(query=query, steps=[step1, step2])

    # Instantiate Executor with available tools
    tools = {
        "web_search": WebSearchTool(provider="duckduckgo"),
        "code_runner": CodeRunnerTool(),
    }
    executor = Executor(tools=tools)

    console.print(f"[dim]Executing {len(ctx.plan.steps)} plan steps sequentially...[/dim]\n")
    await executor.execute(ctx)

    # Render step statuses
    step_table = Table(title="⚡ Execution Status", header_style="bold green")
    step_table.add_column("Step", style="white")
    step_table.add_column("Tool", style="cyan")
    step_table.add_column("Status", style="magenta")
    step_table.add_column("Duration", style="yellow")

    for s in ctx.plan.steps:
        dur = f"{s.duration_ms:.1f} ms" if s.duration_ms else "N/A"
        step_table.add_row(s.description, s.tool_name, s.status.upper(), dur)
    console.print(step_table)

    # Render collected raw evidence
    console.print(f"\n[bold]Collected Raw Evidence Items:[/bold] {len(ctx.raw_evidence)}")
    ev_table = Table(title="📚 Raw Evidence Collected", header_style="bold blue")
    ev_table.add_column("Source", style="cyan")
    ev_table.add_column("Excerpt Snippet", style="white")

    for ev in ctx.raw_evidence[:4]:
        title = ev.source_title or ev.source_url
        if len(title) > 30:
            title = title[:28] + ".."
        ev_table.add_row(title, ev.excerpt[:70].replace("\n", " ") + "...")
    console.print(ev_table)

    # Show code execution output
    code_calls = ctx.get_tool_calls_for("code_runner")
    if code_calls and code_calls[0].result:
        console.print("\n[bold]Code Execution Output:[/bold]")
        console.print(Panel(code_calls[0].result.get("stdout", "").strip(), border_style="dim"))

    console.print(
        "\n[bold yellow]Observations & Next Step:[/bold yellow]\n"
        "1. Notice that raw evidence contains search snippets of varying quality.\n"
        "2. Some sources are high-authority docs, others are forum posts or noise.\n"
        "3. In [bold green]Step 5 (examples/05_source_verification.py)[/bold green], "
        "we filter noise and rank credibility using the Verifier."
    )


if __name__ == "__main__":
    asyncio.run(main())

