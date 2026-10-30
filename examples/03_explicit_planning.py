"""
Progressive Tutorial — Step 3: Explicit Planning Step Before Acting.

WHY THIS STEP EXISTS:
Free-form agents that jump immediately into tool calls often get stuck in
infinite search loops or wander down irrelevant tangents.
A dedicated Planner forces the LLM to think ahead:
1. Decompose the research question into concrete sub-topics.
2. Select the right tool for each sub-topic (search vs page fetch vs code execution).
3. Produce a structured, inspectable checklist before a single tool executes.

Run:
    python examples/03_explicit_planning.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from research_agent.core.context import ResearchContext
from research_agent.core.planner import Planner

console = Console()


async def main() -> None:
    console.print(
        Panel(
            "[bold cyan]Step 3: Explicit Planning Before Acting[/bold cyan]\n"
            "[dim]Generating a structured ResearchPlan with concrete steps and tools[/dim]",
            border_style="cyan",
        )
    )

    query = (
        "Compare the top 3 open-source vector databases for a startup "
        "with 10k QPS and tight cost constraints."
    )
    ctx = ResearchContext(query=query)

    has_llm = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))

    if has_llm:
        from research_agent.agent import _resolve_llm

        planner = Planner(llm=_resolve_llm(), max_steps=5)
        console.print("[dim]Generating structured research plan via LLM...[/dim]")
        plan = await planner.plan(ctx)
    else:
        # Self-contained educational demonstration of what the planner produces
        from research_agent.models import PlanStep, ResearchPlan

        plan = ResearchPlan(
            query=query,
            rationale="To compare vector databases for 10k QPS under cost constraints, we first locate benchmark data, fetch memory guides for the top contenders, and compute RAM cost calculations.",
            steps=[
                PlanStep(
                    description="Search for empirical vector database QPS benchmarks",
                    tool_name="web_search",
                    tool_args={"query": "open source vector database benchmark 10k QPS 2024", "num_results": 5},
                ),
                PlanStep(
                    description="Fetch Qdrant documentation on memory optimization and quantization",
                    tool_name="fetch_page",
                    tool_args={"url": "https://qdrant.tech/documentation/guides/multiple-partitions/"},
                ),
                PlanStep(
                    description="Fetch Milvus architectural scalability overview",
                    tool_name="fetch_page",
                    tool_args={"url": "https://milvus.io/docs/architecture_overview.md"},
                ),
                PlanStep(
                    description="Calculate memory and server cost for 10 million vectors",
                    tool_name="run_code",
                    tool_args={
                        "code": "vectors=10_000_000; dims=768; ram_gb=(vectors*dims*4)/(1024**3); print(f'RAM required: {ram_gb:.1f} GB')",
                        "language": "python",
                    },
                ),
            ],
        )

    # Render plan table
    table = Table(title="📋 Generated Research Plan", header_style="bold magenta")
    table.add_column("#", width=4, style="dim")
    table.add_column("Step Description", style="white")
    table.add_column("Tool", style="cyan", width=14)
    table.add_column("Arguments", style="dim green")

    for i, step in enumerate(plan.steps, start=1):
        args_summary = json.dumps(step.tool_args)
        if len(args_summary) > 40:
            args_summary = args_summary[:38] + ".."
        table.add_row(str(i), step.description, step.tool_name, args_summary)

    console.print(table)
    if plan.rationale:
        console.print(f"\n[bold]Planner Rationale:[/bold] {plan.rationale}")

    console.print(
        "\n[bold yellow]Observations & Next Step:[/bold yellow]\n"
        "1. Having an explicit plan allows us to inspect, modify, or approve actions.\n"
        "2. In [bold green]Step 4 (examples/04_multistep_loop.py)[/bold green], "
        "we run an Executor loop that executes these steps and collects raw evidence."
    )


if __name__ == "__main__":
    asyncio.run(main())

