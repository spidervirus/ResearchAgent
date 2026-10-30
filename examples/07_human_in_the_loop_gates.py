"""
Progressive Tutorial — Step 7: Human-in-the-Loop Approval Gates.

WHY THIS STEP EXISTS:
Autonomous agents should not spend tokens or execute tools unchecked.
Human-in-the-loop (HITL) checkpoints place safety gates at critical transitions:
- PLAN_APPROVAL: Inspect the proposed plan before tools run. Allow the human to:
  * [P]roceed with the generated plan.
  * [E]dit / steer the direction with user notes.
  * [A]bort before any tool or token costs are incurred.
- SYNTHESIS_APPROVAL: Inspect verified evidence before synthesis.

Run:
    python examples/07_human_in_the_loop_gates.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.panel import Panel

from research_agent.core.context import ResearchContext
from research_agent.core.human_gate import HumanGate
from research_agent.models import Evidence, PlanStep, ResearchPlan

console = Console()


async def main() -> None:
    console.print(
        Panel(
            "[bold cyan]Step 7: Human-in-the-Loop Approval Checkpoints[/bold cyan]\n"
            "[dim]Pausing execution at critical stages for human review and steering[/dim]",
            border_style="cyan",
        )
    )

    gate = HumanGate(enabled=True, console=console)

    query = "Evaluate vector database migration risk for 10M vectors"
    ctx = ResearchContext(query=query)

    # Formulate a research plan
    ctx.plan = ResearchPlan(
        query=query,
        rationale="Deconstruct migration into index compatibility, RAM sizing, and write downtime risks.",
        steps=[
            PlanStep(
                description="Search for pgvector to Qdrant migration case studies",
                tool_name="web_search",
                tool_args={"query": "migrate pgvector to qdrant production", "num_results": 4},
            ),
            PlanStep(
                description="Fetch Qdrant payload snapshot & restore documentation",
                tool_name="fetch_page",
                tool_args={"url": "https://qdrant.tech/documentation/concepts/snapshots/"},
            ),
        ],
    )

    console.print("[bold]Triggering Checkpoint 1: PLAN_APPROVAL[/bold]\n")
    # In interactive terminal this prompts: [P]roceed / [E]dit / [A]bort
    approved = await gate.check("plan_approval", ctx, extra_info="Review the proposed steps above.")

    if not approved:
        console.print("[red]Plan rejected or aborted by user. Execution halted safely.[/red]")
        return

    console.print("\n[green]✓ Plan approved by human operator! Proceeding to execution...[/green]")

    # Simulate evidence collection
    ctx.verified_evidence = [
        Evidence(
            source_url="https://qdrant.tech/documentation/concepts/snapshots/",
            source_title="Qdrant Snapshots & Backup",
            excerpt="Snapshots can be restored into a new cluster with zero downtime using rolling read replicas.",
            relevance_score=0.92,
            quality_score=0.9,
        )
    ]

    console.print("\n[bold]Triggering Checkpoint 2: SYNTHESIS_APPROVAL[/bold]\n")
    approved_synth = await gate.check("synthesis_approval", ctx, extra_info="Check gathered evidence before synthesis.")

    if approved_synth:
        console.print("[green]✓ Evidence approved. Synthesizing final answer...[/green]")

    console.print(
        "\n[bold yellow]Observations & Next Step:[/bold yellow]\n"
        "1. Human gates give developers and analysts total oversight of agent execution.\n"
        "2. If stdin is not a TTY (CI/CD or batch jobs), gates automatically auto-approve.\n"
        "3. In [bold green]Step 8 (examples/08_full_agent_mcp.py)[/bold green], "
        "we combine everything: orchestrator + SQLite session memory + MCP tool integration!"
    )


if __name__ == "__main__":
    asyncio.run(main())

