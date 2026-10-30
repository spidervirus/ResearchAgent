"""
Multi-turn Research Session Example.

Demonstrates how session persistence allows the agent to build on
previous findings across multiple consecutive research questions.

Run:
    python examples/multi_turn_session.py
"""

import asyncio
from rich.console import Console

from research_agent import ResearchAgent

console = Console()


async def main() -> None:
    # 1. Initialize agent with automated execution
    agent = ResearchAgent(human_gate=False)

    # 2. Turn 1: High-level comparison
    q1 = "What are the primary performance differences between Qdrant and Milvus for vector search?"
    console.print(f"[bold green]Turn 1:[/bold green] {q1}")
    ans1 = await agent.research(q1)

    console.print(f"\n[bold]Turn 1 Answer:[/bold] {ans1.direct_answer[:300]}...")
    console.print(f"[dim]Active Session ID: {agent.session_id}[/dim]\n")

    # 3. Turn 2: Follow-up question building on Turn 1
    q2 = "Between those two, which has lower RAM cost when scaling past 10 million vectors?"
    console.print(f"[bold green]Turn 2 (Follow-up):[/bold green] {q2}")
    ans2 = await agent.continue_research(q2)

    console.print(f"\n[bold]Turn 2 Answer:[/bold] {ans2.direct_answer}\n")
    console.print(f"[bold]Turn 2 Confidence:[/bold] {ans2.confidence:.0%}")
    console.print(f"[dim]Rationale: {ans2.confidence_rationale}[/dim]")

    # 4. Check persistent sessions stored in SQLite
    sessions = agent.list_sessions()
    console.print(f"\n[dim]Total sessions in SQLite database: {len(sessions)}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())

