"""
Basic Research Example.

Demonstrates running a single research query with ResearchAgent,
accessing the structured findings, confidence score, and citations.

Run:
    python examples/basic_research.py
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from research_agent import ResearchAgent

console = Console()


async def main() -> None:
    # Initialize agent without pausing at human gates for fully automated execution
    agent = ResearchAgent(human_gate=False)

    query = "Compare the top 3 open-source vector databases for a startup with 10k QPS and tight cost constraints"
    console.print(f"[bold cyan]Asking ResearchAgent:[/bold cyan] {query}\n")

    answer = await agent.research(query)

    # 1. Executive Answer
    console.print(Panel(answer.direct_answer, title="🎯 Direct Recommendation", border_style="green"))

    # 2. Confidence Metric
    console.print(f"[bold]Confidence Score:[/bold] {answer.confidence:.0%}")
    console.print(f"[dim]Rationale: {answer.confidence_rationale}[/dim]\n")

    # 3. Key Findings with Evidence
    findings_table = Table(title="📌 Key Findings", header_style="bold cyan")
    findings_table.add_column("#", width=4)
    findings_table.add_column("Finding")
    for i, finding in enumerate(answer.key_findings, start=1):
        findings_table.add_row(str(i), finding)
    console.print(findings_table)
    console.print()

    # 4. Limitations
    if answer.limitations:
        lim_table = Table(title="⚠️ Known Caveats & Limitations", header_style="bold yellow")
        lim_table.add_column("Severity", width=10)
        lim_table.add_column("Description")
        for lim in answer.limitations:
            lim_table.add_row(lim.severity.upper(), lim.description)
        console.print(lim_table)
        console.print()

    # 5. Suggested Next Steps
    if answer.follow_up_questions:
        q_table = Table(title="💡 Recommended Follow-Up Investigations", header_style="bold blue")
        q_table.add_column("Question")
        q_table.add_column("Rationale")
        for q in answer.follow_up_questions:
            q_table.add_row(q.question, q.rationale)
        console.print(q_table)


if __name__ == "__main__":
    asyncio.run(main())

