"""
Command-Line Interface (CLI) for ResearchAgent.

Provides commands to run interactive or automated research, inspect
historical sessions, and render source-backed reports.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from research_agent import __version__
from research_agent.agent import ResearchAgent
from research_agent.models import ResearchAnswer, Severity

console = Console()


def _format_confidence_bar(confidence: float) -> str:
    """Format a visual confidence progress bar."""
    pct = int(confidence * 100)
    filled = int(confidence * 10)
    empty = 10 - filled
    bar = "█" * filled + "░" * empty
    color = "green" if confidence >= 0.75 else "yellow" if confidence >= 0.5 else "red"
    return f"[{color}][{bar}] {pct}%[/{color}]"


def _render_rich_answer(answer: ResearchAnswer) -> None:
    """Render the full structured answer using Rich widgets."""
    # 1. Degraded mode notice if applicable
    if answer.is_degraded:
        reasons_text = "\n• ".join(answer.degradation_reasons)
        console.print(
            Panel(
                f"[bold red]⚠️ PARTIAL FAILURE / DEGRADED ANSWER[/bold red]\n"
                f"[yellow]The agent completed research under partial tool or evidence failures:[/yellow]\n"
                f"• {reasons_text}\n"
                f"[dim]Confidence has been automatically capped to reflect higher uncertainty.[/dim]",
                title="Degraded Execution Notice",
                border_style="yellow",
            )
        )

    # 2. Direct Answer
    console.print()
    console.print(
        Panel(
            Markdown(answer.direct_answer),
            title="🎯 Direct Answer",
            border_style="bold green" if not answer.is_degraded else "bold yellow",
            padding=(1, 2),
        )
    )

    # 3. Confidence & Calibration Breakdown
    conf_bar = _format_confidence_bar(answer.confidence)
    console.print(f"  [bold]Confidence:[/bold] {conf_bar}")
    if answer.confidence_rationale:
        console.print(f"  [dim]Rationale: {answer.confidence_rationale}[/dim]\n")

    if answer.confidence_factors:
        cf = answer.confidence_factors
        cf_table = Table(title="📐 Confidence Calibration Checklist", header_style="bold green")
        cf_table.add_column("Factor", style="white")
        cf_table.add_column("Weight", style="dim", width=8)
        cf_table.add_column("Score", style="yellow", width=8)
        cf_table.add_column("Rubric / Rationale", style="dim")
        cf_table.add_row("Source Quality & Count", "30%", f"{cf.source_count_score:.2f}", ">= 3 authoritative sources for max score")
        cf_table.add_row("Cross-Source Agreement", "25%", f"{cf.agreement_score:.2f}", "Consensus vs contradictions")
        cf_table.add_row("Data Recency", "15%", f"{cf.recency_score:.2f}", "Freshness relative to modern developments")
        cf_table.add_row("Query Constraint Coverage", "30%", f"{cf.constraint_coverage_score:.2f}", "Coverage of all specific question constraints")
        console.print(cf_table)
        console.print()

    # 3. Key Findings
    if answer.key_findings:
        findings_table = Table(title="📌 Key Findings & Sourced Citations", header_style="bold cyan")
        findings_table.add_column("#", style="dim", width=4)
        findings_table.add_column("Finding", style="white")
        for i, f in enumerate(answer.key_findings, 1):
            findings_table.add_row(str(i), f)
        console.print(findings_table)
        console.print()

    # 4. Sources & Evidence Table
    if answer.evidence:
        ev_table = Table(title="📚 Verified Source Evidence", header_style="bold magenta")
        ev_table.add_column("Title / Source", style="cyan")
        ev_table.add_column("Score", style="yellow", width=8)
        ev_table.add_column("URL", style="dim blue")
        for ev in answer.evidence[:10]:
            title = ev.source_title or "Source Document"
            if len(title) > 35:
                title = title[:33] + ".."
            ev_table.add_row(title, f"{ev.composite_score:.2f}", ev.source_url)
        console.print(ev_table)
        console.print()

    # 5. Known Limitations
    if answer.limitations:
        lim_table = Table(title="⚠️ Known Limitations & Assumptions", header_style="bold yellow")
        lim_table.add_column("Severity", width=10)
        lim_table.add_column("Limitation", style="white")
        for lim in answer.limitations:
            sev_color = "red" if lim.severity == Severity.HIGH else "yellow" if lim.severity == Severity.MEDIUM else "green"
            lim_table.add_row(f"[{sev_color}]{lim.severity.upper()}[/{sev_color}]", lim.description)
        console.print(lim_table)
        console.print()

    # 6. Suggested Follow-Ups
    if answer.follow_up_questions:
        q_table = Table(title="💡 Suggested Next Questions / Verification", header_style="bold blue")
        q_table.add_column("Follow-Up Question", style="white")
        q_table.add_column("Why It Matters", style="dim")
        for q in answer.follow_up_questions:
            q_table.add_row(q.question, q.rationale)
        console.print(q_table)
        console.print()

    if answer.token_usage and answer.token_usage.total_tokens > 0:
        tu = answer.token_usage
        console.print(
            f"[dim]⚡ Tokens: {tu.total_tokens:,} ({tu.prompt_tokens:,} in / {tu.completion_tokens:,} out) "
            f"• Est. Cost: ${tu.estimated_cost_usd:.4f}[/dim]"
        )

    if answer.session_id:
        console.print(f"[dim]Session ID: {answer.session_id}[/dim]")


def _answer_to_markdown(answer: ResearchAnswer) -> str:
    """Format answer as Markdown text."""
    lines = [
        f"# Research Report: {answer.query}",
        "",
        f"**Confidence**: {answer.confidence:.0%} — {answer.confidence_rationale}",
        "",
        "## Direct Answer",
        answer.direct_answer,
        "",
        "## Key Findings",
    ]
    for f in answer.key_findings:
        lines.append(f"- {f}")

    lines.extend(["", "## Verified Evidence & Citations"])
    for ev in answer.evidence:
        lines.append(f"- [{ev.source_title or ev.source_url}]({ev.source_url}) (Score: {ev.composite_score:.2f})")
        lines.append(f"  > {ev.excerpt[:200]}...")

    if answer.limitations:
        lines.extend(["", "## Known Limitations"])
        for lim in answer.limitations:
            lines.append(f"- **[{lim.severity.upper()}]** {lim.description}")

    if answer.follow_up_questions:
        lines.extend(["", "## Suggested Follow-Up Questions"])
        for q in answer.follow_up_questions:
            lines.append(f"- **{q.question}**: {q.rationale}")

    return "\n".join(lines)


def run_research(args: argparse.Namespace) -> None:
    verbose = args.output == "rich"
    gate = None
    if args.no_human_gate:
        gate = False
    elif args.human_gate:
        gate = True

    agent = ResearchAgent(
        llm_provider=args.provider,
        llm_model=args.model,
        search_provider=args.search,
        human_gate=gate,
        session_id=args.session_id,
        max_concurrency=args.concurrency,
        verbose=verbose,
    )

    try:
        answer = asyncio.run(agent.research(args.query))
    except Exception as exc:
        console.print(f"[bold red]Research error:[/bold red] {exc}")
        sys.exit(1)

    if args.output == "rich":
        _render_rich_answer(answer)
    elif args.output == "json":
        print(answer.model_dump_json(indent=2))
    elif args.output == "markdown":
        print(_answer_to_markdown(answer))

    if args.save_md:
        md_text = _answer_to_markdown(answer)
        with open(args.save_md, "w", encoding="utf-8") as f:
            f.write(md_text)
        console.print(f"[green]Saved Markdown report to:[/green] {args.save_md}")


def run_sessions(args: argparse.Namespace) -> None:
    store = ResearchAgent(verbose=False).session_store
    sessions = store.list_sessions()
    if not sessions:
        console.print("[dim]No saved sessions found.[/dim]")
        return

    table = Table(title="📜 Research Sessions", header_style="bold cyan")
    table.add_column("Session ID", style="dim", width=12)
    table.add_column("Query", style="white")
    table.add_column("Last Updated", style="magenta")

    for s in sessions:
        sid = s["id"][:10] + ".."
        q = s["query"]
        if len(q) > 50:
            q = q[:48] + ".."
        table.add_row(sid, q, str(s["updated_at"]))

    console.print(table)


def run_show(args: argparse.Namespace) -> None:
    agent = ResearchAgent(verbose=False)
    answer = agent.load_answer(args.session_id)
    if not answer:
        console.print(f"[red]No answer found for session:[/red] {args.session_id}")
        sys.exit(1)
    _render_rich_answer(answer)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="ResearchAgent — Sourced, structured, and auditable AI research.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # research command
    p_res = subparsers.add_parser("research", help="Run a research query and generate a structured answer")
    p_res.add_argument("query", help="The research or problem-solving question")
    p_res.add_argument("-p", "--provider", help="LLM provider: openai | anthropic | ollama")
    p_res.add_argument("-m", "--model", help="Model name (e.g., gpt-4o, claude-3-5-sonnet-20241022, llama3)")
    p_res.add_argument("-s", "--search", help="Search provider: duckduckgo | tavily | serpapi")
    p_res.add_argument("-c", "--concurrency", type=int, default=None, help="Max concurrent tool executions (default: 4)")
    p_res.add_argument("--human-gate", action="store_true", default=None, help="Enable approval checkpoints")
    p_res.add_argument("--no-human-gate", action="store_true", help="Bypass approval checkpoints")
    p_res.add_argument("--session-id", help="Session ID to resume or link")
    p_res.add_argument("-o", "--output", choices=["rich", "json", "markdown"], default="rich", help="Output format")
    p_res.add_argument("--save-md", help="Path to write report as Markdown file")

    # sessions command
    subparsers.add_parser("sessions", help="List historical research sessions")

    # show command
    p_show = subparsers.add_parser("show", help="Display the answer from a past saved session")
    p_show.add_argument("session_id", help="The session ID to display")

    # version command
    subparsers.add_parser("version", help="Print the package version")

    args = parser.parse_args()

    if args.command == "research":
        run_research(args)
    elif args.command == "sessions":
        run_sessions(args)
    elif args.command == "show":
        run_show(args)
    elif args.command == "version":
        print(f"ResearchAgent v{__version__}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
