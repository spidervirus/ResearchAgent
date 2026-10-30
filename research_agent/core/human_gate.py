"""
HumanGate — Human-in-the-loop checkpoint system for research workflows.

Pauses execution at key decision points to allow inspection, modification,
or early aborts:
- PLAN_APPROVAL: Review the generated research plan before tools run.
- SYNTHESIS_APPROVAL: Review gathered & verified evidence before synthesis.
- EXPENSIVE_TOOL: Prompt before invoking costly or high-concurrency tools.

Supports automated bypass in non-interactive environments (CI, piping, background workers).
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from research_agent.core.context import ResearchContext

logger = logging.getLogger(__name__)


class HumanGate:
    """Manages interactive human-in-the-loop approvals.

    Attributes:
        enabled: If False, all checkpoints auto-approve without pausing.
        console: Rich console instance for formatted terminal rendering.
    """

    def __init__(self, enabled: bool = True, console: Optional[Console] = None) -> None:
        self.enabled = enabled
        self.console = console or Console()

    async def check(
        self,
        checkpoint_type: str,
        ctx: ResearchContext,
        extra_info: str = "",
    ) -> bool:
        """Pause and request human approval.

        Args:
            checkpoint_type: Identifier of the checkpoint ("plan_approval", "synthesis_approval", etc.).
            ctx: Current research context.
            extra_info: Optional explanatory message.

        Returns:
            ``True`` to proceed, ``False`` to abort.
        """
        if not self.enabled:
            return True

        # If not running in an interactive terminal, proceed automatically
        if not sys.stdin.isatty():
            return True

        checkpoint = checkpoint_type.lower()

        if "plan" in checkpoint:
            return self._handle_plan_approval(ctx, extra_info)
        elif "synthesis" in checkpoint:
            return self._handle_synthesis_approval(ctx, extra_info)
        else:
            return self._handle_generic_approval(checkpoint_type, extra_info)

    def _handle_plan_approval(self, ctx: ResearchContext, extra_info: str) -> bool:
        """Render plan table and prompt for confirmation."""
        if not ctx.plan or not ctx.plan.steps:
            return True

        table = Table(title="📋 Proposed Research Plan", show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Description", style="white")
        table.add_column("Tool", style="magenta", width=14)
        table.add_column("Target / Query", style="green")

        for i, step in enumerate(ctx.plan.steps, start=1):
            target = ""
            if "query" in step.tool_args:
                target = str(step.tool_args["query"])
            elif "url" in step.tool_args:
                target = str(step.tool_args["url"])
            elif "code" in step.tool_args:
                target = str(step.tool_args["code"])[:30] + "..."

            table.add_row(str(i), step.description, step.tool_name, target)

        self.console.print(table)
        if ctx.plan.rationale:
            self.console.print(f"[dim]Rationale: {ctx.plan.rationale}[/dim]\n")

        if extra_info:
            self.console.print(f"[yellow]{extra_info}[/yellow]")

        prompt = "[bold yellow]Approve plan? [P]roceed / [E]dit / [A]bort (default: P): [/bold yellow]"
        try:
            choice = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return True

        if choice in ("", "p", "proceed", "y", "yes"):
            return True
        elif choice in ("e", "edit"):
            try:
                user_note = input("Enter adjustment instructions for the agent: ").strip()
                if user_note:
                    ctx.metadata["user_edits"] = user_note
            except (EOFError, KeyboardInterrupt):
                pass
            return True
        else:
            self.console.print("[red]Plan rejected by user. Aborting research.[/red]")
            return False

    def _handle_synthesis_approval(self, ctx: ResearchContext, extra_info: str) -> bool:
        """Show gathered evidence summary before final synthesis."""
        evidence = ctx.verified_evidence or ctx.raw_evidence
        count = len(evidence)

        table = Table(title="🔍 Gathered Evidence Summary", show_header=True, header_style="bold green")
        table.add_column("Source", style="cyan")
        table.add_column("Score", style="yellow", width=8)
        table.add_column("Excerpt Preview", style="white")

        for ev in evidence[:6]:
            title = ev.source_title or ev.source_url
            if len(title) > 30:
                title = title[:28] + ".."
            preview = ev.excerpt[:60].replace("\n", " ") + "..."
            table.add_row(title, f"{ev.composite_score:.2f}", preview)

        self.console.print(table)
        self.console.print(f"[dim]Total sources collected: {count}[/dim]\n")

        if extra_info:
            self.console.print(f"[yellow]{extra_info}[/yellow]")

        prompt = "[bold yellow]Proceed to synthesis? [P]roceed / [A]bort (default: P): [/bold yellow]"
        try:
            choice = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return True

        if choice in ("", "p", "proceed", "y", "yes"):
            return True
        else:
            self.console.print("[red]Synthesis cancelled by user.[/red]")
            return False

    def _handle_generic_approval(self, title: str, extra_info: str) -> bool:
        """Generic confirmation prompt."""
        self.console.print(Panel(extra_info or "Confirmation required", title=f"⚠️ {title}"))
        prompt = "[bold yellow]Continue? [y/N]: [/bold yellow]"
        try:
            choice = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return choice in ("y", "yes")

