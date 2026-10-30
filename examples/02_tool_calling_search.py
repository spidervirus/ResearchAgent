"""
Progressive Tutorial — Step 2: Adding Tool Calling (Web Search).

WHY THIS STEP EXISTS:
In Step 1, our LLM had no access to live external data. Here, we add tool calling
using DuckDuckGo search (zero API keys needed) and inspect the raw retrieved data.

WHAT THIS REVEALS:
- The agent can now look up real-time information, release notes, and benchmarks.
- BUT: Single tool calls are reactive. The agent doesn't know what sub-questions
  to investigate next unless guided by an explicit plan.
- BUT: Search snippets are often shallow (~150 characters). Full pages must be fetched.

Run:
    python examples/02_tool_calling_search.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from research_agent.tools.web_search import WebSearchTool

console = Console()


async def main() -> None:
    console.print(
        Panel(
            "[bold cyan]Step 2: Tool Calling (Live Web Search)[/bold cyan]\n"
            "[dim]Giving the agent access to real-time external data using DuckDuckGo[/dim]",
            border_style="cyan",
        )
    )

    search_tool = WebSearchTool(provider="duckduckgo")
    search_query = "top open source vector databases 10k QPS benchmark 2024"

    console.print(f"[bold]Search Query:[/bold] {search_query}\n")
    console.print("[dim]Fetching live search results (no API keys required)...[/dim]")

    result = await search_tool.safe_run(query=search_query, num_results=5)

    if not result.success:
        console.print(f"[red]Search failed:[/red] {result.error}")
        return

    # Render results table
    table = Table(title="🔍 Live Search Results", header_style="bold green")
    table.add_column("#", width=3)
    table.add_column("Title", style="white", max_width=40)
    table.add_column("URL", style="dim blue", max_width=35)
    table.add_column("Snippet Preview", style="dim")

    for i, item in enumerate(result.data, start=1):
        table.add_row(
            str(i),
            item.get("title", ""),
            item.get("url", ""),
            item.get("snippet", "")[:80] + "...",
        )

    console.print(table)
    console.print(f"\n[dim]Elapsed time: {result.metadata.get('elapsed_ms')} ms[/dim]")

    console.print(
        "\n[bold yellow]Observations & Next Step:[/bold yellow]\n"
        "1. We now have real, active URLs and recent snippets.\n"
        "2. But how does an agent know which searches to perform across a complex question?\n"
        "3. In [bold green]Step 3 (examples/03_explicit_planning.py)[/bold green], "
        "we introduce the Planner to map out an explicit checklist before taking action."
    )


if __name__ == "__main__":
    asyncio.run(main())

