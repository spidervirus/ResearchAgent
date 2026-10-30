"""
Progressive Tutorial — Step 1: Plain LLM Call with Research-Style System Prompt.

WHY THIS STEP EXISTS:
Before building tools, planners, or loops, we start with the simplest baseline:
a direct call to a language model with a carefully crafted research prompt.

WHAT THIS REVEALS:
- The model generates fluent, convincing prose.
- BUT: The model's knowledge cut-off limits recent data (e.g. 2024 benchmark updates).
- BUT: The model cannot verify numbers, cite real active URLs, or test its assumptions.
- BUT: It displays false confidence without calibration.

This baseline demonstrates exactly why agentic systems (tools, verification, and planning)
are necessary for real research work.

Run:
    python examples/01_plain_llm_prompt.py
"""

import asyncio
import os
import sys

# Support running directly from git clone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.panel import Panel

console = Console()

# A research-oriented system prompt attempting to enforce rigor through prompting alone.
RESEARCH_SYSTEM_PROMPT = """\
You are an expert technical researcher. Answer the user's research question with:
1. A clear direct answer / recommendation.
2. Key empirical points comparing the top options.
3. Concrete performance numbers and resource trade-offs.
4. An honest assessment of what you might not know or what could be outdated.

Be analytical, objective, and precise. Avoid vague generalities.
"""


async def main() -> None:
    console.print(
        Panel(
            "[bold cyan]Step 1: Plain LLM Research Prompt[/bold cyan]\n"
            "[dim]Testing research quality using prompt engineering alone (no tools, no loop)[/dim]",
            border_style="cyan",
        )
    )

    query = (
        "Compare the top 3 open-source vector databases for a startup "
        "with 10k QPS and tight cost constraints."
    )
    console.print(f"[bold]Query:[/bold] {query}\n")

    # Check available provider
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    if has_openai or has_anthropic:
        from research_agent.agent import _resolve_llm

        llm = _resolve_llm()
        messages = [
            {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        console.print("[dim]Generating response from LLM...[/dim]")
        response = await llm.complete(messages, temperature=0.3)
    else:
        # Informative offline demo fallback
        response = (
            "Direct Recommendation: For 10k QPS on tight budget, Qdrant or Milvus are primary candidates.\n\n"
            "Key Points:\n"
            "- Qdrant (Rust): Single binary, efficient memory usage, supports on-disk payload.\n"
            "- Milvus (Go/C++): Distributed microservice architecture, high throughput at scale.\n"
            "- Chroma (Python): Optimized for developer ergonomics, less suited for sustained 10k QPS.\n\n"
            "Limitations:\n"
            "- Notice that without web tools, this model cannot inspect 2024 benchmarks or real-world\n"
            "  pricing changes. Citations cannot be verified against live documentation."
        )

    console.print(Panel(response, title="🤖 LLM Baseline Output", border_style="green"))

    console.print(
        "\n[bold yellow]Observations & Next Step:[/bold yellow]\n"
        "1. Notice how fluent the answer sounds, but any citations are unverified.\n"
        "2. The model cannot check live release notes or current pricing.\n"
        "3. In [bold green]Step 2 (examples/02_tool_calling_search.py)[/bold green], "
        "we equip the agent with live web search tools to ground its knowledge!"
    )


if __name__ == "__main__":
    asyncio.run(main())

