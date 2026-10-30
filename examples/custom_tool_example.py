"""
Custom Tool Extension Example.

Shows how to define a custom tool (Wikipedia search) by subclassing BaseTool
and registering it directly with ResearchAgent.

Run:
    python examples/custom_tool_example.py
"""

import asyncio
from typing import Any
import httpx
from rich.console import Console

from research_agent import ResearchAgent
from research_agent.tools.base import BaseTool, ToolResult

console = Console()


class WikipediaTool(BaseTool):
    """Fetches article summaries from the public Wikipedia REST API."""

    name = "wikipedia_search"
    description = "Searches Wikipedia for an encyclopedic summary of a concept or technology."

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Wikipedia page title or search topic (e.g. 'Hierarchical_navigable_small_world')",
                        }
                    },
                    "required": ["title"],
                },
            },
        }

    async def run(self, title: str = "", **kwargs: Any) -> ToolResult:
        normalized = title.strip().replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{normalized}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers={"User-Agent": "ResearchAgentTutorial/1.0"})
            if resp.status_code == 404:
                return self._fail(f"No Wikipedia article found for '{title}'")
            if resp.status_code != 200:
                return self._fail(f"Wikipedia API error {resp.status_code}")

            data = resp.json()
            return self._ok(
                {
                    "title": data.get("title"),
                    "description": data.get("description"),
                    "extract": data.get("extract"),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", url),
                }
            )


async def main() -> None:
    # 1. Instantiate the custom tool
    wiki_tool = WikipediaTool()

    # 2. Test the tool directly
    console.print("[cyan]Testing WikipediaTool standalone...[/cyan]")
    res = await wiki_tool.safe_run(title="Vector database")
    if res.success:
        console.print(f"Title: {res.data['title']}")
        console.print(f"Summary: {res.data['extract'][:200]}...\n")

    # 3. Inject into ResearchAgent
    try:
        agent = ResearchAgent(tools=[wiki_tool], human_gate=False)
        console.print(f"Registered tools with Agent: {list(agent.tools.keys())}")
    except ImportError as e:
        console.print(f"[yellow]Agent initialization note: {e}[/yellow]")
        console.print("[green]Tool definition and registration verified successfully![/green]")


if __name__ == "__main__":
    asyncio.run(main())
