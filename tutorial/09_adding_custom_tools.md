# Adding Your Own Tools

> **Chapter 9 of 10** — Step-by-step tutorial on creating custom tools, defining OpenAI-compatible schemas, registering tools with the agent, and handling failures safely.

---

## Anatomy of a Custom Tool

To add a tool to ResearchAgent, subclass `BaseTool` from `research_agent.tools.base` and implement two things:
1. `name` and `description` attributes.
2. The `schema` property (an OpenAI function definition).
3. The asynchronous `run(**kwargs)` method.

---

## Building a Wikipedia Summary Tool

Let's build a real, working tool that queries the public Wikipedia REST API.

Create a file `my_wiki_tool.py`:

```python
import httpx
from typing import Any
from research_agent.tools.base import BaseTool, ToolResult

class WikipediaTool(BaseTool):
    """Fetches encyclopedic summaries directly from Wikipedia."""

    name = "wikipedia_search"
    description = (
        "Search Wikipedia for concise encyclopedic summaries of concepts, "
        "algorithms, people, or technical terms."
    )

    @property
    def schema(self) -> dict[str, Any]:
        """Describe the function parameters to the LLM."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The concept or article title to look up (e.g., 'HNSW')",
                        }
                    },
                    "required": ["topic"],
                },
            },
        }

    async def run(self, topic: str = "", **kwargs: Any) -> ToolResult:
        """Call the Wikipedia REST API."""
        if not topic.strip():
            return self._fail("Topic parameter is required.")

        slug = topic.strip().replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "ResearchAgentTutorial/1.0 (https://github.com/myuser/researchagent)"}
            )

            if resp.status_code == 404:
                return self._fail(f"No Wikipedia article found for topic '{topic}'")
            if resp.status_code != 200:
                return self._fail(f"Wikipedia API error {resp.status_code}")

            data = resp.json()
            return self._ok(
                {
                    "title": data.get("title"),
                    "extract": data.get("extract"),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", url),
                }
            )
```

---

## Testing Your Tool Standalone

Before hooking it up to an agent, always test your tool standalone using `.safe_run()`:

```python
import asyncio

async def test_tool():
    tool = WikipediaTool()
    result = await tool.safe_run(topic="Hierarchical navigable small world")
    
    if result.success:
        print("Success!")
        print("Title:", result.data["title"])
        print("Extract:", result.data["extract"][:150])
    else:
        print("Failed:", result.error)

asyncio.run(test_tool())
```

Notice we called `safe_run()` instead of `run()`. `safe_run()` automatically catches any network timeout or JSON error, times the execution, and returns a clean `ToolResult`.

---

## Registering with ResearchAgent

Inject your custom tool into the agent constructor:

```python
import asyncio
from research_agent import ResearchAgent

async def main():
    wiki = WikipediaTool()
    
    # Pass custom tools in the tools list
    agent = ResearchAgent(tools=[wiki], human_gate=False)
    
    # The agent now has web_search, page_fetcher, code_runner, AND wikipedia_search!
    print("Agent tools:", list(agent.tools.keys()))

    answer = await agent.research("Explain how HNSW indexing works under the hood")
    print(answer.direct_answer)

asyncio.run(main())
```

In the final chapter, we explore **Chapter 10: Running with Local Models** to see how to run ResearchAgent completely offline using Ollama.

