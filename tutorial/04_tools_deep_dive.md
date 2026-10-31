# Tools: Web Search, Page Fetcher, Code Runner

> **Chapter 4 of 10** — How ResearchAgent interfaces with the real world using standard tools, robust result envelopes, and sandboxed execution.

---

## The BaseTool Interface

Every tool in ResearchAgent subclasses `BaseTool` defined in `research_agent/tools/base.py`:

```python
class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool's core logic."""
        ...

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """JSON Schema representation for LLM function calling."""
        ...
```

### The `ToolResult` Envelope

Tools never raise unhandled exceptions into the orchestrator. Instead, the `safe_run()` wrapper catches exceptions, measures wall-clock time, and returns a uniform `ToolResult`:

```python
class ToolResult(BaseModel):
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

This guarantees that if an external website drops a connection or returns 502 Bad Gateway, the orchestrator records the failure and smoothly moves on.

---

## 1. WebSearchTool (`web_search.py`)

`WebSearchTool` abstracts search across multiple providers:

| Provider | Requirement | Strengths |
|----------|-------------|-----------|
| **DuckDuckGo** (default) | Zero API keys | Free, privacy-respecting, zero setup |
| **Tavily** | `TAVILY_API_KEY` | Tuned specifically for AI agents, returns pre-cleaned content |
| **SerpAPI** | `SERPAPI_API_KEY` | Raw Google Search results with rich snippets |

### How DuckDuckGo Works Without Blocking
The tool uses `duckduckgo_search.DDGS().text()` executed in an asynchronous thread-pool worker (`run_in_executor`) with automatic rate-limit politeness delays and tenacity exponential retries.

---

## 2. PageFetcherTool (`page_fetcher.py`)

When an agent needs more than just a search snippet, it fetches the full webpage.

### The Problem with Naive Web Scraping
Webpages contain megabytes of junk: cookie notices, navigation menus, ads, headers, tracking scripts, and footers. Feeding raw HTML into an LLM wastes thousands of tokens and destroys context quality.

### The Trafilatura + BeautifulSoup Pipeline

```
Raw URL ──► httpx (streaming, 15s timeout, browser headers)
                │
                ▼
        Trafilatura Extraction (removes boilerplates, ads, nav)
                │
                ├─► Text found? ──► Return clean markdown/text
                │
                └─► Empty fallback? ──► BeautifulSoup text extractor
```

`PageFetcherTool` also checks MIME types and skips binary files, images, and videos.

---

## 3. CodeRunnerTool (`code_runner.py`)

Often, research requires calculations: converting currency, computing memory formulas, or analyzing tabular data.

`CodeRunnerTool` runs scripts in an isolated child process:
- **Strict Wall-Clock Timeout**: Terminates subprocess groups with `SIGKILL` if execution exceeds timeout (default 10s).
- **Environment Scrubbing**: Inherits zero API keys or environment secrets (`OPENAI_API_KEY`, etc. are stripped).
- **Output Capping**: Limits stdout/stderr to 128KB to prevent memory exhaustion from infinite print loops.
- **Toggle Flag**: Can be completely disabled by setting `CODE_RUNNER_ENABLED=false` in `.env`.

---

## 4. Concurrent Execution & DAG Resolution (`executor.py`)

Rather than running independent plan steps sequentially, `Executor` executes them in concurrent waves:
- **`asyncio.gather()`**: Dispatches independent searches and page fetches concurrently.
- **`asyncio.Semaphore` Throttling**: Limits active simultaneous connections (default: `MAX_CONCURRENT_TOOLS=4`) to respect rate limits.
- **DAG Dependency Resolution**: Steps with `depends_on` wait until prerequisite steps finish (`DONE`). If a prerequisite fails, dependent steps are cleanly `SKIPPED`.
- **Smart URL Injection**: A `fetch_page` step with `url="auto"` automatically receives the top URL from its search prerequisite.

In the next chapter, we look at **Chapter 5: The Verifier & Critic** to see how raw evidence is graded, filtered, and checked for contradictions.

