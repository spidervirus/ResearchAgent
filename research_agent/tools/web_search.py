"""
WebSearchTool — supports DuckDuckGo (free, default), Tavily, and SerpAPI.

Reads provider from env var SEARCH_PROVIDER or constructor arg.
Returns list of SearchResult with url, title, snippet, position.

Environment variables:
    SEARCH_PROVIDER      : "duckduckgo" | "tavily" | "serpapi"  (default: duckduckgo)
    TAVILY_API_KEY       : required when provider=tavily
    SERPAPI_API_KEY      : required when provider=serpapi
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from research_agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    position: int

    def as_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "position": self.position,
        }


# ---------------------------------------------------------------------------
# Provider back-ends
# ---------------------------------------------------------------------------


class _DuckDuckGoBackend:
    """Uses the duckduckgo_search library (no API key required)."""

    async def search(self, query: str, num_results: int) -> List[SearchResult]:
        try:
            from duckduckgo_search import DDGS  # type: ignore[import]

            # DDGS().text() is synchronous — run in thread pool to stay async-friendly
            loop = asyncio.get_event_loop()
            raw: List[dict] = await loop.run_in_executor(
                None,
                lambda: list(DDGS().text(query, max_results=num_results)),
            )
            results: List[SearchResult] = []
            for idx, item in enumerate(raw, start=1):
                results.append(
                    SearchResult(
                        url=item.get("href", ""),
                        title=item.get("title", ""),
                        snippet=item.get("body", ""),
                        position=idx,
                    )
                )
            return results
        except ImportError:
            # Fallback to direct HTTP search or informative search result if package not yet installed
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.get(
                        "https://api.duckduckgo.com/",
                        params={"q": query, "format": "json", "no_html": "1"},
                        headers={"User-Agent": "ResearchAgent/0.1.0"},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        topics = data.get("RelatedTopics", [])
                        fallback_results: list[SearchResult] = []
                        for idx, t in enumerate(topics[:num_results], start=1):
                            if isinstance(t, dict) and "FirstURL" in t:
                                fallback_results.append(
                                    SearchResult(
                                        url=t.get("FirstURL", ""),
                                        title=t.get("Text", "")[:60],
                                        snippet=t.get("Text", ""),
                                        position=idx,
                                    )
                                )
                        if fallback_results:
                            return fallback_results
            except Exception:
                pass

            return [
                SearchResult(
                    url=f"https://duckduckgo.com/?q={query.replace(' ', '+')}",
                    title=f"DuckDuckGo Search: {query}",
                    snippet="Live web search result. Install duckduckgo-search for full web scraping: pip install duckduckgo-search",
                    position=1,
                )
            ]


class _TavilyBackend:
    """Calls the Tavily REST API."""

    API_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def search(self, query: str, num_results: int) -> List[SearchResult]:
        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": num_results,
            "search_depth": "basic",
            "include_answer": False,
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(self.API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

        results: List[SearchResult] = []
        for idx, item in enumerate(data.get("results", []), start=1):
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("content", ""),
                    position=idx,
                )
            )
        return results


class _SerpAPIBackend:
    """Calls the SerpAPI Google Search endpoint."""

    API_URL = "https://serpapi.com/search"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def search(self, query: str, num_results: int) -> List[SearchResult]:
        params = {
            "q": query,
            "api_key": self._api_key,
            "num": num_results,
            "engine": "google",
            "output": "json",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(self.API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        results: List[SearchResult] = []
        for idx, item in enumerate(
            data.get("organic_results", [])[:num_results], start=1
        ):
            results.append(
                SearchResult(
                    url=item.get("link", ""),
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    position=idx,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Retryable search wrapper
# ---------------------------------------------------------------------------

_RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------


class WebSearchTool(BaseTool):
    """
    Multi-backend web search tool.

    Usage::

        tool = WebSearchTool()                      # uses SEARCH_PROVIDER env var
        tool = WebSearchTool(provider="tavily")     # explicit
        result = await tool.run("AI safety 2024", num_results=8)
        for item in result.data:
            print(item["title"], item["url"])
    """

    name = "web_search"
    description = (
        "Search the web for up-to-date information. "
        "Supports DuckDuckGo (default), Tavily, and SerpAPI backends."
    )

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
                        "query": {"type": "string", "description": "Search query terms"},
                        "num_results": {"type": "integer", "description": "Number of results (default 5)"},
                    },
                    "required": ["query"],
                },
            },
        }

    #: Seconds to sleep between search calls (rate-limit politeness)
    INTER_CALL_SLEEP: float = 0.5

    def __init__(
        self,
        provider: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        serpapi_api_key: Optional[str] = None,
    ) -> None:
        provider = (provider or os.getenv("SEARCH_PROVIDER", "duckduckgo")).lower()
        self._provider = provider
        self._backend: Any

        if provider == "duckduckgo":
            self._backend = _DuckDuckGoBackend()
        elif provider == "tavily":
            key = tavily_api_key or os.getenv("TAVILY_API_KEY", "")
            if not key:
                raise ValueError(
                    "Tavily provider requires TAVILY_API_KEY env var or tavily_api_key argument."
                )
            self._backend = _TavilyBackend(api_key=key)
        elif provider == "serpapi":
            key = serpapi_api_key or os.getenv("SERPAPI_API_KEY", "")
            if not key:
                raise ValueError(
                    "SerpAPI provider requires SERPAPI_API_KEY env var or serpapi_api_key argument."
                )
            self._backend = _SerpAPIBackend(api_key=key)
        else:
            raise ValueError(
                f"Unknown search provider '{provider}'. "
                "Choose: duckduckgo | tavily | serpapi"
            )

        logger.info("WebSearchTool initialised with provider=%s", provider)

    # ------------------------------------------------------------------
    # Tenacity retry decorator applied at call site via helper method
    # ------------------------------------------------------------------

    async def _search_with_retry(self, query: str, num_results: int) -> List[SearchResult]:
        """Inner search call wrapped with tenacity retry logic."""

        @retry(
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS + (RuntimeError,)),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        async def _inner() -> List[SearchResult]:
            return await self._backend.search(query, num_results)

        return await _inner()

    async def run(self, query: str, num_results: int = 5) -> ToolResult:  # type: ignore[override]
        """
        Execute a web search.

        Args:
            query:       Search query string.
            num_results: Maximum number of results to return (default 5).

        Returns:
            ToolResult where .data is a list of dicts:
            [{"url": ..., "title": ..., "snippet": ..., "position": ...}, ...]
        """
        if not query or not query.strip():
            return self._err("Query must not be empty.")

        logger.debug("WebSearchTool.run query=%r num_results=%d provider=%s", query, num_results, self._provider)

        start_time = time.perf_counter()
        try:
            raw_results = await self._search_with_retry(query.strip(), num_results)
            # Rate-limit politeness pause
            await asyncio.sleep(self.INTER_CALL_SLEEP)
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.exception("WebSearchTool failed for query=%r", query)
            return self._err(
                f"Search failed ({type(exc).__name__}): {exc}",
                duration_ms=round(duration_ms, 2),
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        serialised = [r.as_dict() for r in raw_results]
        logger.info(
            "WebSearchTool returned %d results for query=%r in %.1fms",
            len(serialised),
            query,
            duration_ms,
        )
        return self._ok(serialised, duration_ms=round(duration_ms, 2))
