"""
tests/conftest.py — Pytest fixtures shared across the test suite.

Provides:
    mock_llm            — MockLLM with canned responses (inherits BaseLLM).
    mock_search_tool    — Fake WebSearchTool that returns deterministic results.
    mock_page_fetcher   — Fake PageFetcherTool that returns canned page text.
    sample_context      — A ResearchContext pre-loaded with a sample query.
    sample_evidence     — Three Evidence objects with varying relevance scores.
"""

from __future__ import annotations

import asyncio
import inspect
from collections import deque
from typing import Any, AsyncIterator, List, Optional

import pytest


def pytest_pyfunc_call(pyfuncitem):
    """Automatically run async test functions via asyncio.run if pytest-asyncio is not installed."""
    if pyfuncitem.config.pluginmanager.hasplugin("asyncio"):
        return None  # Let pytest-asyncio handle it
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        testfunction = pyfuncitem.obj
        funcargs = {
            arg: pyfuncitem.funcargs[arg]
            for arg in pyfuncitem._fixtureinfo.argnames
            if arg in pyfuncitem.funcargs
        }
        asyncio.run(testfunction(**funcargs))
        return True


from research_agent.models import (
    Evidence,
    FollowUpQuestion,
    Limitation,
    PlanStep,
    ResearchAnswer,
    ResearchPlan,
    ToolCall,
    VerificationResult,
)
from research_agent.core.context import ResearchContext
from research_agent.llm.base import BaseLLM
from research_agent.tools.base import BaseTool, ToolResult


# ---------------------------------------------------------------------------
# MockLLM
# ---------------------------------------------------------------------------


class MockLLM(BaseLLM):
    """A deterministic LLM that returns canned responses in order.

    Args:
        responses: Ordered list of strings the mock will return on successive
                   calls to :meth:`complete`.  If the list is exhausted an
                   empty string is returned so tests do not raise IndexError.
    """

    def __init__(self, responses: List[str]) -> None:
        super().__init__(model="mock-model", provider="mock")
        self._queue: deque[str] = deque(responses)

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[dict] = None,
    ) -> str:
        """Return the next canned response (or empty string when exhausted)."""
        return self._queue.popleft() if self._queue else ""

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Yield the next canned response character-by-character."""
        text = self._queue.popleft() if self._queue else ""
        for char in text:
            yield char


# ---------------------------------------------------------------------------
# Mock tools
# ---------------------------------------------------------------------------


class _MockWebSearchTool(BaseTool):
    """Returns a fixed list of search results without hitting the network."""

    name = "web_search"
    description = "Mock web search tool for testing."

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description}}

    async def run(self, query: str = "", num_results: int = 5, **kwargs: Any) -> ToolResult:  # noqa: ARG002
        data = [
            {
                "url": "https://example.com/vector-db-comparison",
                "title": "Vector Database Comparison 2024",
                "snippet": "Comprehensive comparison of Pinecone, Weaviate, and Qdrant.",
                "position": 1,
            },
            {
                "url": "https://arxiv.org/abs/2401.00001",
                "title": "Benchmarking Vector Databases at Scale",
                "snippet": "We benchmark five leading vector databases under 10k QPS load.",
                "position": 2,
            },
            {
                "url": "https://towardsdatascience.com/vector-databases",
                "title": "Understanding Vector Databases",
                "snippet": "An accessible overview of HNSW and IVF indexing strategies.",
                "position": 3,
            },
        ]
        return self._ok(data[:num_results])


class _MockPageFetcherTool(BaseTool):
    """Returns canned page content without performing HTTP requests."""

    name = "page_fetcher"
    description = "Mock page fetcher tool for testing."

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description}}

    async def run(self, url: str = "", **kwargs: Any) -> ToolResult:  # noqa: ARG002
        content = (
            f"[Mock page content for {url}]\n\n"
            "Vector databases store high-dimensional embeddings and enable "
            "approximate nearest-neighbour (ANN) search at scale.  "
            "Key players include Pinecone, Weaviate, Qdrant, Milvus, and Chroma.  "
            "Performance benchmarks show that Qdrant and Milvus excel at "
            "high-throughput scenarios while Chroma is optimised for local "
            "development workflows."
        )
        return self._ok({"text": content, "url": url})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> MockLLM:
    """A MockLLM pre-loaded with a valid JSON research-plan response."""
    canned = [
        '{"steps": [], "rationale": "Mock rationale"}'
    ]
    return MockLLM(responses=canned)


@pytest.fixture
def mock_search_tool() -> _MockWebSearchTool:
    """Fake WebSearchTool that never touches the network."""
    return _MockWebSearchTool()


@pytest.fixture
def mock_page_fetcher() -> _MockPageFetcherTool:
    """Fake PageFetcherTool that returns deterministic HTML-free content."""
    return _MockPageFetcherTool()


@pytest.fixture
def sample_context() -> ResearchContext:
    """A ResearchContext seeded with the canonical test query."""
    return ResearchContext(query="Compare top vector databases")


@pytest.fixture
def sample_evidence() -> List[Evidence]:
    """Three Evidence objects spanning low, medium, and high relevance scores."""
    return [
        Evidence(
            source_url="https://arxiv.org/abs/2401.00001",
            source_title="Benchmarking Vector Databases at Scale",
            excerpt="Academic benchmark of five leading vector databases under 10k QPS.",
            relevance_score=0.9,
            quality_score=0.95,
        ),
        Evidence(
            source_url="https://example.com/vector-db-comparison",
            source_title="Vector Database Comparison 2024",
            excerpt="Side-by-side feature comparison of Pinecone, Weaviate, and Qdrant.",
            relevance_score=0.6,
            quality_score=0.7,
        ),
        Evidence(
            source_url="https://random-blog.io/embeddings",
            source_title="What are embeddings?",
            excerpt="Very introductory content that barely touches the query.",
            relevance_score=0.2,
            quality_score=0.4,
        ),
    ]
