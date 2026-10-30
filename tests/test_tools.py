"""
Tests for tools in research_agent.tools.
"""

import os
import pytest
from unittest.mock import AsyncMock, patch

from research_agent.tools.base import BaseTool, ToolResult
from research_agent.tools.code_runner import CodeRunnerTool
from research_agent.tools.page_fetcher import PageFetcherTool
from research_agent.tools.web_search import WebSearchTool


@pytest.mark.asyncio
async def test_base_tool_safe_run() -> None:
    class FailingTool(BaseTool):
        name = "failing_tool"
        description = "Always raises"

        @property
        def schema(self) -> dict:
            return {}

        async def run(self, **kwargs) -> ToolResult:
            raise ValueError("Intentional failure")

    tool = FailingTool()
    res = await tool.safe_run()
    assert not res.success
    assert "ValueError: Intentional failure" in (res.error or "")
    assert res.metadata.get("tool_name") == "failing_tool"
    assert "elapsed_ms" in res.metadata


@pytest.mark.asyncio
async def test_code_runner_executes_python() -> None:
    runner = CodeRunnerTool(enabled=True)
    res = await runner.safe_run(code="print('Hello ResearchAgent')", language="python")
    assert res.success
    assert "Hello ResearchAgent" in res.data["stdout"]
    assert res.data["exit_code"] == 0


@pytest.mark.asyncio
async def test_code_runner_handles_timeout() -> None:
    runner = CodeRunnerTool(enabled=True)
    # Code with sleep longer than timeout
    res = await runner.safe_run(code="import time; time.sleep(5)", language="python", timeout=1)
    assert not res.success
    assert "timed out" in (res.error or "").lower()


@pytest.mark.asyncio
async def test_code_runner_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODE_RUNNER_ENABLED", "false")
    runner = CodeRunnerTool()
    res = await runner.safe_run(code="print('disabled')")
    assert not res.success
    assert "disabled" in (res.error or "").lower()


@pytest.mark.asyncio
async def test_page_fetcher_strips_html() -> None:
    fetcher = PageFetcherTool()
    html_sample = """
    <html>
      <head><title>Vector DB Performance</title></head>
      <body>
        <nav><a href="/home">Home</a></nav>
        <article>
          <h1>Benchmarking Qdrant at 10k QPS</h1>
          <p>Qdrant is written in Rust and delivers exceptional memory efficiency under high concurrency workloads.</p>
        </article>
        <footer>Copyright 2024</footer>
      </body>
    </html>
    """

    class MockStreamResponse:
        status_code = 200
        url = "https://example.com/vector-article"
        headers = {"content-type": "text/html; charset=utf-8"}

        async def aiter_bytes(self, chunk_size=8192):
            yield html_sample.encode("utf-8")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    with patch("httpx.AsyncClient.stream", return_value=MockStreamResponse()):
        res = await fetcher.safe_run(url="https://example.com/vector-article")
        assert res.success
        text = res.data["content"]
        assert "Qdrant is written in Rust" in text
        assert "Benchmarking Qdrant" in text


@pytest.mark.asyncio
async def test_mcp_tool_wrapper() -> None:
    from research_agent.tools.mcp import MCPTool

    mock_client = AsyncMock()
    mock_client.call_tool.return_value = "MCP result text: cluster healthy"

    schema_def = {
        "type": "object",
        "properties": {"cluster_id": {"type": "string"}},
        "required": ["cluster_id"],
    }
    mcp_tool = MCPTool(
        name="check_cluster",
        description="Inspects cluster status",
        input_schema=schema_def,
        client=mock_client,
    )

    # Check schema formatting
    assert mcp_tool.schema["function"]["name"] == "check_cluster"
    assert mcp_tool.schema["function"]["parameters"] == schema_def

    # Check safe run execution
    res = await mcp_tool.safe_run(cluster_id="prod-us-east-1")
    assert res.success
    assert "cluster healthy" in str(res.data)
    mock_client.call_tool.assert_awaited_once_with("check_cluster", {"cluster_id": "prod-us-east-1"})
