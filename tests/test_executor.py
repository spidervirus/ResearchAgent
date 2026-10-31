"""
tests/test_executor.py — Tests for concurrent execution, DAG dependencies,
throttling, and error handling in Executor.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import pytest

from research_agent.core.context import ResearchContext
from research_agent.core.executor import Executor
from research_agent.models import PlanStep, ResearchPlan, StepStatus
from research_agent.tools.base import BaseTool, ToolResult


class SleepTool(BaseTool):
    """Tool that sleeps for a given duration to test concurrency."""

    name = "sleep_tool"
    description = "Sleeps for duration_s"

    def __init__(self, duration_s: float = 0.1) -> None:
        self.duration_s = duration_s

    @property
    def schema(self) -> dict[str, Any]:
        return {}

    async def run(self, **kwargs: Any) -> ToolResult:
        await asyncio.sleep(self.duration_s)
        return self._ok({"slept": self.duration_s})


class ConcurrencyTrackingTool(BaseTool):
    """Tracks active concurrency to verify Semaphore throttling."""

    name = "track_concurrency"
    description = "Records active concurrent workers"

    def __init__(self) -> None:
        self.active_count = 0
        self.max_observed_concurrency = 0
        self._lock = asyncio.Lock()

    @property
    def schema(self) -> dict[str, Any]:
        return {}

    async def run(self, **kwargs: Any) -> ToolResult:
        async with self._lock:
            self.active_count += 1
            if self.active_count > self.max_observed_concurrency:
                self.max_observed_concurrency = self.active_count

        await asyncio.sleep(0.08)

        async with self._lock:
            self.active_count -= 1

        return self._ok({"active": self.active_count})


@pytest.mark.asyncio
async def test_executor_concurrent_speedup() -> None:
    """Verify that independent steps execute in parallel rather than sequentially."""
    sleep_tool = SleepTool(duration_s=0.1)
    executor = Executor(tools={"sleep_tool": sleep_tool}, max_concurrency=4)

    steps = [
        PlanStep(description=f"Step {i}", tool_name="sleep_tool", tool_args={})
        for i in range(3)
    ]
    ctx = ResearchContext(query="Test concurrency speedup")
    ctx.plan = ResearchPlan(query="Test", steps=steps)

    start = time.perf_counter()
    await executor.execute(ctx)
    elapsed = time.perf_counter() - start

    # 3 sequential steps of 0.1s would take >= 0.30s.
    # Concurrent execution runs in ~0.10s (allow headroom under 0.22s).
    assert elapsed < 0.22, f"Execution took {elapsed:.2f}s, expected < 0.22s for concurrent run"
    assert all(s.status == StepStatus.DONE for s in steps)


@pytest.mark.asyncio
async def test_executor_semaphore_concurrency_limit() -> None:
    """Verify that max_concurrency semaphore limits simultaneous tasks."""
    tracker = ConcurrencyTrackingTool()
    executor = Executor(tools={"track_concurrency": tracker}, max_concurrency=2)

    # 6 steps dispatched with max_concurrency=2
    steps = [
        PlanStep(description=f"Step {i}", tool_name="track_concurrency", tool_args={})
        for i in range(6)
    ]
    ctx = ResearchContext(query="Test semaphore limit")
    ctx.plan = ResearchPlan(query="Test", steps=steps)

    await executor.execute(ctx)

    assert tracker.max_observed_concurrency <= 2, (
        f"Max concurrency observed was {tracker.max_observed_concurrency}, expected <= 2"
    )
    assert all(s.status == StepStatus.DONE for s in steps)


@pytest.mark.asyncio
async def test_executor_dag_dependencies() -> None:
    """Verify that steps with depends_on wait for prerequisite steps to finish."""
    execution_order: list[str] = []

    class OrderLoggingTool(BaseTool):
        name = "order_tool"
        description = "Logs order"

        @property
        def schema(self) -> dict[str, Any]:
            return {}

        async def run(self, step_name: str = "", **kwargs: Any) -> ToolResult:
            await asyncio.sleep(0.04)
            execution_order.append(step_name)
            return self._ok({"step": step_name})

    tool = OrderLoggingTool()
    executor = Executor(tools={"order_tool": tool}, max_concurrency=4)

    # Step 1 and Step 2 run in parallel; Step 3 depends on both
    step1 = PlanStep(id="s1", description="Step 1", tool_name="order_tool", tool_args={"step_name": "step1"})
    step2 = PlanStep(id="s2", description="Step 2", tool_name="order_tool", tool_args={"step_name": "step2"})
    step3 = PlanStep(
        id="s3",
        description="Step 3",
        tool_name="order_tool",
        tool_args={"step_name": "step3"},
        depends_on=["s1", "s2"],
    )

    ctx = ResearchContext(query="Test DAG")
    ctx.plan = ResearchPlan(query="Test DAG", steps=[step1, step2, step3])

    await executor.execute(ctx)

    assert set(execution_order[:2]) == {"step1", "step2"}
    assert execution_order[2] == "step3"
    assert step3.status == StepStatus.DONE


@pytest.mark.asyncio
async def test_executor_prerequisite_failure_skips() -> None:
    """Verify that if a prerequisite fails, dependent steps are marked SKIPPED."""
    class FailingTool(BaseTool):
        name = "fail_tool"
        description = "Fails"

        @property
        def schema(self) -> dict[str, Any]:
            return {}

        async def run(self, **kwargs: Any) -> ToolResult:
            return self._err("Critical database timeout")

    fail_tool = FailingTool()
    sleep_tool = SleepTool()
    executor = Executor(tools={"fail_tool": fail_tool, "sleep_tool": sleep_tool}, max_retries=0)

    step1 = PlanStep(id="s1", description="Failing prerequisite", tool_name="fail_tool", tool_args={})
    step2 = PlanStep(
        id="s2",
        description="Dependent step",
        tool_name="sleep_tool",
        tool_args={},
        depends_on=["s1"],
    )

    ctx = ResearchContext(query="Test prerequisite failure")
    ctx.plan = ResearchPlan(query="Test prerequisite failure", steps=[step1, step2])

    await executor.execute(ctx)

    assert step1.status == StepStatus.FAILED
    assert step2.status == StepStatus.SKIPPED
    assert "Prerequisite" in (step2.error or "")


@pytest.mark.asyncio
async def test_executor_smart_url_injection() -> None:
    """Verify that fetch_page with url='auto' dynamically gets URL from search dependency."""
    class MockSearchTool(BaseTool):
        name = "web_search"
        description = "Mock search"

        @property
        def schema(self) -> dict[str, Any]:
            return {}

        async def run(self, **kwargs: Any) -> ToolResult:
            return self._ok([{"url": "https://qdrant.tech/documentation/", "snippet": "Vector DB"}])

    class MockFetcherTool(BaseTool):
        name = "fetch_page"
        description = "Mock fetcher"

        @property
        def schema(self) -> dict[str, Any]:
            return {}

        async def run(self, url: str = "", **kwargs: Any) -> ToolResult:
            return self._ok({"url": url, "content": "Full Qdrant documentation text"})

    executor = Executor(tools={"web_search": MockSearchTool(), "fetch_page": MockFetcherTool()})

    step1 = PlanStep(
        id="s1",
        description="Search Qdrant",
        tool_name="web_search",
        tool_args={"query": "qdrant docs"},
    )
    step2 = PlanStep(
        id="s2",
        description="Fetch documentation",
        tool_name="fetch_page",
        tool_args={"url": "auto"},
        depends_on=["s1"],
    )

    ctx = ResearchContext(query="Test URL injection")
    ctx.plan = ResearchPlan(query="Test URL injection", steps=[step1, step2])

    await executor.execute(ctx)

    assert step1.status == StepStatus.DONE
    assert step2.status == StepStatus.DONE
    assert len(ctx.raw_evidence) == 1
    assert ctx.raw_evidence[0].source_url == "https://qdrant.tech/documentation/"


@pytest.mark.asyncio
async def test_executor_deadlock_detection() -> None:
    """Verify that circular dependencies fail cleanly rather than freezing."""
    sleep_tool = SleepTool()
    executor = Executor(tools={"sleep_tool": sleep_tool})

    step1 = PlanStep(id="s1", description="Step 1", tool_name="sleep_tool", depends_on=["s2"])
    step2 = PlanStep(id="s2", description="Step 2", tool_name="sleep_tool", depends_on=["s1"])

    ctx = ResearchContext(query="Test circular dependency")
    ctx.plan = ResearchPlan(query="Test circular dependency", steps=[step1, step2])

    await executor.execute(ctx)

    assert step1.status == StepStatus.FAILED
    assert step2.status == StepStatus.FAILED
    assert "Unresolved dependency or cycle" in (step1.error or "")
