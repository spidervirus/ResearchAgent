"""
Executor — Stage 2 of the research pipeline.

Executes PlanStep objects concurrently by calling registered tools.
Supports dependency-aware DAG wave execution and throttles concurrency
using an asyncio.Semaphore. Handles tool dispatch, retries, and converts raw
tool outputs into structured Evidence records in ResearchContext.

Usage::

    executor = Executor(tools={"web_search": search_tool, "fetch_page": fetcher}, max_concurrency=4)
    await executor.execute(ctx)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from research_agent.config import get_settings
from research_agent.core.context import ResearchContext
from research_agent.models import Evidence, PlanStep, StepStatus, ToolCall
from research_agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class Executor:
    """Executes research plan steps against a collection of registered tools.

    Supports dependency-aware concurrent execution (DAG) with concurrency throttling.

    Attributes:
        tools: Mapping from tool name to :class:`~research_agent.tools.base.BaseTool` instance.
        max_retries: Number of retries attempted per step on transient failure.
        max_concurrency: Maximum number of tool executions running simultaneously.
    """

    def __init__(
        self,
        tools: dict[str, BaseTool],
        max_retries: int = 1,
        max_concurrency: Optional[int] = None,
    ) -> None:
        self.tools = dict(tools)
        self.max_retries = max_retries
        self.max_concurrency = (
            max_concurrency
            if max_concurrency is not None
            else get_settings().max_concurrent_tools
        )

    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool instance."""
        self.tools[tool.name] = tool

    async def execute(self, ctx: ResearchContext) -> None:
        """Execute all pending steps in the context plan concurrently.

        Steps without unsatisfied dependencies are executed in parallel waves,
        throttled by ``max_concurrency``.

        Args:
            ctx: The active :class:`~research_agent.core.context.ResearchContext`.
        """
        if not ctx.plan or not ctx.plan.steps:
            logger.warning("No plan or steps found to execute.")
            return

        semaphore = asyncio.Semaphore(self.max_concurrency)
        lock = asyncio.Lock()
        steps = ctx.plan.steps

        # Build index/id lookup maps for dependency resolution
        step_map: dict[str, PlanStep] = {}
        for idx, s in enumerate(steps, start=1):
            step_map[s.id] = s
            step_map[str(idx)] = s
            step_map[f"step_{idx}"] = s
            step_map[f"step{idx}"] = s

        while True:
            pending = [s for s in steps if s.status == StepStatus.PENDING]
            if not pending:
                break

            ready: list[PlanStep] = []
            failed_deps: list[tuple[PlanStep, str]] = []

            for step in pending:
                deps = step.depends_on or []
                all_done = True
                dep_failed = False
                failed_reason = ""

                for dep_ref in deps:
                    ref_str = str(dep_ref).strip()
                    dep_step = step_map.get(ref_str)
                    if dep_step is None:
                        logger.warning(
                            "Step '%s' references unknown dependency '%s'",
                            step.description,
                            ref_str,
                        )
                        continue

                    if dep_step.status in (StepStatus.FAILED, StepStatus.SKIPPED):
                        dep_failed = True
                        failed_reason = (
                            f"Prerequisite step '{dep_step.description}' {dep_step.status}"
                        )
                        break
                    elif dep_step.status != StepStatus.DONE:
                        all_done = False
                        break

                if dep_failed:
                    failed_deps.append((step, failed_reason))
                elif all_done:
                    ready.append(step)

            # Mark any steps with failed prerequisites as skipped
            if failed_deps:
                for step, reason in failed_deps:
                    step.mark_skipped(reason)
                continue

            # If no steps are ready, but pending steps remain -> cycle or deadlock
            if not ready:
                logger.error(
                    "Deadlock or circular dependency detected in plan steps: %s",
                    [s.description for s in pending],
                )
                for step in pending:
                    step.mark_failed("Unresolved dependency or cycle in execution graph")
                break

            logger.info(
                "Executing wave of %d steps concurrently (concurrency limit: %d)...",
                len(ready),
                self.max_concurrency,
            )

            tasks = [
                self._execute_step_throttled(step, ctx, semaphore, lock, step_map)
                for step in ready
            ]
            await asyncio.gather(*tasks)

    async def _execute_step_throttled(
        self,
        step: PlanStep,
        ctx: ResearchContext,
        semaphore: asyncio.Semaphore,
        lock: asyncio.Lock,
        step_map: dict[str, PlanStep],
    ) -> None:
        """Run a single step within the concurrency semaphore throttle."""
        async with semaphore:
            try:
                await self._execute_step(step, ctx, lock, step_map)
            except Exception as exc:
                logger.exception("Unexpected error executing step '%s': %s", step.description, exc)
                async with lock:
                    step.mark_failed(f"Unexpected error: {exc}")

    async def _execute_step(
        self,
        step: PlanStep,
        ctx: ResearchContext,
        lock: asyncio.Lock,
        step_map: dict[str, PlanStep],
    ) -> None:
        """Execute a single plan step with error handling and result extraction."""
        step.mark_running()

        # Resolve tool with common aliases
        tool = self.tools.get(step.tool_name)
        if not tool:
            alias_map = {
                "fetch_page": "page_fetcher",
                "page_fetcher": "fetch_page",
                "run_code": "code_runner",
                "code_runner": "run_code",
                "search": "web_search",
                "web_search": "search",
            }
            mapped_name = alias_map.get(step.tool_name)
            if mapped_name:
                tool = self.tools.get(mapped_name)

        if not tool:
            err = f"Tool '{step.tool_name}' is not registered. Available: {list(self.tools.keys())}"
            logger.error(err)
            async with lock:
                step.mark_failed(err)
                ctx.add_tool_call(
                    ToolCall(
                        tool_name=step.tool_name,
                        args=step.tool_args,
                        error=err,
                    )
                )
            return

        # Smart URL resolution for fetch_page if url is empty or 'auto'
        tool_args = dict(step.tool_args)
        if step.tool_name in ("fetch_page", "page_fetcher"):
            url = tool_args.get("url", "")
            if not url or url.lower() in ("auto", "none"):
                for dep_ref in step.depends_on:
                    dep_step = step_map.get(str(dep_ref).strip())
                    if dep_step and dep_step.result:
                        res = dep_step.result
                        if isinstance(res, list) and res and isinstance(res[0], dict) and "url" in res[0]:
                            tool_args["url"] = res[0]["url"]
                            break
                        elif isinstance(res, dict) and "results" in res and res["results"]:
                            tool_args["url"] = res["results"][0].get("url", "")
                            break

        attempts = 0
        last_result: Optional[ToolResult] = None

        while attempts <= self.max_retries:
            attempts += 1
            last_result = await tool.safe_run(**tool_args)

            # Record tool call and outcome in context under lock
            async with lock:
                ctx.add_tool_call(
                    ToolCall(
                        tool_name=step.tool_name,
                        args=tool_args,
                        result=last_result.data if last_result.success else None,
                        error=last_result.error if not last_result.success else None,
                        duration_ms=last_result.metadata.get("elapsed_ms"),
                    )
                )

                if last_result.success:
                    step.mark_done(last_result.data)
                    self._extract_evidence(step, last_result.data, ctx)
                    return

            logger.warning(
                "Step '%s' failed attempt %d/%d: %s",
                step.description,
                attempts,
                self.max_retries + 1,
                last_result.error,
            )

        # All attempts exhausted
        async with lock:
            step.mark_failed(last_result.error or "Unknown error during execution")

    def _extract_evidence(self, step: PlanStep, data: Any, ctx: ResearchContext) -> None:
        """Extract structured Evidence items from raw tool results."""
        if step.tool_name in ("web_search", "search"):
            results = data
            if isinstance(results, dict) and "results" in results:
                results = results["results"]
            if not isinstance(results, list):
                return

            for item in results:
                url = ""
                title = ""
                snippet = ""
                if isinstance(item, dict):
                    url = item.get("url") or ""
                    title = item.get("title") or ""
                    snippet = item.get("snippet") or item.get("content") or ""
                elif hasattr(item, "url"):
                    url = getattr(item, "url", "")
                    title = getattr(item, "title", "")
                    snippet = getattr(item, "snippet", "")

                if url and snippet:
                    evidence = Evidence(
                        source_url=url,
                        source_title=title,
                        excerpt=snippet,
                        relevance_score=0.5,
                        quality_score=0.5,
                        step_id=step.id,
                    )
                    ctx.add_evidence(evidence)

        elif step.tool_name in ("fetch_page", "page_fetcher"):
            if not isinstance(data, dict):
                return
            url = data.get("url") or step.tool_args.get("url", "")
            title = data.get("title") or ""
            content = data.get("content") or data.get("text") or ""

            if url and content:
                # Check if we already have an evidence entry from search for this URL
                matched = False
                for existing in ctx.raw_evidence:
                    if existing.source_url == url:
                        # Update with richer content from full page fetch
                        if len(content) > len(existing.excerpt):
                            existing.excerpt = content[:4000]
                        if title and not existing.source_title:
                            existing.source_title = title
                        matched = True
                        break

                if not matched:
                    evidence = Evidence(
                        source_url=url,
                        source_title=title,
                        excerpt=content[:4000],
                        relevance_score=0.6,
                        quality_score=0.6,
                        step_id=step.id,
                    )
                    ctx.add_evidence(evidence)
