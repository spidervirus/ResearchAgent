"""
Executor — Stage 2 of the research pipeline.

Executes each PlanStep by calling the appropriate registered tool.
Handles tool dispatch, error handling, retries, and converts raw tool outputs
into structured Evidence records stored in the ResearchContext.

Usage::

    executor = Executor(tools={"web_search": search_tool, "fetch_page": fetcher})
    await executor.execute(ctx)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from research_agent.core.context import ResearchContext
from research_agent.models import Evidence, PlanStep, StepStatus, ToolCall
from research_agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class Executor:
    """Executes research plan steps against a collection of registered tools.

    Attributes:
        tools: Mapping from tool name to :class:`~research_agent.tools.base.BaseTool` instance.
        max_retries: Number of retries attempted per step on transient failure.
    """

    def __init__(
        self,
        tools: dict[str, BaseTool],
        max_retries: int = 1,
    ) -> None:
        self.tools = dict(tools)
        self.max_retries = max_retries

    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool instance."""
        self.tools[tool.name] = tool

    async def execute(self, ctx: ResearchContext) -> None:
        """Execute all pending steps in the context plan.

        Args:
            ctx: The active :class:`~research_agent.core.context.ResearchContext`.
        """
        if not ctx.plan or not ctx.plan.steps:
            logger.warning("No plan or steps found to execute.")
            return

        for step in ctx.plan.steps:
            if step.status != StepStatus.PENDING:
                continue
            await self._execute_step(step, ctx)

    async def _execute_step(self, step: PlanStep, ctx: ResearchContext) -> None:
        """Execute a single plan step with error handling and result extraction."""
        step.mark_running()
        
        # Resolve tool with common aliases
        tool = self.tools.get(step.tool_name)
        if not tool:
            alias_map = {
                "fetch_page": "page_fetcher",
                "run_code": "code_runner",
                "search": "web_search",
            }
            mapped_name = alias_map.get(step.tool_name)
            if mapped_name:
                tool = self.tools.get(mapped_name)

        if not tool:
            err = f"Tool '{step.tool_name}' is not registered. Available: {list(self.tools.keys())}"
            logger.error(err)
            step.mark_failed(err)
            ctx.add_tool_call(
                ToolCall(
                    tool_name=step.tool_name,
                    args=step.tool_args,
                    error=err,
                )
            )
            return

        attempts = 0
        last_result: Optional[ToolResult] = None

        while attempts <= self.max_retries:
            attempts += 1
            last_result = await tool.safe_run(**step.tool_args)

            # Record tool call in context
            ctx.add_tool_call(
                ToolCall(
                    tool_name=step.tool_name,
                    args=step.tool_args,
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
        step.mark_failed(last_result.error or "Unknown error during execution")

    def _extract_evidence(self, step: PlanStep, data: Any, ctx: ResearchContext) -> None:
        """Extract structured Evidence items from raw tool results."""
        if step.tool_name == "web_search":
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

        elif step.tool_name == "fetch_page":
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
