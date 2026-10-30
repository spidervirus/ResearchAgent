"""
ResearchAgent — The main orchestrator.

Coordinates the end-to-end research lifecycle:
  Query → Plan → [Human Gate] → Execute → Verify → [Human Gate] → Synthesize → Persist

Usage::

    import asyncio
    from research_agent import ResearchAgent

    async def main():
        agent = ResearchAgent(human_gate=False)
        answer = await agent.research(
            "Compare top 3 open-source vector databases for 10k QPS"
        )
        print(answer.direct_answer)
        print(f"Confidence: {answer.confidence:.0%}")

    asyncio.run(main())
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.status import Status

from research_agent.config import get_settings
from research_agent.core.context import ResearchContext
from research_agent.core.executor import Executor
from research_agent.core.human_gate import HumanGate
from research_agent.core.planner import Planner
from research_agent.core.synthesizer import Synthesizer
from research_agent.core.verifier import Verifier
from research_agent.llm.anthropic_llm import AnthropicLLM
from research_agent.llm.base import BaseLLM
from research_agent.llm.openai_llm import OpenAILLM
from research_agent.memory.session import SessionStore
from research_agent.models import ResearchAnswer, ResearchPlan
from research_agent.tools.base import BaseTool
from research_agent.tools.code_runner import CodeRunnerTool
from research_agent.tools.page_fetcher import PageFetcherTool
from research_agent.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)


def _resolve_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseLLM:
    """Resolve and instantiate the LLM adapter based on provider preference."""
    p = (provider or os.getenv("DEFAULT_LLM_PROVIDER") or "openai").lower().strip()
    m = model or os.getenv("DEFAULT_LLM_MODEL")

    if p == "anthropic":
        return AnthropicLLM(model=m or "claude-3-5-sonnet-20241022")
    elif p == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OpenAILLM.for_ollama(model=m or "llama3", base_url=base_url)
    else:  # default: openai
        return OpenAILLM(model=m or "gpt-4o-mini")


class ResearchAgent:
    """High-level facade orchestrating all stages of source-backed research.

    Attributes:
        llm: Language model interface for planning, verification, and synthesis.
        tools: Dictionary of registered tools available to the planner and executor.
        session_store: SQLite-backed persistence store for multi-turn sessions.
        human_gate: Approval gate for interactive checkpoints.
        verbose: When True, logs progress to console with rich formatting.
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        search_provider: Optional[str] = None,
        human_gate: Optional[bool] = None,
        session_id: Optional[str] = None,
        max_iterations: Optional[int] = None,
        verbose: bool = True,
        tools: Optional[list[BaseTool]] = None,
        llm: Optional[BaseLLM] = None,
        db_path: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.verbose = verbose
        self.console = Console() if verbose else Console(quiet=True)
        self.session_id = session_id

        # 1. LLM
        self.llm = llm or _resolve_llm(
            llm_provider or settings.default_llm_provider,
            llm_model or settings.default_llm_model,
        )

        # 2. Tools
        self.tools: dict[str, BaseTool] = {}
        sp = search_provider or settings.default_search_provider
        self.register_tool(WebSearchTool(provider=sp))
        self.register_tool(PageFetcherTool())
        self.register_tool(CodeRunnerTool(enabled=settings.code_runner_enabled))

        if tools:
            for t in tools:
                self.register_tool(t)

        # 3. Persistence
        self.session_store = SessionStore(db_path=db_path or settings.session_db_path)

        # 4. Human gate (easily disableable via flag, constructor, or env)
        gate_enabled = human_gate if human_gate is not None else settings.human_gate_enabled
        self.human_gate = HumanGate(enabled=gate_enabled, console=self.console)

        # 5. Pipeline Stages
        self.planner = Planner(llm=self.llm, max_steps=max_iterations or settings.max_plan_steps)
        self.executor = Executor(tools=self.tools)
        self.verifier = Verifier(llm=self.llm, min_relevance_score=settings.min_relevance_score)
        self.synthesizer = Synthesizer(llm=self.llm)

    def register_tool(self, tool: BaseTool) -> None:
        """Register a custom tool with the agent."""
        self.tools[tool.name] = tool
        if hasattr(self, "executor"):
            self.executor.register_tool(tool)

    async def research(self, query: str, session_id: Optional[str] = None) -> ResearchAnswer:
        """Execute a full research cycle for the given question.

        Args:
            query: The natural-language research question.
            session_id: Optional explicit session ID to associate or resume.

        Returns:
            A structured :class:`~research_agent.models.ResearchAnswer`.
        """
        sid = session_id or self.session_id
        ctx = ResearchContext(query=query)
        if sid:
            ctx.session_id = sid
            # Check for prior session state to link context
            prior_ctx = self.session_store.load(sid)
            if prior_ctx and prior_ctx.answer:
                ctx.previous_context = (
                    f"Prior Query: {prior_ctx.query}\n"
                    f"Prior Answer Summary: {prior_ctx.answer.direct_answer}\n"
                    f"Prior Key Findings: {'; '.join(prior_ctx.answer.key_findings[:3])}"
                )

        if self.verbose:
            self.console.print(Panel(f"[bold cyan]{query}[/bold cyan]", title="🔬 Research Query"))

        # --- Stage 1: Planning ---
        with self.console.status("[bold green]Generating research plan...", spinner="dots"):
            plan = await self.planner.plan(ctx)

        # Plan Approval Checkpoint
        approved = await self.human_gate.check("plan_approval", ctx)
        if not approved:
            raise RuntimeError("Research plan was aborted by user.")

        # --- Stage 2: Tool Execution ---
        with self.console.status(f"[bold green]Executing {len(plan.steps)} plan steps...", spinner="dots"):
            await self.executor.execute(ctx)

        if self.verbose:
            ev_count = len(ctx.raw_evidence)
            self.console.print(f"[dim]Collected {ev_count} raw evidence items from tools.[/dim]")

        # --- Stage 3: Verification & Criticism ---
        with self.console.status("[bold green]Verifying evidence and evaluating credibility...", spinner="dots"):
            v_res = await self.verifier.verify(ctx)

        if v_res.issues:
            ctx.metadata["verifier_issues"] = "\n".join(v_res.issues)

        # Synthesis Approval Checkpoint
        approved = await self.human_gate.check("synthesis_approval", ctx)
        if not approved:
            raise RuntimeError("Synthesis aborted by user.")

        # --- Stage 4: Synthesis ---
        with self.console.status("[bold green]Synthesizing structured answer with citations...", spinner="dots"):
            answer = await self.synthesizer.synthesize(ctx)

        # Save session
        self.session_store.save(ctx.session_id, ctx)
        self.session_id = ctx.session_id

        return answer

    async def continue_research(self, followup_query: str, session_id: Optional[str] = None) -> ResearchAnswer:
        """Continue research in an existing session with a follow-up query.

        Args:
            followup_query: The follow-up question building upon prior findings.
            session_id: Target session ID (defaults to current active session).

        Returns:
            A new structured :class:`~research_agent.models.ResearchAnswer`.
        """
        sid = session_id or self.session_id
        if not sid:
            raise ValueError("No active session_id found to continue. Run research() first or supply session_id.")

        prior_ctx = self.session_store.load(sid)
        summary = ""
        if prior_ctx and prior_ctx.answer:
            summary = (
                f"Previous Query: {prior_ctx.query}\n"
                f"Previous Answer: {prior_ctx.answer.direct_answer}\n"
                f"Previous Findings: {'; '.join(prior_ctx.answer.key_findings)}"
            )

        new_ctx = ResearchContext(query=followup_query, session_id=sid, previous_context=summary)

        # Plan
        plan = await self.planner.plan(new_ctx)
        approved = await self.human_gate.check("plan_approval", new_ctx)
        if not approved:
            raise RuntimeError("Follow-up plan rejected.")

        # Execute
        await self.executor.execute(new_ctx)

        # Verify
        await self.verifier.verify(new_ctx)

        # Synthesize
        answer = await self.synthesizer.synthesize(new_ctx)

        # Persist updated session
        self.session_store.save(sid, new_ctx)
        return answer

    def load_answer(self, session_id: str) -> Optional[ResearchAnswer]:
        """Retrieve the answer from a past saved session."""
        ctx = self.session_store.load(session_id)
        return ctx.answer if ctx else None

    def list_sessions(self) -> list[dict]:
        """List all persisted research sessions."""
        return self.session_store.list_sessions()

