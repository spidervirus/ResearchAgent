"""
Planner — converts a raw research query into a structured ResearchPlan.

The Planner calls the LLM with a structured prompt and parses the response
into an ordered list of PlanStep objects.  Each step specifies:
  - A human-readable description of what needs to be done.
  - The tool to invoke (web_search | fetch_page | run_code).
  - The arguments for that tool.

The planner is intentionally conservative: it prefers fewer, high-value steps
over an exhaustive crawl.

Usage::

    planner = Planner(llm=llm, max_steps=8)
    plan = await planner.plan(ctx)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from research_agent.core.context import ResearchContext
from research_agent.llm.base import BaseLLM
from research_agent.models import PlanStep, ResearchPlan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
You are an expert research planner.  Your job is to decompose a research query
into a concise, ordered list of tool-invocation steps that will yield high-quality,
source-backed evidence.

Available tools:
  • web_search  — search the web.  args: { "query": "<string>", "num_results": <int 3-10> }
  • fetch_page  — fetch and extract text from a URL.  args: { "url": "<string>", "max_chars": <int> }
  • run_code    — execute Python to compute, parse, or summarise.  args: { "code": "<python>", "language": "python" }

Rules:
1. Return ONLY a valid JSON object — no prose, no markdown, no code fences.
2. The JSON must have exactly two keys: "rationale" (string) and "steps" (array).
3. Each step object: { "description": "...", "tool_name": "...", "tool_args": { ... } }
4. Aim for 4–8 steps.  Quality over quantity.
5. Always start with at least one web_search step.
6. If the query needs numeric analysis, include a run_code step at the end.
7. Do NOT hallucinate URLs — only use fetch_page for URLs returned by web_search.
8. If previous context is provided, build on it rather than repeating work.
"""


def _build_planner_prompt(ctx: ResearchContext) -> list[dict]:
    """Construct the LLM messages for the planner."""
    user_content = f"Research query: {ctx.query}"
    if ctx.previous_context:
        user_content += (
            f"\n\nPrevious research context (build on this, don't repeat):\n"
            f"{ctx.previous_context}"
        )
    return [
        {"role": "system", "content": _PLANNER_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def _parse_plan(raw: str, ctx: ResearchContext) -> ResearchPlan:
    """Parse the LLM JSON response into a ResearchPlan.

    Handles minor formatting issues (stray backticks, leading/trailing whitespace).

    Args:
        raw: Raw text from the LLM.
        ctx: The current research context.

    Returns:
        A fully populated :class:`ResearchPlan`.

    Raises:
        ValueError: If the response cannot be parsed.
    """
    # Strip common LLM artefacts
    cleaned = raw.strip()
    # Remove markdown code fence if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Try to extract JSON object via regex
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse planner JSON: {exc}") from exc
        else:
            raise ValueError(f"No JSON found in planner response: {exc}") from exc

    rationale: str = data.get("rationale", "")
    raw_steps: list[dict] = data.get("steps", [])

    steps: list[PlanStep] = []
    for raw_step in raw_steps:
        try:
            step = PlanStep(
                description=raw_step.get("description", "Research step"),
                tool_name=raw_step.get("tool_name", "web_search"),
                tool_args=raw_step.get("tool_args", {}),
            )
            steps.append(step)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed plan step: %s — %s", raw_step, exc)

    if not steps:
        # Fallback: single search step so the pipeline never stalls
        steps = [
            PlanStep(
                description=f"Search the web for: {ctx.query}",
                tool_name="web_search",
                tool_args={"query": ctx.query, "num_results": 7},
            )
        ]

    return ResearchPlan(
        query=ctx.query,
        steps=steps,
        rationale=rationale,
    )


class Planner:
    """Converts a research query into a structured :class:`ResearchPlan`.

    Args:
        llm: An :class:`~research_agent.llm.base.BaseLLM` instance.
        max_steps: Hard cap on the number of plan steps (default 10).
        temperature: LLM temperature for planning (lower = more deterministic).

    Example::

        planner = Planner(llm=llm)
        plan = await planner.plan(ctx)
        for step in plan.steps:
            print(step.tool_name, step.description)
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_steps: int = 10,
        temperature: float = 0.2,
    ) -> None:
        self._llm = llm
        self._max_steps = max_steps
        self._temperature = temperature

    async def plan(self, ctx: ResearchContext) -> ResearchPlan:
        """Generate a research plan for the given context.

        Args:
            ctx: The current :class:`ResearchContext`.

        Returns:
            A :class:`ResearchPlan` with ordered steps.

        Raises:
            RuntimeError: If the LLM returns an unparseable response after retries.
        """
        messages = _build_planner_prompt(ctx)
        logger.info("Planner: generating plan for query=%r", ctx.query)

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                raw = await self._llm.complete(
                    messages,
                    temperature=self._temperature,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                plan = _parse_plan(raw, ctx)
                # Enforce step cap
                if len(plan.steps) > self._max_steps:
                    plan.steps = plan.steps[: self._max_steps]

                ctx.plan = plan
                logger.info(
                    "Planner: produced %d steps (attempt %d)",
                    len(plan.steps),
                    attempt + 1,
                )
                return plan
            except Exception as exc:  # noqa: BLE001
                logger.warning("Planner attempt %d failed: %s", attempt + 1, exc)
                last_error = exc

        # All attempts failed — build a minimal fallback plan
        logger.error("Planner: all attempts failed; using fallback plan. Last error: %s", last_error)
        fallback = ResearchPlan(
            query=ctx.query,
            steps=[
                PlanStep(
                    description=f"Search the web for: {ctx.query}",
                    tool_name="web_search",
                    tool_args={"query": ctx.query, "num_results": 7},
                )
            ],
            rationale="Fallback single-step plan (LLM planner failed).",
        )
        ctx.plan = fallback
        return fallback
