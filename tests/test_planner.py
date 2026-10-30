"""
Tests for Planner stage in research_agent.core.planner.
"""

import json
import pytest
from research_agent.core.context import ResearchContext
from research_agent.core.planner import Planner
from tests.conftest import MockLLM


@pytest.mark.asyncio
async def test_planner_produces_plan(sample_context: ResearchContext) -> None:
    canned_plan = {
        "rationale": "Decompose into search and targeted page fetch.",
        "steps": [
            {
                "description": "Search top vector databases",
                "tool_name": "web_search",
                "tool_args": {"query": "vector databases 10k QPS benchmark", "num_results": 5},
            },
            {
                "description": "Fetch benchmark paper",
                "tool_name": "fetch_page",
                "tool_args": {"url": "https://example.com/benchmark"},
            },
        ],
    }

    mock_llm = MockLLM(responses=[json.dumps(canned_plan)])
    planner = Planner(llm=mock_llm, max_steps=5)

    plan = await planner.plan(sample_context)

    assert plan is not None
    assert len(plan.steps) == 2
    assert plan.steps[0].tool_name == "web_search"
    assert plan.steps[1].tool_name == "fetch_page"
    assert sample_context.plan == plan


@pytest.mark.asyncio
async def test_planner_handles_invalid_json(sample_context: ResearchContext) -> None:
    # Model produces invalid non-JSON output
    mock_llm = MockLLM(responses=["Sorry, I cannot format this in JSON!"])
    planner = Planner(llm=mock_llm)

    plan = await planner.plan(sample_context)

    # Should fall back gracefully to a baseline plan
    assert plan is not None
    assert len(plan.steps) >= 1
    assert plan.steps[0].tool_name == "web_search"
    assert sample_context.plan == plan


@pytest.mark.asyncio
async def test_planner_respects_max_steps(sample_context: ResearchContext) -> None:
    many_steps = {
        "rationale": "Excessive steps",
        "steps": [
            {"description": f"Step {i}", "tool_name": "web_search", "tool_args": {}}
            for i in range(12)
        ],
    }
    mock_llm = MockLLM(responses=[json.dumps(many_steps)])
    planner = Planner(llm=mock_llm, max_steps=4)

    plan = await planner.plan(sample_context)

    assert len(plan.steps) <= 4

