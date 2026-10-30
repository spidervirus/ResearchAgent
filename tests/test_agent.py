"""
Tests for full agent orchestrator in research_agent.agent.
"""

import json
import pytest
from research_agent.agent import ResearchAgent
from research_agent.models import ResearchAnswer
from tests.conftest import MockLLM, _MockPageFetcherTool, _MockWebSearchTool


@pytest.mark.asyncio
async def test_agent_full_pipeline_mocked(tmp_path) -> None:
    db_file = tmp_path / "test_sessions.db"

    canned_plan = {
        "rationale": "Search vector benchmarks and fetch top paper.",
        "steps": [
            {
                "description": "Search vector databases",
                "tool_name": "web_search",
                "tool_args": {"query": "vector databases", "num_results": 2},
            },
            {
                "description": "Fetch benchmark page",
                "tool_name": "fetch_page",
                "tool_args": {"url": "https://example.com/vector-db-comparison"},
            },
        ],
    }

    canned_critique = {
        "evaluations": [
            {"index": 0, "relevance_score": 0.9, "quality_score": 0.85},
            {"index": 1, "relevance_score": 0.88, "quality_score": 0.9},
        ],
        "issues": [],
        "notes": "Verified high relevance.",
    }

    canned_synthesis = {
        "direct_answer": "Qdrant is the recommended vector database for 10k QPS startup.",
        "key_findings": ["Qdrant provides low-latency HNSW search under sustained concurrency."],
        "confidence": 0.85,
        "confidence_rationale": "High quality benchmark sources confirm throughput stability.",
        "limitations": [{"description": "Hardware RAM sizing is critical.", "severity": "medium"}],
        "follow_up_questions": [{"question": "What is the quantization trade-off?", "rationale": "Saves RAM"}],
    }

    # Responses queue for:
    # 1. Planner
    # 2. Verifier
    # 3. Synthesizer
    responses = [
        json.dumps(canned_plan),
        json.dumps(canned_critique),
        json.dumps(canned_synthesis),
    ]

    mock_llm = MockLLM(responses=responses)
    mock_search = _MockWebSearchTool()
    mock_fetcher = _MockPageFetcherTool()

    agent = ResearchAgent(
        llm=mock_llm,
        tools=[mock_search, mock_fetcher],
        human_gate=False,
        db_path=str(db_file),
        verbose=False,
    )

    query = "Compare top vector databases for 10k QPS"
    answer = await agent.research(query)

    assert isinstance(answer, ResearchAnswer)
    assert "Qdrant" in answer.direct_answer
    assert answer.confidence == 0.85
    assert len(answer.evidence) > 0

    # Test session persistence
    assert agent.session_id is not None
    loaded_answer = agent.load_answer(agent.session_id)
    assert loaded_answer is not None
    assert loaded_answer.direct_answer == answer.direct_answer

    sessions = agent.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["query"] == query

