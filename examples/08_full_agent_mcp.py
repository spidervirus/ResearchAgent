"""
Progressive Tutorial — Step 8: Full Research Agent + Session Memory + MCP Tools.

WHY THIS STEP EXISTS:
This is the complete, production-grade system that unifies every module:
1. Orchestrator pipeline (Planner → Executor → Verifier → Synthesizer).
2. SQLite Session Memory for persistent multi-turn research across queries.
3. Model Context Protocol (MCP) Tool Support — connect any MCP server
   (databases, local filesystems, Github, custom internal APIs) seamlessly.

Run:
    python examples/08_full_agent_mcp.py
"""

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from research_agent import ResearchAgent
from research_agent.models import FollowUpQuestion, Limitation, ResearchAnswer, Severity
from research_agent.tools.base import BaseTool, ToolResult

console = Console()


# Example of an MCP Tool Adapter representation
class CustomLocalDataMCPTool(BaseTool):
    """Demonstrates exposing an external MCP database/context tool to ResearchAgent."""

    name = "internal_cluster_benchmarks"
    description = "Queries internal infrastructure benchmark database for hardware metrics."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cluster_type": {"type": "string", "description": "e.g. 'c6i.2xlarge' or 'g4dn.xlarge'"},
                        "target_qps": {"type": "integer", "description": "Target concurrency load"},
                    },
                    "required": ["cluster_type"],
                },
            },
        }

    async def run(self, cluster_type: str = "c6i.2xlarge", target_qps: int = 10000, **kwargs) -> ToolResult:
        # Simulate querying an internal MCP data source
        return self._ok(
            {
                "instance": cluster_type,
                "measured_qps": 11420,
                "p95_latency_ms": 7.4,
                "p99_latency_ms": 14.2,
                "monthly_cost_usd": 248.0,
                "source": "Internal Infrastructure Perf Team (MCP Telemetry)",
            }
        )


async def main() -> None:
    console.print(
        Panel(
            "[bold cyan]Step 8: Full Research Agent + Session Memory + MCP Tools[/bold cyan]\n"
            "[dim]The complete system: Planner, Executor, Verifier, Synthesizer, Memory & MCP[/dim]",
            border_style="cyan",
        )
    )

    with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
        # Check if live keys are present; if not, use a demo LLM interface
        has_llm = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
        llm = None
        if not has_llm:
            from research_agent.llm.base import BaseLLM

            class LocalDemoLLM(BaseLLM):
                async def complete(self, messages, **kwargs):
                    return "{}"

                async def stream(self, messages, **kwargs):
                    yield "{}"

            llm = LocalDemoLLM(model="demo-model", provider="demo")

        # 1. Instantiate the full ResearchAgent with custom MCP tool & SQLite session store
        mcp_tool = CustomLocalDataMCPTool()
        agent = ResearchAgent(
            llm=llm,
            tools=[mcp_tool],
            human_gate=False,
            db_path=tmp_db.name,
            verbose=False,
        )

        console.print(f"[bold]Active Tools:[/bold] {list(agent.tools.keys())}\n")

        # 2. Turn 1: Initial Research Query
        q1 = "Compare vector database options for 10k QPS on AWS c6i.2xlarge"
        console.print(f"[bold green]Turn 1 Query:[/bold green] {q1}")

        # Check if live keys are present; if not, use agent's internal architecture with mock for demo
        has_llm = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))

        if has_llm:
            ans1 = await agent.research(q1)
        else:
            # Reconstruct demo answer to showcase session persistence & structure
            from research_agent.models import Evidence, PlanStep, ResearchPlan

            step = PlanStep(description="Query internal MCP benchmark telemetry", tool_name="internal_cluster_benchmarks")
            plan = ResearchPlan(query=q1, steps=[step])
            ev = Evidence(
                source_url="mcp://internal-telemetry/benchmarks",
                source_title="Internal Perf Benchmark on c6i.2xlarge",
                excerpt="Measured 11,420 QPS with p95 of 7.4ms at $248/month cost.",
                relevance_score=0.95,
                quality_score=0.95,
            )
            ans1 = ResearchAnswer(
                query=q1,
                direct_answer="On an AWS c6i.2xlarge node, Qdrant achieves 11,420 QPS at 7.4ms p95 latency, fully meeting the 10k QPS target under $250/mo.",
                key_findings=["Validated 11,420 QPS throughput with single-node deployment [MCP Telemetry](mcp://internal-telemetry)"],
                evidence=[ev],
                confidence=0.92,
                confidence_rationale="Grounded in empirical telemetry from internal hardware benchmarks.",
                limitations=[Limitation(description="Requires reserving 32GB RAM for HNSW graph buffers.", severity=Severity.MEDIUM)],
                follow_up_questions=[FollowUpQuestion(question="How does throughput change under 500 write operations/sec?", rationale="Concurrent indexing can impact latency")],
                plan=plan,
                session_id=agent.session_id or "session_12345",
            )
            # Save into session store
            from research_agent.core.context import ResearchContext

            demo_ctx = ResearchContext(query=q1, session_id=ans1.session_id or "session_12345", answer=ans1)
            agent.session_store.save(demo_ctx.session_id, demo_ctx)
            agent.session_id = demo_ctx.session_id

        console.print(Panel(ans1.direct_answer, title="🎯 Turn 1 Result", border_style="green"))
        console.print(f"Confidence: {ans1.confidence:.0%}")
        console.print(f"[dim]Persisted Session ID: {agent.session_id}[/dim]\n")

        # 3. Inspect Session Persistence
        saved_sessions = agent.list_sessions()
        console.print(f"[bold]Persisted Sessions in SQLite:[/bold] {len(saved_sessions)}")
        for s in saved_sessions:
            console.print(f"  • ID: {s['id'][:12]}.. | Query: {s['query'][:45]}..")

        console.print(
            "\n[bold green]🎉 Complete Progressive Tutorial Finished![/bold green]\n"
            "You have built and understood every moving part of ResearchAgent:\n"
            "  1. Plain LLM call\n"
            "  2. Tool calling with Web Search\n"
            "  3. Explicit Planning\n"
            "  4. Multi-step Execution Loop\n"
            "  5. Source Verification & Credibility Scoring\n"
            "  6. Structured Answer with Confidence & Limitations\n"
            "  7. Human-in-the-Loop Approval Gates\n"
            "  8. Full Orchestrator + Session Persistence + Model Context Protocol (MCP)\n"
        )


if __name__ == "__main__":
    asyncio.run(main())
