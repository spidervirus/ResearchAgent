"""
research_agent — A structured, source-backed AI research agent.

Exports:
    ResearchAgent: Main agent class.
    ResearchAnswer: Final answer model.
    ResearchContext: Shared pipeline state.
    ResearchPlan: Plan produced by the planner.
    Evidence: Single piece of sourced evidence.
"""

from research_agent.agent import ResearchAgent
from research_agent.core.context import ResearchContext
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

__version__ = "0.1.0"

__all__ = [
    "ResearchAgent",
    "Evidence",
    "FollowUpQuestion",
    "Limitation",
    "PlanStep",
    "ResearchAnswer",
    "ResearchPlan",
    "ResearchContext",
    "ToolCall",
    "VerificationResult",
    "__version__",
]
