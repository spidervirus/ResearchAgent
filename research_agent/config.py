"""
Configuration management for ResearchAgent.

Centralizes environment variables, provider settings, security toggles,
and heuristic thresholds with convenient defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _bool_env(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    """Central configuration for ResearchAgent."""

    # LLM Provider Configuration
    default_llm_provider: str = field(
        default_factory=lambda: os.getenv("DEFAULT_LLM_PROVIDER", "openai").lower().strip()
    )
    default_llm_model: str = field(
        default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini").strip()
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    )
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )

    # Search Provider Configuration
    default_search_provider: str = field(
        default_factory=lambda: os.getenv("DEFAULT_SEARCH_PROVIDER", "duckduckgo").lower().strip()
    )
    tavily_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY")
    )
    serpapi_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("SERPAPI_API_KEY")
    )

    # Agent Pipeline Tuning
    min_relevance_score: float = field(
        default_factory=lambda: float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))
    )
    max_plan_steps: int = field(
        default_factory=lambda: int(os.getenv("MAX_PLAN_STEPS", "8"))
    )
    request_timeout: float = field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT", "60.0"))
    )

    # Safety & Human-in-the-Loop Toggles
    # Code runner is opt-in (disabled by default for security)
    code_runner_enabled: bool = field(
        default_factory=lambda: _bool_env("CODE_RUNNER_ENABLED", False)
    )
    human_gate_enabled: bool = field(
        default_factory=lambda: _bool_env("HUMAN_GATE_ENABLED", False)
    )

    # Persistence
    session_db_path: str = field(
        default_factory=lambda: os.getenv(
            "SESSION_DB_PATH",
            str(Path.home() / ".research_agent" / "sessions.db"),
        )
    )


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the global Settings instance, initializing from environment if necessary."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

