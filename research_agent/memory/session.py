"""
SessionStore — persists ResearchContext to SQLite so multi-turn research works.

Database path: SESSION_DB_PATH env var or ~/.research_agent/sessions.db

Schema:
    sessions(
        id           TEXT PRIMARY KEY,
        query        TEXT NOT NULL,
        context_json TEXT NOT NULL,
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    )

Usage::

    store = SessionStore()
    store.save(ctx.session_id, ctx)

    ctx = store.load("some-session-id")
    if ctx:
        print(ctx.query)

    for row in store.list_sessions():
        print(row["id"], row["query"], row["updated_at"])

    store.delete("some-session-id")
"""

from __future__ import annotations

import logging
import os
import pathlib
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from research_agent.core.context import ResearchContext

logger = logging.getLogger(__name__)


def _default_db_path() -> pathlib.Path:
    """Return the default SQLite database path, creating parent dirs as needed."""
    raw = os.getenv("SESSION_DB_PATH", "")
    if raw:
        path = pathlib.Path(raw).expanduser().resolve()
    else:
        path = pathlib.Path.home() / ".research_agent" / "sessions.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class SessionStore:
    """SQLite-backed store for ResearchContext objects using standard library sqlite3."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path:
            self._path = pathlib.Path(db_path).expanduser().resolve()
            self._path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._path = _default_db_path()

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with row factory enabled."""
        conn = sqlite3.connect(str(self._path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the sessions table if it doesn't already exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id           TEXT PRIMARY KEY,
                    query        TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    created_at   TEXT NOT NULL,
                    updated_at   TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save(self, session_id: str, context: ResearchContext) -> None:
        """Upsert a research context into the database."""
        context.touch()
        now_str = datetime.now(tz=timezone.utc).isoformat()
        context_json = context.model_dump_json()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, query, context_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    query = excluded.query,
                    context_json = excluded.context_json,
                    updated_at = excluded.updated_at
                """,
                (session_id, context.query, context_json, now_str, now_str),
            )
            conn.commit()

    def load(self, session_id: str) -> Optional[ResearchContext]:
        """Load and reconstruct a ResearchContext by session ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT context_json FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            try:
                return ResearchContext.model_validate_json(row["context_json"])
            except Exception as exc:
                logger.error("Failed to deserialize session %s: %s", session_id, exc)
                return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions ordered by most recently updated."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, query, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "query": row["query"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]

    def delete(self, session_id: str) -> bool:
        """Delete a session by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0
