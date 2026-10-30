# Memory & Multi-Turn Research

> **Chapter 8 of 10** — Persisting research state to SQLite, resuming previous sessions, and building iterative multi-turn research workflows.

---

## Why Sessions Matter

Real-world research is almost never one-shot. You ask an initial question, review the answer, and immediately think of a follow-up:
- *"Between the two databases you recommended, what is the RAM difference when quantization is enabled?"*
- *"Can you now compare their Kubernetes Helm charts?"*

If the agent restarts from scratch with zero context, it repeats all prior tool calls and loses the thread.

---

## SQLite Session Persistence (`session.py`)

ResearchAgent persists every research run into a local SQLite database (`~/.research_agent/sessions.db` by default, configurable via `SESSION_DB_PATH`).

### Database Schema

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    query        TEXT NOT NULL,
    context_json TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
```

Because `ResearchContext` is a Pydantic v2 model, `to_dict()` and `from_dict()` serialize the entire state—including plan steps, evidence items, tool calls, and final answers—into a single JSON payload.

---

## Multi-Turn Research in Action

### Using the Python API

```python
import asyncio
from research_agent import ResearchAgent

async def run_multi_turn():
    agent = ResearchAgent(human_gate=False)

    # Turn 1: Initial research
    ans1 = await agent.research("Compare Qdrant and Milvus for 10k QPS")
    print(f"Turn 1 Answer: {ans1.direct_answer[:150]}...")
    print(f"Active Session: {agent.session_id}")

    # Turn 2: Follow-up question building directly on Turn 1
    ans2 = await agent.continue_research(
        "Which of those two has a lower RAM footprint when scalar quantization is turned on?"
    )
    print(f"\nTurn 2 Answer: {ans2.direct_answer}")

asyncio.run(run_multi_turn())
```

When `continue_research()` is called:
1. It loads the prior session from SQLite.
2. It extracts prior findings into `ctx.previous_context`.
3. The Planner receives this prior context in its system prompt and formulates steps that **build upon** prior findings without repeating redundant searches.
4. The updated session is saved back to the database under the same session ID.

---

## Inspecting Past Sessions via CLI

List all historical research runs:

```bash
research-agent sessions
```

Output:
```
📜 Research Sessions
┌──────────────┬────────────────────────────────────────────────────┬─────────────────────┐
│ Session ID   │ Query                                              │ Last Updated        │
├──────────────┼────────────────────────────────────────────────────┼─────────────────────┤
│ 4a9d702e..   │ Compare the top 3 open-source vector databases     │ 2024-10-30 03:15:20 │
│ b28cf1a0..   │ Explain Retrieval-Augmented Generation tradeoffs   │ 2024-10-29 18:42:01 │
└──────────────┴────────────────────────────────────────────────────┴─────────────────────┘
```

Re-render any historical answer in full rich formatting:

```bash
research-agent show 4a9d702e-c750-4fc7-bf84-fa6d7e0d3753
```

In the next chapter, we look at **Chapter 9: Adding Your Own Tools** to see how to extend the agent with custom APIs or databases.

