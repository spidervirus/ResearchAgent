# Human-in-the-Loop: When to Trust, When to Check

> **Chapter 7 of 10** — Designing safe agentic workflows with interactive checkpoints, approving or modifying plans on the fly, and running headlessly in production.

---

## The Philosophy of Agent Gates

Autonomous agents that run unconstrained can waste API credits, spam external services, or go down rabbit holes. The solution is **Human-in-the-Loop (HITL) checkpoints**: strategic pause points where a human can:
- **Inspect**: Review what the agent intends to do before expensive operations run.
- **Steer**: Provide user guidance or refine the research direction.
- **Abort**: Cancel execution early if the plan is off-target.

---

## The Three Checkpoint Types

`research_agent/core/human_gate.py` implements three checkpoint levels:

### 1. `PLAN_APPROVAL`
Fires immediately after the Planner generates the step list. Displays the proposed plan table in the terminal:

```
📋 Proposed Research Plan
┌───┬──────────────────────────────────┬──────────────┬────────────────────────┐
│ # │ Description                      │ Tool         │ Target / Query         │
├───┼──────────────────────────────────┼──────────────┼────────────────────────┤
│ 1 │ Search vector DB QPS benchmarks  │ web_search   │ vector db 10k QPS      │
│ 2 │ Fetch Qdrant memory architecture │ page_fetcher │ https://qdrant.tech/.. │
│ 3 │ Compute memory footprint         │ code_runner  │ vectors = 10_000_000.. │
└───┴──────────────────────────────────┴──────────────┴────────────────────────┘

Approve plan? [P]roceed / [E]dit / [A]bort (default: P):
```

Options:
- `P` (Proceed): Execute the plan as-is.
- `E` (Edit): Type feedback (e.g. *"Focus specifically on Rust-based databases"*). The feedback is added to `ctx.metadata["user_edits"]`.
- `A` (Abort): Gracefully stops the run before any tools execute.

### 2. `SYNTHESIS_APPROVAL`
Fires after tool execution and verification, showing a summary of the evidence collected before synthesis consumes tokens.

### 3. `EXPENSIVE_TOOL`
Fires before invoking potentially slow, rate-limited, or costly tools (like code interpreters or deep crawl sweeps).

---

## Configuring the Human Gate

### In the CLI
By default, the CLI uses interactive gates in an interactive terminal. You can bypass them with:

```bash
research-agent research "Compare vector DBs" --no-human-gate
```

### In Python Code
When writing automated scripts, background jobs, or CI/CD pipelines, disable the gate:

```python
agent = ResearchAgent(human_gate=False)
```

### Automatic Headless Detection
If ResearchAgent detects that standard input is not a TTY (for example, when running inside Docker, a cron job, or a Unix pipe like `echo "query" | research-agent`), it **automatically auto-approves** checkpoints without hanging.

In the next chapter, we explore **Chapter 8: Memory & Sessions** to see how multi-turn research is persisted.

