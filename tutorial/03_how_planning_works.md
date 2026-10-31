# Deep Dive: The Planner

> **Chapter 3 of 10** — Why agents need explicit plans, how the Planner prompts the LLM for structured plans, and how plan steps are represented and validated.

---

## Why Planning Matters

In naive LLM agents, an agent runs in a free-form loop: it thinks, calls a tool, gets a response, and repeats until it decides to stop. While flexible, this approach suffers from major failure modes in real research:
1. **Looping traps**: The agent gets obsessed with one tangential search query.
2. **Context window bloat**: Repeating full tool outputs in the prompt degrades reasoning.
3. **No audit trail**: You cannot see upfront what the agent intends to do or stop it from wasting tokens.

**ResearchAgent** separates **Planning** from **Execution**:

```
                       ┌──────────────────────────────┐
                       │       User Research Query    │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │           Planner            │
                       │   (Structured JSON schema)   │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                        ResearchPlan [Step 1, Step 2, ...]
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
           [Human Approval Gate]              [Executor Dispatches]
```

---

## The Planner Prompt Structure

The planner prompt located in `research_agent/core/planner.py` uses a strict instruction contract:

```python
_PLANNER_SYSTEM = """\
You are an expert research planner.  Your job is to decompose a research query
into a concise, ordered list of tool-invocation steps that will yield high-quality,
source-backed evidence.

Available tools:
  • web_search  — search the web. args: { "query": "<string>", "num_results": <int 3-10> }
  • fetch_page  — fetch and extract text from a URL. args: { "url": "<string>", "max_chars": <int> }
  • run_code    — execute Python to compute, parse, or summarise. args: { "code": "<python>", "language": "python" }

Rules:
1. Return ONLY a valid JSON object — no prose, no markdown, no code fences.
2. The JSON must have exactly two keys: "rationale" (string) and "steps" (array).
3. Each step object: { "description": "...", "tool_name": "...", "tool_args": { ... }, "depends_on": [] }
4. Steps with empty "depends_on": [] execute concurrently in parallel!
5. If a step depends on an earlier step, list the 1-based index in "depends_on", e.g. ["1"].
6. Aim for 3–6 steps. Quality over quantity.
"""
```

### Example LLM Output

For the question: *"Compare the top 3 open-source vector databases for 10k QPS"*

```json
{
  "rationale": "We first locate empirical benchmarks for high-throughput vector databases, then fetch the benchmark documentation for Qdrant and Milvus, and finally run a python snippet to compute memory cost estimates.",
  "steps": [
    {
      "description": "Search for vector database QPS benchmarks",
      "tool_name": "web_search",
      "tool_args": {
        "query": "open source vector database benchmark 10k QPS 2024",
        "num_results": 5
      }
    },
    {
      "description": "Fetch official Qdrant scaling and memory guide",
      "tool_name": "fetch_page",
      "tool_args": {
        "url": "https://qdrant.tech/documentation/guides/multiple-partitions/"
      }
    },
    {
      "description": "Calculate RAM footprint for 10M 768-dim float32 vectors",
      "tool_name": "run_code",
      "tool_args": {
        "code": "vectors = 10_000_000\ndim = 768\nbytes_per_vec = dim * 4\nraw_gb = (vectors * bytes_per_vec) / (1024**3)\nprint(f'Raw vector RAM: {raw_gb:.2f} GB; HNSW graph overhead (+20%): {raw_gb * 1.2:.2f} GB')",
        "language": "python"
      }
    }
  ]
}
```

---

## Defensive Parsing & Fallbacks

Language models can occasionally wrap JSON in markdown blocks (````json ... ````) or append conversational preamble.

The planner includes a resilient multi-stage parser in `planner.py`:
1. Strips triple-backtick markers and whitespace.
2. Attempts direct `json.loads`.
3. If that fails, extracts the outermost `{ ... }` block via regular expression.
4. If the model completely fails to output JSON, the planner generates a deterministic fallback plan:
   - Step 1: Initial web search on the user's query.
   - Step 2: High-authority search targeting documentation.

This guarantees the pipeline never crashes on a syntax hiccup.

In the next chapter, we explore **Chapter 4: Tools Deep Dive** to see how the Executor runs search, extracts clean webpage text, and isolates code execution.

