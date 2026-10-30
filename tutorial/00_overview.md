# ResearchAgent: A Tour of the System

> **Chapter 0 of 10** — Start here. Every subsequent chapter builds on concepts introduced in this one.

---

## The Problem This Solves

Ask a raw LLM a research question and you'll typically get one of three outcomes:

1. **A confident-sounding hallucination.** The model synthesises a plausible-sounding answer from its training data without fetching any real sources.
2. **An outdated answer.** Training data has a cutoff. Anything after it is invisible.
3. **A wall of text with no citations.** You can't verify any of it.

ResearchAgent is built to solve all three. It is a *structured, source-backed* research agent that:

- **Retrieves real-time information** from the web (DuckDuckGo, Tavily, or SerpAPI)
- **Reads source pages** and extracts only the relevant content
- **Scores and filters evidence** by relevance and source quality
- **Synthesises a structured answer** with confidence scores and inline citations
- **Tells you what it doesn't know** — limitations are a first-class output, not an afterthought
- **Invites human review** at key checkpoints before expensive or irreversible steps

The result is an answer you can audit, not just accept.

---

## Architecture Overview

ResearchAgent follows a linear pipeline. Data flows forward through four main stages, with a **Human Gate** that can pause execution at configurable checkpoints.

```
                    ┌─────────────────────────────────────┐
                    │        ResearchAgent (orchestrator)  │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────▼──────────────────┐
                    │              PLANNER                 │
                    │  • Receives the research question    │
                    │  • Calls LLM to produce a plan       │
                    │  • Returns: ResearchPlan (steps)     │
                    └─────────────────┬──────────────────┘
                                      │
                          [Human Gate: PLAN_APPROVAL]
                                      │
                    ┌─────────────────▼──────────────────┐
                    │             EXECUTOR                 │
                    │  • Iterates over plan steps          │
                    │  • Dispatches to tools:              │
                    │      • WebSearchTool                 │
                    │      • PageFetcherTool               │
                    │      • CodeRunnerTool                │
                    │  • Collects raw Evidence objects     │
                    └─────────────────┬──────────────────┘
                                      │
                         [Human Gate: EXPENSIVE_TOOL]
                                      │
                    ┌─────────────────▼──────────────────┐
                    │             VERIFIER                 │
                    │  • Scores evidence: relevance + quality│
                    │  • Applies domain heuristics         │
                    │  • Filters below threshold           │
                    │  • Returns: VerificationResult       │
                    └─────────────────┬──────────────────┘
                                      │
                    ┌─────────────────▼──────────────────┐
                    │           SYNTHESIZER                │
                    │  • Receives verified evidence        │
                    │  • Calls LLM in JSON mode            │
                    │  • Produces: ResearchAnswer          │
                    │      • direct_answer                 │
                    │      • key_findings                  │
                    │      • confidence + rationale        │
                    │      • limitations                   │
                    │      • follow_up_questions           │
                    └─────────────────┬──────────────────┘
                                      │
                       [Human Gate: SYNTHESIS_APPROVAL]
                                      │
                    ┌─────────────────▼──────────────────┐
                    │            FINAL OUTPUT              │
                    │  ResearchAnswer (JSON / Rich CLI)    │
                    └─────────────────────────────────────┘
```

### Shared State: ResearchContext

All pipeline stages share a single `ResearchContext` object. Think of it as the "working memory" of one research session:

```
ResearchContext
├── query              # The original question
├── session_id         # Unique ID, used for persistence
├── plan               # ResearchPlan (set by Planner)
├── raw_evidence       # All Evidence collected (set by Executor)
├── verified_evidence  # Filtered subset (set by Verifier)
├── tool_calls         # Audit log of every tool invocation
├── messages           # LLM conversation history
├── answer             # Final ResearchAnswer (set by Synthesizer)
└── metadata           # Arbitrary inter-stage communication
```

This is defined in [`research_agent/core/context.py`](../research_agent/core/context.py).

---

## Suggested Reading Order

This tutorial is designed to be read linearly. Each chapter either introduces a new concept or goes deeper into one you've already seen.

| Chapter | Title | Read when you want to… |
|---------|-------|------------------------|
| **00** (this file) | Overview | Understand the big picture |
| **01** | Installation & Configuration | Get the system running |
| **02** | Your First Research Query | See it work end-to-end |
| **03** | Deep Dive: The Planner | Understand how plans are generated |
| **04** | Tools Deep Dive | Understand how data is fetched |
| **05** | The Verifier | Understand evidence filtering |
| **06** | The Synthesizer | Understand structured output |
| **07** | Human-in-the-Loop | Understand the review gates |
| **08** | Memory & Sessions | Understand persistence |
| **09** | Adding Custom Tools | Extend the agent |
| **10** | Local Models with Ollama | Run without cloud APIs |

> **Tip:** After reading chapters 00–02, you can jump to any chapter in any order. Each chapter is self-contained enough to be referenced independently.

---

## What You'll Be Able to Build

After completing this tutorial, you will be able to:

1. **Run research queries** from the CLI and the Python API
2. **Interpret structured output** — confidence scores, limitations, citations
3. **Customise the Planner** — change step count, add custom tool descriptions
4. **Add your own tools** — WikipediaTool, ArXiv search, PDF reader, SQL query
5. **Configure human review gates** for your workflow
6. **Persist and continue research sessions** across multiple queries
7. **Run the full stack locally** with Ollama (no cloud API required)

---

## Key Design Decisions

### 1. ReAct-style Architecture (Reason + Act)

ResearchAgent follows the [ReAct](https://arxiv.org/abs/2210.03629) pattern: the LLM **reasons** about what to do next, then **acts** by calling a tool, then reasons again about the result. This creates a feedback loop that handles unexpected results gracefully.

In practice this means:
- The Planner produces a *pre-committed* plan (unlike pure ReAct, which plans step-by-step)
- The plan can be overridden by a human at the gate
- Individual tool results feed back into the context for downstream stages

### 2. Structured Output Everywhere

Every stage produces a typed Pydantic model, not a raw string:

- `ResearchPlan` — what to do
- `Evidence` — what was found
- `VerificationResult` — what's trustworthy
- `ResearchAnswer` — the final output

This means every stage is **testable in isolation** and every output is **auditable**.

### 3. Confidence Scores

The Synthesizer always produces a `confidence` float (0–1) alongside `confidence_rationale`. This is not cosmetic — it is derived from:
- The number of independent sources that corroborate the answer
- The quality scores of those sources
- The presence of contradictory evidence

A 0.9 confidence means strong corroboration from high-quality sources. A 0.4 confidence means the answer is speculative and you should treat it accordingly.

### 4. Surfacing Unknowns

The `limitations` field in `ResearchAnswer` is not an error condition. It is a **feature**. A good research agent tells you:
- "I couldn't access paywalled sources"
- "The evidence contradicts itself on this point"
- "This question requires domain expertise I cannot verify"

Hiding these would make the agent less trustworthy, not more.

### 5. Human Gates as a First-Class Feature

Automation is great — until an agent does something expensive or wrong without asking. ResearchAgent's human gate system lets you decide, per-environment, how much autonomy the agent gets:
- `HUMAN_GATE_ENABLED=true` — interactive review at key checkpoints (good for exploration)
- `HUMAN_GATE_ENABLED=false` — fully automated (good for CI/CD and batch jobs)

---

## Project Layout

```
ResearchAgent/
├── pyproject.toml                   # Package metadata and dependencies
├── .env.example                     # Template for your API keys
│
└── research_agent/
    ├── __init__.py                  # Public exports
    ├── models.py                    # All Pydantic data models
    │
    ├── core/
    │   └── context.py               # ResearchContext — shared pipeline state
    │
    ├── llm/
    │   ├── base.py                  # BaseLLM abstract interface
    │   └── openai_llm.py            # OpenAI + Ollama adapter
    │
    └── tools/
        ├── base.py                  # BaseTool + ToolResult
        ├── web_search.py            # WebSearchTool (DDG / Tavily / SerpAPI)
        ├── page_fetcher.py          # PageFetcherTool (trafilatura + BS4)
        └── code_runner.py           # CodeRunnerTool (subprocess sandbox)
```

---

## Next Step

Head to [Chapter 01: Installation & Configuration](./01_installation.md) to get the system running on your machine.
