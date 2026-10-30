# Your First Research Query

> **Chapter 2 of 10** — Run your very first research query using both the CLI and Python API, understand every section of the structured output, and learn how confidence scores work.

---

## 1. Running via the Command-Line Interface (CLI)

Once you've configured your `.env` file with an LLM key (or Ollama), let's run our first research question:

```bash
research-agent research "What is Retrieval-Augmented Generation (RAG) and when should a startup choose it over fine-tuning?"
```

### What Happens Step-by-Step

When you press Enter, ResearchAgent activates its 4-stage pipeline:

```
[1/4] 🧠 Planner     Decomposes query into web search & targeted documentation queries
[2/4] ⚡ Executor    Invokes DuckDuckGo/Tavily, fetches candidate URLs, extracts clean body text
[3/4] 🔍 Verifier    Evaluates source credibility, penalizes low-quality domains, filters noise
[4/4] ✍️ Synthesizer Generates direct recommendation, citations, confidence score & caveats
```

### Understanding the Output

In your terminal, the output renders using Rich panels and tables:

```
╭────────────────────────────── 🎯 Direct Answer ──────────────────────────────╮
│ Startups should choose RAG when knowledge changes frequently, data privacy   │
│ is required without model retraining, or budget is limited. Fine-tuning      │
│ should only be chosen when changing the model's tone, syntax, or reasoning  │
│ format rather than teaching it new factual data.                             │
╰──────────────────────────────────────────────────────────────────────────────╯
  Confidence: [████████░░] 85%
  Rationale: Strong consensus across official documentation from OpenAI, Anthropic, and academic surveys.

📌 Key Findings & Sourced Citations
┌───┬──────────────────────────────────────────────────────────────────────────┐
│ # │ Finding                                                                  │
├───┼──────────────────────────────────────────────────────────────────────────┤
│ 1 │ RAG introduces dynamic context at inference time without retraining costs│
│   │ [OpenAI RAG Guide](https://platform.openai.com/docs/guides/rag)         │
│ 2 │ Fine-tuning costs 10-50x more for knowledge updates and risks hallucinated│
│   │ facts if not grounded [ArXiv: Retrieval Survey](https://arxiv.org/abs/..)│
└───┴──────────────────────────────────────────────────────────────────────────┘

⚠️ Known Limitations & Assumptions
┌──────────┬───────────────────────────────────────────────────────────────────┐
│ Severity │ Limitation                                                        │
├──────────┼───────────────────────────────────────────────────────────────────┤
│ MEDIUM   │ Retrieval latency adds 100-300ms overhead to user-facing queries   │
│ LOW      │ Embeddings pricing changes across providers over time             │
└──────────┴───────────────────────────────────────────────────────────────────┘

💡 Suggested Next Questions / Verification
┌───────────────────────────────────────┬──────────────────────────────────────┐
│ Follow-Up Question                    │ Why It Matters                       │
├───────────────────────────────────────┼──────────────────────────────────────┤
│ What is the cost difference between   │ Vector database hosting is often the │
│ self-hosted vs managed vector DBs?    │ dominant ongoing cost in RAG.        │
└───────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 2. Using the Python API

For embedding into applications or automated pipelines, use the asynchronous Python API:

```python
import asyncio
from research_agent import ResearchAgent

async def main():
    # Instantiate the agent
    agent = ResearchAgent(
        llm_provider="openai",        # "openai", "anthropic", or "ollama"
        llm_model="gpt-4o-mini",      # or "claude-3-5-sonnet-20241022", "llama3"
        human_gate=False,             # Disable CLI approval prompts for scripts
    )

    query = "Compare the top 3 open-source vector databases for 10k QPS"
    answer = await agent.research(query)

    print("=== Direct Recommendation ===")
    print(answer.direct_answer)

    print(f"\nConfidence: {answer.confidence:.0%}")
    print(f"Rationale: {answer.confidence_rationale}")

    print("\n=== Key Citations ===")
    for ev in answer.high_quality_evidence:
        print(f"- {ev.source_title}: {ev.source_url} (Score: {ev.composite_score:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 3. Emitting JSON and Markdown

If you want to feed ResearchAgent output into other systems or save reports:

### JSON Output
```bash
research-agent research "Explain vector quantization" --output json > report.json
```

### Save as Markdown File
```bash
research-agent research "Compare Qdrant vs Milvus" --save-md qdrant_vs_milvus.md
```

In the next chapter, we dive under the hood of **Chapter 3: Deep Dive: The Planner** to see how natural language gets converted into executable tool steps.

