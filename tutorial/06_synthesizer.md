# The Synthesizer: From Evidence to Structured Answers

> **Chapter 6 of 10** — Transforming evidence into a decisive answer, generating grounded citations, calibrating confidence scores, and surfacing known limitations.

---

## What Makes an Answer "Structured"?

Most LLM outputs are free-form markdown paragraphs. The problem with free-form text is that:
1. You can't easily parse out the direct answer programmatically.
2. Citations are often vague ("studies show...") or hallucinated.
3. The model hides its uncertainty behind confident-sounding prose.

**ResearchAgent** replaces free-form generation with a strictly modeled `ResearchAnswer`:

```python
class ResearchAnswer(BaseModel):
    query: str
    direct_answer: str                        # Concise, decisive stance
    key_findings: list[str]                   # Bulleted findings with [Markdown](URL) citations
    evidence: list[Evidence]                  # Complete audit trail of verified sources
    confidence: float                         # 0.0 to 1.0 calibrated score
    confidence_rationale: str                 # Exact explanation of confidence
    limitations: list[Limitation]             # Caveats with severity (LOW, MEDIUM, HIGH)
    follow_up_questions: list[FollowUpQuestion] # Actionable next verification steps
    plan: ResearchPlan                        # The plan that produced this answer
```

---

## 1. Grounded Citations

The Synthesizer requires every key finding to cite a specific source URL extracted during tool execution:

```
"key_findings": [
  "Qdrant maintains under 10ms latency at 10k QPS on modest CPU cores [Qdrant Benchmarks](https://qdrant.tech/benchmarks)",
  "Milvus separates query, index, and data nodes, introducing higher operational overhead for small teams [Milvus Architecture](https://milvus.io/docs/architecture)"
]
```

Because the Synthesizer only sees URLs that were genuinely fetched by the Executor, citation hallucination is virtually eliminated.

---

## 2. Calibrating the Confidence Score

The confidence score is not an arbitrary number; the synthesizer is instructed to calibrate it based on three pillars:

| Confidence Range | Meaning | Typical Conditions |
|------------------|---------|--------------------|
| **0.85 – 1.00** | High Confidence | Multiple independent, authoritative sources agree; empirical data directly answers the query. |
| **0.65 – 0.84** | Moderate Confidence | Good consensus, but depends on unstated assumptions (e.g. vector dimensionality) or lacks fresh benchmarks. |
| **0.40 – 0.64** | Low Confidence | Conflicting sources, limited empirical evidence, or rapidly changing topic. |
| **< 0.40** | Speculative | Very little reliable source data retrieved; answer is based on extrapolation. |

---

## 3. Surfacing Limitations as a First-Class Feature

In professional research, knowing **what could be wrong or missing** is just as important as the answer itself.

Each `Limitation` includes a severity rating:
- `HIGH`: A factor that could invalidate the recommendation entirely (e.g., "The pricing model changed last month, which may double estimated cloud costs").
- `MEDIUM`: A variable that requires situational tuning (e.g., "Benchmark numbers assume 768-dimension vectors; 1536-dim embeddings will halve QPS").
- `LOW`: Minor caveats or theoretical considerations.

In the next chapter, we look at **Chapter 7: Human-in-the-Loop** to see how human approval gates can be placed at high-stakes checkpoints.

