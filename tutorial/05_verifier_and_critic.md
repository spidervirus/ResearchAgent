# The Verifier: Filtering Noise from Real Evidence

> **Chapter 5 of 10** — Why verification is essential for reliable AI research, how domain reputation heuristics work, and how the LLM critic checks for contradictions.

---

## The Verification Problem

A standard web search query returns a mix of authoritative papers, marketing blog posts, outdated documentation, and SEO spam. If you feed all of this directly into the final synthesizer, you get:
- Contradictory claims presented as facts.
- Outdated metrics (e.g., benchmark numbers from 2021).
- Unreliable claims from low-credibility personal blogs or forum threads.

**ResearchAgent introduces Stage 3: The Verifier.**

```
 Raw Evidence Set (from Tools)
               │
               ▼
 [Domain Reputation & Quality Heuristics]
    • Boost academic / official docs (+0.25)
    • Penalize forums / low-signal sites (-0.20)
    • Check excerpt length & density
               │
               ▼
 [LLM Semantic Critique]
    • Evaluate relevance to exact question (0.0 - 1.0)
    • Detect contradictions between sources
    • Identify critical coverage gaps
               │
               ▼
 [Threshold Filter] (min_relevance_score >= 0.35)
               │
               ▼
 Verified, Ranked Evidence (Passed to Synthesizer)
```

---

## 1. Domain Reputation Heuristics

`research_agent/core/verifier.py` applies deterministic heuristics to every source URL before running semantic analysis:

```python
_REPUTATION_BOOSTS = {
    "arxiv.org": 0.25,        # Peer-reviewed academic benchmarks
    "github.com": 0.15,       # Source code and official release notes
    "docs.": 0.15,            # Official software documentation
    "readthedocs.io": 0.15,
    ".edu": 0.20,             # Educational / research institutions
    ".gov": 0.20,
    "nature.com": 0.25,
    "ieee.org": 0.25,
    "acm.org": 0.25,
    "wikipedia.org": 0.10,
}

_REPUTATION_PENALTIES = {
    "reddit.com": -0.15,      # Informal opinion / anecdotes
    "quora.com": -0.20,
    "medium.com": -0.05,      # Variable quality blog posts
    "pinterest.com": -0.30,
}
```

If an excerpt has fewer than 50 characters, it is also penalized to avoid shallow snippets that lack context.

---

## 2. LLM Semantic Critique

Next, the Verifier formats the top candidate excerpts and sends them to the LLM with an analytical rubric:

1. **Relevance**: Does this source actually answer the question asked, or is it discussing a related but irrelevant topic?
2. **Quality & Depth**: Does it contain specific numbers, architectural explanations, or methodology?
3. **Cross-Source Contradictions**: Do two sources disagree? If Source A claims Qdrant requires 32GB RAM while Source B claims 8GB RAM, the verifier flags this discrepancy into `ctx.metadata["verifier_issues"]`.

---

## 3. Threshold Filtering & Composite Scoring

Each evidence item has two scores:
- `relevance_score` (0.0 to 1.0)
- `quality_score` (0.0 to 1.0)

They are combined into a weighted **composite score**:
$$\text{composite\_score} = 0.6 \times \text{relevance} + 0.4 \times \text{quality}$$

Evidence with a relevance score below `min_relevance_score` (default 0.3) is filtered out. The remaining evidence is sorted descending by composite score, ensuring the Synthesizer sees the most authoritative sources first.

In the next chapter, we cover **Chapter 6: The Synthesizer** to see how structured answers, confidence scores, and limitations are generated.

