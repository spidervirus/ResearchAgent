# Running with Local Models (Ollama)

> **Chapter 10 of 10** — Run ResearchAgent 100% locally with zero cloud API keys, complete privacy, and zero token costs using Ollama.

---

## Why Local Models for Research?

Running research locally offers major advantages:
- 🔒 **Privacy**: Sensitive proprietary queries, internal IP, and customer questions never leave your machine.
- 💰 **Zero Cost**: No recurring API bills or rate limits when conducting long research sweeps.
- 🔌 **Offline Mode**: Perform local analysis and code execution even on an airplane or air-gapped machine.

---

## 1. Setting Up Ollama

Install Ollama from [ollama.com](https://ollama.com):

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version
```

### Pull a Recommended Model

For structured planning and research synthesis, models with good instruction-following and JSON capabilities work best:

```bash
# Recommended baseline (8B parameters — runs well on 16GB RAM)
ollama pull llama3:latest

# Alternative: Mistral
ollama pull mistral:latest

# Alternative: Qwen2.5 (excellent JSON formatting)
ollama pull qwen2.5:7b
```

Make sure the Ollama server is running:

```bash
ollama serve
# Running at http://localhost:11434
```

---

## 2. Configuring ResearchAgent for Ollama

In your `.env` file or environment variables:

```bash
DEFAULT_LLM_PROVIDER=ollama
DEFAULT_LLM_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_SEARCH_PROVIDER=duckduckgo
```

Notice that with `DEFAULT_SEARCH_PROVIDER=duckduckgo`, **the entire agent stack requires zero paid API keys!**

---

## 3. Running via CLI with Ollama

Pass the `--provider` and `--model` flags directly:

```bash
research-agent research "What are the security implications of using eBPF in Kubernetes?" \
  --provider ollama \
  --model llama3 \
  --search duckduckgo
```

---

## 4. Using the Python API with Ollama

```python
import asyncio
from research_agent import ResearchAgent
from research_agent.llm.openai_llm import OpenAILLM

async def main():
    # Factory method connects to Ollama's OpenAI-compatible endpoint
    local_llm = OpenAILLM.for_ollama(
        model="llama3",
        base_url="http://localhost:11434",
    )

    agent = ResearchAgent(llm=local_llm, human_gate=False)

    answer = await agent.research(
        "Compare RocksDB vs LevelDB for high-write embedded workloads"
    )

    print("=== Local Answer ===")
    print(answer.direct_answer)
    print(f"\nConfidence: {answer.confidence:.0%}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Tips for Getting the Best Results with Local Models

1. **Use 7B/8B+ models**: Smaller 3B models often struggle to produce strict valid JSON plans. `llama3:8b` or `qwen2.5:7b` are ideal sweet spots.
2. **Defensive Parsing Handles Minor Hiccups**: ResearchAgent's built-in fallback parser automatically handles models that wrap JSON in explanatory text.
3. **Hardware Acceleration**: On Apple Silicon (M1/M2/M3/M4) or machines with an NVIDIA GPU, Ollama utilizes Metal or CUDA automatically, delivering 30–70 tokens per second.

---

## Summary: What You Built

Over these 10 chapters, you explored the complete architecture of a reliable, source-backed AI research agent:
1. **Pydantic Data Models**: Strict typing for steps, evidence, and answers.
2. **Planner**: Structured JSON task decomposition.
3. **Tools**: Multi-backend search, streaming HTML extraction, and sandboxed code execution.
4. **Verifier**: Domain reputation heuristics + LLM criticism for filtering noise and detecting contradictions.
5. **Synthesizer**: Calibrated confidence scores, grounded citations, and transparent limitations.
6. **Human Gate**: Interactive approval checkpoints.
7. **Persistence**: SQLite session state for multi-turn iterative research.
8. **Pluggable Architecture**: Bring your own tools, cloud models, or local LLMs.

