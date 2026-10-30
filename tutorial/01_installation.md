# Installation & Configuration

> **Chapter 1 of 10** — Get ResearchAgent running on your machine. This chapter covers every step from a fresh environment to your first working command.

---

## Prerequisites

Before installing, make sure you have:

| Requirement | Minimum Version | Check with |
|-------------|----------------|------------|
| Python | 3.10+ | `python --version` |
| pip | 22+ | `pip --version` |
| Git | any | `git --version` (only needed for dev install) |

> **Why Python 3.10?** ResearchAgent uses structural pattern matching, `match`/`case` union types (`X | Y`), and `from __future__ import annotations` features that require 3.10+. Pydantic v2 also performs best on 3.10+.

### Check your Python version

```bash
python --version
# Python 3.11.9 ✓

python3 --version
# If 'python' gives 2.x, use 'python3' throughout this tutorial
```

If you need to install Python, use [pyenv](https://github.com/pyenv/pyenv) (recommended) or the official [python.org installer](https://www.python.org/downloads/).

```bash
# Install pyenv (macOS / Linux)
curl https://pyenv.run | bash

# Install Python 3.11
pyenv install 3.11.9
pyenv global 3.11.9
```

---

## Installation

You have two options:

### Option A: Install from PyPI (recommended for users)

```bash
pip install research-agent
```

This installs the `research-agent` CLI and the `research_agent` Python package.

### Option B: Clone and install in editable mode (recommended for developers)

```bash
# Clone the repository
git clone https://github.com/your-org/ResearchAgent.git
cd ResearchAgent

# Create a virtual environment (strongly recommended)
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
```

Editable mode (`-e`) means changes you make to the source files are immediately reflected — no reinstall needed.

### Verify the installation

```bash
research-agent --version
# research-agent, version 0.1.0

research-agent --help
# Usage: research-agent [OPTIONS] COMMAND [ARGS]...
# ...
```

---

## Configuration: The `.env` File

ResearchAgent is configured through environment variables. The easiest way to set these is with a `.env` file in your project root.

Copy the example file:

```bash
cp .env.example .env
```

Now open `.env` in your editor. Here's what each variable does:

```bash
# ===== LLM Provider (pick one or more) =====

# OpenAI API key — needed if DEFAULT_LLM_PROVIDER=openai
# Get yours at: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-...

# Anthropic API key — needed if DEFAULT_LLM_PROVIDER=anthropic
# Get yours at: https://console.anthropic.com/
ANTHROPIC_API_KEY=sk-ant-...

# For local models via Ollama (no key needed — see Chapter 10):
# OLLAMA_BASE_URL=http://localhost:11434

# ===== Search Provider (pick one) =====

# Option 1: Free, no key needed — DuckDuckGo is the default
# Nothing to set — DuckDuckGo works out of the box.

# Option 2: Tavily — better, more reliable search results
# Get your key at: https://app.tavily.com/
TAVILY_API_KEY=tvly-...

# Option 3: SerpAPI — Google results via API
# Get your key at: https://serpapi.com/manage-api-key
SERPAPI_KEY=...

# ===== Agent Defaults =====

# Which LLM to use: openai | anthropic | ollama
DEFAULT_LLM_PROVIDER=openai

# Which model to use within that provider
# OpenAI options: gpt-4o-mini (cheap, fast), gpt-4o (better quality)
# Anthropic options: claude-3-5-sonnet-20241022
# Ollama options: llama3, mistral, phi3
DEFAULT_LLM_MODEL=gpt-4o-mini

# Which search provider to use: duckduckgo | tavily | serpapi
DEFAULT_SEARCH_PROVIDER=duckduckgo

# How many search results to fetch per query
MAX_SEARCH_RESULTS=5

# How many pages to fetch and read per research session
MAX_PAGES_TO_FETCH=3

# Maximum ReAct loop iterations before giving up
MAX_ITERATIONS=10

# Whether to pause for human review at checkpoints
# Set to false for automated pipelines
HUMAN_GATE_ENABLED=true

# Where to store session data (SQLite database)
# ~ is expanded to your home directory
SESSION_DB_PATH=~/.research_agent/sessions.db
```

### Minimal working configuration

If you just want to get started quickly with OpenAI and DuckDuckGo:

```bash
# .env (minimal)
OPENAI_API_KEY=sk-your-key-here
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_SEARCH_PROVIDER=duckduckgo
HUMAN_GATE_ENABLED=false
```

---

## Getting API Keys

### OpenAI

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Click **"Create new secret key"**
4. Copy the key — you won't see it again after this screen
5. Add billing information (usage-based pricing, ~\$0.15/1M tokens for `gpt-4o-mini`)

> **Cost note:** A typical research query with `gpt-4o-mini` costs roughly \$0.001–\$0.003. You can run hundreds of queries for less than a dollar.

### Anthropic

1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Create an account and navigate to **API Keys**
3. Create a new key and add it to your `.env`

### Tavily (optional but recommended)

1. Go to [app.tavily.com](https://app.tavily.com/)
2. Sign up for a free account (1,000 searches/month free)
3. Copy your API key from the dashboard

---

## Test Your Setup

Run a simple test query to confirm everything works:

```bash
# Disable the human gate for the quick test so it runs non-interactively
HUMAN_GATE_ENABLED=false research-agent research "What is the capital of France?"
```

You should see output similar to:

```
🔍 ResearchAgent v0.1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Query: What is the capital of France?

📋 Planning...
  ✓ Plan created: 2 steps

⚙️  Executing step 1/2: Search for "capital of France"
  ✓ web_search: 5 results in 842ms

⚙️  Executing step 2/2: Fetch top result for details
  ✓ page_fetcher: 1,204 words in 1,203ms

🔬 Verifying evidence...
  ✓ 5 of 5 pieces passed verification

🧠 Synthesizing answer...
  ✓ Answer ready

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Answer: Paris is the capital of France.
Confidence: 97%

Key Findings:
  • Paris has been the capital of France since 987 AD
  • Population of the metropolitan area: ~12 million
  • Located in north-central France on the Seine River

Sources: 3 verified
Limitations: none
```

If you see this, you're ready for [Chapter 02](./02_your_first_query.md).

---

## Using Ollama (Local Models)

If you prefer to run without any cloud API, you can use [Ollama](https://ollama.com/) to run models locally. This is covered in depth in [Chapter 10](./10_extending_with_local_models.md), but here's the quick setup:

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh  # macOS / Linux

# 2. Pull a model
ollama pull llama3

# 3. Configure your .env
# DEFAULT_LLM_PROVIDER=ollama
# DEFAULT_LLM_MODEL=llama3
# OLLAMA_BASE_URL=http://localhost:11434
```

---

## Troubleshooting

### `command not found: research-agent`

Your virtual environment's `bin/` directory isn't on your `PATH`. Make sure you've activated the venv:

```bash
source .venv/bin/activate
research-agent --version  # should work now
```

### `OPENAI_API_KEY is not set`

The `.env` file isn't being loaded. Make sure it's in the directory where you run the command:

```bash
ls -la .env   # should exist
cat .env | grep OPENAI_API_KEY  # should show your key
```

Alternatively, export the variable directly:

```bash
export OPENAI_API_KEY=sk-your-key-here
research-agent research "test query"
```

### `Error: Tavily provider requires TAVILY_API_KEY`

You've set `DEFAULT_SEARCH_PROVIDER=tavily` but haven't set `TAVILY_API_KEY`. Either add the key or switch back to `duckduckgo`:

```bash
DEFAULT_SEARCH_PROVIDER=duckduckgo
```

### `ModuleNotFoundError: No module named 'research_agent'`

The package isn't installed in your active Python environment. Reinstall:

```bash
pip install research-agent
# or
pip install -e .
```

### `RateLimitError` from OpenAI

You've hit OpenAI's rate limits. Wait a minute and try again, or lower `MAX_SEARCH_RESULTS` to reduce the number of LLM calls per query.

### `duckduckgo_search` returning 0 results

DuckDuckGo sometimes rate-limits heavy usage. Wait a few minutes, or switch to Tavily for more reliable results.

---

## Next Step

You have a working installation. Head to [Chapter 02: Your First Research Query](./02_your_first_query.md) to see the full pipeline in action.
