# LangChain Tool-Calling Agents

Small, readable demos of **LLM agents with tools** (OpenRouter / OpenAI-compatible API).

## Demos

| File | What it shows |
|------|----------------|
| `agent.py` | OpenAI-tools agent with math tools (`add`, `multiply`) |
| `meteo.py` | ReAct agent: country → timezone → current weather (public APIs) |
| `test_mat.py` | Tiny pytest example for tool functions |

## Stack

LangChain · LangGraph · OpenAI SDK (via OpenRouter) · requests · python-dotenv

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set API_KEY
python agent.py
# or
python meteo.py
```

## Why this repo

Shows agent patterns recruiters look for: tool schemas, ReAct / tools agent, env-based secrets, external API tools — without a heavyweight framework dump.
