# LangChain tool-calling agent

A small, readable tool-calling agent built with LangChain 1.x `create_agent` (a LangGraph ReAct loop):
calculator tools and a "current time in a country" tool, served by an open-weight model on Groq.

```
$ python -m toolagent.agent "What time is it in Japan right now, and what is 12.5 times 8?"

Current time in Japan: 2026-11-22 09:49 (Asia/Tokyo)
12.5 × 8 = 100.0
  tools: time_in_country, multiply
```

## What changed in the 2026 rewrite

* **Migrated to LangChain 1.x.** The original code used `AgentExecutor` / `create_openai_tools_agent`,
  which no longer exist, so it no longer ran.
* **Removed two fragile external APIs.** The time tool chained REST Countries and timeapi.io. REST Countries
  v3.1 has since been deprecated, and its zone labels ("UTC+01:00") were never accepted by timeapi.io anyway.
  The tool now uses the IANA timezone database (`tzdata`): no network, real zone names, and daylight-saving
  time handled.
* **Real tests.** The old test file imported a missing module and only tested Python's `sum`. The tool logic
  is now plain functions with tests for DST, multi-zone countries, aliases and error cases. One optional
  test runs the agent end to end and checks that it actually called the tools.

## Run

```bash
pip install -r requirements.txt && export PYTHONPATH=src
cp .env.example .env          # GROQ_API_KEY
make run                      # interactive
make test                     # 9 offline tests (+1 with GROQ_API_KEY)
```

## Files

| File | Role |
|---|---|
| `src/toolagent/tools.py` | calculator + timezone tools (plain functions wrapped with `@tool`) |
| `src/toolagent/agent.py` | agent construction, `ask()` helper returning the answer and the tools used, CLI |
| `tests/test_tools.py` | unit tests + optional end-to-end agent test |

For a production-grade agent (MCP tool server, structured output, deterministic guardrails, evaluation
against ground truth), see [FraudGuard](../fraud-detection-platform/).
