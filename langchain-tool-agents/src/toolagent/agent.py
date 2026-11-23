"""Tool-calling agent (LangChain 1.x `create_agent`, a LangGraph ReAct loop under the hood).

    cp .env.example .env      # GROQ_API_KEY
    python -m toolagent.agent           # interactive
    python -m toolagent.agent "What time is it in Japan, and what is 12.5 * 8?"
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent

from toolagent.tools import ALL_TOOLS

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the calculator tools for every arithmetic operation instead of "
    "computing yourself, and the time tool for any question about the current time in a country. "
    "If a tool returns an error, say so plainly."
)


def build_agent(llm=None):
    if llm is None:
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=os.getenv("AGENT_MODEL", "openai/gpt-oss-20b"), temperature=0)
    return create_agent(llm, ALL_TOOLS, system_prompt=SYSTEM_PROMPT)


def ask(agent, question: str) -> tuple[str, list[str]]:
    result = agent.invoke({"messages": [("user", question)]})
    tools_used = [m.name for m in result["messages"] if m.type == "tool"]
    return result["messages"][-1].content, tools_used


def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Set GROQ_API_KEY in .env (see .env.example)")
    agent = build_agent()
    questions = sys.argv[1:] or iter(lambda: input("\nQuestion (empty to quit): ").strip(), "")
    for question in questions:
        answer, tools_used = ask(agent, question)
        print(f"\n{answer}\n  tools: {', '.join(tools_used) or 'none'}")


if __name__ == "__main__":
    main()
