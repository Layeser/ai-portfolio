"""Minimal OpenAI-tools agent with math tools (OpenRouter-compatible)."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


tools = [multiply, add]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can use tools."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def main() -> None:
    if not os.getenv("API_KEY"):
        raise SystemExit("Set API_KEY in .env (see .env.example)")
    while True:
        question = input("\nQuestion (empty to quit): ").strip()
        if not question:
            break
        result = agent_executor.invoke({"input": question})
        print("\nFinal:", result["output"])


if __name__ == "__main__":
    main()
