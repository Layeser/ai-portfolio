"""Alert-investigation agent (LangGraph + MCP).

    START -> investigate -> write_report -> apply_policy -> END

* investigate : a tool-calling LLM agent gathers evidence through the MCP
                server (alert details, customer profile, surrounding activity).
* write_report: the transcript is turned into a typed `InvestigationReport`
                (structured output, validated by Pydantic).
* apply_policy: deterministic guardrails. The LLM *recommends*; hard business
                rules decide what needs a human. The agent never blocks a card
                on its own, and every number it quotes is traced back to a tool
                output (untraced values are flagged for the analyst).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

DEFAULT_MODEL = os.getenv("FRAUDGUARD_LLM", "openai/gpt-oss-120b")

INVESTIGATOR_PROMPT = """You are a senior card-fraud analyst investigating one alert.
Use the tools to collect evidence before concluding:
1. get_alert for the transaction, its signals, the expert rules fired and the model's reasons;
2. get_customer_profile (as_of = alert timestamp) to know the customer's normal behaviour;
3. get_recent_activity around the alert timestamp to see what happened before and after.
Compare the alert with the customer's habits: device, country, amount, merchant, hour, velocity.
Remember legitimate explanations exist (travel, new phone, one-off large purchase).
Finish with a short factual analysis. Quote concrete values; never invent data."""

REPORT_PROMPT = """Turn the investigation below into the final report.
Base every evidence item on values that appear in the tool outputs.
verdict: likely_fraud | likely_legitimate | needs_human_review (when evidence is mixed).
recommended_action must be consistent with the verdict."""


class InvestigationReport(BaseModel):
    tx_id: str
    verdict: Literal["likely_fraud", "likely_legitimate", "needs_human_review"]
    confidence: Literal["low", "medium", "high"]
    summary: str = Field(description="2-4 sentences for the analyst")
    evidence: list[str] = Field(description="Concrete facts taken from tool outputs", min_length=1)
    recommended_action: Literal[
        "block_card_and_contact_customer", "contact_customer", "close_alert", "escalate_to_senior_analyst"
    ]


class FinalDecision(BaseModel):
    report: InvestigationReport
    model_score: float
    requires_human_approval: bool
    policy_notes: list[str]
    untraced_values: list[str] = Field(default_factory=list)


NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def untraced_numbers(evidence: list[str], tool_outputs: str) -> list[str]:
    """Numbers quoted in the evidence that appear nowhere in the tool outputs.

    LLMs are unreliable at arithmetic; a derived figure ("3.9x the max") must be
    checked by the analyst, so it is surfaced instead of silently trusted.
    """
    known = {round(float(n), 2) for n in NUMBER_RE.findall(tool_outputs)}
    missing = []
    for item in evidence:
        for raw in NUMBER_RE.findall(item):
            value = round(float(raw.replace(",", ".")), 2)
            if value not in known and raw not in missing:
                missing.append(raw)
    return missing


class State(TypedDict, total=False):
    tx_id: str
    messages: list
    tools_used: list[str]
    alert: dict
    report: InvestigationReport
    decision: FinalDecision


def apply_policy(report: InvestigationReport, alert: dict, block_threshold: float,
                 tool_outputs: str | None = None) -> FinalDecision:
    """Deterministic guardrails on top of the LLM recommendation."""
    notes: list[str] = []
    report = report.model_copy()
    score = float(alert.get("fraud_score", 0.0))

    if report.verdict == "likely_legitimate" and score >= block_threshold:
        notes.append(f"Model score {score:.3f} is above the block threshold: an LLM cannot clear it alone.")
        report.verdict, report.recommended_action = "needs_human_review", "escalate_to_senior_analyst"
    if report.confidence == "low" and report.verdict != "needs_human_review":
        notes.append("Low confidence: routed to a human.")
        report.verdict, report.recommended_action = "needs_human_review", "escalate_to_senior_analyst"

    consistent = {
        "likely_fraud": {"block_card_and_contact_customer", "contact_customer"},
        "likely_legitimate": {"close_alert", "contact_customer"},
        "needs_human_review": {"escalate_to_senior_analyst", "contact_customer"},
    }
    if report.recommended_action not in consistent[report.verdict]:
        notes.append(f"Action '{report.recommended_action}' inconsistent with verdict: escalated.")
        report.recommended_action = "escalate_to_senior_analyst"

    untraced = untraced_numbers(report.evidence, tool_outputs) if tool_outputs is not None else []
    if untraced:
        notes.append(f"{len(untraced)} quoted value(s) not found in tool outputs "
                     "(derived or invented): verify.")

    return FinalDecision(
        report=report,
        model_score=score,
        requires_human_approval=report.recommended_action != "close_alert" or bool(untraced),
        policy_notes=notes,
        untraced_values=untraced,
    )


def _transcript(messages: list) -> str:
    lines = []
    for m in messages:
        if isinstance(m, ToolMessage):
            lines.append(f"[tool {m.name}] {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            lines.append(f"[analyst] {m.content}")
    return "\n".join(lines)


def mcp_client() -> MultiServerMCPClient:
    # The server runs in a subprocess: make sure it imports this same source tree (no install step needed).
    src = str(Path(__file__).resolve().parents[2])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")]))}
    return MultiServerMCPClient(
        {
            "cases": {
                "command": sys.executable,
                "args": ["-m", "fraudguard.agent.mcp_server"],
                "transport": "stdio",
                "env": env,
            }
        }
    )


async def build_graph(llm=None, tools=None, block_threshold: float | None = None):
    """Compile the graph. `llm` / `tools` can be injected (tests use fakes)."""
    from fraudguard.agent.case_store import CaseStore

    llm = llm or ChatGroq(model=DEFAULT_MODEL, temperature=0, max_retries=4, timeout=60)
    tools = tools if tools is not None else await mcp_client().get_tools()
    if block_threshold is None:
        from fraudguard.config import ARTIFACTS_DIR
        from fraudguard.model import META_FILE

        block_threshold = json.loads((ARTIFACTS_DIR / META_FILE).read_text())["thresholds"]["block"]
    investigator = create_agent(llm, tools, system_prompt=INVESTIGATOR_PROMPT)
    reporter = llm.with_structured_output(InvestigationReport)
    store = CaseStore()

    async def investigate(state: State) -> State:
        result = await investigator.ainvoke(
            {"messages": [HumanMessage(f"Investigate fraud alert {state['tx_id']}.")]},
            {"recursion_limit": 20},
        )
        used = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]
        return {"messages": result["messages"], "tools_used": used}

    async def write_report(state: State) -> State:
        report = await reporter.ainvoke(
            [SystemMessage(REPORT_PROMPT),
             HumanMessage(f"Alert {state['tx_id']}\n\n{_transcript(state['messages'])}")]
        )
        report.tx_id = state["tx_id"]
        return {"report": report}

    async def policy(state: State) -> State:
        alert = store.get_alert(state["tx_id"])  # ground the score on the system of record, not the LLM
        tool_outputs = "\n".join(str(m.content) for m in state["messages"] if isinstance(m, ToolMessage))
        decision = apply_policy(state["report"], alert, block_threshold, tool_outputs)
        return {"alert": alert, "decision": decision}

    graph = StateGraph(State)
    graph.add_node("investigate", investigate)
    graph.add_node("write_report", write_report)
    graph.add_node("apply_policy", policy)
    graph.add_edge(START, "investigate")
    graph.add_edge("investigate", "write_report")
    graph.add_edge("write_report", "apply_policy")
    graph.add_edge("apply_policy", END)
    return graph.compile()
