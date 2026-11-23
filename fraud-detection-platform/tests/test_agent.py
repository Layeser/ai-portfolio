import asyncio
import json
import os

import pytest
from fraudguard.agent.case_store import CaseStore
from fraudguard.agent.graph import InvestigationReport, apply_policy, untraced_numbers


def _report(**kw):
    base = {"tx_id": "T1", "verdict": "likely_legitimate", "confidence": "high", "summary": "s",
            "evidence": ["e"], "recommended_action": "close_alert"}
    return InvestigationReport(**{**base, **kw})


def test_policy_llm_cannot_clear_a_high_score_alert():
    d = apply_policy(_report(), {"fraud_score": 0.97}, block_threshold=0.5)
    assert d.report.verdict == "needs_human_review"
    assert d.requires_human_approval and d.policy_notes


def test_policy_low_confidence_goes_to_human():
    d = apply_policy(_report(confidence="low"), {"fraud_score": 0.1}, block_threshold=0.5)
    assert d.report.recommended_action == "escalate_to_senior_analyst"


def test_policy_fixes_inconsistent_action_and_keeps_human_in_loop():
    d = apply_policy(_report(verdict="likely_fraud", recommended_action="close_alert"),
                     {"fraud_score": 0.9}, block_threshold=0.5)
    assert d.report.recommended_action == "escalate_to_senior_analyst"
    ok = apply_policy(_report(), {"fraud_score": 0.1}, block_threshold=0.5)
    assert ok.report.verdict == "likely_legitimate" and not ok.requires_human_approval


def test_untraced_numbers_flags_derived_or_invented_values():
    tools = '{"amount": 77.57, "max_amount": 27.97, "tx_count_1h": 3.0}'
    evidence = ["Amount 77.57 is 3.9x the max 27.97", "3 transactions in the last hour"]
    assert untraced_numbers(evidence, tools) == ["3.9"]
    d = apply_policy(_report(evidence=evidence), {"fraud_score": 0.1}, 0.5, tool_outputs=tools)
    assert d.untraced_values == ["3.9"] and d.requires_human_approval


def test_case_store_hides_ground_truth(tiny_project):
    store = CaseStore(tiny_project["artifacts"], tiny_project["data"])
    alert = store.list_open_alerts(1)[0]
    details = store.get_alert(alert["tx_id"])
    profile = store.get_customer_profile(alert["customer_id"], alert["timestamp"])
    activity = store.get_recent_activity(alert["customer_id"], alert["timestamp"])
    dumped = json.dumps([details, profile, activity])
    assert "is_fraud" not in dumped and "fraud_scenario" not in dumped
    assert details["rule_hits"] is not None and profile["tx_last_90d"] >= 0
    assert store.get_alert("nope") == {"error": "unknown alert nope"}


def test_mcp_server_exposes_tools(tiny_project, monkeypatch):
    """Spawns the real MCP server over stdio and calls a tool through the protocol."""
    monkeypatch.setenv("FRAUDGUARD_ARTIFACTS_DIR", str(tiny_project["artifacts"]))
    monkeypatch.setenv("FRAUDGUARD_DATA_DIR", str(tiny_project["data"]))
    from fraudguard.agent.graph import mcp_client

    async def run():
        tools = {t.name: t for t in await mcp_client().get_tools()}
        alerts = await tools["list_open_alerts"].ainvoke({"limit": 2})
        return tools, alerts

    tools, alerts = asyncio.run(run())
    assert {"list_open_alerts", "get_alert", "get_customer_profile", "get_recent_activity"} <= set(tools)
    assert "tx_id" in str(alerts)


@pytest.mark.llm
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
def test_agent_end_to_end_with_real_llm(tiny_project, monkeypatch):
    monkeypatch.setenv("FRAUDGUARD_ARTIFACTS_DIR", str(tiny_project["artifacts"]))
    monkeypatch.setenv("FRAUDGUARD_DATA_DIR", str(tiny_project["data"]))
    from fraudguard.agent.graph import build_graph

    tx_id = CaseStore(tiny_project["artifacts"], tiny_project["data"]).list_open_alerts(1)[0]["tx_id"]

    async def run():
        graph = await build_graph(block_threshold=0.5)
        return await graph.ainvoke({"tx_id": tx_id})

    state = asyncio.run(run())
    assert state["decision"].report.tx_id == tx_id
    assert "get_alert" in state["tools_used"]
