"""MCP server exposing the fraud case data as tools.

Any MCP-compatible client (the LangGraph agent here, Claude Desktop, an IDE)
can plug into it. Run standalone with:

    python -m fraudguard.agent.mcp_server          # stdio transport
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from fraudguard.agent.case_store import CaseStore

mcp = FastMCP("fraudguard-cases", log_level="WARNING")
store = CaseStore()


@mcp.tool()
def list_open_alerts(limit: int = 10) -> list[dict]:
    """List the open fraud alerts, highest fraud score first."""
    return store.list_open_alerts(limit)


@mcp.tool()
def get_alert(tx_id: str) -> dict:
    """Full details of one alert: transaction, behavioural signals, expert rules fired
    and the model's top reasons (SHAP contributions in log-odds)."""
    return store.get_alert(tx_id)


@mcp.tool()
def get_customer_profile(customer_id: str, as_of: str) -> dict:
    """Customer's usual behaviour over the 90 days before `as_of` (ISO timestamp):
    amounts, devices, countries, merchant categories, active hours."""
    return store.get_customer_profile(customer_id, as_of)


@mcp.tool()
def get_recent_activity(
    customer_id: str, around: str, hours_before: int = 24, hours_after: int = 6
) -> list[dict]:
    """Customer transactions in a time window around `around` (ISO timestamp).
    `is_alert` marks transactions that also triggered an alert."""
    return store.get_recent_activity(customer_id, around, hours_before, hours_after)


if __name__ == "__main__":
    mcp.run()
