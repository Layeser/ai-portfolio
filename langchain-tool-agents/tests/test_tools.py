import os
from datetime import UTC, datetime

import pytest
from toolagent.tools import add, current_time_in_country, divide, multiply, resolve_country

NOON_UTC_WINTER = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
NOON_UTC_SUMMER = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)


def test_calculator_tools():
    assert add.invoke({"a": 2, "b": 3}) == 5
    assert multiply.invoke({"a": 12.5, "b": 8}) == 100
    assert divide.invoke({"a": 1, "b": 0}) == "Error: division by zero."


@pytest.mark.parametrize("text, code", [("France", "FR"), ("japan", "JP"), ("USA", "US"), ("Côte d'Ivoire", "CI"),
                                        ("GB", "GB"), ("Atlantis", None)])
def test_resolve_country(text, code):
    assert resolve_country(text) == code


def test_daylight_saving_time_is_handled():
    assert "2026-01-15 13:00 (Europe/Paris)" in current_time_in_country("France", NOON_UTC_WINTER)
    assert "2026-07-15 14:00 (Europe/Paris)" in current_time_in_country("France", NOON_UTC_SUMMER)
    assert "21:00 (Asia/Tokyo)" in current_time_in_country("Japan", NOON_UTC_WINTER)


def test_multi_zone_country_and_errors():
    usa = current_time_in_country("United States", NOON_UTC_WINTER)
    assert "07:00 (America/New_York)" in usa and "other zones" in usa
    assert current_time_in_country("Atlantis") == "Unknown country: Atlantis"
    assert current_time_in_country("  ") == "Please provide a country name."


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
def test_agent_uses_both_tools():
    from toolagent.agent import ask, build_agent

    answer, tools_used = ask(build_agent(), "What time is it in Japan, and what is 12.5 multiplied by 8?")
    assert {"multiply", "time_in_country"} <= set(tools_used) and "100" in answer
