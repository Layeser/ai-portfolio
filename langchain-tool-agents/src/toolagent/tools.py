"""Tools exposed to the agent. Plain functions first (unit-testable), wrapped as LangChain tools below.

The time tool first used two public APIs (REST Countries -> timeapi.io). REST Countries v3.1 was
deprecated and the zone labels it returned ("UTC+01:00") were not accepted by timeapi.io, so the tool
now uses the IANA timezone database shipped with the `tzdata` package: no network, no API key,
real zone names with daylight-saving time.
"""

from __future__ import annotations

import importlib.resources
import unicodedata
from datetime import UTC, datetime
from functools import cache
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

ALIASES = {"usa": "US", "us": "US", "united states": "US", "uk": "GB", "england": "GB", "great britain": "GB",
           "russia": "RU", "south korea": "KR", "north korea": "KP", "vietnam": "VN", "iran": "IR",
           "ivory coast": "CI", "cote d'ivoire": "CI", "czech republic": "CZ", "bolivia": "BO", "syria": "SY",
           "tanzania": "TZ", "venezuela": "VE", "laos": "LA", "taiwan": "TW", "moldova": "MD"}


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower().strip())
    return "".join(c for c in text if not unicodedata.combining(c))


def _read_tab(name: str) -> list[list[str]]:
    content = (importlib.resources.files("tzdata") / "zoneinfo" / name).read_text(encoding="utf-8")
    return [line.split("\t") for line in content.splitlines() if line and not line.startswith("#")]


@cache
def country_index() -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    """(normalised name -> ISO code, ISO code -> display name, ISO code -> IANA zones)."""
    names = {row[0]: row[1] for row in _read_tab("iso3166.tab")}
    by_name = {_normalise(name): code for code, name in names.items()}
    by_name.update({alias: code for alias, code in ALIASES.items()})
    zones: dict[str, list[str]] = {}
    for row in _read_tab("zone1970.tab"):
        for code in row[0].split(","):
            zones.setdefault(code, []).append(row[2])
    return by_name, names, zones


def resolve_country(country: str) -> str | None:
    by_name, names, _ = country_index()
    key = _normalise(country)
    if key.upper() in names:
        return key.upper()
    if key in by_name:
        return by_name[key]
    matches = [code for name, code in by_name.items() if name.startswith(key)]
    return matches[0] if len(set(matches)) == 1 else None


def current_time_in_country(country: str, now: datetime | None = None) -> str:
    if not country or not country.strip():
        return "Please provide a country name."
    code = resolve_country(country)
    if code is None:
        return f"Unknown country: {country}"
    _, names, zones = country_index()
    now = now or datetime.now(UTC)
    country_zones = zones.get(code, [])
    if not country_zones:
        return f"No timezone known for {names[code]}."
    shown = [f"{now.astimezone(ZoneInfo(z)):%Y-%m-%d %H:%M} ({z})" for z in country_zones[:5]]
    more = f" and {len(country_zones) - 5} other zones" if len(country_zones) > 5 else ""
    return f"Current time in {names[code]}: " + "; ".join(shown) + more


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: float, b: float) -> float | str:
    """Divide a by b."""
    return "Error: division by zero." if b == 0 else a / b


@tool
def time_in_country(country: str) -> str:
    """Current local time in a country (English name or ISO code, e.g. 'France', 'Japan', 'US').
    Lists every timezone for countries that have several."""
    return current_time_in_country(country)


ALL_TOOLS = [add, multiply, divide, time_in_country]
