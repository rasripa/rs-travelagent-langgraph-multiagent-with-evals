"""Fixture access for the mock MCP servers.

Every MCP server reads from the same JSON fixtures so the agents can never be
shown contradicting facts. Loads are cached per process; mutations made by
write tools (book_ticket, change_or_cancel_ticket, claim_delay_repay) are held
in a per-process overlay so that restarting a server resets the demo to a known
baseline.
"""
from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from typing import Any

from common.config import FIXTURES_DIR, REFERENCE_DATE  # noqa: F401  (re-export)


# ---------------------------------------------------------------- loading
@lru_cache(maxsize=None)  # noqa: UP033 - cache() needs 3.9+, keep explicit
def _load(name: str) -> dict[str, Any]:
    with open(FIXTURES_DIR / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)


def travellers() -> dict[str, Any]:
    return _load("travellers")["travellers"]


def saved_journeys() -> dict[str, Any]:
    return _load("travellers")["saved_journeys"]


def stations() -> dict[str, Any]:
    return _load("stations")["stations"]


def station_aliases() -> dict[str, str]:
    return _load("stations")["aliases"]


def services() -> list[dict[str, Any]]:
    return _load("services")["services"]


def network_disruption() -> list[dict[str, Any]]:
    return _load("services")["network_disruption"]


def fares() -> dict[str, Any]:
    return _load("fares")["fares"]


def fare_rules() -> dict[str, Any]:
    return _load("fares")["fare_rules_summary"]


def bookings_baseline() -> dict[str, Any]:
    return _load("bookings")["bookings"]


def delay_repay_bands() -> list[dict[str, Any]]:
    return _load("bookings")["delay_repay_bands"]


def claim_window_days() -> int:
    return _load("bookings")["claim_window_days"]


# ---------------------------------------------------------------- helpers
def resolve_station(value: str) -> str | None:
    """Map a code or a human name ('London Euston', 'euston') to a CRS code."""
    if not value:
        return None
    probe = value.strip()
    if probe.upper() in stations():
        return probe.upper()
    aliases = {k.lower(): v for k, v in station_aliases().items()}
    return aliases.get(probe.lower())


def money(value: float | Decimal) -> float:
    """Round half-up to 2dp. Bankers rounding would break the fixture maths."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    )


def apply_discount(price_gbp: float, discount_percent: float | None) -> float:
    if not discount_percent:
        return money(price_gbp)
    return money(Decimal(str(price_gbp)) * (1 - Decimal(str(discount_percent)) / 100))


def find_service(service_id: str) -> dict[str, Any] | None:
    return next((s for s in services() if s["service_id"] == service_id), None)


def mcp_error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    """Uniform error envelope so agents get a predictable shape on failure."""
    return {"ok": False, "error": {"code": code, "message": message, **extra}}


def mcp_ok(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def railcards() -> dict[str, Any]:
    """Public railcard reference data (discount percent, restrictions).

    Deliberately NOT traveller PII: which railcard a person holds comes from the
    Traveller Profile server; what that railcard is worth comes from here.
    """
    data = dict(_load("fares")["railcards"])
    data.pop("_comment", None)
    return data
