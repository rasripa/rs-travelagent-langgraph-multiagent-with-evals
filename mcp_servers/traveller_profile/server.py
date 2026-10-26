"""Traveller Profile MCP server -- the SHARED server.

Both sub-agents depend on this one, for different reasons:
  Journey Planner  -> railcard drives which fares are priced
  Journey Support  -> railcard and loyalty tier drive refund and compensation
                      entitlement

It is also the only server in the demo that returns personal data, which is
what makes the access-control story concrete: it is reachable by both
principals, while Journey Search and Ticket Operations are reachable by exactly
one each.

Gateway tool names are prefixed with the target name, so these surface to the
agents as TravellerProfile___get_traveller_profile and so on.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import fixtures as fx  # noqa: E402
from common.mcp_runtime import build_server, serve  # noqa: E402

mcp = build_server(
    "traveller_profile",
    instructions=(
        "Traveller account data for a rail booking platform: identity, seat and "
        "accessibility preferences, railcard, loyalty tier and saved journeys. "
        "Call this before pricing fares or assessing a refund, because the "
        "railcard and loyalty tier change both answers."
    ),
)


@mcp.tool()
def get_traveller_profile(traveller_id: str) -> dict[str, Any]:
    """Return the full profile for a traveller.

    Includes seat preferences and accessibility requirements, which must be
    honoured when booking or rebooking.

    Args:
        traveller_id: Traveller identifier, for example TRV-1001.
    """
    profile = fx.travellers().get(traveller_id.strip().upper())
    if not profile:
        return fx.mcp_error(
            "TRAVELLER_NOT_FOUND",
            f"No traveller with id {traveller_id}.",
            known_ids=sorted(fx.travellers().keys()),
        )

    home = fx.stations().get(profile["home_station"], {})
    return fx.mcp_ok(
        traveller={
            "traveller_id": profile["traveller_id"],
            "name": profile["name"],
            "email": profile["email"],
            "home_station": {
                "crs": profile["home_station"],
                "name": home.get("name", profile["home_station"]),
            },
            "seat_preferences": profile["seat_preferences"],
            "accessibility": profile["accessibility"],
            "railcard": profile["railcard"],
            "loyalty": profile["loyalty"],
        }
    )


@mcp.tool()
def get_railcard_and_loyalty(traveller_id: str) -> dict[str, Any]:
    """Return only the railcard and loyalty tier for a traveller.

    Narrower than get_traveller_profile: use this when all you need is the
    discount entitlement or the loyalty tier, and not the traveller's contact
    details or accessibility requirements.

    Args:
        traveller_id: Traveller identifier, for example TRV-1001.
    """
    profile = fx.travellers().get(traveller_id.strip().upper())
    if not profile:
        return fx.mcp_error(
            "TRAVELLER_NOT_FOUND",
            f"No traveller with id {traveller_id}.",
            known_ids=sorted(fx.travellers().keys()),
        )

    railcard = profile["railcard"]
    return fx.mcp_ok(
        traveller_id=profile["traveller_id"],
        has_railcard=railcard is not None,
        railcard=railcard,
        loyalty=profile["loyalty"],
        note=(
            "No railcard held; quote undiscounted fares."
            if railcard is None
            else f"Pass railcard code {railcard['code']} to get_fares to price the discount."
        ),
    )


@mcp.tool()
def list_saved_journeys(traveller_id: str) -> dict[str, Any]:
    """List the journeys a traveller has saved or travels regularly.

    Useful for resolving a vague request such as "my usual trip to Manchester"
    into a concrete origin and destination.

    Args:
        traveller_id: Traveller identifier, for example TRV-1001.
    """
    key = traveller_id.strip().upper()
    if key not in fx.travellers():
        return fx.mcp_error(
            "TRAVELLER_NOT_FOUND",
            f"No traveller with id {traveller_id}.",
            known_ids=sorted(fx.travellers().keys()),
        )

    saved = fx.saved_journeys().get(key, [])
    enriched = [
        {
            **journey,
            "origin_name": fx.stations().get(journey["origin"], {}).get("name"),
            "destination_name": fx.stations().get(journey["destination"], {}).get("name"),
        }
        for journey in saved
    ]
    return fx.mcp_ok(traveller_id=key, count=len(enriched), saved_journeys=enriched)


if __name__ == "__main__":
    serve(mcp)
