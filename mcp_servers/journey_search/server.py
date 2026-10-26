"""Journey Search MCP server -- reachable by the Journey Planner ONLY.

Pre-travel capability: what services run, what they cost, and committing a
booking. The Journey Support agent is denied this target by Cedar policy, which
is why Support has to delegate to the Planner over A2A when it needs an
alternative service during disruption rather than searching for one itself.

Gateway tool names surface as JourneySearch___search_journeys and so on.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import fixtures as fx  # noqa: E402
from common.config import REFERENCE_DATE  # noqa: E402
from common.mcp_runtime import build_server, serve  # noqa: E402

mcp = build_server(
    "journey_search",
    instructions=(
        "Rail timetable, fare and booking capability. Search services between "
        "two stations on a date, price the available fare types with any "
        "railcard discount, and commit a booking. Disruption status is returned "
        "with every service."
    ),
)

# Bookings created during this process. Restarting the server resets the demo.
_NEW_BOOKINGS: dict[str, Any] = {}
_NEXT_REF = [77401]


def _route_key(origin: str, destination: str) -> str:
    return f"{origin}-{destination}"


def _disruption_for(origin: str, destination: str, date: str) -> list[dict[str, Any]]:
    route = _route_key(origin, destination)
    return [
        d
        for d in fx.network_disruption()
        if d["date"] == date and d["affected_route"] == route
    ]


@mcp.tool()
def search_journeys(
    origin: str,
    destination: str,
    date: str | None = None,
    include_cancelled: bool = True,
) -> dict[str, Any]:
    """Search rail services between two stations on a date.

    Station names or CRS codes are both accepted ("Manchester" or "MAN").
    Each result carries its live status, so always check `status` before
    recommending a service: a cancelled service must never be offered, and a
    traveller needing a wheelchair space must be offered one where
    `wheelchair_spaces_available` is greater than zero.

    Args:
        origin: Origin station name or CRS code.
        destination: Destination station name or CRS code.
        date: Travel date as YYYY-MM-DD. Defaults to the current booking date.
        include_cancelled: Keep cancelled services in the result, flagged as
            such, so the traveller can be told their service is cancelled.
    """
    o = fx.resolve_station(origin)
    d = fx.resolve_station(destination)
    if not o or not d:
        return fx.mcp_error(
            "STATION_NOT_FOUND",
            f"Could not resolve origin={origin!r} destination={destination!r}.",
            known_stations={k: v["name"] for k, v in fx.stations().items()},
        )
    if o == d:
        return fx.mcp_error(
            "SAME_STATION", "Origin and destination are the same station."
        )

    travel_date = (date or REFERENCE_DATE).strip()
    matches = [
        s
        for s in fx.services()
        if s["origin"] == o and s["destination"] == d and s["date"] == travel_date
    ]
    if not include_cancelled:
        matches = [s for s in matches if s["status"] != "cancelled"]
    matches.sort(key=lambda s: s["departure"])

    results = [
        {
            "service_id": s["service_id"],
            "operator": s["operator"],
            "departure": s["departure"],
            "arrival": s["arrival"],
            "duration_minutes": s["duration_minutes"],
            "changes": s["changes"],
            "status": s["status"],
            "delay_minutes": s["delay_minutes"],
            "disruption_reason": s["disruption_reason"],
            "seats_available": s["seats_available"],
            "wheelchair_spaces_available": s["wheelchair_spaces_available"],
            "replacement_advice": s.get("replacement_advice"),
        }
        for s in matches
    ]

    return fx.mcp_ok(
        origin={"crs": o, "name": fx.stations()[o]["name"]},
        destination={"crs": d, "name": fx.stations()[d]["name"]},
        date=travel_date,
        count=len(results),
        services=results,
        network_disruption=_disruption_for(o, d, travel_date),
        note=(
            "No services found for that route and date."
            if not results
            else "Check `status` on each service before recommending it."
        ),
    )


@mcp.tool()
def get_fares(
    origin: str,
    destination: str,
    railcard_code: str | None = None,
    travel_class: str | None = None,
) -> dict[str, Any]:
    """Price the fare types available on a route, with any railcard discount.

    The railcard code comes from the Traveller Profile server; this server does
    not hold traveller data. Pass it to get discounted prices, and note that
    railcard time restrictions apply to some fare types only -- the returned
    `railcard.restricted_fare_types` says which.

    Args:
        origin: Origin station name or CRS code.
        destination: Destination station name or CRS code.
        railcard_code: Railcard code such as TTG or YNG. Omit for no discount.
        travel_class: Filter to "Standard" or "First". Omit for all classes.
    """
    o = fx.resolve_station(origin)
    d = fx.resolve_station(destination)
    if not o or not d:
        return fx.mcp_error(
            "STATION_NOT_FOUND",
            f"Could not resolve origin={origin!r} destination={destination!r}.",
        )

    route_fares = fx.fares().get(_route_key(o, d))
    if not route_fares:
        return fx.mcp_error(
            "NO_FARES_FOR_ROUTE",
            f"No published fares for {o} to {d}.",
            available_routes=sorted(fx.fares().keys()),
        )

    railcard = None
    if railcard_code:
        railcard = fx.railcards().get(railcard_code.strip().upper())
        if not railcard:
            return fx.mcp_error(
                "RAILCARD_NOT_FOUND",
                f"Unknown railcard code {railcard_code!r}.",
                known_codes=sorted(fx.railcards().keys()),
            )

    discount = railcard["discount_percent"] if railcard else None
    priced = []
    for fare in route_fares:
        if travel_class and fare["travel_class"].lower() != travel_class.strip().lower():
            continue
        full = fx.money(fare["price_gbp"])
        payable = fx.apply_discount(full, discount)
        rules = fx.fare_rules()[fare["fare_type"]]
        priced.append(
            {
                "fare_type": fare["fare_type"],
                "travel_class": fare["travel_class"],
                "full_price_gbp": full,
                "price_gbp": payable,
                "discount_applied_gbp": fx.money(full - payable),
                "refundable": rules["refundable"],
                "changeable": rules["changeable"],
                "change_fee_gbp": rules["change_fee_gbp"],
                "train_specific": rules["train_specific"],
                "conditions": fare["conditions"],
                "railcard_time_restricted": bool(
                    railcard and fare["fare_type"] in railcard["restricted_fare_types"]
                ),
            }
        )

    priced.sort(key=lambda f: f["price_gbp"])
    return fx.mcp_ok(
        route=f"{o}-{d}",
        railcard=railcard,
        count=len(priced),
        fares=priced,
    )


@mcp.tool()
def book_ticket(
    traveller_id: str,
    service_id: str,
    fare_type: str,
    travel_class: str = "Standard",
    railcard_code: str | None = None,
) -> dict[str, Any]:
    """Book a ticket on a specific service and return the booking reference.

    Refuses to book a cancelled service. Advance fares are tied to the booked
    service and are neither changeable nor refundable, so confirm the traveller
    accepts that before calling this with fare_type="Advance".

    Args:
        traveller_id: Traveller identifier, for example TRV-1001.
        service_id: Service to book, from search_journeys.
        fare_type: "Advance", "Off-Peak" or "Anytime".
        travel_class: "Standard" or "First".
        railcard_code: Railcard code to apply, for example TTG.
    """
    service = fx.find_service(service_id.strip())
    if not service:
        return fx.mcp_error("SERVICE_NOT_FOUND", f"No service {service_id}.")
    if service["status"] == "cancelled":
        return fx.mcp_error(
            "SERVICE_CANCELLED",
            f"Service {service_id} is cancelled and cannot be booked.",
            disruption_reason=service["disruption_reason"],
            replacement_advice=service.get("replacement_advice"),
        )

    fare_type = fare_type.strip().title().replace("Off-peak", "Off-Peak")
    if fare_type not in fx.fare_rules():
        return fx.mcp_error(
            "INVALID_FARE_TYPE",
            f"Unknown fare type {fare_type!r}.",
            valid=sorted(fx.fare_rules().keys()),
        )

    route = _route_key(service["origin"], service["destination"])
    candidates = [
        f
        for f in fx.fares().get(route, [])
        if f["fare_type"] == fare_type
        and f["travel_class"].lower() == travel_class.strip().lower()
    ]
    if not candidates:
        return fx.mcp_error(
            "FARE_UNAVAILABLE",
            f"No {fare_type} {travel_class} fare on {route}.",
        )

    railcard = fx.railcards().get(railcard_code.strip().upper()) if railcard_code else None
    if railcard_code and not railcard:
        return fx.mcp_error(
            "RAILCARD_NOT_FOUND", f"Unknown railcard code {railcard_code!r}."
        )

    base = fx.money(candidates[0]["price_gbp"])
    paid = fx.apply_discount(base, railcard["discount_percent"] if railcard else None)

    seat_pool = service["seats_available"].get(travel_class.strip().lower(), 0)
    if seat_pool <= 0:
        return fx.mcp_error(
            "NO_SEATS", f"No {travel_class} seats available on {service_id}."
        )

    ref = f"BKG-{_NEXT_REF[0]}"
    _NEXT_REF[0] += 1
    rules = fx.fare_rules()[fare_type]
    booking = {
        "booking_reference": ref,
        "traveller_id": traveller_id.strip().upper(),
        "service_id": service["service_id"],
        "origin": service["origin"],
        "destination": service["destination"],
        "travel_date": service["date"],
        "departure": service["departure"],
        "arrival": service["arrival"],
        "fare_type": fare_type,
        "travel_class": travel_class.strip().title(),
        "base_fare_gbp": base,
        "railcard_applied": railcard["name"] if railcard else None,
        "amount_paid_gbp": paid,
        "status": "confirmed",
        "refundable": rules["refundable"],
        "changeable": rules["changeable"],
        "change_fee_gbp": rules["change_fee_gbp"],
    }
    _NEW_BOOKINGS[ref] = booking

    return fx.mcp_ok(
        booking=booking,
        confirmation=(
            f"Booked {ref}: {service['operator']} {service['departure']} "
            f"{service['origin']} to {service['destination']} on {service['date']}, "
            f"{fare_type} {booking['travel_class']}, {paid:.2f} GBP."
        ),
    )


if __name__ == "__main__":
    serve(mcp)
