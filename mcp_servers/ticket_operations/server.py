"""Ticket Operations MCP server -- reachable by the Journey Support agent ONLY.

Post-booking capability: retrieve a booking, change or cancel it within the
fare rules, and assess a Delay Repay claim. The Journey Planner is denied this
target by Cedar policy -- a planning agent has no business cancelling tickets.

Note what this server deliberately cannot do: it cannot search for an
alternative service. That capability lives on Journey Search, which Support
cannot reach, so Support must delegate to the Planner over A2A. That is the
constraint that makes the peer A2A hop genuine rather than decorative.

Gateway tool names surface as TicketOperations___get_booking and so on.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import fixtures as fx  # noqa: E402
from common.config import REFERENCE_DATE  # noqa: E402
from common.mcp_runtime import build_server, serve  # noqa: E402

mcp = build_server(
    "ticket_operations",
    instructions=(
        "Post-booking ticket servicing for a rail platform: look up a booking, "
        "change or cancel it subject to the fare rules, and assess Delay Repay "
        "compensation. Fare rules are enforced here, not advisory: an Advance "
        "ticket on an undisrupted service will be refused."
    ),
)

# A service delayed by at least this much counts as an operator-caused
# disruption, which suspends the Advance no-change / no-refund restrictions.
# This threshold is also stated in knowledgebase/fare-rules.md -- keep the two
# in step so the agent is never shown contradicting rules.
SIGNIFICANT_DELAY_MINUTES = 60

# Per-process mutations layered over the fixture baseline, so a server restart
# returns the demo to a known state.
_OVERLAY: dict[str, dict[str, Any]] = {}


def _get_booking(ref: str) -> dict[str, Any] | None:
    key = ref.strip().upper()
    if key in _OVERLAY:
        return _OVERLAY[key]
    baseline = fx.bookings_baseline().get(key)
    return dict(baseline) if baseline else None


def _operator_exception(service: dict[str, Any] | None) -> tuple[bool, str | None]:
    """Has the operator disrupted this service enough to waive fare rules?"""
    if not service:
        return False, None
    if service["status"] == "cancelled":
        return True, f"Service cancelled: {service['disruption_reason']}"
    delay = service.get("delay_minutes") or 0
    if delay >= SIGNIFICANT_DELAY_MINUTES:
        return True, f"Service delayed {delay} minutes: {service['disruption_reason']}"
    return False, None


@mcp.tool()
def get_booking(booking_reference: str) -> dict[str, Any]:
    """Retrieve a booking and the live status of the service it is on.

    Always call this before telling a traveller what they can do with a ticket.
    The response includes `entitlements`, which is the authoritative answer on
    whether this specific ticket can be changed or refunded right now -- it
    already accounts for operator disruption overriding the fare rules.

    Args:
        booking_reference: Booking reference, for example BKG-77301.
    """
    booking = _get_booking(booking_reference)
    if not booking:
        return fx.mcp_error(
            "BOOKING_NOT_FOUND",
            f"No booking with reference {booking_reference}.",
            known_references=sorted(fx.bookings_baseline().keys()),
        )

    service = fx.find_service(booking["service_id"])
    waived, reason = _operator_exception(service)
    rules = fx.fare_rules()[booking["fare_type"]]

    return fx.mcp_ok(
        booking=booking,
        service_status=(
            {
                "service_id": service["service_id"],
                "operator": service["operator"],
                "status": service["status"],
                "delay_minutes": service["delay_minutes"],
                "disruption_reason": service["disruption_reason"],
                "replacement_advice": service.get("replacement_advice"),
                "wheelchair_spaces_available": service["wheelchair_spaces_available"],
            }
            if service
            else None
        ),
        entitlements={
            "fare_type": booking["fare_type"],
            "fare_rules_changeable": rules["changeable"],
            "fare_rules_refundable": rules["refundable"],
            "change_fee_gbp": rules["change_fee_gbp"],
            "operator_disruption_waiver": waived,
            "waiver_reason": reason,
            "can_change_now": bool(rules["changeable"] or waived),
            "can_refund_now": bool(rules["refundable"] or waived),
            "fee_payable_gbp": 0.0 if waived else (rules["change_fee_gbp"] or 0.0),
        },
    )


@mcp.tool()
def change_or_cancel_ticket(
    booking_reference: str,
    action: str,
    new_service_id: str | None = None,
) -> dict[str, Any]:
    """Change a booking to a different service, or cancel it for a refund.

    Fare rules are enforced. An Advance ticket on an undisrupted service will be
    refused with code FARE_RULE_PROHIBITS -- report that refusal to the
    traveller plainly rather than retrying. If the operator cancelled or
    significantly delayed the booked service, the restrictions are waived
    automatically and no administration fee is charged.

    This server cannot search for alternative services. To change a booking you
    must already have a new_service_id.

    Args:
        booking_reference: Booking reference, for example BKG-77301.
        action: "change" or "cancel".
        new_service_id: Required when action is "change".
    """
    booking = _get_booking(booking_reference)
    if not booking:
        return fx.mcp_error(
            "BOOKING_NOT_FOUND", f"No booking with reference {booking_reference}."
        )

    act = action.strip().lower()
    if act not in {"change", "cancel"}:
        return fx.mcp_error(
            "INVALID_ACTION", f"action must be 'change' or 'cancel', got {action!r}."
        )
    if booking["status"] in {"cancelled", "refunded"}:
        return fx.mcp_error(
            "ALREADY_CLOSED",
            f"Booking {booking['booking_reference']} is already {booking['status']}.",
        )
    if booking["status"] == "completed":
        return fx.mcp_error(
            "JOURNEY_COMPLETED",
            "This journey has already been travelled and cannot be changed or "
            "cancelled. If it arrived late, assess Delay Repay instead.",
        )

    service = fx.find_service(booking["service_id"])
    waived, reason = _operator_exception(service)
    rules = fx.fare_rules()[booking["fare_type"]]
    permitted = (rules["changeable"] if act == "change" else rules["refundable"]) or waived

    if not permitted:
        return fx.mcp_error(
            "FARE_RULE_PROHIBITS",
            f"A {booking['fare_type']} ticket cannot be "
            f"{'changed' if act == 'change' else 'refunded'}. The booked service "
            f"is running normally, so no disruption waiver applies.",
            fare_type=booking["fare_type"],
            booking_reference=booking["booking_reference"],
            amount_paid_gbp=booking["amount_paid_gbp"],
            remedy=(
                "The traveller would need to buy a new ticket. If their service "
                "is later disrupted, or it arrives 15 or more minutes late, "
                "Delay Repay may still apply."
            ),
        )

    fee = 0.0 if waived else fx.money(rules["change_fee_gbp"] or 0.0)
    updated = dict(booking)

    if act == "cancel":
        refund = fx.money(booking["amount_paid_gbp"] - fee)
        updated["status"] = "refunded"
        updated["refund_gbp"] = refund
        updated["admin_fee_gbp"] = fee
        _OVERLAY[updated["booking_reference"]] = updated
        return fx.mcp_ok(
            action="cancel",
            booking=updated,
            refund_gbp=refund,
            admin_fee_gbp=fee,
            waiver_applied=waived,
            waiver_reason=reason,
            confirmation=(
                f"Cancelled {updated['booking_reference']}. Refund "
                f"{refund:.2f} GBP"
                + (" with no administration fee (operator disruption)." if waived
                   else f" after a {fee:.2f} GBP administration fee.")
            ),
        )

    if not new_service_id:
        return fx.mcp_error(
            "NEW_SERVICE_REQUIRED",
            "action='change' needs new_service_id. This server cannot search "
            "for alternatives; obtain a service id from journey search first.",
        )

    new_service = fx.find_service(new_service_id.strip())
    if not new_service:
        return fx.mcp_error("SERVICE_NOT_FOUND", f"No service {new_service_id}.")
    if new_service["status"] == "cancelled":
        return fx.mcp_error(
            "SERVICE_CANCELLED",
            f"Service {new_service_id} is itself cancelled. Choose another.",
        )
    if (
        new_service["origin"] != booking["origin"]
        or new_service["destination"] != booking["destination"]
    ):
        return fx.mcp_error(
            "ROUTE_MISMATCH",
            f"Service {new_service_id} runs {new_service['origin']} to "
            f"{new_service['destination']}, but the booking is "
            f"{booking['origin']} to {booking['destination']}.",
        )

    needs_space = bool(
        (booking.get("seat") or {}).get("type") == "wheelchair space"
        or booking.get("assistance_booked")
    )
    if needs_space and new_service["wheelchair_spaces_available"] <= 0:
        return fx.mcp_error(
            "NO_ACCESSIBLE_SPACE",
            f"Service {new_service_id} has no wheelchair space available. A "
            "traveller with a booked wheelchair space must be moved to a "
            "service with a confirmed space, not merely the next service.",
        )

    updated["service_id"] = new_service["service_id"]
    updated["travel_date"] = new_service["date"]
    updated["departure"] = new_service["departure"]
    updated["arrival"] = new_service["arrival"]
    updated["status"] = "confirmed"
    updated["admin_fee_gbp"] = fee
    updated["changed_from_service_id"] = booking["service_id"]
    if needs_space:
        updated["assistance_booked"] = True
        updated["assistance_reconfirmed"] = True
    _OVERLAY[updated["booking_reference"]] = updated

    return fx.mcp_ok(
        action="change",
        booking=updated,
        admin_fee_gbp=fee,
        waiver_applied=waived,
        waiver_reason=reason,
        assistance_reconfirmed=needs_space or None,
        confirmation=(
            f"Moved {updated['booking_reference']} to {new_service['operator']} "
            f"{new_service['departure']} ({new_service['service_id']}), arriving "
            f"{new_service['arrival']}"
            + (" at no charge (operator disruption)." if waived
               else f", administration fee {fee:.2f} GBP.")
            + (" Ramp assistance re-confirmed on the new service." if needs_space else "")
        ),
    )


@mcp.tool()
def claim_delay_repay(booking_reference: str) -> dict[str, Any]:
    """Assess and submit a Delay Repay claim for a completed journey.

    Compensation is banded on the delay to arrival and calculated on the amount
    actually paid, after any railcard discount. Delays under 15 minutes do not
    qualify. Claims must be made within the claim window.

    Args:
        booking_reference: Booking reference, for example BKG-77304.
    """
    booking = _get_booking(booking_reference)
    if not booking:
        return fx.mcp_error(
            "BOOKING_NOT_FOUND", f"No booking with reference {booking_reference}."
        )
    if booking.get("delay_repay_claimed"):
        return fx.mcp_error(
            "ALREADY_CLAIMED",
            f"Delay Repay has already been claimed on {booking['booking_reference']}. "
            "One claim per booking per journey.",
        )

    service = fx.find_service(booking["service_id"])
    delay = booking.get("delay_minutes")
    if delay is None and service:
        delay = service.get("delay_minutes")
    if booking["status"] != "completed":
        return fx.mcp_error(
            "JOURNEY_NOT_COMPLETED",
            "Delay Repay is assessed on the delay to arrival, so it can only be "
            "claimed after the journey has been travelled.",
            booking_status=booking["status"],
        )
    if not delay or delay < 15:
        return fx.mcp_error(
            "BELOW_THRESHOLD",
            f"The journey arrived {delay or 0} minutes late. Delays under 15 "
            "minutes do not qualify for compensation.",
            delay_minutes=delay or 0,
        )

    window = fx.claim_window_days()
    days_since = (
        date.fromisoformat(REFERENCE_DATE)
        - date.fromisoformat(booking["travel_date"])
    ).days
    if days_since > window:
        return fx.mcp_error(
            "CLAIM_WINDOW_EXPIRED",
            f"The journey was {days_since} days ago and claims must be made "
            f"within {window} days.",
            days_since_travel=days_since,
            claim_window_days=window,
        )

    band = next(
        (
            b
            for b in fx.delay_repay_bands()
            if delay >= b["min_minutes"]
            and (b["max_minutes"] is None or delay <= b["max_minutes"])
        ),
        None,
    )
    if not band:
        return fx.mcp_error(
            "NO_BAND_MATCH", f"No compensation band matches a {delay} minute delay."
        )

    paid = fx.money(booking["amount_paid_gbp"])
    amount = fx.money(paid * band["compensation_percent"] / 100)

    updated = dict(booking)
    updated["delay_repay_claimed"] = True
    updated["delay_repay_amount_gbp"] = amount
    _OVERLAY[updated["booking_reference"]] = updated

    return fx.mcp_ok(
        claim={
            "booking_reference": booking["booking_reference"],
            "claim_reference": f"DR-{booking['booking_reference'].split('-')[-1]}",
            "status": "submitted",
            "delay_minutes": delay,
            "band": f"{band['min_minutes']} to {band['max_minutes'] or 'unlimited'} minutes",
            "compensation_percent": band["compensation_percent"],
            "basis": band["basis"],
            "amount_paid_gbp": paid,
            "compensation_gbp": amount,
            "days_since_travel": days_since,
            "claim_window_days": window,
        },
        calculation=(
            f"{delay} minutes late falls in the {band['min_minutes']} to "
            f"{band['max_minutes'] or 'unlimited'} minute band, so "
            f"{band['compensation_percent']}% of the {paid:.2f} GBP paid = "
            f"{amount:.2f} GBP."
        ),
        confirmation=(
            f"Delay Repay claim submitted for {booking['booking_reference']}: "
            f"{amount:.2f} GBP. Normally processed within 10 working days."
        ),
    )


if __name__ == "__main__":
    serve(mcp)
