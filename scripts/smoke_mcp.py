"""Smoke-test the three MCP servers over streamable HTTP.

Starts each server as a subprocess on its local loopback port, runs tools/list
and a representative tools/call against it, and asserts the fixture maths comes
out exactly as the knowledge base and the architecture document claim.

This is the test that would have caught the railcard/booking inconsistency, so
it runs the specific numbers the demo narrates on stage.

    .venv/bin/python scripts/smoke_mcp.py
"""
from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamablehttp_client  # noqa: E402

from common.config import LOCAL_BIND_HOST, MCP_PORTS  # noqa: E402

SERVERS = {
    "traveller_profile": "mcp_servers/traveller_profile/server.py",
    "journey_search": "mcp_servers/journey_search/server.py",
    "ticket_operations": "mcp_servers/ticket_operations/server.py",
}

EXPECTED_TOOLS = {
    "traveller_profile": {
        "get_traveller_profile",
        "get_railcard_and_loyalty",
        "list_saved_journeys",
    },
    "journey_search": {"search_journeys", "get_fares", "book_ticket"},
    "ticket_operations": {"get_booking", "change_or_cancel_ticket", "claim_delay_repay"},
}

PASS, FAIL = [], []

# ticket_operations holds per-process mutation state (bookings cancelled, Delay
# Repay claimed). Reusing a running instance would make these tests pass once
# and then fail on a second run, so it always gets a private instance on its own
# port. The other two servers are read-only and safe to reuse.
PRIVATE_PORTS = {"ticket_operations": 8113}


def check(label: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(label)
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" -- {detail}" if detail and not condition else ""))


def port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.3)
        return probe.connect_ex((LOCAL_BIND_HOST, port)) == 0


def start(name: str, script: str) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(ROOT), "TRAVEL_RUN_MODE": "local"}
    if name in PRIVATE_PORTS:
        env["MCP_PORT"] = str(PRIVATE_PORTS[name])
    return subprocess.Popen(
        [sys.executable, str(ROOT / script)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def port_for(name: str) -> int:
    return PRIVATE_PORTS.get(name, MCP_PORTS[name])


async def call(session: ClientSession, tool: str, **args):
    result = await session.call_tool(tool, args)
    import json

    return json.loads(result.content[0].text)


async def test_traveller_profile(session: ClientSession) -> None:
    names = {t.name for t in (await session.list_tools()).tools}
    check("traveller_profile exposes exactly its 3 tools",
          names == EXPECTED_TOOLS["traveller_profile"], str(names))

    r = await call(session, "get_traveller_profile", traveller_id="TRV-1001")
    check("TRV-1001 is Sarah Chen", r["ok"] and r["traveller"]["name"] == "Sarah Chen")
    check("TRV-1001 holds a Two Together Railcard",
          r["traveller"]["railcard"]["code"] == "TTG")

    r = await call(session, "get_traveller_profile", traveller_id="TRV-1003")
    check("TRV-1003 needs a wheelchair space",
          r["traveller"]["accessibility"]["wheelchair_space_required"] is True)

    r = await call(session, "get_railcard_and_loyalty", traveller_id="TRV-1003")
    check("TRV-1003 has no railcard", r["has_railcard"] is False)

    r = await call(session, "list_saved_journeys", traveller_id="TRV-1001")
    check("TRV-1001 has 2 saved journeys", r["count"] == 2)

    r = await call(session, "get_traveller_profile", traveller_id="TRV-9999")
    check("unknown traveller returns a structured error",
          r["ok"] is False and r["error"]["code"] == "TRAVELLER_NOT_FOUND")


async def test_journey_search(session: ClientSession) -> None:
    names = {t.name for t in (await session.list_tools()).tools}
    check("journey_search exposes exactly its 3 tools",
          names == EXPECTED_TOOLS["journey_search"], str(names))

    r = await call(session, "search_journeys", origin="London Euston",
                   destination="Manchester", date="2026-11-06")
    check("station aliases resolve to CRS codes",
          r["ok"] and r["origin"]["crs"] == "EUS" and r["destination"]["crs"] == "MAN")
    by_id = {s["service_id"]: s for s in r["services"]}
    check("the 07:15 is cancelled",
          by_id["AW0715-20261106"]["status"] == "cancelled")
    check("the 09:15 is delayed 12 minutes",
          by_id["AW0915-20261106"]["delay_minutes"] == 12)
    check("network disruption DSR-4471 is surfaced",
          any(d["disruption_id"] == "DSR-4471" for d in r["network_disruption"]))
    check("services are returned in departure order",
          [s["departure"] for s in r["services"]]
          == sorted(s["departure"] for s in r["services"]))

    r = await call(session, "get_fares", origin="EUS", destination="MAN",
                   railcard_code="TTG")
    adv = next(f for f in r["fares"]
               if f["fare_type"] == "Advance" and f["travel_class"] == "Standard")
    check("EUS-MAN Advance with TTG prices at 24.64 (matches BKG-77301)",
          adv["price_gbp"] == 24.64, f"got {adv['price_gbp']}")
    check("Advance is not railcard time-restricted",
          adv["railcard_time_restricted"] is False)
    anytime = next(f for f in r["fares"]
                   if f["fare_type"] == "Anytime" and f["travel_class"] == "Standard")
    check("Anytime IS railcard time-restricted for TTG",
          anytime["railcard_time_restricted"] is True)
    check("Advance is flagged non-refundable and non-changeable",
          adv["refundable"] is False and adv["changeable"] is False)
    check("fares are returned cheapest first",
          [f["price_gbp"] for f in r["fares"]]
          == sorted(f["price_gbp"] for f in r["fares"]))

    r = await call(session, "get_fares", origin="EUS", destination="BHM",
                   railcard_code="TTG")
    op = next(f for f in r["fares"] if f["fare_type"] == "Off-Peak")
    check("EUS-BHM Off-Peak with TTG prices at 29.97 (matches BKG-77304)",
          op["price_gbp"] == 29.97, f"got {op['price_gbp']}")

    r = await call(session, "book_ticket", traveller_id="TRV-1001",
                   service_id="AW0715-20261106", fare_type="Advance")
    check("booking a cancelled service is refused",
          r["ok"] is False and r["error"]["code"] == "SERVICE_CANCELLED")

    r = await call(session, "book_ticket", traveller_id="TRV-1001",
                   service_id="AW0745-20261106", fare_type="Advance",
                   railcard_code="TTG")
    check("booking the 07:45 succeeds at the discounted fare",
          r["ok"] and r["booking"]["amount_paid_gbp"] == 24.64,
          str(r.get("error") or r["booking"]["amount_paid_gbp"]))


async def test_ticket_operations(session: ClientSession) -> None:
    names = {t.name for t in (await session.list_tools()).tools}
    check("ticket_operations exposes exactly its 3 tools",
          names == EXPECTED_TOOLS["ticket_operations"], str(names))

    r = await call(session, "get_booking", booking_reference="BKG-77301")
    check("BKG-77301 is a 24.64 Advance", r["booking"]["amount_paid_gbp"] == 24.64)
    check("BKG-77301's service shows as cancelled",
          r["service_status"]["status"] == "cancelled")
    check("disruption waives the Advance restrictions",
          r["entitlements"]["operator_disruption_waiver"] is True
          and r["entitlements"]["can_refund_now"] is True)
    check("waived change carries no fee",
          r["entitlements"]["fee_payable_gbp"] == 0.0)

    r = await call(session, "get_booking", booking_reference="BKG-77303")
    check("BKG-77303 Advance on a normal service cannot be refunded",
          r["entitlements"]["can_refund_now"] is False)

    r = await call(session, "change_or_cancel_ticket",
                   booking_reference="BKG-77303", action="cancel")
    check("cancelling an undisrupted Advance is refused by fare rule",
          r["ok"] is False and r["error"]["code"] == "FARE_RULE_PROHIBITS",
          str(r.get("error")))

    r = await call(session, "change_or_cancel_ticket",
                   booking_reference="BKG-77301", action="change",
                   new_service_id="AW0745-20261106")
    check("disrupted Advance CAN be moved, free of charge",
          r["ok"] and r["admin_fee_gbp"] == 0.0 and r["waiver_applied"] is True,
          str(r.get("error")))

    r = await call(session, "change_or_cancel_ticket",
                   booking_reference="BKG-77302", action="change",
                   new_service_id="LN0900-20261110")
    check("wheelchair booking rebooks only onto a service with a space",
          r["ok"] and r.get("assistance_reconfirmed") is True, str(r.get("error")))

    r = await call(session, "claim_delay_repay", booking_reference="BKG-77304")
    check("47-minute delay lands in the 30-59 band at 50%",
          r["ok"] and r["claim"]["compensation_percent"] == 50, str(r.get("error")))
    check("Delay Repay on 29.97 paid = 14.99",
          r["claim"]["compensation_gbp"] == 14.99,
          f"got {r['claim'].get('compensation_gbp')}")
    check("claim is inside the 28-day window",
          r["claim"]["days_since_travel"] <= r["claim"]["claim_window_days"])

    r = await call(session, "claim_delay_repay", booking_reference="BKG-77304")
    check("a second claim on the same booking is refused",
          r["ok"] is False and r["error"]["code"] == "ALREADY_CLAIMED")

    r = await call(session, "claim_delay_repay", booking_reference="BKG-77303")
    check("Delay Repay on a journey not yet travelled is refused",
          r["ok"] is False and r["error"]["code"] == "JOURNEY_NOT_COMPLETED",
          str(r.get("error")))


TESTS = {
    "traveller_profile": test_traveller_profile,
    "journey_search": test_journey_search,
    "ticket_operations": test_ticket_operations,
}


async def test_tool_access_control() -> None:
    """The demo centrepiece: per-TOOL filtering on the SHARED server.

    Goes through the real load_tools() path, so it exercises the same code the
    agents use rather than just re-reading the config.
    """
    from common import policy_shim
    from common.tools_mcp import load_tools

    print("\ntool access control (shared TravellerProfile target)")

    catalogues = {}
    for agent in ("journey_planner", "journey_support"):
        catalogues[agent] = {tool.name for tool in await load_tools(agent)}

    planner, support = catalogues["journey_planner"], catalogues["journey_support"]

    check("journey_planner gets 5 tools", len(planner) == 5, str(sorted(planner)))
    check("journey_support gets 4 tools", len(support) == 4, str(sorted(support)))

    prefix = "TravellerProfile___"
    planner_shared = {n for n in planner if n.startswith(prefix)}
    support_shared = {n for n in support if n.startswith(prefix)}

    check("Planner sees 2 tools on the SHARED target",
          len(planner_shared) == 2, str(sorted(planner_shared)))
    check("Support sees 1 tool on the SHARED target",
          len(support_shared) == 1, str(sorted(support_shared)))
    check("both agents DO reach the shared target",
          bool(planner_shared) and bool(support_shared))
    check("the shared catalogues genuinely differ",
          planner_shared != support_shared)

    check("Support cannot see get_traveller_profile",
          f"{prefix}get_traveller_profile" not in support)
    check("Support cannot see list_saved_journeys",
          f"{prefix}list_saved_journeys" not in support)
    check("Support CAN see get_railcard_and_loyalty",
          f"{prefix}get_railcard_and_loyalty" in support)
    check("Planner cannot see get_railcard_and_loyalty",
          f"{prefix}get_railcard_and_loyalty" not in planner)

    # Target-level denials still hold.
    check("Planner is denied the whole TicketOperations target",
          not any(n.startswith("TicketOperations___") for n in planner))
    check("Support is denied the whole JourneySearch target",
          not any(n.startswith("JourneySearch___") for n in support))

    # The shim and the Cedar source agree, since both derive from the same dict.
    for agent in ("journey_planner", "journey_support"):
        expected = set(policy_shim.granted_qualified_tools(agent))
        check(f"{agent} catalogue matches its declared grant",
              catalogues[agent] == expected,
              f"got {sorted(catalogues[agent])} want {sorted(expected)}")


async def main() -> int:
    procs: dict[str, subprocess.Popen] = {}
    # Reuse servers that are already running (./scripts/run_local.sh start) and
    # only spawn the ones that are down. Spawning onto a bound port would fail
    # with a confusing "did not start" rather than "already running".
    reused = [
        name
        for name in SERVERS
        if name not in PRIVATE_PORTS and port_busy(MCP_PORTS[name])
    ]
    if reused:
        print(f"reusing already-running server(s): {', '.join(reused)}")
    print(
        "starting a private ticket_operations on port "
        f"{PRIVATE_PORTS['ticket_operations']} for the mutation tests"
    )

    try:
        for name, script in SERVERS.items():
            if name not in reused:
                procs[name] = start(name, script)
        if procs:
            time.sleep(4.0)

        for name in SERVERS:
            proc = procs.get(name)
            if proc is not None and proc.poll() is not None:
                err = proc.stderr.read().decode()[-1500:] if proc.stderr else ""
                print(f"\n{name}: SERVER FAILED TO START\n{err}")
                FAIL.append(f"{name} did not start")
                continue

            url = f"http://{LOCAL_BIND_HOST}:{port_for(name)}/mcp"
            print(f"\n{name}  ({url})")
            try:
                async with streamablehttp_client(url) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        await TESTS[name](session)
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] transport/session error: {exc!r}")
                FAIL.append(f"{name} session error: {exc!r}")

        # Runs against the shared servers above, through the real tool loader.
        try:
            await test_tool_access_control()
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] tool access control: {exc!r}")
            FAIL.append(f"tool access control: {exc!r}")
    finally:
        for proc in procs.values():
            proc.terminate()
        for proc in procs.values():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    print(f"\n{'=' * 62}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("\nfailures:")
        for f in FAIL:
            print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
