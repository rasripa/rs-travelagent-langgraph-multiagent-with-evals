"""Show each agent's tool catalogue side by side -- the Cedar demo artefact.

This is the slide to put on screen for the access-control story. It performs a
real tools/list per agent and prints what comes back, highlighting the SHARED
TravellerProfile target where the two sub-agents see different tools:

    journey_planner  2 tools   get_traveller_profile, list_saved_journeys
    journey_support  1 tool    get_railcard_and_loyalty

Same server. Same gateway endpoint. Same MCP handshake. Different catalogues.

In agentcore mode this queries the live gateway as each agent's execution role,
so what you see is Cedar's actual decision. In local mode it queries the MCP
servers directly and applies the policy shim, which mirrors the same grants.

    .venv/bin/python scripts/show_tool_access.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import policy_shim  # noqa: E402
from common.config import RUN_MODE  # noqa: E402

SHARED_TARGET = "TravellerProfile"
AGENTS = ["journey_planner", "journey_support", "train_travel_agent"]
WIDTH = 78


async def catalogue(agent_key: str) -> list[str]:
    """Real tools/list for one agent, via whichever path is active."""
    from common.tools_mcp import load_tools

    try:
        tools = await load_tools(agent_key)
        return sorted(tool.name for tool in tools)
    except Exception as exc:  # noqa: BLE001
        return [f"<error: {type(exc).__name__}: {exc}>"]


def all_tools_on(target: str) -> set[str]:
    """Every tool that exists on a target, across all grants."""
    found: set[str] = set()
    for agent_key in AGENTS:
        found.update(policy_shim.granted_tools(agent_key, target))
    return found


async def main() -> int:
    mode = "AgentCore Gateway (live Cedar)" if RUN_MODE == "agentcore" else \
        "local MCP servers (policy shim mirroring Cedar)"
    print(f"\n{'=' * WIDTH}")
    print("TOOL ACCESS CONTROL -- what each agent can see")
    print(f"source: {mode}")
    print("=" * WIDTH)

    catalogues: dict[str, list[str]] = {}
    for agent_key in AGENTS:
        if agent_key == "train_travel_agent" and RUN_MODE != "agentcore":
            # Locally the orchestrator uses the in-process KB index, so it has
            # no gateway catalogue to list.
            catalogues[agent_key] = ["search_rail_policy  (local index)"]
            continue
        print()
        catalogues[agent_key] = await catalogue(agent_key)

    print(f"\n{'-' * WIDTH}")
    for agent_key in AGENTS:
        names = catalogues[agent_key]
        print(f"\n{agent_key}  ({len(names)} tools)")
        for name in names:
            print(f"    {name}")

    # The headline comparison.
    print(f"\n{'=' * WIDTH}")
    print(f"THE SHARED SERVER: {SHARED_TARGET}")
    print("=" * WIDTH)
    print("Both sub-agents are permitted this target. Same endpoint, same")
    print("handshake. Cedar filters tools/list per principal, so they receive")
    print("different catalogues from it:\n")

    universe = sorted(all_tools_on(SHARED_TARGET))
    planner = set(policy_shim.granted_tools("journey_planner", SHARED_TARGET))
    support = set(policy_shim.granted_tools("journey_support", SHARED_TARGET))

    print(f"    {'tool':<34}{'Planner':<12}{'Support'}")
    print(f"    {'-' * 34}{'-' * 12}{'-' * 10}")
    for tool_name in universe:
        p = "visible" if tool_name in planner else "HIDDEN"
        s = "visible" if tool_name in support else "HIDDEN"
        print(f"    {tool_name:<34}{p:<12}{s}")

    print(f"\n    Journey Planner sees {len(planner)} tool(s).")
    print(f"    Journey Support  sees {len(support)} tool(s).")
    print(
        "\n    Denied tools are ABSENT from the catalogue, not merely "
        "un-callable."
        "\n    Journey Support's model never learns get_traveller_profile "
        "exists."
    )
    print(
        "\n    Why: Support needs entitlement (what discount was paid, what"
        "\n    loyalty tier applies), not identity. get_railcard_and_loyalty"
        "\n    exists precisely so servicing a booking does not require"
        "\n    reading a traveller's contact details or journey history."
    )

    if RUN_MODE != "agentcore":
        print(
            f"\n{'-' * WIDTH}\n"
            "NOTE: this is the local shim. It is advisory and in-process --\n"
            "the agent code could bypass it. That is the argument FOR the\n"
            "gateway: enforcement belongs outside the agent. Run this again\n"
            "with TRAVEL_RUN_MODE=agentcore to see Cedar enforce it for real."
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
