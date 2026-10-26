"""Best-effort trajectory capture from outside the agent processes.

THE HONEST LIMITATION, STATED UP FRONT
--------------------------------------
This harness drives the system through its real network interfaces and is not
allowed to modify the application, so it cannot see the agents' internal tool
calls directly. What it can observe differs sharply by mode:

  local mode      The MCP servers log one line per tool invocation
                  ("Processing request of type CallToolRequest"), but NOT the
                  tool NAME. So locally we recover tool-call counts per tool
                  SERVER and nothing finer. The A2A sub-agents and the
                  orchestrator run uvicorn at log_level="warning" and emit no
                  access log at all, so an A2A delegation is never observed
                  directly -- it is INFERRED from which MCP server saw traffic.
                  The policy agent's local knowledge-base tool runs in-process
                  against a local index with no MCP server behind it, so its
                  tool use leaves NO log evidence whatsoever and is reported as
                  unobservable rather than as zero.

  agentcore mode  OpenTelemetry spans in the CloudWatch Logs group `aws/spans`
                  carry real tool names, real A2A hops and token counts. That
                  is where a full trajectory actually lives. See perf.py.

Attribution works by diffing byte offsets in `.run/logs/*.log` around each
case, which REQUIRES SEQUENTIAL EXECUTION: with concurrency > 1 the log deltas
of overlapping cases interleave and cannot be separated. The runners therefore
disable local trajectory capture whenever concurrency > 1, rather than
reporting a plausible-looking attribution that is wrong.

Nothing here fabricates a tool name. Every entry carries the evidence it came
from, and layers we did not observe are marked unobservable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = Path(os.getenv("EVAL_LOG_DIR", PROJECT_ROOT / ".run" / "logs"))

# Log file -> the gateway target name the tools on it are prefixed with. These
# prefixes match the tool names in common/config.py AGENT_TOOL_GRANTS, so an
# expected tool like "JourneySearch___get_fares" reduces to "JourneySearch"
# and can be compared against what we observed.
MCP_LOGS = {
    "traveller_profile.log": "TravellerProfile",
    "journey_search.log": "JourneySearch",
    "ticket_operations.log": "TicketOperations",
}

TOOL_CALL_MARKER = "Processing request of type CallToolRequest"
LIST_TOOLS_MARKER = "Processing request of type ListToolsRequest"

# Which sub-agent owns each tool server, from AGENT_TOOL_GRANTS. Used only to
# INFER a delegation; TravellerProfile is shared and so proves nothing on its
# own.
EXCLUSIVE_SERVER_OWNER = {
    "JourneySearch": "journey_planner",
    "TicketOperations": "journey_support",
}

# The policy agent's only tool. Locally it is an in-process function
# (common.kb_tool.search_rail_policy) over a cached index; deployed it is the
# gateway's managed knowledge-base connector (RailPolicyKB___Retrieve). Both
# names are treated as the same capability when comparing against
# expected_tools.
POLICY_TOOL_NAMES = ("RailPolicyKB___Retrieve", "search_rail_policy")
POLICY_SERVER = "RailPolicyKB"


def tool_server_of(tool_name: str) -> str:
    """Reduce a tool name to its gateway target, e.g. JourneySearch___x -> JourneySearch."""
    if tool_name in POLICY_TOOL_NAMES:
        return POLICY_SERVER
    head, sep, _ = tool_name.partition("___")
    return head if sep else tool_name


class LogOffsets:
    """Byte offsets of the local stack's logs at one moment in time."""

    def __init__(self, offsets: dict[str, int], available: bool) -> None:
        self.offsets = offsets
        self.available = available


def snapshot() -> LogOffsets:
    """Record the current size of each MCP server log."""
    offsets: dict[str, int] = {}
    found = False
    for filename in MCP_LOGS:
        path = LOG_DIR / filename
        if path.exists():
            found = True
            offsets[filename] = path.stat().st_size
        else:
            offsets[filename] = 0
    return LogOffsets(offsets, available=found)


def observe(before: LogOffsets) -> dict[str, Any]:
    """Read what the MCP servers logged since `before` and describe it.

    Returns a dict with:
      tool_calls_by_server  observed CallToolRequest counts, per tool server
      trajectory            one entry per observed call, as "<Server>___?" --
                            the server is evidence, the tool name is not
                            recoverable locally
      inferred_delegations  sub-agents that must have run, from exclusive servers
      unobservable          layers this mode cannot see at all
      evidence              how each figure was derived
    """
    result: dict[str, Any] = {
        "tool_calls_by_server": {},
        "trajectory": [],
        "inferred_delegations": [],
        "unobservable": [],
        "evidence": "mcp_server_logs (byte-offset diff per case)",
        "available": before.available,
    }

    if not before.available:
        result["unobservable"] = [
            f"all tool calls (no MCP server logs found under {LOG_DIR})"
        ]
        result["evidence"] = "none"
        return result

    for filename, server in MCP_LOGS.items():
        path = LOG_DIR / filename
        if not path.exists():
            continue
        start = before.offsets.get(filename, 0)
        try:
            with path.open("r", errors="replace") as handle:
                handle.seek(start)
                delta = handle.read()
        except OSError:
            continue

        calls = delta.count(TOOL_CALL_MARKER)
        if calls:
            result["tool_calls_by_server"][server] = calls
            # "?" is deliberate and load-bearing: the log line records that a
            # tool on this server ran but never which one. Emitting a guessed
            # name here would be the single most misleading thing this module
            # could do.
            result["trajectory"].extend([f"{server}___?"] * calls)
            owner = EXCLUSIVE_SERVER_OWNER.get(server)
            if owner and owner not in result["inferred_delegations"]:
                result["inferred_delegations"].append(owner)

    result["unobservable"] = [
        "tool names (MCP logs record CallToolRequest without the tool name)",
        "A2A delegation hops (sub-agent uvicorn runs at log_level=warning, "
        "no access log)",
        f"policy knowledge-base tool ({POLICY_TOOL_NAMES[1]} runs in-process "
        "locally and writes no log line)",
    ]
    return result


def compare_expected(expected_tools: list[str], observed: dict[str, Any]) -> dict[str, Any]:
    """Compare expected tools against what was observable, at server granularity.

    Deliberately NOT turned into a pass/fail score. Locally the observation is
    too coarse to justify failing a case on it, so this is reported as evidence
    for a human reading the dashboard, not as a metric.
    """
    expected_servers = sorted({tool_server_of(t) for t in expected_tools})
    observed_servers = sorted(observed.get("tool_calls_by_server", {}))

    # The policy server is never observable locally, so its absence is not a miss.
    unobservable_servers = {POLICY_SERVER}
    missing = [
        s
        for s in expected_servers
        if s not in observed_servers and s not in unobservable_servers
    ]
    return {
        "expected_tools": list(expected_tools),
        "expected_servers": expected_servers,
        "observed_servers": observed_servers,
        "servers_not_observed": missing,
        "servers_unobservable_in_this_mode": sorted(
            s for s in expected_servers if s in unobservable_servers
        ),
    }
