"""Local mirror of the AgentCore Gateway Cedar policies.

In AgentCore mode the gateway is the enforcement point: each sub-agent calls it
with SigV4 signed by its own IAM execution role, Cedar evaluates that role as an
AgentCore::IamEntity principal, and -- importantly -- tools denied by policy do
not appear in tools/list at all. The agent never sees them.

Enforcement is per TOOL, not per server. That is the point worth demonstrating:
both sub-agents are granted the same shared TravellerProfile target, hit the
same gateway endpoint, and get back different tool catalogues from it --
two tools for the Journey Planner, one for Journey Support.

Locally there is no gateway, so this module reproduces that filtering from
AGENT_TOOL_GRANTS in common/config.py, which is also the source the Cedar
policies in deploy/gateway/policies.cedar are generated from. One source of
truth, so the local demo and the deployed demo grant exactly the same thing.

What this shim does NOT reproduce: it is advisory, in-process, and trivially
bypassable by the agent code it is protecting. That is the whole argument for
the gateway -- enforcement belongs outside the agent process. Say that out loud
during the demo rather than implying the shim is a security control.
"""
from __future__ import annotations

from common.config import AGENT_TOOL_GRANTS, MCP_PORTS

# Local MCP server key -> AgentCore Gateway target name. The gateway prefixes
# every tool with its target name and three underscores, so the same tool is
# called TravellerProfile___get_traveller_profile once deployed.
TARGET_NAMES = {
    "traveller_profile": "TravellerProfile",
    "journey_search": "JourneySearch",
    "ticket_operations": "TicketOperations",
}

GATEWAY_PREFIX_SEPARATOR = "___"


def _grant(agent_key: str) -> dict[str, list[str]]:
    try:
        return AGENT_TOOL_GRANTS[agent_key]
    except KeyError:
        raise KeyError(
            f"No tool grant defined for agent {agent_key!r}. "
            f"Known agents: {sorted(AGENT_TOOL_GRANTS)}"
        ) from None


def granted_targets(agent_key: str) -> list[str]:
    """Gateway target names this agent is permitted to reach."""
    return list(_grant(agent_key))


def granted_tools(agent_key: str, target: str) -> list[str]:
    """Unqualified tool names this agent may call on a given target."""
    return list(_grant(agent_key).get(target, []))


def granted_qualified_tools(agent_key: str) -> list[str]:
    """Every tool this agent may call, in gateway Target___tool form."""
    return [
        f"{target}{GATEWAY_PREFIX_SEPARATOR}{tool}"
        for target, tools in _grant(agent_key).items()
        for tool in tools
    ]


def is_tool_granted(agent_key: str, target: str, tool: str) -> bool:
    return tool in _grant(agent_key).get(target, [])


def granted_server_keys(agent_key: str) -> list[str]:
    """Local MCP server keys this agent is permitted to reach."""
    allowed = set(granted_targets(agent_key))
    return [key for key, target in TARGET_NAMES.items() if target in allowed]


def denied_server_keys(agent_key: str) -> list[str]:
    allowed = set(granted_server_keys(agent_key))
    return [key for key in TARGET_NAMES if key not in allowed]


def qualified_tool_name(server_key: str, tool_name: str) -> str:
    """Rename a local tool to the name the gateway would give it.

    Doing this locally means the agent's prompt can reference exact tool names
    and behave identically in both run modes.
    """
    return f"{TARGET_NAMES[server_key]}{GATEWAY_PREFIX_SEPARATOR}{tool_name}"


def local_server_config(agent_key: str) -> dict[str, dict[str, str]]:
    """MultiServerMCPClient config containing only the granted servers.

    Denied servers are not configured at all, so the agent process holds no
    connection to a server it has no business reaching. Tool-level filtering
    within a granted server is applied separately, in common/tools_mcp.py.
    """
    from common.config import LOCAL_BIND_HOST

    return {
        key: {
            "url": f"http://{LOCAL_BIND_HOST}:{MCP_PORTS[key]}/mcp",
            "transport": "streamable_http",
        }
        for key in granted_server_keys(agent_key)
    }


def describe_grant(agent_key: str) -> str:
    """One-line summary for demo output, showing per-target tool counts."""
    grant = _grant(agent_key)
    parts = [f"{target}({len(tools)})" for target, tools in grant.items()]
    denied = ", ".join(TARGET_NAMES[k] for k in denied_server_keys(agent_key)) or "none"
    return f"{agent_key}: {' '.join(parts)}  denied targets=[{denied}]"


def describe_shared_asymmetry(shared_target: str = "TravellerProfile") -> list[str]:
    """Render the shared-server asymmetry, which is the core demo point."""
    lines = [f"shared target {shared_target} -- same endpoint, different catalogues:"]
    for agent_key in ("journey_planner", "journey_support"):
        tools = granted_tools(agent_key, shared_target)
        lines.append(f"  {agent_key:<16} {len(tools)} tool(s): {', '.join(tools)}")
    return lines
