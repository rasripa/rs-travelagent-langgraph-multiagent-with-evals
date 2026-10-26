"""Load a sub-agent's MCP tools, in whichever run mode is active.

local      -> connect straight to the three mock MCP servers on loopback, and
              apply common/policy_shim.py so each agent sees only its granted
              targets. Tools are renamed to the gateway's Target___tool form so
              the agent behaves identically in both modes.

agentcore  -> connect to ONE AgentCore Gateway endpoint, signing with SigV4
              under this runtime's own IAM execution role. Cedar evaluates that
              role and filters tools/list server-side, so denied tools are
              simply absent from the catalogue. Names already arrive
              gateway-prefixed.

The deliberate design point: the agent code below does not branch on which
tools it is allowed to have. It asks for its catalogue and uses what it gets.
Entitlement is configuration, not application logic.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.tools import BaseTool  # noqa: E402
from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: E402

from common import policy_shim  # noqa: E402
from common.config import RUN_MODE  # noqa: E402

# AGENTPERF06-BP01: tool selection accuracy degrades past roughly 10-15 tools.
# Each sub-agent here is scoped to 6. Warn loudly if that ever drifts.
TOOL_COUNT_WARN_THRESHOLD = 10


def gateway_url() -> str:
    """Resolved dynamically: env var, then Parameter Store."""
    from common import config_store

    return config_store.require("gateway_url")


async def load_tools(agent_key: str) -> list[BaseTool]:
    """Return the MCP tools this agent is entitled to."""
    if RUN_MODE == "agentcore":
        tools = await _load_from_gateway(agent_key)
        source = f"AgentCore Gateway ({len(tools)} tools after Cedar filtering)"
    else:
        tools = await _load_from_local_servers(agent_key)
        source = f"local MCP servers ({len(tools)} tools after policy shim)"

    print(f"  [{agent_key}] tools from {source}")
    for tool in tools:
        print(f"      - {tool.name}")

    if len(tools) > TOOL_COUNT_WARN_THRESHOLD:
        warnings.warn(
            f"{agent_key} has {len(tools)} tools, above the "
            f"{TOOL_COUNT_WARN_THRESHOLD}-tool threshold where selection "
            f"accuracy starts to degrade. Consider narrowing its grant.",
            stacklevel=2,
        )
    return tools


async def _load_from_local_servers(agent_key: str) -> list[BaseTool]:
    connections = policy_shim.local_server_config(agent_key)
    if not connections:
        raise RuntimeError(f"Agent {agent_key!r} is granted no MCP targets.")

    client = MultiServerMCPClient(connections)
    tools: list[BaseTool] = []
    filtered: list[str] = []

    for server_key in connections:
        target = policy_shim.TARGET_NAMES[server_key]
        try:
            server_tools = await client.get_tools(server_name=server_key)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not reach the {server_key} MCP server at "
                f"{connections[server_key]['url']}. Start the servers first: "
                f"./scripts/run_local.sh start.  ({type(exc).__name__}: {exc})"
            ) from exc

        for tool in server_tools:
            # Tool-level filtering, mirroring Cedar. A denied tool is dropped
            # from the catalogue entirely, exactly as the gateway would omit it
            # from tools/list -- the model never learns it exists.
            if not policy_shim.is_tool_granted(agent_key, target, tool.name):
                filtered.append(f"{target}___{tool.name}")
                continue
            # Rename to the gateway's Target___tool form so prompts and traces
            # look the same locally as they do once deployed.
            tool.name = policy_shim.qualified_tool_name(server_key, tool.name)
            tools.append(tool)

    if filtered:
        print(f"      (policy hid {len(filtered)}: {', '.join(filtered)})")
    return tools


async def _load_from_gateway(agent_key: str) -> list[BaseTool]:
    from common.aws_sigv4 import signer_for_agentcore
    from common.config import AWS_REGION

    # SigV4 with the runtime's own execution role. No token endpoint, no client
    # secret, no expiry to manage -- and the role ARN is exactly what the
    # gateway's Cedar policies match on as AgentCore::IamEntity.
    client = MultiServerMCPClient(
        {
            "gateway": {
                "url": gateway_url(),
                "transport": "streamable_http",
                "auth": signer_for_agentcore(AWS_REGION),
                "timeout": 60.0,
            }
        }
    )
    # tools/list is paginated; get_tools drains the pages for us.
    tools = await client.get_tools(server_name="gateway")

    # Cross-check what Cedar actually returned against what we expect to be
    # granted. A mismatch usually means the policy engine is still LOG_ONLY, or
    # the Cedar actions drifted from AGENT_TOOL_GRANTS.
    expected = set(policy_shim.granted_qualified_tools(agent_key))
    received = {tool.name for tool in tools}
    if received != expected:
        unexpected = sorted(received - expected)
        missing = sorted(expected - received)
        warnings.warn(
            f"Gateway tool catalogue for {agent_key} does not match the "
            f"expected grant.\n"
            f"  unexpectedly visible: {unexpected or 'none'}\n"
            f"  expected but absent:  {missing or 'none'}\n"
            f"If extra tools are visible the policy engine may still be in "
            f"LOG_ONLY mode.",
            stacklevel=2,
        )
    return tools
