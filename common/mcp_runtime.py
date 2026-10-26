"""Shared bootstrap for the three mock MCP servers.

AgentCore Runtime's MCP protocol contract requires the container to listen on
0.0.0.0:8000 and expose /mcp as a POST endpoint, with a stateless
streamable-HTTP transport because the runtime provides session isolation itself
and injects an Mcp-Session-Id header. See:
https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-mcp.html

Locally we bind loopback on a per-server port instead, so three servers can run
side by side on one machine without being reachable off-host.
"""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from common.config import LOCAL_BIND_HOST, MCP_PORTS, RUN_MODE

# AgentCore's required MCP listener.
AGENTCORE_MCP_HOST = "0.0.0.0"  # nosec B104 - mandated by the runtime contract
AGENTCORE_MCP_PORT = 8000


def build_server(server_key: str, instructions: str) -> FastMCP:
    """Create a FastMCP server configured for the current run mode.

    stateless_http=True is required by AgentCore for basic MCP servers. These
    servers expose tools only -- no prompts and no resources -- because the
    AgentCore Gateway policy engine governs MCP tools only: prompts and
    resources that a target advertises are always allowed and are auto-synced.
    Exposing none keeps the enforced surface equal to the advertised surface.
    """
    if RUN_MODE == "agentcore":
        host, port = AGENTCORE_MCP_HOST, AGENTCORE_MCP_PORT
    else:
        host, port = LOCAL_BIND_HOST, MCP_PORTS[server_key]

    # Allow an explicit override (AgentCore sets no PORT var, but local tooling
    # and container tests find it convenient).
    port = int(os.getenv("MCP_PORT", str(port)))

    return FastMCP(
        name=server_key,
        instructions=instructions,
        host=host,
        port=port,
        stateless_http=True,
    )


def serve(server: FastMCP) -> None:
    """Run the server over streamable HTTP, the only transport AgentCore accepts.

    Tracing is initialised here rather than in each server module, so all three
    MCP servers appear as spans in the same trace as the agents that call them.
    Without this the trace stops at the gateway.
    """
    from common import observability

    observability.init(f"mcp-{server.name}")
    server.run(transport="streamable-http")
