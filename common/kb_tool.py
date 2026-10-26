"""The orchestrator's policy knowledge-base tool.

Two sources, one interface:

  agentcore  -> an Amazon Bedrock MANAGED Knowledge Base, attached to the same
                AgentCore Gateway as the MCP servers via the built-in
                `bedrock-knowledge-bases` connector. The orchestrator discovers
                it with tools/list like any other tool, signs with SigV4, and
                never sees the knowledge base id -- the gateway binds it.

  local      -> the markdown files in knowledgebase/, embedded with Titan and
                cached to .kb_index.json, with a lexical fallback. Keeps the
                demo runnable on a laptop with no provisioning.

The connector exposes two tools, `Retrieve` and `AgenticRetrieveStream`. Cedar
grants the orchestrator only `Retrieve`: a single hybrid search is fast, cheap
and deterministic, which is what a scripted demo wants. The denial of
`AgenticRetrieveStream` is a second, smaller illustration of per-tool control --
this time capping cost and latency rather than protecting data.

Why the KB is a tool on the orchestrator and not a third sub-agent: a policy
question resolves in one model turn against retrieved text. Wrapping that in an
agentic loop with its own identity and network hop would buy nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.tools import BaseTool, tool  # noqa: E402

from common.config import RUN_MODE  # noqa: E402


@tool
def search_rail_policy(query: str) -> str:
    """Search rail fare rules, railcard terms, Delay Repay compensation rules,
    ticket validity and peak times, and luggage and accessible-travel policy.

    Use this for any question about what the rules ARE -- entitlements,
    restrictions, compensation bands, peak windows, luggage allowances. Do not
    use it to look up a specific traveller, booking or train service; delegate
    those to the specialist agents.

    Always quote the retrieved policy rather than recalling it from memory, and
    cite the source document.

    Args:
        query: The policy question, in natural language.
    """
    from knowledgebase import indexer

    hits = indexer.search(query, top_k=4)
    if not hits:
        return (
            "No matching policy found in the knowledge base. Say so plainly "
            "rather than guessing at the rule."
        )

    return "\n\n".join(
        f"--- source: {hit['source']} > {hit['section']} "
        f"(relevance {hit['score']}) ---\n{hit['text']}"
        for hit in hits
    )


async def load_kb_tools(agent_key: str = "train_travel_agent") -> list[BaseTool]:
    """Return the knowledge-base tool(s) for the current run mode."""
    if RUN_MODE != "agentcore":
        from knowledgebase import indexer

        print(f"  [{agent_key}] knowledge base: local index "
              f"({indexer.load_index()['mode']})")
        return [search_rail_policy]

    print(f"  [{agent_key}] knowledge base: AgentCore Gateway (managed KB)")
    from common.tools_mcp import load_tools

    tools = await load_tools(agent_key)
    if not tools:
        raise RuntimeError(
            "The gateway returned no knowledge-base tools for the "
            "orchestrator. Check that the RailPolicyKB connector target is "
            "READY and that the Cedar policy permits RailPolicyKB___Retrieve."
        )
    return tools
