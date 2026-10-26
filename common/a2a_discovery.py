"""Runtime discovery of sub-agents from their A2A agent cards.

Why this module exists
----------------------
The router must stay domain-agnostic, and a new sub-agent must be deployable
WITHOUT editing the router. Those two requirements rule out the obvious
implementation, which is one hand-written delegation tool per sub-agent with
routing rules baked into the router's prompt: that makes the router a
domain participant and makes "add an agent" a code change to Tier 1 -- the
exact bottleneck a multi-agent split is supposed to remove.

So the router does not know what its sub-agents do. It knows only where to
look. At startup it:

  1. reads the sub-agent REGISTRY  (one config value, see registered_agents)
  2. fetches each agent's CARD     (/.well-known/agent-card.json)
  3. reads the card's SKILLS       (id, name, description, tags, examples)
  4. synthesises one delegation TOOL per agent, whose description is built
     from that card

The model then routes by reading tool descriptions, which is the same
mechanism it already uses for every other tool. Adding a Tier-2 agent is
therefore: deploy it, publish its URL, append its key to the registry. The
router picks it up on its next cold start with no redeploy of Tier 1.

What a card can and cannot carry
--------------------------------
A2A skills are advertisements, not schemas: id, name, description, tags,
examples. There is no input schema for a skill, so delegation is free text --
which is why the synthesised tool takes a single self-contained `request`
string and why the generated description tells the caller to include every
identifier the peer will need. That is a property of the protocol, not a
shortcut here.

Failure behaviour is deliberate: a sub-agent whose card cannot be fetched is
SKIPPED with a warning rather than failing the router's startup. One
unreachable Tier-2 agent must not take the assistant down; the router simply
cannot route to it, and says so when asked.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# How long to wait for a single card fetch before giving up on that agent.
CARD_TIMEOUT_SECONDS = 20.0

# Registry default, used when nothing is configured. Deliberately NOT a
# hardcoded list inside the router: it lives here next to the discovery code,
# and the deployed stack overrides it from Parameter Store.
DEFAULT_REGISTRY = ("journey_planner", "journey_support", "policy_advisor")


def registered_agents() -> list[str]:
    """Sub-agent keys the router should try to discover.

    Resolution order is the same as every other dynamic value: environment,
    then Parameter Store (`/travel-demo/sub-agents`), then the default above.
    Publishing a new key is what makes a new Tier-2 agent routable.
    """
    from common import config_store

    raw = config_store.get("sub_agents")
    if raw:
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if keys:
            return keys
    return list(DEFAULT_REGISTRY)


async def fetch_card(agent_key: str) -> Any | None:
    """Fetch one sub-agent's agent card, or None if it cannot be reached."""
    import httpx
    from a2a.client import A2ACardResolver

    from common import a2a_client
    from common.config import a2a_base_url

    try:
        base_url = a2a_base_url(agent_key)
    except RuntimeError as exc:
        logger.warning("no address for sub-agent %s: %s", agent_key, exc)
        return None

    auth, headers = a2a_client.auth_for()
    try:
        async with httpx.AsyncClient(
            timeout=CARD_TIMEOUT_SECONDS, headers=headers or {}, auth=auth
        ) as http_client:
            resolver = A2ACardResolver(
                httpx_client=http_client, base_url=base_url.rstrip("/")
            )
            return await resolver.get_agent_card()
    except Exception as exc:  # noqa: BLE001
        # Includes 403 (missing GetAgentCard permission), connection refused
        # locally, and a cold runtime that has not finished starting.
        logger.warning(
            "could not fetch the agent card for %s at %s (%s: %s)",
            agent_key, base_url, type(exc).__name__, exc,
        )
        return None


def describe_card(card: Any) -> str:
    """Render a card's skills as a delegation-tool description.

    This text is the ONLY thing the router knows about the sub-agent, so it
    has to carry enough for the model to route correctly: what the agent is
    for, each skill, and example utterances where the card provides them.
    """
    lines: list[str] = []
    name = getattr(card, "name", "") or "(unnamed agent)"
    description = (getattr(card, "description", "") or "").strip()
    lines.append(f"Delegate to {name}.")
    if description:
        lines.append(description)

    skills = list(getattr(card, "skills", None) or [])
    if skills:
        lines.append("")
        lines.append("It advertises these skills:")
        for skill in skills:
            skill_name = getattr(skill, "name", None) or getattr(skill, "id", "?")
            skill_desc = (getattr(skill, "description", "") or "").strip()
            lines.append(f"  - {skill_name}: {skill_desc}")
            examples = list(getattr(skill, "examples", None) or [])
            for example in examples[:2]:
                lines.append(f"      e.g. {example}")

    lines.append("")
    lines.append(
        "Pass ONE self-contained instruction. The peer agent shares no "
        "conversation state with you, so include every identifier it needs "
        "(traveller id, booking reference, stations, date) and what the "
        "traveller wants to happen."
    )
    return "\n".join(lines)


def tool_name_for(agent_key: str) -> str:
    """Delegation tool name for a sub-agent key."""
    safe = "".join(c if c.isalnum() else "_" for c in agent_key).strip("_")
    return f"ask_{safe.lower()}"


def _delegation_tool(agent_key: str, card: Any) -> Any:
    """Build one LangChain tool that forwards a request to this sub-agent."""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    from common import a2a_client
    from common.config import a2a_base_url

    class DelegationInput(BaseModel):
        request: str = Field(
            description=(
                "A complete, self-contained instruction for the peer agent, "
                "including every identifier it needs."
            )
        )

    async def delegate(request: str) -> str:
        auth, headers = a2a_client.auth_for()
        result = await a2a_client.ask_agent(
            a2a_base_url(agent_key), request, auth=auth, headers=headers
        )
        return a2a_client.render(result)

    return StructuredTool.from_function(
        coroutine=delegate,
        name=tool_name_for(agent_key),
        description=describe_card(card),
        args_schema=DelegationInput,
    )


async def discover_delegation_tools() -> tuple[list[Any], list[str]]:
    """Discover every registered sub-agent and return (tools, skill_lines).

    Cards are fetched concurrently: on a cold deployment several Tier-2
    runtimes may be starting at once, and doing this serially would add their
    cold starts together.

    The second return value is a human-readable summary for the startup log,
    so an operator can see what the router actually discovered rather than
    what it was supposed to.
    """
    keys = registered_agents()
    cards = await asyncio.gather(*(fetch_card(k) for k in keys))

    tools: list[Any] = []
    summary: list[str] = []
    for agent_key, card in zip(keys, cards, strict=True):
        if card is None:
            summary.append(f"  {agent_key}: UNREACHABLE -- not routable this run")
            continue
        tools.append(_delegation_tool(agent_key, card))
        skills = list(getattr(card, "skills", None) or [])
        skill_names = ", ".join(
            str(getattr(s, "id", None) or getattr(s, "name", "?")) for s in skills
        )
        summary.append(
            f"  {agent_key}: {tool_name_for(agent_key)} "
            f"<- {len(skills)} skill(s) [{skill_names}]"
        )

    if not tools:
        logger.error(
            "discovered NO sub-agents from registry %s. The router has nothing "
            "to delegate to and can only answer that it cannot help.", keys
        )
    return tools, summary
