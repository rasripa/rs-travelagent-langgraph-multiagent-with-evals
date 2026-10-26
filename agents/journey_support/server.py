"""A2A server entrypoint for the Journey Support sub-agent.

Local:      uvicorn on 127.0.0.1:9102
AgentCore:  uvicorn on 0.0.0.0:9000, which is the A2A protocol contract port

This agent is both an A2A server (the orchestrator calls it) and an A2A client
(it calls the Journey Planner for alternatives it cannot search for itself).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.journey_support.agent import (  # noqa: E402
    AGENT_KEY,
    AGENT_NAME,
    SKILLS,
    build,
)
from common.a2a_server import (  # noqa: E402
    LangGraphAgentExecutor,
    advertised_url,
    build_a2a_app,
    build_agent_card,
    serve,
)
from common import observability  # noqa: E402
from common.config import A2A_PORTS  # noqa: E402

DESCRIPTION = (
    "Looks after travellers who already hold a booking. Retrieves bookings and "
    "their live entitlements, changes or cancels within the fare rules, handles "
    "disruption rebooking including assisted travel, and assesses Delay Repay "
    "compensation. Cannot search for or price new journeys."
)


def public_url() -> str:
    return advertised_url("JOURNEY_SUPPORT_A2A_URL", A2A_PORTS[AGENT_KEY])


def main() -> None:
    print(f"starting {AGENT_NAME} (A2A)")
    observability.init(AGENT_NAME)
    agent = asyncio.run(build())

    card = build_agent_card(
        name=AGENT_NAME,
        description=DESCRIPTION,
        url=public_url(),
        skills=SKILLS,
    )
    app = build_a2a_app(card, LangGraphAgentExecutor(agent, AGENT_NAME))

    print(f"  agent card: {public_url()}.well-known/agent-card.json")
    serve(app, local_port=A2A_PORTS[AGENT_KEY])


if __name__ == "__main__":
    main()
