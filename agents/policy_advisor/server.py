"""A2A server entrypoint for the Policy Advisor sub-agent.

Local:      uvicorn on 127.0.0.1:9103
AgentCore:  uvicorn on 0.0.0.0:9000, which is the A2A protocol contract port
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.policy_advisor.agent import (  # noqa: E402
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
    "Answers questions about rail travel rules from the published policy "
    "knowledge base: fare conditions and refund eligibility, railcard "
    "restrictions, Delay Repay compensation bands, ticket validity, luggage "
    "and accessibility policy. Holds no traveller data, bookings or timetable, "
    "and cannot act on a specific booking."
)


def public_url() -> str:
    return advertised_url("POLICY_ADVISOR_A2A_URL", A2A_PORTS[AGENT_KEY])


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
