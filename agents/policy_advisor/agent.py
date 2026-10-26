"""Policy Advisor -- the policy and FAQ sub-agent.

Owns the rules: fare conditions, refund and change eligibility, railcard
restrictions, Delay Repay bands, luggage and accessibility policy. It answers
from the knowledge base and nothing else -- it holds no traveller data, no
bookings and no timetable.

Tool entitlement (1 tool):
  RailPolicyKB    Retrieve                                        (exclusive)

Why this is a sub-agent and not a tool on the router
----------------------------------------------------
The knowledge base used to hang off the orchestrator, which meant the router
answered policy questions itself. That made Tier 1 a domain participant: it
owned rail-specific knowledge, it needed a rail-specific prompt, and any change
to how policy is answered was a change to the router. A router that is also a
domain expert cannot be owned by a platform team and handed new domains by
other teams.

Moving it here costs one extra A2A hop on policy questions and buys a router
whose only job is routing. That is the trade the architecture asks for.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from a2a.types import AgentSkill  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402

from common.agent_factory import build_agent, describe_tools  # noqa: E402
from common.config import REFERENCE_DATE  # noqa: E402
from common.kb_tool import load_kb_tools  # noqa: E402

AGENT_KEY = "policy_advisor"
AGENT_NAME = "Policy Advisor"

SYSTEM_PROMPT = f"""You are the Policy Advisor agent, a specialist that
answers questions about rail travel RULES. You are called by the Train Travel
Agent orchestrator.

Today's date is {REFERENCE_DATE}.

What you own:
- Fare conditions: which ticket types are refundable, changeable, or tied to a
  specific service, and what fees apply.
- Railcard rules: discounts, minimum fares, time restrictions.
- Delay Repay: the compensation bands and what qualifies.
- Ticket validity, luggage allowances, and accessibility policy.

How to answer:
- ALWAYS retrieve before answering. These are published rules, not things to
  recall from memory, and a wrong refund rule is a customer-harm bug.
- Quote the rule and the condition attached to it. "Advance tickets are
  non-refundable, except where the operator cancels the service" is useful;
  "it depends" is not.
- If the knowledge base does not cover it, say so plainly. Do not infer a
  policy that might be wrong.
- State percentages and thresholds exactly as the source gives them. Do not
  compute a money amount for a specific booking -- you do not have booking
  data, and the agent that does will calculate it.

What you must NOT do:
- Do not look up a traveller, a booking, a fare quote or a timetable. You have
  no tools for any of that and no access to traveller data. If asked, state
  that you only hold the rules and let the caller route it elsewhere.
- Do not promise an outcome on a specific booking. You explain the rule; acting
  on it belongs to another agent.
"""

SKILLS = [
    AgentSkill(
        id="explain_fare_rules",
        name="Explain fare and refund rules",
        description=(
            "Explain whether a ticket type can be refunded or changed, what "
            "fee applies, what restrictions are attached, and the exceptions "
            "that apply when an operator cancels or severely delays a service."
        ),
        tags=["policy", "fares", "refunds", "rules"],
        examples=[
            "I booked an Advance ticket. Can I get a refund if I change my mind?",
            "What is the admin fee on an Off-Peak refund?",
        ],
    ),
    AgentSkill(
        id="explain_compensation",
        name="Explain Delay Repay compensation",
        description=(
            "Explain the Delay Repay bands, what counts as a qualifying delay, "
            "what proportion of the fare paid each band returns, and how "
            "cancellation and disruption affect entitlement."
        ),
        tags=["policy", "compensation", "delay-repay"],
        examples=[
            "How much compensation do I get for a 45 minute delay?",
            "Does a cancellation qualify for Delay Repay?",
        ],
    ),
    AgentSkill(
        id="explain_railcards_and_validity",
        name="Explain railcards, ticket validity and accessibility policy",
        description=(
            "Explain railcard discounts and their time restrictions and "
            "minimum fares, when a ticket is valid on which services, luggage "
            "allowances, and the assistance available to passengers with "
            "accessibility requirements."
        ),
        tags=["policy", "railcards", "validity", "accessibility", "luggage"],
        examples=[
            "What time restrictions apply to a Two Together Railcard?",
            "How much luggage can I take, and how do I book assistance?",
        ],
    ),
]


async def build() -> CompiledStateGraph:
    tools = await load_kb_tools(AGENT_KEY)
    prompt = (
        f"{SYSTEM_PROMPT}\n\nThe tools available to you:\n{describe_tools(tools)}\n"
    )
    return build_agent(AGENT_KEY, tools, prompt)
