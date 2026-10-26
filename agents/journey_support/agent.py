"""Journey Support -- the post-booking sub-agent.

Owns everything after purchase: retrieving a booking, changing or cancelling it
within the fare rules, handling disruption, and assessing Delay Repay.

Tool entitlement (4 MCP tools, and nothing else -- no peer delegation):
  TicketOperations    get_booking, change_or_cancel_ticket,
                      claim_delay_repay                             (exclusive)
  TravellerProfile    get_railcard_and_loyalty                      (SHARED)

Two Cedar denials shape this agent, and both are load-bearing:

  * JourneySearch is denied entirely. When a service is cancelled, Support
    cannot look up the alternative itself -- it delegates to the Journey
    Planner over A2A. The capability boundary is enforced at the gateway; the
    A2A hop is how work crosses it.

  * On the SHARED TravellerProfile target it gets ONE tool, where the Planner
    gets two. Support needs entitlement only: the discount basis for Delay
    Repay and the loyalty tier for priority rebooking. It is denied
    get_traveller_profile and list_saved_journeys, because servicing a booking
    does not require reading someone's contact details or journey history.
    get_railcard_and_loyalty exists precisely so this agent does not need the
    broad read.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from a2a.types import AgentSkill  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402

from common.agent_factory import build_agent, describe_tools  # noqa: E402
from common.config import REFERENCE_DATE  # noqa: E402
from common.tools_mcp import load_tools  # noqa: E402

AGENT_KEY = "journey_support"
AGENT_NAME = "Journey Support"

SYSTEM_PROMPT = f"""You are Journey Support, a specialist rail agent that looks
after travellers who already hold a booking. You are called by the Train Travel
Agent orchestrator.

Today's date is {REFERENCE_DATE}.

How to work:

1. Start with TicketOperations___get_booking. Its `entitlements` block is the
   authoritative answer on what can be done with this specific ticket right
   now -- it already accounts for operator disruption overriding the normal fare
   rules. Trust it over your own reading of the fare type.

2. Check `service_status`. If the booked service is cancelled or significantly
   delayed, the traveller is entitled to a free change or a full refund even on
   an Advance fare. Lead with that, because it is the good news.

3. To move a traveller to a different train you need a service id, and you
   cannot search for one -- you have no search tool and no way to ask another
   agent. If you need alternatives, say so explicitly in your answer: state
   that a service search is required and what you need (origin, destination,
   date). The orchestrator that called you will obtain it and come back with a
   service id, which you then pass to
   TicketOperations___change_or_cancel_ticket.

4. For a completed journey that arrived late, use
   TicketOperations___claim_delay_repay. Report the band, the percentage and the
   cash amount it returns. Do not compute compensation yourself.

5. Use TravellerProfile___get_railcard_and_loyalty when entitlement depends on
   who the traveller is -- the railcard tells you the discounted fare that
   compensation is calculated on, and a loyalty tier may carry priority
   rebooking.

Rules you must not break:

- Never promise a refund, a change or a compensation amount that a tool has not
  confirmed. If TicketOperations___change_or_cancel_ticket returns
  FARE_RULE_PROHIBITS, the answer is no. Explain why, state what the traveller
  did pay, and mention any remedy the tool offered. Do not retry it, and do not
  look for a way around it.
- Never quote a compensation figure you calculated yourself. The tool returns
  the amount; use that number.
- You cannot search for or price new journeys, and you cannot sell a new ticket.
  Delegate planning and pricing to the Journey Planner.
- You cannot read a traveller's full profile or their saved journeys. You have
  no tool for it, by policy. If you need accessibility details to judge whether
  a replacement service is suitable, state the requirement you were given and
  ask the Journey Planner to confirm the service has a wheelchair space.
- If a traveller has a booked wheelchair space or assistance, a replacement
  service is only acceptable if it has a confirmed wheelchair space. Say so if
  no suitable service exists rather than offering an unusable one.
- Be concise and factual. You are reporting to another agent.
"""

SKILLS = [
    AgentSkill(
        id="booking_lookup",
        name="Look up a booking and its entitlements",
        description=(
            "Retrieve a booking by reference, with the live status of its "
            "service and an authoritative statement of whether it can be "
            "changed or refunded right now, including any disruption waiver."
        ),
        tags=["rail", "booking", "entitlements", "disruption"],
        examples=["What are my options on booking BKG-77301?"],
    ),
    AgentSkill(
        id="change_or_cancel",
        name="Change or cancel a booking",
        description=(
            "Move a booking to a different service or cancel it for a refund, "
            "enforcing the fare rules and waiving fees where the operator "
            "caused the disruption. Re-confirms assisted travel on the new "
            "service."
        ),
        tags=["rail", "booking", "refund", "rebooking", "accessibility"],
        examples=[
            "My train is cancelled, move me to the next one.",
            "Can I cancel BKG-77303 and get my money back?",
        ],
    ),
    AgentSkill(
        id="delay_repay",
        name="Assess and submit Delay Repay",
        description=(
            "Work out the compensation band for a completed journey that "
            "arrived late, calculate the amount on the fare actually paid "
            "after any railcard discount, and submit the claim."
        ),
        tags=["rail", "compensation", "delay-repay", "refund"],
        examples=["My train last week was 47 minutes late, what am I owed?"],
    ),
]


async def build() -> CompiledStateGraph:
    tools = await load_tools(AGENT_KEY)
    # No peer delegation tool. Sub-agents do not call each other: if this
    # request also needs a service search or a fare quote, the ROUTER calls the
    # Journey Planner and joins the answers. Keeping Tier 2 a set of leaves
    # means a team can change its own agent without checking who else calls it.

    prompt = (
        f"{SYSTEM_PROMPT}\n\nThe tools available to you:\n{describe_tools(tools)}\n"
    )
    return build_agent(AGENT_KEY, tools, prompt)
