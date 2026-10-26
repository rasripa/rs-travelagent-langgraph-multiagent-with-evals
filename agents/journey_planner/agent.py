"""Journey Planner -- the pre-travel sub-agent.

Owns everything up to and including the moment of purchase: what services run,
what they cost, and committing the booking.

Tool entitlement (5 tools, well inside the 5-10 band where selection stays
accurate):
  JourneySearch       search_journeys, get_fares, book_ticket      (exclusive)
  TravellerProfile    get_traveller_profile, list_saved_journeys   (SHARED)

It is denied TicketOperations entirely by Cedar policy -- a planning agent that
could cancel tickets is a blast radius, not a feature.

Note what it gets on the SHARED TravellerProfile target: two tools. Journey
Support, hitting the same target through the same gateway endpoint, sees only
one. The Planner needs the full profile because the railcard drives pricing and
the accessibility requirements drive which services are even usable.
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

AGENT_KEY = "journey_planner"
AGENT_NAME = "Journey Planner"

SYSTEM_PROMPT = f"""You are the Journey Planner, a specialist rail agent that
plans and books train travel. You are called by the Train Travel Agent
orchestrator, and sometimes by the Journey Support agent when it needs an
alternative service during disruption.

Today's date is {REFERENCE_DATE}.

How to work:

1. If you are given a traveller id, call TravellerProfile___get_traveller_profile
   FIRST. It returns the railcard, which changes every price you are about to
   quote, and the accessibility requirements, which change which services are
   even usable. Never quote a fare before you know whether a railcard applies.
   Use TravellerProfile___list_saved_journeys to resolve a vague request such as
   "my usual trip to Manchester" into a concrete origin and destination.

2. Use JourneySearch___search_journeys to find services. Check the `status`
   field on every result:
   - NEVER recommend a service whose status is "cancelled". Say it is cancelled.
   - If a service is delayed, say so and give the delay in minutes.
   - If the traveller needs a wheelchair space, only offer services where
     `wheelchair_spaces_available` is greater than zero.

3. Use JourneySearch___get_fares to price the options, passing the railcard code
   from the profile you fetched in step 1. Report the actual price payable, not
   the undiscounted price. If a fare is flagged `railcard_time_restricted`,
   mention that the restriction applies to that fare type.

4. Only call JourneySearch___book_ticket when the traveller has clearly chosen a
   service and a fare type. Before booking an Advance fare, state plainly that
   it is tied to that one train and is neither refundable nor changeable.

Rules you must not break:

- Never invent a service, a price, a time or a booking reference. Every factual
  claim must come from a tool result.
- If a tool returns ok=false, report what it said. Do not retry the same call
  with the same arguments, and do not work around a refusal.
- You cannot change, cancel or refund an existing booking, and you cannot
  process compensation. You have no tools for it. If asked, say that it is the
  Journey Support agent's responsibility.
- Keep answers short and concrete: times, prices, service ids. You are talking
  to another agent, not to the traveller, so skip the pleasantries and give it
  the facts it needs.
"""

SKILLS = [
    AgentSkill(
        id="search_services",
        name="Search rail services",
        description=(
            "Find train services between two stations on a date, with live "
            "status, journey time, changes, seat and wheelchair-space "
            "availability, and any disruption affecting the route."
        ),
        tags=["rail", "timetable", "search", "disruption"],
        examples=[
            "What trains run from London Euston to Manchester on 2026-11-06?",
            "Find an alternative to the cancelled 07:15 to Manchester.",
        ],
    ),
    AgentSkill(
        id="price_fares",
        name="Price fares with railcard discounts",
        description=(
            "Quote the Advance, Off-Peak and Anytime fares available on a "
            "route, applying the traveller's railcard discount and flagging "
            "refundability, changeability and any time restriction."
        ),
        tags=["rail", "fares", "pricing", "railcard"],
        examples=[
            "What does London to Manchester cost for traveller TRV-1001?",
            "Price the cheapest flexible fare to Birmingham.",
        ],
    ),
    AgentSkill(
        id="book_ticket",
        name="Book a ticket",
        description=(
            "Commit a booking on a chosen service and fare type, applying any "
            "railcard, and return the booking reference."
        ),
        tags=["rail", "booking", "purchase"],
        examples=["Book TRV-1001 onto the 07:45 to Manchester on an Advance fare."],
    ),
]


async def build() -> CompiledStateGraph:
    tools = await load_tools(AGENT_KEY)
    prompt = (
        f"{SYSTEM_PROMPT}\n\nThe tools available to you:\n{describe_tools(tools)}\n"
    )
    return build_agent(AGENT_KEY, tools, prompt)
