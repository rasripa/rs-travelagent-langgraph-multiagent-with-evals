"""Shared configuration for the Train Travel Agent demo.

Everything is overridable by environment variable so the same code runs in
local mode and on AgentCore without edits.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- AWS / region
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

# ---------------------------------------------------------------- models
# These are now resolved DYNAMICALLY through common/config_store.py: an
# explicit env var wins, then Parameter Store (when TRAVEL_CONFIG_SOURCE=ssm),
# then the built-in default. Changing the parameter repoints a running
# deployment without a redeploy.
#
# Claude via a cross-Region inference profile. The current Sonnet models have
# no In-Region endpoint, so the geo-prefixed profile ID is required -- "us."
# for us-west-2.
#
# Alternatives (same "us." prefix):
#   us.anthropic.claude-sonnet-5-5
#   us.anthropic.claude-sonnet-5
#   us.anthropic.claude-haiku-5-5            (cheaper/faster, weaker tool use)
def model_id() -> str:
    from common import config_store

    return config_store.require("model_id")


def embed_model_id() -> str:
    """Knowledge-base embeddings. Titan v2 is available in us-west-2."""
    from common import config_store

    return config_store.require("embed_model_id")

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "fixtures"
KB_DIR = ROOT / "knowledgebase"
KB_INDEX_PATH = ROOT / "knowledgebase" / ".kb_index.json"

# ---------------------------------------------------------------- determinism
# The demo must never depend on the real clock. All timetable and booking
# fixtures are anchored to this date; "Friday" in the demo script is 2026-11-06.
REFERENCE_DATE = os.getenv("TRAVEL_REFERENCE_DATE", "2026-11-02")  # a Monday

# ---------------------------------------------------------------- local mode
# Loopback only. These are developer-mode listeners and must not be published
# beyond the local host; AgentCore mode replaces them with managed runtimes.
LOCAL_BIND_HOST = os.getenv("TRAVEL_LOCAL_BIND", "127.0.0.1")

MCP_PORTS = {
    "traveller_profile": int(os.getenv("PORT_TRAVELLER_PROFILE", "8101")),
    "journey_search": int(os.getenv("PORT_JOURNEY_SEARCH", "8102")),
    "ticket_operations": int(os.getenv("PORT_TICKET_OPERATIONS", "8103")),
}

# A2A servers. Port 9000 is the AgentCore A2A contract port; locally we need
# distinct ports, so only the deployed containers bind 9000.
A2A_PORTS = {
    "journey_planner": int(os.getenv("PORT_JOURNEY_PLANNER", "9101")),
    "journey_support": int(os.getenv("PORT_JOURNEY_SUPPORT", "9102")),
    "policy_advisor": int(os.getenv("PORT_POLICY_ADVISOR", "9103")),
}

# No peer-A2A feature flag any more. Sub-agents do not call each other at all:
# every hop is router -> sub-agent -> router. Removing the flag removes the
# possibility of a deployment where the topology is quietly different.

# "local" uses the policy shim in common/policy_shim.py; "agentcore" relies on
# the real Cedar policy engine at the gateway.
RUN_MODE = os.getenv("TRAVEL_RUN_MODE", "local")

# ---------------------------------------------------------------- agent identity
# Which TOOLS each agent is entitled to, per gateway target. This is the single
# source of truth for the local shim, and mirrors the deployed Cedar policies
# exactly.
#
# Note the asymmetry on TravellerProfile, the SHARED server. Both sub-agents
# reach the same target through the same gateway endpoint, and they receive
# DIFFERENT tool catalogues from it:
#
#   journey_planner -> 2 tools   it prices fares (needs the railcard) and picks
#                               services (needs accessibility), and it resolves
#                               vague requests like "my usual trip"
#   journey_support -> 1 tool    it only needs entitlement: the discount basis
#                               for Delay Repay and the loyalty tier for
#                               priority rebooking. It has no business reading
#                               a traveller's full profile or journey history.
#
# get_railcard_and_loyalty exists precisely SO THAT Support does not need the
# broad get_traveller_profile. That is privilege minimisation with a reason,
# not an arbitrary cut -- and it is what makes the tools/list demo land.
AGENT_TOOL_GRANTS = {
    "journey_planner": {
        "JourneySearch": ["search_journeys", "get_fares", "book_ticket"],
        "TravellerProfile": ["get_traveller_profile", "list_saved_journeys"],
    },
    "journey_support": {
        "TicketOperations": [
            "get_booking",
            "change_or_cancel_ticket",
            "claim_delay_repay",
        ],
        "TravellerProfile": ["get_railcard_and_loyalty"],
    },
    # The policy knowledge base belongs to a Tier-2 agent, not to the router.
    # The target name is hardcoded rather than configurable because the Cedar
    # policies name it too, and the two must not drift.
    "policy_advisor": {
        "RailPolicyKB": ["Retrieve"],
    },
    # The ROUTER HOLDS NO GATEWAY TOOLS AT ALL. This empty grant is the
    # architecture, not an oversight: Tier 1 classifies intent and delegates,
    # so it has no business reaching a tool server. Cedar is default-deny, so
    # an empty grant means every tool call it attempted would be refused.
    "train_travel_agent": {},
}

# ---------------------------------------------------------------- A2A endpoints
# Local mode resolves to loopback. AgentCore mode needs the deployed runtime
# endpoint, which deploy/agentcore/deploy.py writes into deploy/agentcore/.env
# as JOURNEY_PLANNER_A2A_URL / JOURNEY_SUPPORT_A2A_URL.
def a2a_base_url(agent_key: str) -> str:
    """Base URL for a sub-agent's A2A endpoint.

    Resolved dynamically: env var, then Parameter Store, then loopback. That
    means repointing a sub-agent does not need the caller redeployed.
    """
    from common import config_store

    override = config_store.get(f"{agent_key}_a2a_url")
    if override:
        return override.rstrip("/")
    if RUN_MODE == "agentcore":
        raise RuntimeError(
            f"TRAVEL_RUN_MODE=agentcore but neither {agent_key.upper()}_A2A_URL "
            f"nor {config_store.parameter_name(f'{agent_key}_a2a_url')} is set."
        )
    return f"http://{LOCAL_BIND_HOST}:{A2A_PORTS[agent_key]}"
