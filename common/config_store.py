"""Dynamic configuration, resolved at use time rather than frozen at startup.

Resolution order for every key:

  1. An explicit environment variable  -- escape hatch, and how local mode and
                                          non-AWS hosting work unchanged
  2. AWS Systems Manager Parameter Store -- when TRAVEL_CONFIG_SOURCE=ssm
  3. A built-in default

Why a store at all rather than more environment variables:

  * DYNAMIC. Values are re-read on a short TTL, so changing a parameter takes
    effect in a running container without a redeploy or a new runtime version.
    Swapping TRAVEL_MODEL_ID is the obvious case -- repoint the model for all
    six runtimes with one put_parameter.
  * ONE WRITER, MANY READERS. The gateway-URL outage came from six env blocks
    that each had to be right, and one of them was not. A shared parameter
    path cannot express that asymmetry.

Why environment variables still win when set: they keep this module optional.
With TRAVEL_CONFIG_SOURCE unset there is no SSM call, no IAM requirement and no
AWS dependency, so the local stack and any non-AWS host behave exactly as
before. Only the deployed runtimes opt in.

The TTL matters: ssm:GetParameter is rate-limited per account, and these
values are read on the hot path (the model is resolved on every agent
invocation). The cache keeps that to one call per key per TTL window, while
still being "dynamic" on a timescale that matters to an operator.
"""
from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# Parameter path prefix. Keys below are appended, e.g. /travel-demo/model-id
DEFAULT_PREFIX = "/travel-demo"

# How long a resolved value is reused before SSM is consulted again.
DEFAULT_TTL_SECONDS = 60

# key -> (environment variable, SSM parameter name, built-in default)
#
# The env var names are unchanged from before this module existed, so nothing
# that already sets them has to change.
SPEC: dict[str, tuple[str, str, str | None]] = {
    "model_id": (
        "TRAVEL_MODEL_ID",
        "model-id",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ),
    "embed_model_id": (
        "TRAVEL_EMBED_MODEL_ID",
        "embed-model-id",
        "amazon.titan-embed-text-v2:0",
    ),
    "gateway_url": ("TRAVEL_GATEWAY_URL", "gateway-url", None),
    "memory_id": ("TRAVEL_MEMORY_ID", "memory-id", None),
    "journey_planner_a2a_url": (
        "JOURNEY_PLANNER_A2A_URL",
        "journey-planner-a2a-url",
        None,
    ),
    "journey_support_a2a_url": (
        "JOURNEY_SUPPORT_A2A_URL",
        "journey-support-a2a-url",
        None,
    ),
    "policy_advisor_a2a_url": (
        "POLICY_ADVISOR_A2A_URL",
        "policy-advisor-a2a-url",
        None,
    ),
    # The sub-agent REGISTRY: a comma-separated list of agent keys the router
    # discovers at startup. This is the one value that has to change to add a
    # Tier-2 agent, and it is read at use time -- so a new agent becomes
    # routable without touching the router's code or redeploying it.
    "sub_agents": (
        "TRAVEL_SUB_AGENTS",
        "sub-agents",
        "journey_planner,journey_support,policy_advisor",
    ),
}

_LOCK = threading.Lock()
_CACHE: dict[str, tuple[str | None, float]] = {}
_client = None
_warned: set[str] = set()


def source() -> str:
    """'ssm' or 'env'. Defaults to env, so SSM is strictly opt-in."""
    return os.getenv("TRAVEL_CONFIG_SOURCE", "env").strip().lower()


def prefix() -> str:
    return os.getenv("TRAVEL_CONFIG_PREFIX", DEFAULT_PREFIX).rstrip("/")


def ttl_seconds() -> int:
    try:
        return int(os.getenv("TRAVEL_CONFIG_TTL", str(DEFAULT_TTL_SECONDS)))
    except ValueError:
        return DEFAULT_TTL_SECONDS


def parameter_name(key: str) -> str:
    _, suffix, _ = SPEC[key]
    return f"{prefix()}/{suffix}"


def _ssm():
    global _client
    if _client is None:
        import boto3

        from common.config import AWS_REGION

        _client = boto3.client("ssm", region_name=AWS_REGION)
    return _client


def _from_ssm(key: str) -> str | None:
    try:
        response = _ssm().get_parameter(Name=parameter_name(key))
        return response["Parameter"]["Value"]
    except Exception as exc:  # noqa: BLE001
        # Never fail a request because a parameter is missing -- fall through
        # to the default. Warn once per key so logs stay readable.
        if key not in _warned:
            _warned.add(key)
            logger.warning(
                "could not read %s from SSM (%s: %s); using the default",
                parameter_name(key), type(exc).__name__, exc,
            )
        return None


def get(key: str, default: str | None = None) -> str | None:
    """Resolve a config key. Cheap enough to call on the hot path."""
    if key not in SPEC:
        raise KeyError(f"unknown config key {key!r}. Known: {sorted(SPEC)}")

    env_name, _, builtin = SPEC[key]

    # 1. Explicit environment variable always wins.
    value = os.getenv(env_name)
    if value:
        return value

    # 2. SSM, when opted in, behind a TTL cache.
    if source() == "ssm":
        now = time.monotonic()
        with _LOCK:
            cached = _CACHE.get(key)
            if cached and cached[1] > now:
                if cached[0] is not None:
                    return cached[0]
            else:
                cached = None
        if cached is None:
            resolved = _from_ssm(key)
            with _LOCK:
                _CACHE[key] = (resolved, now + ttl_seconds())
            if resolved:
                return resolved

    # 3. Built-in default.
    return default if default is not None else builtin


def require(key: str) -> str:
    """Resolve a key that has no sensible default."""
    value = get(key)
    if not value:
        env_name, _, _ = SPEC[key]
        raise RuntimeError(
            f"{key} is not configured. Set {env_name}, or publish "
            f"{parameter_name(key)} to Parameter Store and set "
            f"TRAVEL_CONFIG_SOURCE=ssm."
        )
    return value


def invalidate(key: str | None = None) -> None:
    """Drop cached values so the next read goes back to SSM."""
    with _LOCK:
        if key is None:
            _CACHE.clear()
        else:
            _CACHE.pop(key, None)


def describe() -> str:
    """One-line status for startup output."""
    if source() != "ssm":
        return "config: environment variables"
    return f"config: Parameter Store {prefix()}/* (TTL {ttl_seconds()}s), env overrides win"


def snapshot() -> dict[str, str]:
    """Resolve everything, reporting where each value came from. Diagnostics."""
    out = {}
    for key in SPEC:
        env_name, _, _ = SPEC[key]
        if os.getenv(env_name):
            origin = f"env:{env_name}"
        elif source() == "ssm":
            origin = f"ssm:{parameter_name(key)}"
        else:
            origin = "default"
        value = get(key)
        out[key] = f"{value or '(unset)'}   <- {origin}"
    return out
