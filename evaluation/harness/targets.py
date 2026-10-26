"""Target adapters: one interface for invoking anything in the system.

    output = await run_case(case)

Both runners call exactly this, so the only difference between them is which
judge scores the result -- never how the system was driven.

Every path goes through a REAL public interface of the system under test. The
orchestrator is called over HTTP POST /invocations, exactly as scripts/demo.py
does it; a sub-agent is called over A2A by reusing the application's own
`common.a2a_client.ask_agent`. Nothing in the application is modified, patched
or monkeypatched: the harness imports core modules read-only and otherwise
talks to open ports.

Modes, selected by the EVAL_MODE environment variable (default "local"):

  local       orchestrator on 127.0.0.1:8090, sub-agents on 127.0.0.1:9101-9103
  agentcore   orchestrator via boto3 bedrock-agentcore invoke_agent_runtime,
              sub-agents via their deployed A2A URLs with SigV4
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.harness import perf, trajectory  # noqa: E402

LOCAL_ORCHESTRATOR_URL = os.getenv("EVAL_ORCHESTRATOR_URL", "http://127.0.0.1:8090")
LOCAL_A2A_PORTS = {
    "journey_planner": int(os.getenv("PORT_JOURNEY_PLANNER", "9101")),
    "journey_support": int(os.getenv("PORT_JOURNEY_SUPPORT", "9102")),
    "policy_advisor": int(os.getenv("PORT_POLICY_ADVISOR", "9103")),
}

# This subsystem's target name -> the agent key the core application uses.
# "policy_advisor" is a clearer name for the agent that owns the rail policy
# and FAQ knowledge base, but the application keys it "policy_advisor" in
# common/config.py (A2A_PORTS, AGENT_TOOL_GRANTS) and in
# common/a2a_discovery.py (DEFAULT_REGISTRY). Renaming it there would be a
# change to the core application, which is out of scope for the evaluation
# subsystem, so the two names are reconciled here instead.
AGENT_KEYS = {
    "journey_planner": "journey_planner",
    "journey_support": "journey_support",
    "policy_advisor": "policy_advisor",
}

DISPLAY_NAMES = {
    "train_travel_agent": "Train Travel Agent (router)",
    "journey_planner": "Journey Planner",
    "journey_support": "Journey Support",
    "policy_advisor": "Rail Policy Advisor",
}

DEPLOY_STATE_PATH = PROJECT_ROOT / "deploy" / "agentcore" / ".deploy-state.json"

ORCHESTRATOR_TIMEOUT = float(os.getenv("EVAL_TIMEOUT_SECONDS", "300"))
# botocore defaults to a 60s read timeout, which is below what one full
# multi-agent turn costs once cold starts and a fan-out are in play. The server
# finishes; the client gives up first and reports what looks like a hang.
AGENTCORE_READ_TIMEOUT = 300


def mode() -> str:
    value = os.getenv("EVAL_MODE", "local").strip().lower()
    if value not in ("local", "agentcore"):
        raise ValueError(f"EVAL_MODE must be 'local' or 'agentcore', got {value!r}")
    return value


def _deploy_state() -> dict[str, Any]:
    if not DEPLOY_STATE_PATH.exists():
        return {}
    try:
        return json.loads(DEPLOY_STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def resolve_a2a_url(target: str) -> str:
    """Base URL for a sub-agent, for the current mode."""
    agent_key = AGENT_KEYS[target]
    if mode() == "local":
        return f"http://127.0.0.1:{LOCAL_A2A_PORTS[agent_key]}"

    override = os.getenv(f"{agent_key.upper()}_A2A_URL")
    if override:
        return override.rstrip("/")

    runtimes = _deploy_state().get("agent_runtimes", {})
    url = (runtimes.get(agent_key) or {}).get("url")
    if url:
        return url.rstrip("/")

    raise RuntimeError(
        f"EVAL_MODE=agentcore but no deployed A2A URL for {target!r} "
        f"(agent key {agent_key!r}). Set {agent_key.upper()}_A2A_URL, or deploy "
        f"the agent so {DEPLOY_STATE_PATH.name} carries "
        f"agent_runtimes.{agent_key}.url."
    )


def orchestrator_arn() -> str:
    arn = os.getenv("TRAVEL_ORCHESTRATOR_ARN")
    if arn:
        return arn
    runtimes = _deploy_state().get("agent_runtimes", {})
    arn = (runtimes.get("train_travel_agent") or {}).get("arn")
    if arn:
        return arn
    raise RuntimeError(
        "EVAL_MODE=agentcore but no orchestrator ARN. Set "
        "TRAVEL_ORCHESTRATOR_ARN or deploy the orchestrator."
    )


def preflight() -> dict[str, Any]:
    """Report which targets look reachable, before spending money on judges."""
    import httpx

    report: dict[str, Any] = {"mode": mode(), "targets": {}}
    if mode() != "local":
        report["note"] = (
            "agentcore mode: reachability is not probed, because an "
            "InvokeAgentRuntime call is a billed agent turn."
        )
        return report

    try:
        response = httpx.get(f"{LOCAL_ORCHESTRATOR_URL}/ping", timeout=5.0)
        report["targets"]["train_travel_agent"] = (
            "up" if response.status_code == 200 else f"HTTP {response.status_code}"
        )
    except Exception as exc:  # noqa: BLE001
        report["targets"]["train_travel_agent"] = f"down ({type(exc).__name__})"

    for target in AGENT_KEYS:
        url = resolve_a2a_url(target)
        try:
            response = httpx.get(f"{url}/.well-known/agent-card.json", timeout=5.0)
            report["targets"][target] = (
                "up" if response.status_code == 200 else f"HTTP {response.status_code}"
            )
        except Exception as exc:  # noqa: BLE001
            report["targets"][target] = f"down ({type(exc).__name__})"
    return report


async def _call_orchestrator_local(
    prompt: str, actor_id: str, session_id: str
) -> tuple[str, str | None, float]:
    import httpx

    payload = {"prompt": prompt, "actor_id": actor_id, "session_id": session_id}
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=ORCHESTRATOR_TIMEOUT) as client:
            response = await client.post(
                f"{LOCAL_ORCHESTRATOR_URL}/invocations", json=payload
            )
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}: {exc}", time.monotonic() - started

    transport = time.monotonic() - started
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        return "", f"HTTP {response.status_code}: {response.text[:400]}", transport

    if response.status_code != 200 or body.get("status") != "success":
        return "", str(body.get("error", response.text[:400])), transport
    return body.get("response", ""), None, transport


async def _call_orchestrator_agentcore(
    prompt: str, actor_id: str, session_id: str
) -> tuple[str, str | None, float]:
    import boto3
    from botocore.config import Config

    def invoke() -> tuple[str, str | None]:
        client = boto3.client(
            "bedrock-agentcore",
            region_name=os.getenv("AWS_REGION", "us-west-2"),
            config=Config(
                read_timeout=AGENTCORE_READ_TIMEOUT,
                connect_timeout=10,
                # Retries are off deliberately: an agent invocation is neither
                # free nor idempotent. A retry re-runs the model and the tool
                # calls, and with memory enabled writes a second set of events.
                retries={"max_attempts": 0},
            ),
        )
        request: dict[str, Any] = {
            "agentRuntimeArn": orchestrator_arn(),
            "qualifier": "DEFAULT",
            "payload": json.dumps({"prompt": prompt, "actor_id": actor_id}).encode(),
        }
        # The API enforces 33-256 characters on a runtime session id. A uuid4
        # string is 36 and passes; a shorter one fails as an opaque
        # ValidationException.
        if 33 <= len(session_id) <= 256:
            request["runtimeSessionId"] = session_id

        try:
            response = client.invoke_agent_runtime(**request)
        except Exception as exc:  # noqa: BLE001
            return "", f"{type(exc).__name__}: {exc}"

        body = response["response"].read()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return body.decode(errors="replace"), None
        if parsed.get("status") != "success":
            return "", str(parsed.get("error", parsed))
        return parsed.get("response", ""), None

    started = time.monotonic()
    answer, error = await asyncio.to_thread(invoke)
    return answer, error, time.monotonic() - started


async def _call_sub_agent(
    target: str, prompt: str, session_id: str
) -> tuple[str, str | None, float]:
    """Call a Tier-2 agent over A2A, reusing the application's own client."""
    from common import a2a_client

    auth, headers = (None, {})
    if mode() == "agentcore":
        # auth_for() returns a SigV4 signer only when the application's own
        # RUN_MODE is agentcore, since with IAM the caller's identity IS its
        # execution role. Set it here so the harness signs the same way the
        # orchestrator would.
        os.environ.setdefault("TRAVEL_RUN_MODE", "agentcore")
        auth, headers = a2a_client.auth_for(session_id)

    url = resolve_a2a_url(target)
    started = time.monotonic()
    result = await a2a_client.ask_agent(
        url, prompt, auth=auth, headers=headers or None
    )
    transport = time.monotonic() - started

    if result.get("ok"):
        return result.get("answer", ""), None, transport
    return "", str(result.get("error", "unknown A2A error")), transport


async def run_case(case: dict[str, Any], capture_trajectory: bool = True) -> dict[str, Any]:
    """Invoke one case against its target and return a uniform result.

    Returns:
        {"output": str, "trajectory": list[str], "timings": dict,
         "session_id": str, "error": str | None}

    Never raises for an agent-side failure: a failed case is data the runners
    must still be able to score and report, not an exception that aborts a run
    halfway through.
    """
    target = case["target"]
    if target not in DISPLAY_NAMES:
        return {
            "output": "",
            "trajectory": [],
            "timings": {},
            "session_id": "",
            "error": f"unknown target {target!r}",
        }

    session_id = case.get("session_id") or str(uuid4())
    prompt = case["input"]

    log_before = trajectory.snapshot() if capture_trajectory and mode() == "local" else None
    watch = perf.Stopwatch()
    started_wall_ms = int(time.time() * 1000)

    try:
        if target == "train_travel_agent":
            if mode() == "local":
                answer, error, transport = await _call_orchestrator_local(
                    prompt, case["actor_id"], session_id
                )
            else:
                answer, error, transport = await _call_orchestrator_agentcore(
                    prompt, case["actor_id"], session_id
                )
        else:
            # A Tier-2 agent shares no conversation state, so the instruction
            # has to be self-contained. The traveller id is appended only when
            # the case text does not already carry it, so the dataset stays the
            # authority on what the agent was actually asked.
            instruction = prompt
            if case["actor_id"] not in prompt:
                instruction = f"{prompt}\n\n(Traveller id: {case['actor_id']})"
            answer, error, transport = await _call_sub_agent(
                target, instruction, session_id
            )
    except Exception as exc:  # noqa: BLE001
        answer, error, transport = "", f"{type(exc).__name__}: {exc}", 0.0

    total = watch.stop()
    ended_wall_ms = int(time.time() * 1000)

    observed: dict[str, Any] = {}
    trajectory_names: list[str] = []
    if log_before is not None:
        observed = trajectory.observe(log_before)
        trajectory_names = list(observed.get("trajectory", []))

    if mode() == "agentcore":
        # _agentcore_timings blocks twice: it sleeps to let spans land, then
        # calls CloudWatch synchronously. Called inline it would block the
        # single event loop that asyncio.gather uses to run the whole sweep,
        # serialising every case behind each other's span lag and making
        # --concurrency a no-op. Push it to a worker thread.
        timings = await asyncio.to_thread(
            _agentcore_timings, total, session_id, started_wall_ms, ended_wall_ms
        )
    else:
        timings = perf.local_breakdown(
            total,
            transport_seconds=transport,
            tool_calls_by_server=observed.get("tool_calls_by_server"),
        )

    if observed:
        timings["trajectory_observation"] = {
            k: v for k, v in observed.items() if k != "trajectory"
        }
        timings["trajectory_comparison"] = trajectory.compare_expected(
            case.get("expected_tools", []), observed
        )
    elif capture_trajectory:
        timings["trajectory_observation"] = {
            "available": False,
            "evidence": "none",
            "unobservable": [
                "all tool calls (trajectory capture disabled: it requires "
                "local mode and sequential execution)"
            ],
        }

    return {
        "output": answer,
        "trajectory": trajectory_names,
        "timings": timings,
        "session_id": session_id,
        "error": error,
    }


def _agentcore_timings(
    total: float, session_id: str, started_ms: int, ended_ms: int
) -> dict[str, Any]:
    """Pull the span breakdown for a deployed turn, falling back cleanly."""
    # Spans land in CloudWatch asynchronously, so widen the window at both
    # ends. Without the tail allowance the last spans of the turn are routinely
    # missed and the breakdown under-reports.
    lag_seconds = float(os.getenv("EVAL_SPAN_LAG_SECONDS", "20"))
    if lag_seconds > 0:
        time.sleep(lag_seconds)

    spans = perf.fetch_spans(started_ms - 5_000, ended_ms + int(lag_seconds * 1000) + 5_000)
    if not spans:
        result = perf.local_breakdown(total)
        result["source"] = "local wall clock (no spans returned)"
        result["unavailable_reason"] = (
            f"No spans found in {perf.SPANS_LOG_GROUP} for this window. Check "
            "that AGENT_OBSERVABILITY_ENABLED=true on the runtimes and that "
            "CloudWatch Transaction Search is enabled for the account."
        )
        return result

    breakdown = perf.breakdown_from_spans(spans, session_id=session_id)
    breakdown["wall_clock_seconds"] = round(total, 4)
    if not breakdown.get("traces_matched"):
        # Session correlation failed; fall back to the unfiltered window and
        # say so, rather than reporting an empty breakdown as if it were zero.
        breakdown = perf.breakdown_from_spans(spans, session_id=None)
        breakdown["wall_clock_seconds"] = round(total, 4)
        breakdown["correlation"] = (
            "session id not found on any span; breakdown covers every trace in "
            "the time window and may include concurrent activity"
        )
    else:
        breakdown["correlation"] = "matched on session id"
    return breakdown
