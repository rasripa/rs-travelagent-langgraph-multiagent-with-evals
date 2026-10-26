"""Local demo web server for the Train Travel Agent chat experience.

Why a server at all, rather than a static page calling the agent directly: the
deployed orchestrator sits behind AgentCore Runtime, which requires SigV4. A
browser cannot sign a SigV4 request without shipping AWS credentials to the
client, which would be worse than this proxy by a wide margin. So this process
holds the credentials, signs, and serves the page.

SECURITY POSTURE, stated plainly:

  * Binds 127.0.0.1 only. This is a local demo tool on the presenter's laptop;
    it is NOT reachable from another machine and must not be made so. There is
    no authentication on it, which is acceptable precisely because the only
    client that can connect is a browser on this host.
  * It forwards prompts to an agent using YOUR AWS credentials. Anyone who can
    reach this port can invoke the agent as you.
  * If you need to show it to someone on another machine, use an SSH local
    port-forward (ssh -L 8000:127.0.0.1:8000 ...) rather than widening the
    bind. Widening the bind would publish an unauthenticated endpoint that
    invokes Bedrock on your credentials.

Two modes, chosen by TRAVEL_RUN_MODE:

  local      -> POSTs to the local orchestrator on 127.0.0.1:8090
  agentcore  -> calls InvokeAgentRuntime on the deployed orchestrator with
                SigV4, reading the ARN from TRAVEL_ORCHESTRATOR_ARN

    .venv/bin/python frontend/server.py
    open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import FileResponse, JSONResponse  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402
from starlette.staticfiles import StaticFiles  # noqa: E402

from common.config import AWS_REGION, RUN_MODE  # noqa: E402
from frontend.evaluation_api import (  # noqa: E402
    eval_dashboard,
    eval_report,
    eval_run,
    eval_status,
)

STATIC_DIR = Path(__file__).parent / "static"

# Loopback only. See the module docstring before changing this.
BIND_HOST = "127.0.0.1"
BIND_PORT = int(os.getenv("TRAVEL_FRONTEND_PORT", "8000"))

LOCAL_ORCHESTRATOR = "http://127.0.0.1:8090/invocations"
REQUEST_TIMEOUT = 300.0


def orchestrator_arn() -> str:
    arn = os.getenv("TRAVEL_ORCHESTRATOR_ARN")
    if arn:
        return arn
    state_path = ROOT / "deploy" / "agentcore" / ".deploy-state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        arn = (
            state.get("agent_runtimes", {})
            .get("train_travel_agent", {})
            .get("arn")
        )
        if arn:
            return arn
    raise RuntimeError(
        "TRAVEL_ORCHESTRATOR_ARN is not set and no deploy state was found. "
        "Run the deployment, then: source deploy/agentcore/.env"
    )


async def call_local(prompt: str, actor_id: str | None, session_id: str) -> dict:
    payload: dict[str, Any] = {"prompt": prompt, "session_id": session_id}
    if actor_id:
        payload["actor_id"] = actor_id

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(
            LOCAL_ORCHESTRATOR,
            json=payload,
            headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id},
        )

    # Read the body BEFORE deciding on the status. The orchestrator returns a
    # useful message in `error` (an expired credential, a missing model grant),
    # and raise_for_status() would replace it with a bare "500 Internal Server
    # Error" -- which is exactly the wrong thing to be staring at mid-demo.
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        raise RuntimeError(
            f"orchestrator returned HTTP {response.status_code}: "
            f"{response.text[:400]}"
        ) from None

    if response.status_code != 200 or body.get("status") != "success":
        raise RuntimeError(body.get("error") or f"HTTP {response.status_code}")
    return body


async def local_get(path: str, params: dict[str, Any] | None = None) -> dict:
    """Read-only passthrough to the local orchestrator's side endpoints."""
    base = LOCAL_ORCHESTRATOR.rsplit("/", 1)[0]
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{base}{path}", params=params or {})
    response.raise_for_status()
    return response.json()


def call_agentcore(
    prompt: str, actor_id: str | None, session_id: str
) -> dict:
    import boto3
    from botocore.config import Config as BotoConfig

    payload: dict[str, Any] = {"prompt": prompt, "session_id": session_id}
    if actor_id:
        payload["actor_id"] = actor_id

    client = boto3.client(
        "bedrock-agentcore",
        region_name=AWS_REGION,
        # Match REQUEST_TIMEOUT used on the local path. botocore defaults to a
        # 60s read timeout, which is below the cost of a full fan-out turn with
        # cold starts, so the browser would see a failure while the agent was
        # still working.
        config=BotoConfig(
            read_timeout=int(REQUEST_TIMEOUT),
            connect_timeout=10,
            # Never retry an agent invocation: it re-runs the model and tools,
            # and writes a second set of memory events.
            retries={"max_attempts": 0},
        ),
    )
    response = client.invoke_agent_runtime(
        agentRuntimeArn=orchestrator_arn(),
        qualifier="DEFAULT",
        runtimeSessionId=session_id,
        payload=json.dumps(payload).encode(),
    )
    raw = response["response"].read()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return {"response": raw.decode(errors="replace")}
    if body.get("status") != "success":
        raise RuntimeError(body.get("error", "the agent returned an error"))
    return body


async def chat(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "body must be JSON"}, status_code=400)

    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"error": "prompt is required"}, status_code=400)

    actor_id = payload.get("actor_id") or None
    # AgentCore session ids must be at least 33 characters; a uuid4 hex pair is
    # comfortably over and keeps each browser conversation on its own session.
    session_id = payload.get("session_id") or (
        uuid.uuid4().hex + uuid.uuid4().hex[:8]
    )

    import time

    started = time.monotonic()
    try:
        if RUN_MODE == "agentcore":
            import asyncio

            body = await asyncio.to_thread(
                call_agentcore, prompt, actor_id, session_id
            )
        else:
            body = await call_local(prompt, actor_id, session_id)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            {"error": f"{type(exc).__name__}: {exc}"}, status_code=502
        )

    return JSONResponse(
        {
            "response": body.get("response", ""),
            "session_id": body.get("session_id") or session_id,
            "actor_id": body.get("actor_id"),
            "turns_recalled": body.get("turns_recalled", 0),
            "long_term_recalled": body.get("long_term_recalled", False),
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "mode": RUN_MODE,
        }
    )


async def actors(_request: Request) -> JSONResponse:
    """Travellers this deployment knows, for the identity picker."""
    if RUN_MODE == "agentcore":
        # Read the fixtures directly: cheaper than an agent round trip, and
        # the orchestrator reads the same file.
        from common import memory

        return JSONResponse(
            {
                "memory_enabled": bool(os.getenv("TRAVEL_MEMORY_ID")),
                "actors": [
                    {"actor_id": aid, "name": name}
                    for aid, name in sorted(memory.known_actors().items())
                ],
            }
        )
    try:
        return JSONResponse(await local_get("/actors"))
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            {"memory_enabled": False, "actors": [], "error": str(exc)}
        )


async def sessions(request: Request) -> JSONResponse:
    """Recent sessions for a traveller, so the UI can offer to resume one."""
    actor_id = request.query_params.get("actor_id")
    if not actor_id:
        return JSONResponse({"error": "actor_id is required"}, status_code=400)

    if RUN_MODE == "agentcore":
        from common import memory

        if not memory.enabled():
            return JSONResponse({"memory_enabled": False, "sessions": []})
        try:
            resolved = memory.resolve_actor(actor_id)
        except memory.UnknownActor as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(
            {
                "memory_enabled": True,
                "actor_id": resolved,
                "sessions": memory.list_recent_sessions(resolved, limit=3),
            }
        )

    try:
        return JSONResponse(await local_get("/sessions", {"actor_id": actor_id}))
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"memory_enabled": False, "sessions": [],
                             "error": str(exc)})


async def config(_request: Request) -> JSONResponse:
    """Tell the page which mode it is talking to, for the status badge."""
    target = "unknown"
    if RUN_MODE == "agentcore":
        try:
            target = orchestrator_arn().split("/")[-1]
        except RuntimeError as exc:
            target = f"NOT CONFIGURED: {exc}"
    else:
        target = LOCAL_ORCHESTRATOR
    return JSONResponse({"mode": RUN_MODE, "region": AWS_REGION, "target": target})


async def index(_request: Request) -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app = Starlette(
    routes=[
        Route("/", index),
        Route("/api/chat", chat, methods=["POST"]),
        Route("/api/config", config, methods=["GET"]),
        Route("/api/actors", actors, methods=["GET"]),
        Route("/api/sessions", sessions, methods=["GET"]),
        # Evaluation controls. These SPAWN PROCESSES -- see
        # frontend/evaluation_api.py for why that is constrained and safe here.
        Route("/api/eval/status", eval_status, methods=["GET"]),
        Route("/api/eval/run", eval_run, methods=["POST"]),
        Route("/api/eval/report", eval_report, methods=["POST"]),
        Route("/api/eval/dashboard", eval_dashboard, methods=["GET"]),
        Mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static"),
    ]
)


def main() -> None:
    if not STATIC_DIR.exists():
        raise SystemExit(f"missing {STATIC_DIR}")

    print("Train Travel Agent -- demo frontend")
    print(f"  mode    : {RUN_MODE}")
    print(f"  region  : {AWS_REGION}")
    if RUN_MODE == "agentcore":
        try:
            print(f"  agent   : {orchestrator_arn()}")
        except RuntimeError as exc:
            print(f"  agent   : NOT CONFIGURED -- {exc}")
    else:
        print(f"  agent   : {LOCAL_ORCHESTRATOR}")
    print(f"\n  open http://{BIND_HOST}:{BIND_PORT}")
    print(
        "\n  NOTE: loopback only, and unauthenticated by design -- anyone who\n"
        "  can reach this port invokes the agent with your AWS credentials,\n"
        "  AND can start an evaluation run, which executes local commands and\n"
        "  spends money on Bedrock.\n"
        "  To show it from another machine use an SSH tunnel:\n"
        f"      ssh -L {BIND_PORT}:127.0.0.1:{BIND_PORT} <this-host>\n"
        "  Do not widen the bind.\n"
    )
    uvicorn.run(app, host=BIND_HOST, port=BIND_PORT, log_level="warning")


if __name__ == "__main__":
    main()
