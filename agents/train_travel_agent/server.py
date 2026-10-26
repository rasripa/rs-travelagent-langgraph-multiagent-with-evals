"""HTTP entrypoint for the Train Travel Agent orchestrator.

This is the only agent a caller outside the system talks to, so it implements
AgentCore Runtime's HTTP protocol contract rather than A2A:

  POST /invocations   the agent interaction endpoint
  GET  /ping          health check, returning {"status": "Healthy"}
  port 8080, host 0.0.0.0, ARM64 container

See https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/getting-started-custom.html

The orchestrator is NOT an A2A server. Nothing delegates work to it -- it is
the entry point. Giving it an Agent Card would advertise a capability surface
nobody consumes.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import uvicorn  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

from agents.train_travel_agent.agent import AGENT_NAME, build  # noqa: E402
from common import config_store, memory, observability  # noqa: E402
from common.agent_factory import run_agent  # noqa: E402
from common.config import LOCAL_BIND_HOST, RUN_MODE, model_id  # noqa: E402

AGENTCORE_HTTP_HOST = "0.0.0.0"  # nosec B104 - mandated by the runtime contract
AGENTCORE_HTTP_PORT = 8080
LOCAL_PORT = 8090

# The agent is built LAZILY, on the first invocation, not at startup.
#
# AgentCore gives a runtime 30 seconds to initialise, and building this agent
# makes a network round trip: SigV4-signed MCP initialize + tools/list against
# the gateway for the knowledge-base tool. Doing that before binding the port
# spends the startup budget on I/O, and -- worse -- ANY failure in it kills the
# process before it ever listens, which AgentCore reports as "Runtime
# initialization time exceeded". That hides the real error behind a message
# about latency.
#
# Binding first means the port is open in milliseconds, /ping is immediately
# healthy, and a build failure comes back on /invocations as the actual
# exception. It also avoids freezing the gateway tool catalogue into a
# platformVersion V2 snapshot.
_AGENT = None
_AGENT_LOCK = asyncio.Lock()


async def get_agent():
    """Return the agent, building it once on first use."""
    global _AGENT
    if _AGENT is not None:
        return _AGENT
    async with _AGENT_LOCK:
        if _AGENT is None:
            print("  building agent (first invocation)")
            _AGENT = await build()
    return _AGENT


async def ping(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "Healthy"})


async def invocations(request: Request) -> JSONResponse:
    # Every rejection below is LOGGED as well as returned. AgentCore passes the
    # container's status code to the caller but discards the response body, so
    # an unlogged 400 is invisible from both sides -- the client sees "check
    # your CloudWatch logs" and CloudWatch has nothing in it.
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        print("rejected 400: request body is not JSON", flush=True)
        return JSONResponse(
            {"error": "Request body must be JSON.", "status": "error"},
            status_code=400,
        )

    prompt = body.get("prompt") or body.get("input") or body.get("message")
    if not isinstance(prompt, str) or not prompt.strip():
        print(
            f"rejected 400: no usable prompt (keys present: "
            f"{sorted(body) if isinstance(body, dict) else type(body).__name__})",
            flush=True,
        )
        return JSONResponse(
            {
                "error": 'Provide a non-empty string in "prompt".',
                "status": "error",
            },
            status_code=400,
        )

    # Session id: prefer AgentCore's own header so a runtime session and a
    # memory session are the same thing, then the request body, then nothing.
    session_id = (
        request.headers.get("X-Amzn-Bedrock-AgentCore-Runtime-Session-Id")
        or body.get("session_id")
        or ""
    )

    actor_id = None
    memory_context = ""
    history: list = []
    if memory.enabled():
        try:
            actor_id = memory.resolve_actor(body.get("actor_id"))
        except memory.UnknownActor as exc:
            print(f"rejected 400: {exc}", flush=True)
            return JSONResponse(
                {"error": str(exc), "status": "error"}, status_code=400
            )
        history = memory.load_turns(actor_id, session_id)
        memory_context = memory.memory_context_block(actor_id, prompt)

    try:
        agent = await get_agent()
    except Exception as exc:  # noqa: BLE001
        # A build failure is almost always missing or wrong configuration.
        # Return it verbatim rather than letting it look like a model error.
        return JSONResponse(
            {
                "error": (
                    f"agent initialisation failed: {type(exc).__name__}: {exc}"
                ),
                "status": "error",
            },
            status_code=503,
        )

    try:
        answer = await run_agent(
            agent, prompt, history=history, extra_context=memory_context
        )
    except Exception as exc:  # noqa: BLE001
        # Report the real error. A generic 500 here is what makes a multi-agent
        # system impossible to debug from the outside.
        return JSONResponse(
            {"error": f"{type(exc).__name__}: {exc}", "status": "error"},
            status_code=500,
        )

    if actor_id:
        memory.save_turn(actor_id, session_id, prompt, answer)

    return JSONResponse(
        {
            "response": answer,
            "status": "success",
            "session_id": session_id,
            "actor_id": actor_id,
            "turns_recalled": len(history),
            "long_term_recalled": bool(memory_context),
        }
    )


async def sessions(request: Request) -> JSONResponse:
    """Recent sessions for a traveller, so a UI can offer to resume one."""
    actor = request.query_params.get("actor_id")
    if not memory.enabled():
        return JSONResponse({"memory_enabled": False, "sessions": []})
    try:
        actor_id = memory.resolve_actor(actor)
    except memory.UnknownActor as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(
        {
            "memory_enabled": True,
            "actor_id": actor_id,
            "sessions": memory.list_recent_sessions(actor_id, limit=3),
        }
    )


async def actors(_request: Request) -> JSONResponse:
    """Which travellers this deployment knows, for a UI picker."""
    return JSONResponse(
        {
            "memory_enabled": memory.enabled(),
            "actors": [
                {"actor_id": aid, "name": name}
                for aid, name in sorted(memory.known_actors().items())
            ],
        }
    )


async def config(request: Request) -> JSONResponse:
    """Report resolved configuration and where each value came from.

    Useful on stage: change the model parameter, call this with ?refresh=1,
    and show the new value taking effect without a redeploy.
    """
    if request.query_params.get("refresh"):
        config_store.invalidate()
    return JSONResponse(
        {
            "source": config_store.describe(),
            "resolved": config_store.snapshot(),
        }
    )


app = Starlette(
    routes=[
        Route("/ping", ping, methods=["GET"]),
        Route("/invocations", invocations, methods=["POST"]),
        Route("/sessions", sessions, methods=["GET"]),
        Route("/actors", actors, methods=["GET"]),
        Route("/config", config, methods=["GET"]),
    ]
)


def main() -> None:
    print(f"starting {AGENT_NAME} (HTTP)")
    observability.init(AGENT_NAME)
    observability.instrument_app(app)
    print(f"  {config_store.describe()}")
    print(f"  model: {model_id()}")
    print(f"  {memory.describe_state()}")

    if RUN_MODE == "agentcore":
        host, port = AGENTCORE_HTTP_HOST, AGENTCORE_HTTP_PORT
    else:
        host, port = LOCAL_BIND_HOST, LOCAL_PORT

    # Bind immediately; the agent is built on the first invocation. See the
    # note next to _AGENT for why this ordering matters.
    print(f"  POST http://{host}:{port}/invocations")
    print("  agent will be built on the first request")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
