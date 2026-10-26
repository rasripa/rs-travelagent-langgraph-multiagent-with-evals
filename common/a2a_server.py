"""Expose a LangGraph agent as an A2A server that satisfies AgentCore's contract.

LangGraph has no built-in A2A executor -- Strands ships StrandsA2AExecutor which
builds the Agent Card for you, but with LangGraph the AgentExecutor is yours to
write. That is the whole of the work below: translate an A2A RequestContext into
a LangGraph invocation and publish the result back onto the EventQueue.

AgentCore's A2A contract, which this module implements:
  * JSON-RPC 2.0 over HTTP, served at /
  * bound to 0.0.0.0, port 9000
  * Agent Card at /.well-known/agent-card.json
  * /ping returning Healthy or HealthyBusy
  * ARM64 container

Two deliberate choices worth narrating during the demo:

InMemoryTaskStore is used because each AgentCore Runtime session gets its own
microVM. A shared task store would be the right answer if tasks had to outlive a
session or be resumed from another instance; here they do not, and pretending
otherwise would be a distributed-state bug waiting to happen. The sample code
everyone copies uses InMemoryTaskStore without saying why -- this is the why.

/ping deliberately does NOT touch task state. AgentCore uses the health probe
to decide whether a session is still active; if a ping advanced the session's
last-update timestamp, an idle session would never be reaped by the idle timeout
and would instead survive all the way to MaxLifetime, quietly holding a microVM.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a2a.server.agent_execution import AgentExecutor, RequestContext  # noqa: E402
from a2a.server.apps import A2AStarletteApplication  # noqa: E402
from a2a.server.events import EventQueue  # noqa: E402
from a2a.server.request_handlers import DefaultRequestHandler  # noqa: E402
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater  # noqa: E402
from a2a.types import (  # noqa: E402
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    TaskState,
)
from a2a.utils import new_agent_text_message  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import PlainTextResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

from common.agent_factory import run_agent  # noqa: E402
from common import observability  # noqa: E402

# The AgentCore A2A contract port. Local runs override it so two sub-agents can
# coexist on one machine.
AGENTCORE_A2A_HOST = "0.0.0.0"  # nosec B104 - mandated by the runtime contract
AGENTCORE_A2A_PORT = 9000


class LangGraphAgentExecutor(AgentExecutor):
    """Bridge a compiled LangGraph graph to the A2A task lifecycle."""

    def __init__(self, agent: CompiledStateGraph, agent_name: str) -> None:
        self._agent = agent
        self._agent_name = agent_name
        self._in_flight = 0

    @property
    def busy(self) -> bool:
        return self._in_flight > 0

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        # A2A requires a task to be submitted before it can transition to
        # working. Skip the submit if the caller already created the task.
        if not context.current_task:
            await updater.submit()
        await updater.start_work()

        self._in_flight += 1
        try:
            prompt = context.get_user_input()
            if not prompt.strip():
                await updater.failed(
                    new_agent_text_message(
                        "Empty request: no text part found in the A2A message."
                    )
                )
                return

            # The inbound HTTP span is created by the Starlette instrumentation
            # and already parented to the caller's trace; this span names the
            # agent turn inside it, so the trace reads as a call graph.
            with observability.span(
                f"agent.{self._agent_name}",
                **{
                    "agent.name": self._agent_name,
                    "a2a.task_id": context.task_id,
                    "a2a.context_id": context.context_id,
                    "a2a.prompt_chars": len(prompt),
                },
            ):
                answer = await run_agent(self._agent, prompt)
            await updater.complete(new_agent_text_message(answer))

        except Exception as exc:  # noqa: BLE001
            # Surface the real failure to the caller. A silent generic error
            # here is what makes multi-agent systems undebuggable.
            await updater.failed(
                new_agent_text_message(
                    f"{self._agent_name} failed: {type(exc).__name__}: {exc}"
                )
            )
        finally:
            self._in_flight -= 1

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Mark the task cancelled.

        A LangGraph ainvoke is not interruptible mid-step, so this records the
        caller's intent rather than tearing down a running graph. Saying that
        plainly is better than implying pre-emption we do not have.
        """
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.update_status(
            TaskState.canceled,
            new_agent_text_message(
                "Cancellation recorded. Any in-flight model call will finish "
                "before the worker is released."
            ),
            final=True,
        )


def build_agent_card(
    name: str,
    description: str,
    url: str,
    skills: list[AgentSkill],
    version: str = "1.0.0",
) -> AgentCard:
    """Build the Agent Card served at /.well-known/agent-card.json.

    Note what a Card skill can and cannot carry: id, name, description, tags,
    examples. There is no typed parameter schema -- a caller discovers that a
    skill exists and what it is for, then sends free text. That is exactly why
    the tool hop in this architecture is MCP and not A2A: MCP tools advertise a
    JSON Schema inputSchema, so the model gets typed, validated arguments.
    A2A is used only where it earns its keep, at the agent-to-agent boundary.
    """
    return AgentCard(
        name=name,
        description=description,
        url=url,
        version=version,
        default_input_modes=["text", "text/plain"],
        default_output_modes=["text", "text/plain"],
        capabilities=AgentCapabilities(streaming=False),
        skills=skills,
    )


def build_a2a_app(
    agent_card: AgentCard,
    executor: LangGraphAgentExecutor,
) -> Starlette:
    """Assemble the Starlette app: JSON-RPC at /, card, and the health probe."""
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler).build()

    async def ping(_request: Request) -> PlainTextResponse:
        # Read-only by design. See the module docstring.
        return PlainTextResponse("HealthyBusy" if executor.busy else "Healthy")

    app.router.routes.append(Route("/ping", ping, methods=["GET"]))

    # Inbound tracing: continues the caller's trace so the orchestrator ->
    # sub-agent hop is one trace, not two. No-op when tracing is disabled.
    observability.instrument_app(app)
    return app


def advertised_url(agent_env_var: str, local_port: int) -> str:
    """The URL to publish in the Agent Card.

    This matters more than it looks. A2A clients route by the card's `url`
    field, NOT by the address you connected to -- so a card advertising
    0.0.0.0:9000 means every delegation fails, after the card itself resolved
    fine. Verified by test in scripts/smoke_a2a.py.

    Resolution order:
      1. AGENTCORE_RUNTIME_URL -- injected by AgentCore Runtime. Read as a
         plain environment variable, so this stays free of any platform SDK.
      2. <AGENT>_A2A_URL -- set explicitly by deploy/agentcore/deploy.py once
         the runtime ARN is known.
      3. loopback, for local development.

    In AgentCore mode with neither variable set we raise, rather than publish a
    card pointing at 0.0.0.0. Failing at startup is far cheaper than a
    delegation that fails mid-conversation.
    """
    for candidate in (os.getenv("AGENTCORE_RUNTIME_URL"), os.getenv(agent_env_var)):
        if candidate:
            return candidate if candidate.endswith("/") else candidate + "/"

    from common.config import LOCAL_BIND_HOST, RUN_MODE

    if RUN_MODE == "agentcore":
        raise RuntimeError(
            "Running in agentcore mode but neither AGENTCORE_RUNTIME_URL nor "
            f"{agent_env_var} is set. The Agent Card would advertise "
            "0.0.0.0:9000, which no caller can reach -- A2A clients route by "
            "card.url. Set one of them to this runtime's invocation URL "
            "(https://bedrock-agentcore.<region>.amazonaws.com/runtimes/"
            "<url-encoded-arn>/invocations/)."
        )
    return f"http://{LOCAL_BIND_HOST}:{local_port}/"


def serve(app: Starlette, local_port: int) -> None:
    """Run the A2A server, on 0.0.0.0:9000 in AgentCore, loopback locally."""
    import uvicorn

    from common.config import LOCAL_BIND_HOST, RUN_MODE

    if RUN_MODE == "agentcore":
        host, port = AGENTCORE_A2A_HOST, AGENTCORE_A2A_PORT
    else:
        host, port = LOCAL_BIND_HOST, local_port

    uvicorn.run(app, host=host, port=port, log_level="warning")
