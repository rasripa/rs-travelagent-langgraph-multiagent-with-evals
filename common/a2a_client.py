"""A2A client used by the orchestrator, and by Journey Support for its peer hop.

LangGraph has no A2A client node either, so delegation is wrapped as an ordinary
LangChain tool. From the orchestrator model's point of view "ask the Journey
Planner" is just another tool call; underneath it is a JSON-RPC task sent to a
separately deployed agent, with its own identity and its own tool entitlements.

Peer hop: Journey Support can call Journey Planner through exactly this path.
That is not duplication -- Support is denied the JourneySearch MCP target by
Cedar policy, so when it needs an alternative service during disruption it has
to ask the agent that owns search. The capability boundary is enforced at the
gateway, and the A2A hop is how work crosses it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory  # noqa: E402
from a2a.types import (  # noqa: E402
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TextPart,
    TransportProtocol,
)

# A sub-agent runs a full agentic loop with several tool calls, so the timeout
# has to be generous. AgentCore Gateway's own tool-call ceiling is 15 minutes;
# this is well inside it.
DEFAULT_TIMEOUT_SECONDS = 180.0


from common import observability  # noqa: E402

# AgentCore requires a session id on every runtime invocation; it is what the
# service keys session isolation on. Sent only in AWS mode, so the local path
# stays free of platform-specific headers.
AGENTCORE_SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"


def auth_for(session_id: str | None = None) -> tuple[Any | None, dict[str, str]]:
    """Return (httpx auth, extra headers) for calling a sub-agent.

    Local mode returns (None, {}): the sub-agents are loopback-only developer
    processes with no authenticator in front of them.

    AWS mode returns a SigV4 signer. Note there is no per-agent argument: with
    IAM the caller's identity IS its execution role, so there is nothing to
    look up and no secret to hold. That role ARN is also what the gateway's
    Cedar policies match on, so one identity mechanism covers both the
    agent-to-agent hop and the tool hop.
    """
    from common.config import AWS_REGION, RUN_MODE

    if RUN_MODE != "agentcore":
        return None, {}

    from common.aws_sigv4 import signer_for_agentcore

    return (
        signer_for_agentcore(AWS_REGION),
        {AGENTCORE_SESSION_HEADER: session_id or str(uuid4())},
    )


def _is_unreachable(exc: BaseException) -> bool:
    """True if this failure is "nothing is listening there"."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (httpx.ConnectError, httpx.ConnectTimeout)):
            return True
        current = current.__cause__ or current.__context__

    # The A2A SDK stringifies the transport error rather than always chaining it.
    text = str(exc).lower()
    return (
        "all connection attempts failed" in text
        or "network communication error" in text
        or "connection refused" in text
    )


def _text_of(message: Message) -> str:
    chunks = []
    for part in message.parts or []:
        root = getattr(part, "root", part)
        text = getattr(root, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


def _text_of_task(task: Task) -> str:
    """Pull the agent's answer out of a completed task.

    Checks the status message first, then artifacts, because an executor may
    legitimately return its result either way.
    """
    status_message = getattr(task.status, "message", None)
    if status_message:
        text = _text_of(status_message)
        if text:
            return text

    for artifact in task.artifacts or []:
        chunks = []
        for part in artifact.parts or []:
            root = getattr(part, "root", part)
            text = getattr(root, "text", None)
            if text:
                chunks.append(text)
        if chunks:
            return "\n".join(chunks).strip()

    for message in reversed(task.history or []):
        if message.role == Role.agent:
            text = _text_of(message)
            if text:
                return text

    return ""


async def ask_agent(
    base_url: str,
    prompt: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    auth: Any | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Send one task to an A2A agent and return its final result.

    Transport-agnostic: `auth` is any httpx.Auth and `headers` are whatever the
    hosting platform requires. Pass nothing and this is a plain A2A client
    against a plain HTTP endpoint.

    Returns a dict rather than raising, so a delegation failure becomes text the
    orchestrator model can reason about and report, instead of a stack trace
    that kills the conversation mid-demo.
    """
    try:
        # Name the delegation explicitly so the trace shows which agent asked
        # which peer for what, rather than just an anonymous HTTP span.
        with observability.span(
            "a2a.delegate",
            **{"a2a.target_url": base_url, "a2a.prompt_chars": len(prompt)},
        ):
            return await _send(base_url, prompt, timeout, auth, headers)
    except Exception as exc:  # noqa: BLE001
        # A2ACardResolver wraps transport failures in A2AClientHTTPError, so a
        # bare `except httpx.ConnectError` never fires. Walk the cause chain
        # instead, otherwise "agent isn't running" reads as an opaque HTTP 503.
        if _is_unreachable(exc):
            return {
                "ok": False,
                "error": (
                    f"Could not reach the agent at {base_url}. Start the agents "
                    f"first: ./scripts/run_local.sh start.  ({exc})"
                ),
            }
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


async def _send(
    base_url: str,
    prompt: str,
    timeout: float,
    auth: Any | None,
    headers: dict[str, str] | None,
) -> dict[str, Any]:
    # One client for both the card fetch and the JSON-RPC call, so the signer
    # and headers apply to each. A2ACardResolver appends
    # /.well-known/agent-card.json to base_url, which on AgentCore resolves to
    # <runtime-url>/.well-known/agent-card.json -- the documented path-based
    # mounting pattern.
    async with httpx.AsyncClient(
        timeout=timeout, headers=headers or {}, auth=auth
    ) as http_client:
        card = await A2ACardResolver(
            httpx_client=http_client, base_url=base_url.rstrip("/")
        ).get_agent_card()

        config = ClientConfig(
            httpx_client=http_client,
            streaming=False,
            supported_transports=[TransportProtocol.jsonrpc],
        )
        client = ClientFactory(config).create(card)

        request = Message(
            message_id=str(uuid4()),
            role=Role.user,
            parts=[Part(root=TextPart(text=prompt))],
        )

        answer, state = "", None
        async for event in client.send_message(request):
            if isinstance(event, Message):
                text = _text_of(event)
                if text:
                    answer = text
                continue

            task = event[0] if isinstance(event, tuple) else event
            if isinstance(task, Task):
                state = task.status.state
                text = _text_of_task(task)
                if text:
                    answer = text

        if state == TaskState.failed:
            return {
                "ok": False,
                "agent": card.name,
                "error": answer or "The remote agent reported a failure.",
            }
        if not answer:
            return {
                "ok": False,
                "agent": card.name,
                "error": f"The remote agent returned no text (state={state}).",
            }
        return {"ok": True, "agent": card.name, "answer": answer}


def render(result: dict[str, Any]) -> str:
    """Flatten an ask_agent result into tool-return text."""
    if result.get("ok"):
        return f"[reply from {result['agent']}]\n{result['answer']}"
    return (
        "[delegation failed]\n"
        f"{result.get('error', 'unknown error')}\n"
        "Tell the traveller you could not complete that step. Do not invent "
        "the answer the other agent would have given."
    )
