"""Invoke the deployed Train Travel Agent orchestrator on AgentCore Runtime.

    source deploy/agentcore/.env
    python deploy/agentcore/invoke.py "my train is cancelled, what can I do?"
    python deploy/agentcore/invoke.py --beat 3

Uses SigV4 via boto3. The orchestrator's own calls onward to the sub-agents are
also SigV4, signed with its execution role -- the same identity mechanism all
the way down, with no token endpoint and no secret anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import boto3  # noqa: E402
from botocore.config import Config  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402

from common.config import AWS_REGION  # noqa: E402
from scripts.demo import BEATS  # noqa: E402

STATE_PATH = Path(__file__).parent / ".deploy-state.json"

# A full multi-agent turn plus cold starts can exceed two minutes. See invoke().
INVOKE_READ_TIMEOUT_SECONDS = 300


def orchestrator_arn() -> str:
    arn = os.getenv("TRAVEL_ORCHESTRATOR_ARN")
    if arn:
        return arn
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
        arn = state.get("agent_runtimes", {}).get("train_travel_agent", {}).get("arn")
        if arn:
            return arn
    raise SystemExit(
        "No orchestrator ARN. Run the deployment, then:\n"
        "  source deploy/agentcore/.env"
    )


def invoke(prompt: str, actor_id: str, session_id: str = "") -> str:
    # botocore's default read timeout is 60s, which is BELOW what a full
    # multi-agent turn costs: the disruption beat fans out to a sub-agent, a
    # peer A2A hop and several MCP tools, and pays a cold start on any runtime
    # that has been idle. The server completes; the client gives up first and
    # reports a ReadTimeoutError that looks like a hang.
    #
    # Retries are disabled deliberately. A retried agent invocation is not
    # free or idempotent -- it re-runs the model and the tool calls, and with
    # memory enabled it writes a second set of events.
    client = boto3.client(
        "bedrock-agentcore",
        region_name=AWS_REGION,
        config=Config(
            read_timeout=INVOKE_READ_TIMEOUT_SECONDS,
            connect_timeout=10,
            retries={"max_attempts": 0},
        ),
    )

    # actor_id is mandatory whenever the orchestrator has memory enabled: a
    # conversation has to belong to someone before it can be remembered, and
    # resolve_actor() validates the value against the traveller fixtures.
    payload: dict[str, str] = {"prompt": prompt, "actor_id": actor_id}

    request: dict[str, Any] = {
        "agentRuntimeArn": orchestrator_arn(),
        "qualifier": "DEFAULT",
    }
    # AgentCore's own session id is what the orchestrator prefers, so a runtime
    # session and a memory session are the same thing. Pass it here and the
    # turns accumulate into one conversation.
    if session_id:
        # The API enforces 33-256 characters. A UUID (36) passes, which is what
        # the orchestrator hands back; a short hand-typed id would otherwise
        # fail as an opaque ValidationException.
        if not 33 <= len(session_id) <= 256:
            raise SystemExit(
                f"--session must be 33-256 characters (got {len(session_id)}). "
                f"Use the session id the agent returned, or omit it to start "
                f"a new conversation."
            )
        request["runtimeSessionId"] = session_id

    request["payload"] = json.dumps(payload).encode()

    try:
        response = client.invoke_agent_runtime(**request)
    except ClientError as exc:
        # AgentCore returns the container's status code but DISCARDS its
        # response body, so a deliberate 400 from the agent arrives as an
        # opaque "check your CloudWatch logs". The most common cause by far is
        # a rejected or missing actor_id, so say so rather than sending the
        # operator to CloudWatch for a validation error.
        code = exc.response.get("Error", {}).get("Code", "")
        if code == "RuntimeClientError" and "(400)" in str(exc):
            raise SystemExit(
                f"The agent rejected the request (HTTP 400). The body is not\n"
                f"returned through InvokeAgentRuntime, so the usual causes:\n"
                f"  - actor_id {actor_id!r} is not a known traveller\n"
                f"    (known: TRV-1001, TRV-1002, TRV-1003)\n"
                f"  - the prompt was empty\n"
                f"Rejections are logged; check:\n"
                f"  /aws/bedrock-agentcore/runtimes/travel_demo_train_travel_agent-*"
            ) from exc
        raise

    body = response["response"].read()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return body.decode(errors="replace")

    if parsed.get("status") != "success":
        return f"[agent error] {parsed.get('error', parsed)}"

    answer = parsed.get("response", json.dumps(parsed, indent=2))
    # Memory provenance, so the demo can show recall happening.
    trailer = (
        f"\n{'-' * 70}\n"
        f"session {parsed.get('session_id', '?')}  actor {parsed.get('actor_id', '?')}"
        f"  turns recalled {parsed.get('turns_recalled', 0)}"
        f"  long-term {'yes' if parsed.get('long_term_recalled') else 'no'}"
    )
    return answer + trailer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="what to ask the agent")
    parser.add_argument("--beat", type=int, help=f"run demo beat 1-{len(BEATS)}")
    parser.add_argument(
        "--actor",
        help="traveller id (TRV-1001/2/3). Defaults to the beat's own "
        "traveller, or TRV-1001 for a free-form prompt.",
    )
    parser.add_argument(
        "--session",
        default="",
        help="reuse a session id so turns accumulate into one conversation",
    )
    args = parser.parse_args()

    if args.beat:
        if not 1 <= args.beat <= len(BEATS):
            print(f"--beat must be 1-{len(BEATS)}", file=sys.stderr)
            return 2
        beat = BEATS[args.beat - 1]
        actor = args.actor or beat["actor_id"]
        print(f"BEAT {args.beat}  {beat['title']}\n")
        print("watch for:")
        for line in beat["watch"]:
            print(f"  - {line}")
        print(f"\ntraveller {actor} asks:\n  \"{beat['prompt']}\"\n{'-' * 70}")
        print(invoke(beat["prompt"], actor, args.session))
        return 0

    if not args.prompt:
        parser.error("give a prompt, or use --beat N")
    print(invoke(args.prompt, args.actor or "TRV-1001", args.session))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
