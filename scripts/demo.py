"""Drive the demo through the orchestrator's real entry point.

Calls POST /invocations on the Train Travel Agent exactly as an external caller
would, so what you see on stage is the deployed interface and not a test
harness shortcut.

    .venv/bin/python scripts/demo.py              # all beats
    .venv/bin/python scripts/demo.py --beat 3     # one beat
    .venv/bin/python scripts/demo.py --list       # show the running order
    .venv/bin/python scripts/demo.py --ask "..."  # free-form question

Each beat prints what to watch for before it runs, so you can narrate it while
the model is still thinking.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

ORCHESTRATOR = "http://127.0.0.1:8090"
TIMEOUT = 300.0

BEATS = [
    {
        "title": "Policy question -- routed, not answered by the router",
        "actor_id": "TRV-1001",
        "prompt": (
            "I booked an Advance ticket. Can I get a refund if I change my mind?"
        ),
        "watch": [
            "The router holds NO knowledge base, so it cannot answer this itself.",
            "It routes to the Policy Advisor, which retrieves from the KB.",
            "The answer should cite the fare rules, not recall them from memory.",
            "Kill the Policy Advisor and re-run: the router must say it could not",
            "  reach it. A confident answer with the agent down means it invented one.",
        ],
    },
    {
        "title": "Plan and price -- orchestrator delegates to Journey Planner",
        "actor_id": "TRV-1001",
        "prompt": (
            "I'm traveller TRV-1001. I need to get from London to Manchester on "
            "2026-11-06. What are my options and what will it cost?"
        ),
        "watch": [
            "Planner calls the SHARED Traveller Profile server first, for the railcard.",
            "Then Journey Search for services and fares, with the discount applied.",
            "The 07:15 is cancelled: it must be reported as such, never offered.",
            "Expect the Advance at 24.64 GBP -- 37.00 less the Two Together discount.",
        ],
    },
    {
        "title": "Disruption -- two specialists, fanned out by the router",
        "actor_id": "TRV-1001",
        "prompt": (
            "I'm traveller TRV-1001 and my booking is BKG-77301. I've just been "
            "told my train is cancelled. What are my rights, and can you get me "
            "to Manchester this morning?"
        ),
        "watch": [
            "This needs TWO specialists: the rule from the Policy Advisor and the",
            "  action on the booking from Journey Support.",
            "Support sees the disruption waiver -- a non-refundable Advance becomes free to change.",
            "Support has NO search tool AND cannot ask a peer: sub-agents never",
            "  call each other. It reports that a search is needed.",
            "The ROUTER then obtains alternatives from the Planner and joins the",
            "  answers. Fan-out at Tier 1 is the architecture doing real work.",
        ],
    },
    {
        "title": "Policy refusal -- relayed honestly, not softened",
        "actor_id": "TRV-1002",
        "prompt": (
            "I'm traveller TRV-1002. Please cancel booking BKG-77303 and refund "
            "me. I've changed my plans."
        ),
        "watch": [
            "Advance fare on a service running normally: no disruption waiver.",
            "The tool returns FARE_RULE_PROHIBITS and the agent must not retry.",
            "Watch that the refusal reaches the traveller as a clear no.",
            "The 5.59 GBP actually paid should be stated, with any remedy offered.",
        ],
    },
    {
        "title": "Delay Repay -- the number comes from the tool, not the model",
        "actor_id": "TRV-1001",
        "prompt": (
            "I'm traveller TRV-1001. My journey on booking BKG-77304 arrived 47 "
            "minutes late last week. Am I owed anything?"
        ),
        "watch": [
            "47 minutes falls in the 30-59 band: 50% of the fare PAID.",
            "Paid was 29.97 (45.00 less the railcard), so compensation is 14.99 GBP.",
            "The agent must quote the tool's figure, not compute its own.",
            "Note it is calculated on the discounted fare, not the headline price.",
        ],
    },
]


def rule(char: str = "-", width: int = 78) -> str:
    return char * width


def check_up() -> bool:
    try:
        response = httpx.get(f"{ORCHESTRATOR}/ping", timeout=5.0)
        return response.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def ask(prompt: str, actor_id: str, session_id: str = "") -> tuple[bool, str, float]:
    """Ask the orchestrator one question.

    actor_id is required whenever memory is enabled -- the orchestrator
    validates it and rejects the request otherwise, because a conversation has
    to belong to someone before it can be remembered. Locally memory is
    usually off and the field is ignored, so sending it always keeps the local
    and deployed paths identical.
    """
    payload: dict[str, str] = {"prompt": prompt, "actor_id": actor_id}
    if session_id:
        payload["session_id"] = session_id

    started = time.monotonic()
    try:
        response = httpx.post(
            f"{ORCHESTRATOR}/invocations", json=payload, timeout=TIMEOUT
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}", time.monotonic() - started

    elapsed = time.monotonic() - started
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        return False, f"HTTP {response.status_code}: {response.text[:400]}", elapsed

    if response.status_code != 200 or body.get("status") != "success":
        return False, body.get("error", response.text[:400]), elapsed
    return True, body.get("response", ""), elapsed


def run_beat(index: int, beat: dict, session_id: str = "") -> bool:
    print(f"\n{rule('=')}")
    print(f"BEAT {index}  {beat['title']}")
    print(rule('='))
    print("\nwatch for:")
    for line in beat["watch"]:
        print(f"  - {line}")
    print(f"\ntraveller {beat['actor_id']} asks:\n  \"{beat['prompt']}\"")
    print(f"\n{rule()}")

    ok, answer, elapsed = ask(beat["prompt"], beat["actor_id"], session_id)
    if ok:
        print(answer)
        print(f"\n{rule()}\n({elapsed:.1f}s)")
    else:
        print(f"FAILED after {elapsed:.1f}s:\n{answer}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beat", type=int, help="run a single beat, 1-5")
    parser.add_argument("--list", action="store_true", help="show the running order")
    parser.add_argument("--ask", type=str, help="ask a free-form question")
    parser.add_argument(
        "--actor",
        default="TRV-1001",
        help="traveller id for --ask (required when memory is on; TRV-1001/2/3)",
    )
    parser.add_argument(
        "--session",
        default="",
        help="reuse a session id so turns accumulate into one conversation",
    )
    args = parser.parse_args()

    if args.list:
        print("demo running order:\n")
        for i, beat in enumerate(BEATS, 1):
            print(f"  {i}. {beat['title']}")
        return 0

    if not check_up():
        print(
            f"The orchestrator is not responding at {ORCHESTRATOR}.\n"
            "Start the stack first:\n  ./scripts/run_local.sh start",
            file=sys.stderr,
        )
        return 1

    if args.ask:
        ok, answer, elapsed = ask(args.ask, args.actor, args.session)
        print(answer if ok else f"FAILED: {answer}")
        print(f"\n({elapsed:.1f}s)")
        return 0 if ok else 1

    if args.beat:
        if not 1 <= args.beat <= len(BEATS):
            print(f"--beat must be 1-{len(BEATS)}", file=sys.stderr)
            return 2
        return 0 if run_beat(args.beat, BEATS[args.beat - 1], args.session) else 1

    failures = 0
    for i, beat in enumerate(BEATS, 1):
        if not run_beat(i, beat, args.session):
            failures += 1

    print(f"\n{rule('=')}")
    print(f"{len(BEATS) - failures}/{len(BEATS)} beats completed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
