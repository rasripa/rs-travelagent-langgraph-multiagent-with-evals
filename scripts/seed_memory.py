"""Seed a prior conversation so long-term memory has something to recall.

Why this exists: AgentCore Memory extraction is ASYNCHRONOUS. Writing an event
triggers a background job that produces summaries and preferences some time
later. If you demo "it remembers me" in a fresh deployment, memory is empty and
the beat falls flat through no fault of the code.

So: run this once after deploying, wait, then verify records exist before you
present.

    source deploy/agentcore/.env
    python scripts/seed_memory.py              # write seed conversations
    python scripts/seed_memory.py --check      # has extraction produced records?

It writes events directly rather than driving the agent, so it costs no model
tokens and takes seconds.
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from datetime import datetime, timedelta, UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import memory  # noqa: E402

# Prior conversations, written as if they happened on earlier days. Each one is
# shaped to give the extraction strategies something concrete to pull out:
# a stable preference, and a summary worth recalling.
SEED = {
    "TRV-1001": [
        ("I'm travelling London to Manchester again next month. Same as usual "
         "please -- I always want a quiet coach, a table seat, and "
         "forward-facing if you can.",
         "Noted. I've found the London Euston to Manchester Piccadilly route. "
         "I'll look for a quiet coach with a forward-facing table seat. Your "
         "Two Together Railcard brings the Advance fare down to 24.64 GBP."),
        ("Yes book that. And I prefer morning departures, I don't like "
         "travelling after about 6pm.",
         "Booked. I've recorded that you prefer morning departures and would "
         "rather avoid evening travel after 18:00."),
    ],
    "TRV-1003": [
        ("I need to get to Edinburgh. I use a wheelchair, so I need a "
         "wheelchair space and ramp assistance at both ends.",
         "Understood. I've reserved a wheelchair space on the 09:00 King's "
         "Cross to Edinburgh Waverley and booked ramp assistance at both "
         "stations. Your Gold tier includes priority rebooking if anything "
         "is disrupted."),
        ("Thanks. Please always book assistance for me without asking, I "
         "always need it.",
         "Noted -- I'll arrange ramp assistance on every booking by default "
         "rather than asking each time."),
    ],
}


def seed() -> int:
    if not memory.enabled():
        print(
            "TRAVEL_MEMORY_ID is not set, so there is no memory to seed.\n"
            "  source deploy/agentcore/.env",
            file=sys.stderr,
        )
        return 1

    print(f"memory: {memory.memory_id()}\n")
    written = 0
    for actor_id, turns in SEED.items():
        try:
            actor = memory.resolve_actor(actor_id)
        except memory.UnknownActor as exc:
            print(f"  skipping {actor_id}: {exc}")
            continue

        # A distinct session id per seed run, dated earlier so it reads as a
        # genuinely prior conversation.
        session_id = f"seed-{actor.lower()}-{uuid.uuid4().hex[:8]}"
        when = datetime.now(UTC) - timedelta(days=3)

        print(f"{actor} ({memory.known_actors()[actor]}) -> session {session_id}")
        client = memory._data_plane()  # noqa: SLF001 - deliberate, seeding only
        for index, (user_text, agent_text) in enumerate(turns):
            client.create_event(
                memoryId=memory.memory_id(),
                actorId=actor,
                sessionId=session_id,
                eventTimestamp=when + timedelta(minutes=index * 2),
                payload=[
                    {"conversational": {"role": "USER",
                                        "content": {"text": user_text}}},
                    {"conversational": {"role": "ASSISTANT",
                                        "content": {"text": agent_text}}},
                ],
            )
            written += 1
            print(f"    turn {index + 1} written")

    print(f"\n{written} turns written across {len(SEED)} travellers.")
    print(
        "\nExtraction runs in the background. Wait a few minutes, then:\n"
        "  python scripts/seed_memory.py --check"
    )
    return 0


def check() -> int:
    if not memory.enabled():
        print("TRAVEL_MEMORY_ID is not set.", file=sys.stderr)
        return 1

    client = memory._data_plane()  # noqa: SLF001
    print(f"memory: {memory.memory_id()}\n")
    total = 0
    for actor_id in sorted(memory.known_actors()):
        try:
            response = client.list_memory_records(
                memoryId=memory.memory_id(),
                namespace=memory.NS_PREFERENCES.format(actor_id=actor_id),
                maxResults=20,
            )
            prefs = response.get("memoryRecordSummaries", [])
        except Exception as exc:  # noqa: BLE001
            print(f"  {actor_id}: list failed ({type(exc).__name__}: {exc})")
            continue

        sessions = memory.list_recent_sessions(actor_id, limit=3)
        print(f"  {actor_id:10} preferences={len(prefs):2}  sessions={len(sessions)}")
        for record in prefs[:5]:
            content = record.get("content") or {}
            text = content.get("text") if isinstance(content, dict) else None
            if text:
                print(f"      - {text.strip()[:110]}")
        total += len(prefs)

    if total == 0:
        print(
            "\nNo preference records yet. Either extraction has not finished "
            "(wait and retry), or no events have been written.\n"
            "Check for failures with ListMemoryExtractionJobs."
        )
        return 1
    print(f"\n{total} preference record(s) present. Personalisation will demo.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report whether extraction has produced records")
    parser.add_argument("--wait", type=int, metavar="SECONDS",
                        help="seed, wait, then check")
    args = parser.parse_args()

    if args.check:
        return check()
    result = seed()
    if result == 0 and args.wait:
        print(f"\nwaiting {args.wait}s for extraction...")
        time.sleep(args.wait)
        return check()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
