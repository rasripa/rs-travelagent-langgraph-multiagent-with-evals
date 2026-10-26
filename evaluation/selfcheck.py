"""Validate the dataset against every coverage requirement.

    python -m evaluation.selfcheck

Exits non-zero on the first unmet requirement, so it is usable as a gate before
spending money on a judged run. Checks structure (via harness.cases, which is
the same validation the runners apply) and then the coverage rules the dataset
is supposed to guarantee.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.harness import cases as case_loader  # noqa: E402

MIN_TOTAL = 24
MIN_PER_SUB_AGENT = 4
MIN_ORCHESTRATOR = 8
MIN_MULTI_SPECIALIST = 2
MIN_RESPONSIBLE_AI = 6
MIN_CANCELLED_ASSERTIONS = 2
MIN_ROUTER_DELEGATES = 1

SUB_AGENTS = ("journey_planner", "journey_support", "policy_advisor")

# Each required responsible-AI theme, and a predicate that recognises it. Keyed
# on case id prefixes rather than on text matching, so renaming a prompt does
# not silently drop coverage.
RAI_THEMES = {
    "prompt injection in a user message": (
        "pa-ra-injection-06",
        "orc-ra-injection-09",
    ),
    "extracting another traveller's personal data": (
        "jp-ra-cross-traveller-pii-06",
        "orc-ra-other-traveller-data-12",
    ),
    "payment card details": ("orc-ra-payment-card-10",),
    "out-of-domain request declined": ("orc-ra-out-of-domain-11",),
    "bypassing a fare rule / falsifying a claim": ("js-ra-falsify-claim-06",),
    "leading question inviting a wrong policy answer": ("pa-ra-leading-question-07",),
}


def main() -> int:
    failures: list[str] = []
    notes: list[str] = []

    try:
        cases = case_loader.load()
    except case_loader.DatasetError as exc:
        print(f"FAIL  dataset is invalid: {exc}", file=sys.stderr)
        return 1
    print(f"PASS  dataset parses and every case validates ({len(cases)} cases)")

    def check(label: str, ok: bool, detail: str = "") -> None:
        if ok:
            print(f"PASS  {label}{(' -- ' + detail) if detail else ''}")
        else:
            failures.append(f"{label}{(' -- ' + detail) if detail else ''}")
            print(f"FAIL  {label}{(' -- ' + detail) if detail else ''}")

    counts = case_loader.counts(cases)
    check(
        f"at least {MIN_TOTAL} cases",
        len(cases) >= MIN_TOTAL,
        f"found {len(cases)}",
    )

    for agent in SUB_AGENTS:
        direct = [
            c for c in cases if c["target"] == agent and c["tier"] == "sub_agent"
        ]
        check(
            f"at least {MIN_PER_SUB_AGENT} cases directly against {agent}",
            len(direct) >= MIN_PER_SUB_AGENT,
            f"found {len(direct)}",
        )

    orchestrator = [c for c in cases if c["tier"] == "orchestrator"]
    check(
        f"at least {MIN_ORCHESTRATOR} orchestrator end-to-end cases",
        len(orchestrator) >= MIN_ORCHESTRATOR,
        f"found {len(orchestrator)}",
    )

    multi = [c for c in cases if c.get("requires_multiple_specialists")]
    multi_valid = [c for c in multi if len(c.get("expected_tools", [])) >= 2]
    check(
        f"at least {MIN_MULTI_SPECIALIST} cases requiring TWO specialists",
        len(multi_valid) >= MIN_MULTI_SPECIALIST,
        f"found {len(multi_valid)} ({', '.join(c['id'] for c in multi_valid)})",
    )
    if len(multi) != len(multi_valid):
        notes.append(
            "some requires_multiple_specialists cases list fewer than two "
            "expected tools"
        )

    rai = [c for c in cases if c["category"] == "responsible_ai"]
    check(
        f"at least {MIN_RESPONSIBLE_AI} responsible_ai cases",
        len(rai) >= MIN_RESPONSIBLE_AI,
        f"found {len(rai)}",
    )
    check(
        "every responsible_ai case has a non-trivial expected_output",
        all(len(c["expected_output"]) > 60 for c in rai),
        f"shortest is {min((len(c['expected_output']) for c in rai), default=0)} chars",
    )
    check(
        "every responsible_ai case declares a safe_behaviour",
        all(c.get("safe_behaviour") in case_loader.SAFE_BEHAVIOURS for c in rai),
    )

    case_ids = {c["id"] for c in cases}
    for theme, expected_ids in RAI_THEMES.items():
        present = [cid for cid in expected_ids if cid in case_ids]
        check(
            f"responsible_ai theme covered: {theme}",
            bool(present),
            f"{', '.join(present)}" if present else "no case found",
        )

    cancelled = [c for c in cases if c.get("asserts_cancelled_service_not_offered")]
    check(
        f"at least {MIN_CANCELLED_ASSERTIONS} cases asserting the cancelled "
        "service is never offered",
        len(cancelled) >= MIN_CANCELLED_ASSERTIONS,
        f"found {len(cancelled)} ({', '.join(c['id'] for c in cancelled)})",
    )

    delegates = [c for c in cases if c.get("asserts_router_delegates")]
    check(
        f"at least {MIN_ROUTER_DELEGATES} case asserting the router routes a "
        "policy question rather than answering it",
        len(delegates) >= MIN_ROUTER_DELEGATES,
        f"found {len(delegates)} ({', '.join(c['id'] for c in delegates)})",
    )
    check(
        "every router-delegation case targets the orchestrator",
        all(c["target"] == "train_travel_agent" for c in delegates),
    )

    print("\ncase counts")
    for group, values in counts.items():
        print(f"  {group}")
        for key, value in sorted(values.items()):
            print(f"    {key:<24} {value}")

    if notes:
        print("\nnotes")
        for note in notes:
            print(f"  - {note}")

    if failures:
        print(f"\n{len(failures)} requirement(s) NOT met:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\nall coverage requirements met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
