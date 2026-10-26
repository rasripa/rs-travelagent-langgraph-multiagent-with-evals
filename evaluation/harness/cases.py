"""Load, validate and filter the evaluation dataset.

The dataset is the contract between the two runners. Both read it through this
module so a filter flag means the same thing in each, and so a malformed case
fails here with a precise message rather than halfway through a paid run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = EVAL_ROOT / "datasets" / "cases.json"

# Canonical target keys. `policy_advisor` is this subsystem's name for the
# Tier-2 agent that owns the rail policy and FAQ knowledge base. The core
# application still keys that agent `policy_advisor` internally, so the
# alias below is accepted on input and normalised away -- see
# targets.AGENT_KEYS for where the two names are reconciled.
TARGETS = (
    "train_travel_agent",
    "journey_planner",
    "journey_support",
    "policy_advisor",
)

TARGET_ALIASES = {"policy_advisor": "policy_advisor"}

TIERS = ("orchestrator", "sub_agent")
CATEGORIES = ("quality", "responsible_ai")
SAFE_BEHAVIOURS = ("refuse", "comply_safely")

# Which tier each target must declare. The orchestrator is the only Tier-1
# agent; everything else is a Tier-2 specialist reached directly over A2A.
EXPECTED_TIER = {
    "train_travel_agent": "orchestrator",
    "journey_planner": "sub_agent",
    "journey_support": "sub_agent",
    "policy_advisor": "sub_agent",
}

REQUIRED_FIELDS = (
    "id",
    "target",
    "tier",
    "category",
    "actor_id",
    "input",
    "expected_output",
)

VALID_ACTORS = ("TRV-1001", "TRV-1002", "TRV-1003")


class DatasetError(ValueError):
    """The dataset is malformed. Raised before any case is run."""


def normalise_target(target: str) -> str:
    return TARGET_ALIASES.get(target, target)


def load(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Read and validate the dataset, returning the case list."""
    cases_path = Path(path) if path else DEFAULT_CASES_PATH
    if not cases_path.exists():
        raise DatasetError(f"No dataset at {cases_path}")

    try:
        document = json.loads(cases_path.read_text())
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{cases_path} is not valid JSON: {exc}") from exc

    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise DatasetError(f"{cases_path} has no non-empty 'cases' array.")

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, case in enumerate(raw_cases):
        cases.append(_validate(case, index, seen))
    return cases


def _validate(case: Any, index: int, seen: set[str]) -> dict[str, Any]:
    where = f"cases[{index}]"
    if not isinstance(case, dict):
        raise DatasetError(f"{where} is not an object.")

    missing = [f for f in REQUIRED_FIELDS if not case.get(f)]
    if missing:
        raise DatasetError(f"{where} (id={case.get('id')!r}) is missing: {missing}")

    case = dict(case)
    case_id = case["id"]
    if case_id in seen:
        raise DatasetError(f"{where}: duplicate case id {case_id!r}.")
    seen.add(case_id)

    case["target"] = normalise_target(case["target"])
    if case["target"] not in TARGETS:
        raise DatasetError(
            f"{case_id}: target {case['target']!r} is not one of {list(TARGETS)}."
        )
    if case["tier"] not in TIERS:
        raise DatasetError(f"{case_id}: tier {case['tier']!r} is not one of {TIERS}.")
    if case["tier"] != EXPECTED_TIER[case["target"]]:
        raise DatasetError(
            f"{case_id}: target {case['target']} must declare tier "
            f"{EXPECTED_TIER[case['target']]!r}, found {case['tier']!r}."
        )
    if case["category"] not in CATEGORIES:
        raise DatasetError(
            f"{case_id}: category {case['category']!r} is not one of {CATEGORIES}."
        )
    if case["actor_id"] not in VALID_ACTORS:
        raise DatasetError(
            f"{case_id}: actor_id {case['actor_id']!r} is not a known traveller "
            f"{list(VALID_ACTORS)}."
        )

    tools = case.setdefault("expected_tools", [])
    if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
        raise DatasetError(f"{case_id}: expected_tools must be a list of strings.")

    # Responsible-AI cases must say which safe behaviour is correct, because the
    # two demand opposite verdicts from a refusal detector: declining is right
    # for a data-exfiltration attempt and wrong for a leading policy question,
    # where the correct behaviour is to answer accurately. Defaulting to
    # "refuse" would silently mark every corrected-premise case as a failure.
    if case["category"] == "responsible_ai":
        behaviour = case.setdefault("safe_behaviour", "refuse")
        if behaviour not in SAFE_BEHAVIOURS:
            raise DatasetError(
                f"{case_id}: safe_behaviour {behaviour!r} is not one of "
                f"{SAFE_BEHAVIOURS}."
            )
    elif "safe_behaviour" in case:
        raise DatasetError(
            f"{case_id}: safe_behaviour only applies to responsible_ai cases."
        )

    return case


def select(
    cases: list[dict[str, Any]],
    target: str | None = None,
    category: str | None = None,
    tier: str | None = None,
    case_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Filter a case list. Every filter is optional and they compose."""
    chosen = cases
    if target:
        wanted = normalise_target(target)
        chosen = [c for c in chosen if c["target"] == wanted]
    if category:
        chosen = [c for c in chosen if c["category"] == category]
    if tier:
        chosen = [c for c in chosen if c["tier"] == tier]
    if case_ids:
        wanted_ids = set(case_ids)
        chosen = [c for c in chosen if c["id"] in wanted_ids]
    if limit is not None and limit >= 0:
        chosen = chosen[:limit]
    return chosen


def counts(cases: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Case counts by target, tier and category, for reporting."""
    summary: dict[str, dict[str, int]] = {
        "by_target": {},
        "by_tier": {},
        "by_category": {},
    }
    for case in cases:
        for key, field in (
            ("by_target", "target"),
            ("by_tier", "tier"),
            ("by_category", "category"),
        ):
            summary[key][case[field]] = summary[key].get(case[field], 0) + 1
    return summary
