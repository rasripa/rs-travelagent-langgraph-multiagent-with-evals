"""The shared result schema both runners write, and its summary roll-up.

Keeping this in one module is what makes the side-by-side dashboard possible:
"pass rate" and "mean score" are computed by the same code for both runners, so
a difference between them is a difference in the JUDGES, not in the arithmetic.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.harness import perf

EVAL_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = EVAL_ROOT / "results"

SCHEMA_VERSION = 1


def timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def build(
    runner: str,
    mode: str,
    judge_model: str,
    started_at: str,
    case_results: list[dict[str, Any]],
    dataset_path: str,
    evaluator_names: list[str],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the full result document, summary included."""
    return {
        "schema_version": SCHEMA_VERSION,
        "runner": runner,
        "started_at": started_at,
        "finished_at": now_iso(),
        "mode": mode,
        "judge_model": judge_model,
        "dataset": dataset_path,
        "evaluators": evaluator_names,
        "notes": notes or [],
        "cases": case_results,
        "summary": summarise(case_results),
    }


def _metric_rows(case_results: list[dict[str, Any]]) -> list[tuple[str, float, bool]]:
    rows: list[tuple[str, float, bool]] = []
    for case in case_results:
        for metric, result in (case.get("scores") or {}).items():
            score = result.get("score")
            if not isinstance(score, (int, float)):
                continue
            rows.append((metric, float(score), bool(result.get("passed"))))
    return rows


def _aggregate(
    case_results: list[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for case in case_results:
        key = str(case.get(field, "unknown"))
        bucket = buckets.setdefault(
            key, {"cases": 0, "scored": 0, "passed": 0, "score_sum": 0.0, "errors": 0}
        )
        bucket["cases"] += 1
        if case.get("error"):
            bucket["errors"] += 1
        for result in (case.get("scores") or {}).values():
            score = result.get("score")
            if not isinstance(score, (int, float)):
                continue
            bucket["scored"] += 1
            bucket["score_sum"] += float(score)
            if result.get("passed"):
                bucket["passed"] += 1

    for bucket in buckets.values():
        scored = bucket.pop("scored")
        total = bucket.pop("score_sum")
        bucket["assertions"] = scored
        bucket["mean_score"] = round(total / scored, 4) if scored else None
        bucket["pass_rate"] = round(bucket["passed"] / scored, 4) if scored else None
    return buckets


def summarise(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll per-case scores up by metric, target and category, plus perf."""
    by_metric: dict[str, dict[str, Any]] = {}
    for metric, score, passed in _metric_rows(case_results):
        bucket = by_metric.setdefault(
            metric, {"assertions": 0, "passed": 0, "score_sum": 0.0}
        )
        bucket["assertions"] += 1
        bucket["score_sum"] += score
        bucket["passed"] += int(passed)
    for bucket in by_metric.values():
        count = bucket["assertions"]
        bucket["mean_score"] = round(bucket.pop("score_sum") / count, 4) if count else None
        bucket["pass_rate"] = round(bucket["passed"] / count, 4) if count else None

    rows = _metric_rows(case_results)
    total_assertions = len(rows)
    total_passed = sum(1 for _, _, passed in rows if passed)
    mean_score = (
        round(sum(score for _, score, _ in rows) / total_assertions, 4)
        if total_assertions
        else None
    )

    latencies = [
        float(case["latency_seconds"])
        for case in case_results
        if isinstance(case.get("latency_seconds"), (int, float))
    ]
    per_case_perf = [case.get("perf") or {} for case in case_results]

    return {
        "cases": len(case_results),
        "cases_with_errors": sum(1 for c in case_results if c.get("error")),
        "assertions": total_assertions,
        "assertions_passed": total_passed,
        "pass_rate": round(total_passed / total_assertions, 4) if total_assertions else None,
        "mean_score": mean_score,
        "by_metric": by_metric,
        "by_target": _aggregate(case_results, "target"),
        "by_category": _aggregate(case_results, "category"),
        "by_tier": _aggregate(case_results, "tier"),
        "perf": perf.summarise(latencies, per_case_perf),
    }


def write(document: dict[str, Any], out_path: str | Path | None = None) -> Path:
    """Write the result document, defaulting to results/<runner>-<timestamp>.json."""
    if out_path:
        path = Path(out_path)
    else:
        path = RESULTS_DIR / f"{document['runner']}-{timestamp_slug()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, default=str))
    return path


def case_record(
    case: dict[str, Any],
    invocation: dict[str, Any],
    scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """One row of the shared `cases` array."""
    timings = invocation.get("timings") or {}
    return {
        "id": case["id"],
        "target": case["target"],
        "tier": case["tier"],
        "category": case["category"],
        "safe_behaviour": case.get("safe_behaviour"),
        "input": case["input"],
        "expected_output": case["expected_output"],
        "expected_tools": case.get("expected_tools", []),
        "scores": scores,
        "output": invocation.get("output", ""),
        "trajectory": invocation.get("trajectory", []),
        "latency_seconds": round(float(timings.get("total_seconds", 0.0) or 0.0), 4),
        "perf": timings,
        "session_id": invocation.get("session_id", ""),
        "error": invocation.get("error"),
    }
