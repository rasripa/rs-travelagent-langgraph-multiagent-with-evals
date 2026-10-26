"""Evaluation controls for the demo frontend: run a suite, build the report.

Mounted by frontend/server.py. Kept in its own module because it does something
categorically different from the chat proxy -- it SPAWNS PROCESSES -- and that
deserves to be reviewable on its own.

SECURITY POSTURE, stated plainly
--------------------------------
These endpoints execute local commands. That is a command-execution surface,
and the only reason it is acceptable here is that every part of the command is
chosen by THIS file, never by the caller:

  * The interpreter and script paths are module constants.
  * `runner` is looked up in RUNNERS; anything else is rejected before any
    process is created. The value is never interpolated into a path.
  * `scope` is looked up in SCOPES, which maps a short key to a FIXED argv
    fragment. The caller cannot pass arbitrary flags, filenames or values.
  * Commands are built as an argv LIST and run without a shell, so there is no
    string for a metacharacter to escape into.
  * One job at a time. A second request while a job is running is refused
    rather than queued, so the page cannot be used to fork-bomb the host.

The server binds 127.0.0.1 and has no authentication, which is the same posture
as the chat proxy it sits beside: acceptable only because nothing off-host can
reach it. Do not widen that bind. An unauthenticated endpoint that runs local
commands and spends money on Bedrock is not something to publish, and port
forwarding is the right way to show it to someone else.

Evaluation is also deliberately NOT importable from the application: the
runners live in their own virtual environment with their own dependencies, and
this module shells out to that interpreter rather than importing it.
"""
from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - argv-only, no shell; see module docstring
import threading
import time
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]

# The evaluation venv is separate on purpose -- its dependencies must never
# reach the deployment package.
EVAL_PYTHON = ROOT / "evaluation" / ".venv" / "bin" / "python"
PROJECT_PYTHON = ROOT / ".venv" / "bin" / "python"

RUNNERS: dict[str, dict[str, Any]] = {
    "strands": {
        "label": "Strands Evals",
        "script": ROOT / "evaluation" / "runners" / "strands_runner.py",
    },
    "langsmith": {
        "label": "LangSmith (offline)",
        "script": ROOT / "evaluation" / "runners" / "langsmith_runner.py",
    },
}

DASHBOARD_SCRIPT = ROOT / "evaluation" / "report" / "build_dashboard.py"
DASHBOARD_HTML = ROOT / "evaluation" / "report" / "dashboard.html"
RESULTS_DIR = ROOT / "evaluation" / "results"
LOG_DIR = ROOT / ".run" / "logs"

# Caller-selectable scope -> fixed argv fragment. This is an allowlist, not a
# template: a key that is not here never becomes part of a command.
SCOPES: dict[str, list[str]] = {
    "all": [],
    "quick": ["--limit", "4"],
    "responsible_ai": ["--category", "responsible_ai"],
    "orchestrator": ["--tier", "orchestrator"],
    "sub_agents": ["--tier", "sub_agent"],
    "journey_planner": ["--target", "journey_planner"],
    "journey_support": ["--target", "journey_support"],
    "policy_advisor": ["--target", "policy_advisor"],
}

# Caller-selectable mode -> fixed argv fragment. Another allowlist, not a
# template. Defaults to the frontend's own run mode so the two tabs cannot
# disagree: it would be actively misleading for the badge to read "AgentCore
# Runtime" while the evaluation buttons silently scored the local stack.
MODES: dict[str, list[str]] = {
    "local": ["--mode", "local"],
    "agentcore": ["--mode", "agentcore"],
}

DEFAULT_CONCURRENCY = 12
MAX_CONCURRENCY = 24
LOG_TAIL_LINES = 40

_lock = threading.Lock()
_job: dict[str, Any] | None = None


def _log_path(runner: str) -> Path:
    return LOG_DIR / f"eval-{runner}.log"


def _tail(path: Path, lines: int = LOG_TAIL_LINES) -> list[str]:
    if not path.exists():
        return []
    try:
        content = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    return [line for line in content[-lines:] if line.strip()]


def _latest_results() -> dict[str, Any]:
    """Headline numbers from the most recent run of each runner."""
    summaries: dict[str, Any] = {}
    if not RESULTS_DIR.exists():
        return summaries
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        runner = data.get("runner")
        if not runner:
            continue
        previous = summaries.get(runner)
        if previous and str(data.get("started_at", "")) <= previous["started_at"]:
            continue
        summary = data.get("summary") or {}
        perf = summary.get("perf") or {}
        summaries[runner] = {
            "runner": runner,
            "label": RUNNERS.get(runner, {}).get("label", runner),
            "file": path.name,
            "started_at": str(data.get("started_at", "")),
            "mode": data.get("mode"),
            "judge_model": data.get("judge_model"),
            "cases": summary.get("cases"),
            "assertions": summary.get("assertions"),
            "assertions_passed": summary.get("assertions_passed"),
            "pass_rate": summary.get("pass_rate"),
            "mean_score": summary.get("mean_score"),
            "errors": summary.get("cases_with_errors"),
            "p50": perf.get("p50_seconds"),
            "p95": perf.get("p95_seconds"),
            "responsible_ai": (summary.get("by_category") or {}).get(
                "responsible_ai"
            ),
        }
    return summaries


def _default_mode() -> str:
    """The frontend's own run mode, so the Evaluation tab matches the Chat tab."""
    from common.config import RUN_MODE

    return RUN_MODE if RUN_MODE in MODES else "local"


def _state() -> dict[str, Any]:
    job = _job
    running = bool(job and job["process"].poll() is None)
    payload: dict[str, Any] = {
        "running": running,
        "eval_env_ready": EVAL_PYTHON.exists(),
        "dashboard_exists": DASHBOARD_HTML.exists(),
        "runners": {k: v["label"] for k, v in RUNNERS.items()},
        "scopes": sorted(SCOPES),
        "modes": sorted(MODES),
        "default_mode": _default_mode(),
        "results": _latest_results(),
    }
    if job:
        returncode = job["process"].poll()
        payload["job"] = {
            "kind": job["kind"],
            "runner": job.get("runner"),
            "scope": job.get("scope"),
            "mode": job.get("mode"),
            "label": job["label"],
            "started_at": job["started_at"],
            "elapsed_seconds": round(time.time() - job["started"], 1),
            "returncode": returncode,
            "ok": returncode == 0 if returncode is not None else None,
            "log": _tail(job["log_path"]),
        }
    return payload


async def eval_status(_request: Request) -> JSONResponse:
    return JSONResponse(_state())


async def eval_run(request: Request) -> JSONResponse:
    """Start one evaluation run. Refuses if another job is in flight."""
    global _job

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}

    runner = str(body.get("runner", ""))
    if runner not in RUNNERS:
        return JSONResponse(
            {
                "error": f"unknown runner {runner!r}; expected one of "
                f"{sorted(RUNNERS)}"
            },
            status_code=400,
        )

    scope = str(body.get("scope", "all"))
    if scope not in SCOPES:
        return JSONResponse(
            {"error": f"unknown scope {scope!r}; expected one of {sorted(SCOPES)}"},
            status_code=400,
        )

    from common.config import RUN_MODE

    default_mode = RUN_MODE if RUN_MODE in MODES else "local"
    mode = str(body.get("mode") or default_mode)
    if mode not in MODES:
        return JSONResponse(
            {"error": f"unknown mode {mode!r}; expected one of {sorted(MODES)}"},
            status_code=400,
        )

    try:
        concurrency = int(body.get("concurrency", DEFAULT_CONCURRENCY))
    except (TypeError, ValueError):
        return JSONResponse({"error": "concurrency must be an integer"}, 400)
    concurrency = max(1, min(concurrency, MAX_CONCURRENCY))

    if not EVAL_PYTHON.exists():
        return JSONResponse(
            {
                "error": (
                    "the evaluation environment is not set up. Create it once:\n"
                    "  cd evaluation && uv venv --python 3.12 .venv\n"
                    "  VIRTUAL_ENV=.venv uv pip install -r requirements-eval.txt"
                )
            },
            status_code=409,
        )

    with _lock:
        if _job and _job["process"].poll() is None:
            return JSONResponse(
                {
                    "error": f"{_job['label']} is still running; wait for it to "
                    f"finish before starting another run"
                },
                status_code=409,
            )

        # argv list, no shell, every element chosen here -- see module docstring.
        command = [
            str(EVAL_PYTHON),
            str(RUNNERS[runner]["script"]),
            "--concurrency",
            str(concurrency),
            *MODES[mode],
            *SCOPES[scope],
        ]
        env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _log_path(runner)
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(  # noqa: S603 # nosec B603
            command, stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env
        )
        _job = {
            "kind": "run",
            "runner": runner,
            "scope": scope,
            "mode": mode,
            "label": f"{RUNNERS[runner]['label']} ({scope}, {mode})",
            "process": process,
            "log_path": log_path,
            "started": time.time(),
            "started_at": time.strftime("%H:%M:%S"),
        }

    return JSONResponse({"started": True, **_state()})


async def eval_report(_request: Request) -> JSONResponse:
    """Rebuild the comparison dashboard from whatever results exist."""
    global _job

    with _lock:
        if _job and _job["process"].poll() is None:
            return JSONResponse(
                {"error": f"{_job['label']} is still running"}, status_code=409
            )

    has_results = RESULTS_DIR.exists() and any(RESULTS_DIR.glob("*.json"))
    if not has_results:
        return JSONResponse(
            {"error": "no evaluation results yet -- run at least one suite first"},
            status_code=409,
        )

    python = PROJECT_PYTHON if PROJECT_PYTHON.exists() else EVAL_PYTHON
    completed = subprocess.run(  # noqa: S603 # nosec B603
        [str(python), str(DASHBOARD_SCRIPT)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        return JSONResponse(
            {
                "error": "building the dashboard failed",
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-2000:],
            },
            status_code=500,
        )
    return JSONResponse(
        {
            "built": True,
            "output": completed.stdout.strip().splitlines(),
            **_state(),
        }
    )


async def eval_dashboard(_request: Request) -> Any:
    """Serve the generated dashboard, so it opens in a tab from the UI."""
    if not DASHBOARD_HTML.exists():
        return JSONResponse(
            {"error": "no dashboard yet -- use 'Generate report & compare' first"},
            status_code=404,
        )
    return FileResponse(DASHBOARD_HTML, media_type="text/html")
