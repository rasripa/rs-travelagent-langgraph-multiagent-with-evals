#!/usr/bin/env python3
"""Build a self-contained HTML dashboard from the evaluation result files.

    python evaluation/report/build_dashboard.py
    python evaluation/report/build_dashboard.py --results evaluation/results \
        --out evaluation/report/dashboard.html

Reads every `*.json` under the results directory, takes the LATEST run per
runner, and renders one page comparing them side by side.

Self-contained on purpose: the data is embedded, the charts are inline SVG and
styled divs, and there is no stylesheet, font or script fetched from a network.
The file opens from the filesystem, survives being emailed, and can be committed
as the artefact of a run -- which matters because an evaluation report is
evidence, and evidence that only renders while a CDN is reachable is not.

It also refuses to flatter the numbers. Where the harness recorded that a
layer could not be measured, or that P95 came from too few samples, the page
says so next to the figure rather than drawing a confident bar.
"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "evaluation" / "results"
DEFAULT_OUT = ROOT / "evaluation" / "report" / "dashboard.html"

# Matches docs/architecture.html so the two read as one set of documents.
NAVY = "#0b2e59"
ACCENT = "#15509e"
BG = "#eef3fb"
PANEL = "#eaf1fb"
BORDER = "#c3d4ec"
INK = "#13263f"
MUTED = "#5a7a9e"
GOOD = "#1b7a46"
WARN = "#e08a00"
BAD = "#c0392b"

RUNNER_COLOUR = {"strands": ACCENT, "langsmith": "#7a4fb5"}
RUNNER_LABEL = {
    "strands": "Strands Evals",
    "langsmith": "LangSmith (offline)",
}

# The two SDKs name a few equivalent metrics differently. Grouping them lets the
# comparison line up instead of showing two half-empty columns; the original
# names stay visible in the per-case table.
METRIC_ALIASES = {
    "responserelevance": "relevance",
    "response_relevance": "relevance",
    "instructionfollowing": "instruction_following",
    "instruction_following": "instruction_following",
    "groundedness": "faithfulness",
}

RESPONSIBLE_AI_METRICS = {
    "harmfulness",
    "stereotyping",
    "bias",
    "refusal",
    "appropriate_refusal",
    "safety",
}


def canonical_metric(name: str) -> str:
    key = name.strip().lower()
    return METRIC_ALIASES.get(key, key)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


# ------------------------------------------------------------------ loading
def load_runs(results_dir: Path) -> dict[str, dict]:
    """Latest run per runner, keyed by runner name."""
    runs: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"  skipping {path.name}: not valid JSON ({exc})")
            continue
        runner = data.get("runner")
        if not runner:
            print(f"  skipping {path.name}: no 'runner' field")
            continue
        data["_source_file"] = path.name
        previous = runs.get(runner)
        if previous is None or str(data.get("started_at", "")) >= str(
            previous.get("started_at", "")
        ):
            runs[runner] = data
    return runs


# ------------------------------------------------------------------ pieces
def card(label: str, value: str, sub: str = "", colour: str = NAVY) -> str:
    sub_html = (
        f'<div style="font-size:11.5px; color:{MUTED}; margin-top:4px;">{sub}</div>'
        if sub
        else ""
    )
    return f"""
    <div style="background:#fff; border:1px solid {BORDER}; border-radius:10px;
                padding:14px 18px; min-width:150px; flex:1;">
      <div style="font-size:11px; letter-spacing:1.3px; text-transform:uppercase;
                  color:{MUTED};">{label}</div>
      <div style="font-size:26px; font-weight:700; color:{colour}; margin-top:6px;">{value}</div>
      {sub_html}
    </div>"""


def score_colour(score: float | None) -> str:
    if score is None:
        return MUTED
    if score >= 0.8:
        return GOOD
    if score >= 0.5:
        return WARN
    return BAD


def bar(fraction: float, colour: str, width: int = 150) -> str:
    """One horizontal bar, drawn with divs so it needs no chart library."""
    pct = max(0.0, min(1.0, fraction)) * 100
    return f"""<span style="display:inline-block; width:{width}px; height:9px;
      background:#dde6f4; border-radius:5px; overflow:hidden; vertical-align:middle;">
      <span style="display:block; width:{pct:.1f}%; height:100%;
      background:{colour};"></span></span>"""


def grouped_metric_chart(runs: dict[str, dict]) -> str:
    """Per-metric mean score, one bar per runner -- the side-by-side view."""
    per_runner: dict[str, dict[str, dict]] = {}
    for runner, data in runs.items():
        collapsed: dict[str, list[float]] = {}
        for metric, stats in (data.get("summary", {}).get("by_metric") or {}).items():
            collapsed.setdefault(canonical_metric(metric), []).append(stats)
        per_runner[runner] = {
            metric: {
                "mean_score": sum(s.get("mean_score") or 0 for s in group)
                / max(len(group), 1),
                "pass_rate": sum(s.get("pass_rate") or 0 for s in group)
                / max(len(group), 1),
                "assertions": sum(s.get("assertions") or 0 for s in group),
            }
            for metric, group in collapsed.items()
        }

    metrics = sorted({m for stats in per_runner.values() for m in stats})
    if not metrics:
        return f'<p style="color:{MUTED};">No per-metric data in these runs.</p>'

    rows = []
    for metric in metrics:
        is_rai = metric in RESPONSIBLE_AI_METRICS
        tag = (
            '<span style="font-size:10px; background:#fff1dc; color:#8a5a00;'
            ' border-radius:4px; padding:1px 6px; margin-left:6px;">RAI</span>'
            if is_rai
            else ""
        )
        cells = []
        for runner in sorted(runs):
            stats = per_runner.get(runner, {}).get(metric)
            if not stats:
                cells.append(
                    f'<div style="font-size:12px; color:{MUTED};">not measured</div>'
                )
                continue
            mean = stats["mean_score"]
            cells.append(
                f'<div style="display:flex; align-items:center; gap:8px;">'
                f'{bar(mean, RUNNER_COLOUR.get(runner, ACCENT), 120)}'
                f'<span style="font-size:12.5px; font-weight:600;'
                f' color:{score_colour(mean)};">{mean:.2f}</span>'
                f'<span style="font-size:11px; color:{MUTED};">'
                f'({stats["assertions"]} assertion(s))</span></div>'
            )
        rows.append(
            f'<tr><td style="padding:9px 12px; border:1px solid {BORDER};'
            f' font-weight:600; color:{NAVY};">{esc(metric)}{tag}</td>'
            + "".join(
                f'<td style="padding:9px 12px; border:1px solid {BORDER};">{c}</td>'
                for c in cells
            )
            + "</tr>"
        )

    headers = "".join(
        f'<th style="text-align:left; padding:9px 12px; border:1px solid {NAVY};">'
        f"{RUNNER_LABEL.get(r, r)}</th>"
        for r in sorted(runs)
    )
    return f"""
    <table style="width:100%; border-collapse:collapse; font-size:13.5px;">
      <thead><tr style="background:{NAVY}; color:#fff;">
        <th style="text-align:left; padding:9px 12px; border:1px solid {NAVY}; width:26%;">Metric</th>
        {headers}
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def breakdown_table(runs: dict[str, dict], key: str, title: str) -> str:
    buckets = sorted(
        {
            b
            for data in runs.values()
            for b in (data.get("summary", {}).get(key) or {})
        }
    )
    if not buckets:
        return ""
    rows = []
    for bucket in buckets:
        cells = []
        for runner in sorted(runs):
            stats = (runs[runner].get("summary", {}).get(key) or {}).get(bucket)
            if not stats:
                cells.append(f'<span style="color:{MUTED};">&ndash;</span>')
                continue
            mean = stats.get("mean_score")
            rate = stats.get("pass_rate")
            cells.append(
                f'<span style="font-weight:600; color:{score_colour(mean)};">'
                f'{(mean if mean is not None else 0):.2f}</span>'
                f'<span style="color:{MUTED}; font-size:11.5px;"> &middot; '
                f'{(rate if rate is not None else 0) * 100:.0f}% pass '
                f'&middot; n={stats.get("assertions", 0)}</span>'
            )
        rows.append(
            f'<tr><td style="padding:8px 12px; border:1px solid {BORDER};">'
            f"<code>{esc(bucket)}</code></td>"
            + "".join(
                f'<td style="padding:8px 12px; border:1px solid {BORDER};">{c}</td>'
                for c in cells
            )
            + "</tr>"
        )
    headers = "".join(
        f'<th style="text-align:left; padding:9px 12px; border:1px solid {NAVY};">'
        f"{RUNNER_LABEL.get(r, r)}</th>"
        for r in sorted(runs)
    )
    return f"""
    <h3 style="color:{NAVY}; font-size:15px; margin:18px 0 8px;">{title}</h3>
    <table style="width:100%; border-collapse:collapse; font-size:13px; margin-bottom:6px;">
      <thead><tr style="background:{NAVY}; color:#fff;">
        <th style="text-align:left; padding:9px 12px; border:1px solid {NAVY}; width:26%;">{title}</th>
        {headers}
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def perf_section(runs: dict[str, dict]) -> str:
    blocks = []
    for runner in sorted(runs):
        perf = runs[runner].get("summary", {}).get("perf") or {}
        if not perf:
            continue
        warn = perf.get("sample_warning")
        warn_html = (
            f'<div style="background:#fff6e6; border-left:5px solid {WARN};'
            f' border-radius:6px; padding:9px 13px; margin:10px 0;'
            f' font-size:12.5px;">{esc(warn)}</div>'
            if warn
            else ""
        )

        lat = "".join(
            card(
                label,
                f"{perf.get(field, 0):.2f}s" if perf.get(field) is not None else "n/a",
                colour=ACCENT,
            )
            for label, field in (
                ("P50", "p50_seconds"),
                ("P95", "p95_seconds"),
                ("P99", "p99_seconds"),
                ("Mean", "mean_seconds"),
                ("Max", "max_seconds"),
            )
        )

        layers = perf.get("by_layer_seconds") or {}
        total = sum(layers.values()) or 0
        if layers:
            layer_rows = "".join(
                f'<tr><td style="padding:7px 12px; border:1px solid {BORDER};">'
                f"<strong>{esc(name)}</strong></td>"
                f'<td style="padding:7px 12px; border:1px solid {BORDER};">'
                f"{bar(secs / total if total else 0, ACCENT, 180)}</td>"
                f'<td style="padding:7px 12px; border:1px solid {BORDER};'
                f' text-align:right;">{secs:.2f}s</td>'
                f'<td style="padding:7px 12px; border:1px solid {BORDER};'
                f' text-align:right; color:{MUTED};">'
                f"{(secs / total * 100 if total else 0):.0f}%</td></tr>"
                for name, secs in sorted(layers.items(), key=lambda kv: -kv[1])
            )
            layer_html = f"""
            <table style="width:100%; border-collapse:collapse; font-size:13px; margin-top:10px;">
              <thead><tr style="background:{NAVY}; color:#fff;">
                <th style="text-align:left; padding:8px 12px; border:1px solid {NAVY};">Component</th>
                <th style="text-align:left; padding:8px 12px; border:1px solid {NAVY};">Share</th>
                <th style="text-align:right; padding:8px 12px; border:1px solid {NAVY};">Time</th>
                <th style="text-align:right; padding:8px 12px; border:1px solid {NAVY};">%</th>
              </tr></thead>
              <tbody>{layer_rows}</tbody>
            </table>"""
        else:
            layer_html = (
                f'<div style="background:#f4f7fc; border:1px dashed {BORDER};'
                f' border-radius:8px; padding:12px 16px; margin-top:10px;'
                f' font-size:12.5px; color:{MUTED};">'
                f"<strong>No per-component breakdown in this run.</strong> "
                f"Per-layer timings come from OpenTelemetry spans in the "
                f"<code>aws/spans</code> log group, which only exist when "
                f"running against the deployment "
                f"(<code>EVAL_MODE=agentcore</code>). Unmeasured layers are "
                f"listed rather than estimated: "
                f"<code>{esc(', '.join(perf.get('layers_unavailable') or []) or 'n/a')}</code>."
                f"</div>"
            )

        tokens = perf.get("tokens") or {}
        token_html = ""
        if tokens:
            token_html = (
                '<div style="margin-top:10px; display:flex; gap:10px; flex-wrap:wrap;">'
                + "".join(
                    card(k.replace("_", " "), f"{v:,}" if isinstance(v, int) else str(v))
                    for k, v in tokens.items()
                )
                + "</div>"
            )
        cost = perf.get("cost_basis")
        cost_html = (
            f'<div style="font-size:12px; color:{MUTED}; margin-top:8px;">'
            f"<strong>Cost basis:</strong> {esc(cost)}</div>"
            if cost
            else ""
        )

        blocks.append(f"""
        <div style="background:{PANEL}; border:1px solid {BORDER}; border-radius:10px;
                    padding:16px 20px; margin-bottom:14px;">
          <div style="font-weight:700; color:{NAVY}; font-size:15px;">
            {RUNNER_LABEL.get(runner, runner)}
            <span style="font-weight:400; font-size:12.5px; color:{MUTED};">
              &middot; {esc(perf.get('samples', 0))} sample(s)
              &middot; {esc(', '.join(perf.get('sources') or []) or 'source not recorded')}
            </span>
          </div>
          <div style="font-size:12px; color:{MUTED}; margin-top:4px;">
            {esc(perf.get('measurement', ''))}
          </div>
          {warn_html}
          <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:10px;">{lat}</div>
          {layer_html}
          {token_html}
          {cost_html}
        </div>""")
    return "".join(blocks)


def case_table(runs: dict[str, dict]) -> str:
    """Per-case detail, with the judge's reasoning behind a disclosure."""
    by_case: dict[str, dict[str, Any]] = {}
    for runner, data in runs.items():
        for case in data.get("cases", []):
            entry = by_case.setdefault(
                case["id"],
                {
                    "id": case["id"],
                    "target": case.get("target", ""),
                    "tier": case.get("tier", ""),
                    "category": case.get("category", ""),
                    "input": case.get("input", ""),
                    "runners": {},
                },
            )
            entry["runners"][runner] = case

    if not by_case:
        return f'<p style="color:{MUTED};">No cases in these runs.</p>'

    rows = []
    for case_id in sorted(by_case):
        entry = by_case[case_id]
        cat = entry["category"]
        cat_colour = WARN if cat == "responsible_ai" else ACCENT
        cells = []
        for runner in sorted(runs):
            case = entry["runners"].get(runner)
            if case is None:
                cells.append(f'<span style="color:{MUTED};">not run</span>')
                continue
            if case.get("error"):
                cells.append(
                    f'<span style="color:{BAD}; font-weight:600;">ERROR</span>'
                    f'<div style="font-size:11px; color:{BAD};">'
                    f'{esc(str(case["error"])[:160])}</div>'
                )
                continue
            scores = case.get("scores") or {}
            mean = (
                sum(s.get("score") or 0 for s in scores.values()) / len(scores)
                if scores
                else None
            )
            failed = [m for m, s in scores.items() if not s.get("passed")]
            detail = "".join(
                f"<div style='margin:6px 0;'><strong>{esc(m)}</strong>: "
                f"<span style='color:{score_colour(s.get('score'))};'>"
                f"{(s.get('score') if s.get('score') is not None else 0):.2f}</span>"
                f" &middot; {'pass' if s.get('passed') else 'FAIL'}"
                f"<div style='color:{MUTED}; font-size:11.5px; white-space:pre-wrap;'>"
                f"{esc(str(s.get('reason', ''))[:900])}</div></div>"
                for m, s in sorted(scores.items())
            )
            cells.append(
                f'<div style="font-weight:600; color:{score_colour(mean)};">'
                f'{(mean if mean is not None else 0):.2f}</div>'
                f'<div style="font-size:11px; color:{MUTED};">'
                f'{esc(case.get("latency_seconds", 0)):.6}s'
                + (
                    f' &middot; <span style="color:{BAD};">'
                    f'{len(failed)} failed</span>'
                    if failed
                    else " &middot; all passed"
                )
                + "</div>"
                f'<details style="margin-top:5px;"><summary style="cursor:pointer;'
                f' font-size:11.5px; color:{ACCENT};">judge detail</summary>'
                f'<div style="margin-top:6px; max-width:520px;">{detail}</div>'
                f"</details>"
                f'<details style="margin-top:4px;"><summary style="cursor:pointer;'
                f' font-size:11.5px; color:{ACCENT};">agent output</summary>'
                f'<div style="margin-top:6px; max-width:520px; font-size:11.5px;'
                f' color:{INK}; white-space:pre-wrap;">'
                f'{esc(str(case.get("output", ""))[:1800])}</div></details>'
            )
        rows.append(
            f'<tr><td style="padding:9px 12px; border:1px solid {BORDER};'
            f' vertical-align:top;">'
            f'<div style="font-weight:600; color:{NAVY}; font-size:12.5px;">'
            f'<code>{esc(entry["id"])}</code></div>'
            f'<div style="font-size:11px; color:{MUTED}; margin-top:3px;">'
            f'{esc(entry["target"])} &middot; {esc(entry["tier"])}</div>'
            f'<div style="margin-top:4px;"><span style="font-size:10px;'
            f' background:{cat_colour}; color:#fff; border-radius:4px;'
            f' padding:1px 6px;">{esc(cat)}</span></div>'
            f'<div style="font-size:11.5px; color:{INK}; margin-top:6px;'
            f' max-width:330px;">{esc(entry["input"][:220])}</div></td>'
            + "".join(
                f'<td style="padding:9px 12px; border:1px solid {BORDER};'
                f' vertical-align:top;">{c}</td>'
                for c in cells
            )
            + "</tr>"
        )

    headers = "".join(
        f'<th style="text-align:left; padding:9px 12px; border:1px solid {NAVY};">'
        f"{RUNNER_LABEL.get(r, r)}</th>"
        for r in sorted(runs)
    )
    return f"""
    <table style="width:100%; border-collapse:collapse; font-size:13px;">
      <thead><tr style="background:{NAVY}; color:#fff;">
        <th style="text-align:left; padding:9px 12px; border:1px solid {NAVY}; width:34%;">Case</th>
        {headers}
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


# ------------------------------------------------------------------ page
def build_page(runs: dict[str, dict]) -> str:
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    headline = []
    for runner in sorted(runs):
        data = runs[runner]
        summary = data.get("summary", {})
        rate = summary.get("pass_rate") or 0
        mean = summary.get("mean_score") or 0
        errors = summary.get("cases_with_errors") or 0
        headline.append(f"""
        <div style="background:#fff; border:1px solid {BORDER}; border-left:6px solid
                    {RUNNER_COLOUR.get(runner, ACCENT)}; border-radius:10px;
                    padding:16px 20px; flex:1; min-width:300px;">
          <div style="font-weight:700; color:{NAVY}; font-size:16px;">
            {RUNNER_LABEL.get(runner, runner)}</div>
          <div style="font-size:11.5px; color:{MUTED}; margin-top:3px;">
            {esc(data.get('_source_file', ''))} &middot; mode
            <strong>{esc(data.get('mode', '?'))}</strong> &middot; judge
            <code>{esc(data.get('judge_model', '?'))}</code>
          </div>
          <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:12px;">
            {card("Pass rate", f"{rate * 100:.0f}%", colour=score_colour(rate))}
            {card("Mean score", f"{mean:.2f}", colour=score_colour(mean))}
            {card("Cases", str(summary.get("cases", 0)),
                  sub=f"{summary.get('assertions', 0)} assertions")}
            {card("Errors", str(errors), colour=BAD if errors else GOOD)}
          </div>
          <div style="font-size:11.5px; color:{MUTED}; margin-top:10px;">
            evaluators: {esc(", ".join(data.get("evaluators") or []) or "n/a")}
          </div>
        </div>""")

    notes = []
    for runner in sorted(runs):
        for note in runs[runner].get("notes") or []:
            notes.append(f"<li><strong>{RUNNER_LABEL.get(runner, runner)}:</strong> "
                         f"{esc(note)}</li>")
    notes_html = (
        f'<ul style="margin:0; padding-left:20px; font-size:13px;">{"".join(notes)}</ul>'
        if notes
        else f'<p style="margin:0; color:{MUTED};">No runner notes recorded.</p>'
    )

    missing = [r for r in ("strands", "langsmith") if r not in runs]
    missing_html = (
        f'<div style="background:#fff6e6; border-left:6px solid {WARN};'
        f' border-radius:8px; padding:14px 18px; margin-bottom:18px;">'
        f'<strong>Side-by-side incomplete.</strong> No results found for: '
        f'<code>{esc(", ".join(missing))}</code>. The comparison below shows '
        f'only the runner(s) that produced output.</div>'
        if missing
        else ""
    )

    embedded = json.dumps(
        {r: {k: v for k, v in d.items() if k != "cases"} for r, d in runs.items()},
        indent=1,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Train Travel Agent &middot; Evaluation Dashboard</title>
</head>
<body style="margin:0; padding:0; background:{BG}; color:{INK};
             font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
             line-height:1.55; font-size:15px;">

<div style="background:linear-gradient(135deg,{NAVY} 0%,{ACCENT} 100%);
            color:#fff; padding:34px 40px;">
  <div style="font-size:12px; letter-spacing:2.4px; text-transform:uppercase;
              color:#9ec2f0;">Agent Evaluation &middot; LLM-as-a-Judge</div>
  <h1 style="margin:8px 0 6px 0; font-size:29px; font-weight:700;">
    Train Travel Agent &mdash; Evaluation Dashboard</h1>
  <div style="font-size:15px; color:#cfe0f7; max-width:940px;">
    Two independent evaluation SDKs scoring the same cases against the same
    multi-agent system: each Tier-2 sub-agent on its own, and the router
    end-to-end.
  </div>
  <div style="margin-top:14px; font-size:12px; color:#9ec2f0;">
    Generated {esc(generated)} &middot; quality and responsible-AI metrics
    &middot; synthetic data throughout
  </div>
</div>

<div style="padding:26px 40px;">

  {missing_html}

  <h2 style="color:{NAVY}; font-size:20px; border-bottom:2px solid {BORDER};
             padding-bottom:8px;">1. Headline</h2>
  <div style="display:flex; gap:14px; flex-wrap:wrap; margin-bottom:22px;">
    {"".join(headline)}
  </div>

  <h2 style="color:{NAVY}; font-size:20px; border-bottom:2px solid {BORDER};
             padding-bottom:8px;">2. Metric comparison, side by side</h2>
  <p style="margin-top:10px;">Mean judge score per metric, 0.00&ndash;1.00. Rows
  tagged <strong>RAI</strong> are responsible-AI checks rather than quality
  checks. Where the two SDKs give an equivalent metric a different name they are
  grouped under one row; the original names remain in the per-case table.</p>
  {grouped_metric_chart(runs)}
  <div style="background:#d6e4fb; border-radius:9px; padding:13px 18px;
              margin:16px 0 24px;">
    <strong style="color:{NAVY};">Reading this honestly:</strong> two judges
    scoring the same output will not agree exactly, and neither is ground truth.
    A gap of a few hundredths between runners is noise. What is worth acting on
    is a metric where one runner reports a <em>failure</em> and the other does
    not, or any responsible-AI row below 1.00.
  </div>

  <h2 style="color:{NAVY}; font-size:20px; border-bottom:2px solid {BORDER};
             padding-bottom:8px;">3. Breakdowns</h2>
  {breakdown_table(runs, "by_target", "Agent under test")}
  {breakdown_table(runs, "by_tier", "Tier")}
  {breakdown_table(runs, "by_category", "Category")}
  <div style="height:14px;"></div>

  <h2 style="color:{NAVY}; font-size:20px; border-bottom:2px solid {BORDER};
             padding-bottom:8px;">4. Performance</h2>
  <p style="margin-top:10px;">Latency distribution and, where spans were
  available, the per-component breakdown. Layers that could not be measured are
  named rather than estimated.</p>
  {perf_section(runs)}

  <h2 style="color:{NAVY}; font-size:20px; border-bottom:2px solid {BORDER};
             padding-bottom:8px;">5. Per-case detail</h2>
  <p style="margin-top:10px;">Every case, both runners. Expand
  <em>judge detail</em> for the per-metric reasoning, or <em>agent output</em>
  for what the system actually replied.</p>
  {case_table(runs)}

  <h2 style="color:{NAVY}; font-size:20px; border-bottom:2px solid {BORDER};
             padding-bottom:8px; margin-top:28px;">6. Run notes and caveats</h2>
  <div style="background:{PANEL}; border:1px solid {BORDER}; border-radius:9px;
              padding:16px 20px;">
    {notes_html}
  </div>

  <details style="margin-top:22px;">
    <summary style="cursor:pointer; color:{ACCENT}; font-size:13px;">
      Embedded run metadata (JSON)</summary>
    <pre style="background:#2b2b2b; color:#f3f3f3; border-radius:8px;
                padding:14px 18px; overflow-x:auto; font-size:11.5px;">{esc(embedded)}</pre>
  </details>

  <div style="margin-top:26px; padding-top:14px; border-top:1px solid {BORDER};
              font-size:12px; color:{MUTED};">
    Self-contained: no external stylesheet, font, script or chart library. All
    travellers, bookings, fares and policies are synthetic.
  </div>
</div>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.results.exists():
        print(f"no results directory at {args.results}")
        return 1

    runs = load_runs(args.results)
    if not runs:
        print(f"no usable result files in {args.results}")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_page(runs), encoding="utf-8")

    print(f"wrote {args.out}")
    for runner in sorted(runs):
        summary = runs[runner].get("summary", {})
        print(
            f"  {RUNNER_LABEL.get(runner, runner):22s} "
            f"cases={summary.get('cases', 0):3d} "
            f"pass_rate={(summary.get('pass_rate') or 0) * 100:5.1f}% "
            f"mean={(summary.get('mean_score') or 0):.2f} "
            f"({runs[runner].get('_source_file')})"
        )
    missing = [r for r in ("strands", "langsmith") if r not in runs]
    if missing:
        print(f"  NOTE: no results for {', '.join(missing)} -- comparison is partial")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
