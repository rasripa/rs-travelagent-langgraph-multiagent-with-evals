"""Latency measurement and per-component breakdown.

Two sources, in order of preference:

  1. agentcore mode -- OpenTelemetry spans from the CloudWatch Logs group
     `aws/spans`, grouped by traceId and attributed to a layer by
     resource.attributes["service.name"] and scope.name. This is the only
     source that yields a real per-component breakdown, because it is the only
     one that can see inside the agents.

  2. local mode -- wall-clock timing around each call the harness itself makes.
     That is one layer: the client-observed round trip. Everything below it is
     reported as UNAVAILABLE rather than estimated. A breakdown we did not
     measure is worse than no breakdown, because it looks like data.

Per-layer totals use a UNION OF INTERVALS, not a sum of durations. Child spans
nest inside their parents and siblings overlap, so summing durations
double-counts time and routinely produces a "model time" larger than the wall
clock of the whole request.
"""
from __future__ import annotations

import os
import time
from typing import Any

SPANS_LOG_GROUP = os.getenv("EVAL_SPANS_LOG_GROUP", "aws/spans")

# Scope name -> the layer its spans belong to. These are the instrumentation
# scopes emitted by aws-opentelemetry-distro, which requirements.txt pins at
# 0.21.0 and which deploy.py enables on every runtime.
SCOPE_LAYERS = {
    "amazon.opentelemetry.distro.instrumentation.langchain": "agent_turn",
    "opentelemetry.instrumentation.botocore.bedrock-runtime": "model",
    "amazon.opentelemetry.distro.instrumentation.mcp": "tool",
    "opentelemetry.instrumentation.httpx": "a2a_or_http",
}

TOKEN_ATTRIBUTES = (
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
)

# Published on-demand Bedrock pricing is not read from the API here, so cost is
# only derived when the operator supplies rates. Left unset, the dashboard
# reports tokens and says cost is unavailable rather than inventing a number.
PRICE_PER_1K_INPUT = os.getenv("EVAL_PRICE_PER_1K_INPUT_USD")
PRICE_PER_1K_OUTPUT = os.getenv("EVAL_PRICE_PER_1K_OUTPUT_USD")


class Stopwatch:
    """Monotonic wall-clock timer for one case."""

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.elapsed: float | None = None

    def stop(self) -> float:
        self.elapsed = time.monotonic() - self.started
        return self.elapsed


def local_breakdown(
    total_seconds: float,
    transport_seconds: float | None = None,
    tool_calls_by_server: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Describe what local mode can and cannot measure.

    `transport_seconds` is the HTTP/A2A round trip the harness timed itself.
    Everything inside the agent -- model time, tool time, the A2A hop -- is not
    visible from here and is reported as unavailable, NOT as total minus
    something.
    """
    measured = {"client_round_trip": round(total_seconds, 4)}
    if transport_seconds is not None:
        measured["transport"] = round(transport_seconds, 4)

    return {
        "source": "local wall clock",
        "total_seconds": round(total_seconds, 4),
        "measured": measured,
        "derived_by_subtraction": {},
        "unavailable": [
            "agent_turn",
            "model",
            "tool",
            "a2a_or_http",
        ],
        "unavailable_reason": (
            "Local mode has no span export. Per-layer timings require "
            "agentcore mode, where OpenTelemetry spans are read from the "
            f"{SPANS_LOG_GROUP} CloudWatch Logs group."
        ),
        "tool_calls_by_server": dict(tool_calls_by_server or {}),
        "tokens": {},
        "token_source": "unavailable in local mode",
    }


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Total time covered by a set of possibly overlapping [start, end) intervals."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    total += current_end - current_start
    return total


def _nanos_to_seconds(value: Any) -> float | None:
    try:
        return float(value) / 1_000_000_000.0
    except (TypeError, ValueError):
        return None


def fetch_spans(
    start_ms: int,
    end_ms: int,
    log_group: str = SPANS_LOG_GROUP,
    region: str | None = None,
) -> list[dict[str, Any]]:
    """Read raw spans from CloudWatch Logs between two epoch-millisecond marks.

    filter_log_events legitimately returns pages with ZERO events but a
    nextToken, so paging must follow the token rather than stopping on an empty
    page. Stopping early is the classic way to silently lose most of a trace.
    """
    import json

    import boto3

    client = boto3.client("logs", region_name=region or os.getenv("AWS_REGION", "us-west-2"))

    spans: list[dict[str, Any]] = []
    token: str | None = None
    pages = 0
    while True:
        request: dict[str, Any] = {
            "logGroupName": log_group,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 10_000,
        }
        if token:
            request["nextToken"] = token
        try:
            response = client.filter_log_events(**request)
        except Exception:  # noqa: BLE001
            # A missing log group or no Transaction Search means no breakdown,
            # not a failed evaluation run.
            break

        for event in response.get("events", []):
            try:
                spans.append(json.loads(event["message"]))
            except (json.JSONDecodeError, KeyError):
                continue

        token = response.get("nextToken")
        pages += 1
        # Follow the token even when this page was empty; only an absent token
        # ends the scan. The page cap is a runaway guard, not a stop condition.
        if not token or pages > 500:
            break
    return spans


def breakdown_from_spans(
    spans: list[dict[str, Any]],
    session_id: str | None = None,
) -> dict[str, Any]:
    """Attribute span time to layers and services, using a union of intervals."""
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        trace_id = span.get("traceId") or span.get("trace_id") or ""
        by_trace.setdefault(trace_id, []).append(span)

    layer_intervals: dict[str, list[tuple[float, float]]] = {}
    service_intervals: dict[str, list[tuple[float, float]]] = {}
    root_intervals: list[tuple[float, float]] = []
    all_intervals: list[tuple[float, float]] = []
    tokens = {"input_tokens": 0, "output_tokens": 0}
    matched_traces: list[str] = []

    for trace_id, trace_spans in by_trace.items():
        if session_id and not _trace_mentions(trace_spans, session_id):
            continue
        matched_traces.append(trace_id)

        for span in trace_spans:
            start = _nanos_to_seconds(span.get("startTimeUnixNano"))
            end = _nanos_to_seconds(span.get("endTimeUnixNano"))
            if start is None or end is None or end < start:
                continue

            all_intervals.append((start, end))

            scope = ((span.get("scope") or {}).get("name")) or ""
            layer = SCOPE_LAYERS.get(scope)
            if layer:
                layer_intervals.setdefault(layer, []).append((start, end))

            resource_attributes = (span.get("resource") or {}).get("attributes") or {}
            service = resource_attributes.get("service.name") or "unknown"
            service_intervals.setdefault(service, []).append((start, end))

            if not span.get("parentSpanId"):
                root_intervals.append((start, end))

            attributes = span.get("attributes") or {}
            for attribute in TOKEN_ATTRIBUTES:
                value = attributes.get(attribute)
                if value is None:
                    continue
                try:
                    tokens[attribute.rsplit(".", 1)[-1]] += int(value)
                except (TypeError, ValueError):
                    continue

    by_layer = {k: round(_union_seconds(v), 4) for k, v in layer_intervals.items()}
    by_service = {k: round(_union_seconds(v), 4) for k, v in service_intervals.items()}

    # The turn's total. Prefer the union of root spans, but a deployed turn
    # routinely has none: the true root is the CALLING client (invoke.py, the
    # frontend proxy, this harness), which is not instrumented, so every span
    # the runtime emits carries a parentSpanId. Treating that as a zero-second
    # turn is wrong and silently empties the performance panel, so fall back to
    # the envelope of every matched span and say which basis was used.
    if root_intervals:
        trace_total = round(_union_seconds(root_intervals), 4)
        total_basis = "union of root spans (spans with no parentSpanId)"
    else:
        trace_total = round(_union_seconds(all_intervals), 4)
        total_basis = (
            "union of every matched span -- no span lacked a parentSpanId, "
            "because the turn's root is the calling client and is not "
            "instrumented"
        )

    # Time inside the trace that no instrumented scope claimed: framework
    # overhead, serialisation, cold start. Explicitly labelled as derived by
    # subtraction so nobody mistakes it for something that was measured.
    attributed = by_layer.get("agent_turn", 0.0)
    unattributed = round(max(trace_total - attributed, 0.0), 4) if trace_total else 0.0

    total_tokens = tokens["input_tokens"] + tokens["output_tokens"]
    result: dict[str, Any] = {
        "source": f"opentelemetry spans from {SPANS_LOG_GROUP}",
        "traces_matched": len(matched_traces),
        "total_seconds": trace_total,
        "total_basis": total_basis,
        "measured": {
            "by_layer": by_layer,
            "by_service": by_service,
            "note": (
                "Each figure is a UNION of span intervals, so nested and "
                "overlapping spans are counted once. Layer totals therefore "
                "do not sum to the trace total."
            ),
        },
        "derived_by_subtraction": {
            "unattributed_in_root_trace": unattributed,
        },
        "unavailable": [],
        "tokens": {**tokens, "total_tokens": total_tokens} if total_tokens else {},
        "token_source": (
            "gen_ai.usage.* on bedrock-runtime spans"
            if total_tokens
            else "no token attributes found on the matched spans"
        ),
    }
    cost = derive_cost(tokens["input_tokens"], tokens["output_tokens"])
    if cost is not None:
        result["cost_usd"] = cost
        result["cost_basis"] = (
            f"derived: {PRICE_PER_1K_INPUT} USD/1k input, "
            f"{PRICE_PER_1K_OUTPUT} USD/1k output (operator-supplied rates)"
        )
    else:
        result["cost_basis"] = (
            "unavailable: set EVAL_PRICE_PER_1K_INPUT_USD and "
            "EVAL_PRICE_PER_1K_OUTPUT_USD to derive a cost per turn"
        )
    return result


def _trace_mentions(trace_spans: list[dict[str, Any]], session_id: str) -> bool:
    """True if any span in the trace carries this session id."""
    for span in trace_spans:
        attributes = span.get("attributes") or {}
        for value in attributes.values():
            if isinstance(value, str) and session_id in value:
                return True
    return False


def derive_cost(input_tokens: int, output_tokens: int) -> float | None:
    """Cost per turn, ONLY if the operator supplied rates. Never guessed."""
    if not PRICE_PER_1K_INPUT or not PRICE_PER_1K_OUTPUT:
        return None
    try:
        rate_in = float(PRICE_PER_1K_INPUT)
        rate_out = float(PRICE_PER_1K_OUTPUT)
    except ValueError:
        return None
    return round((input_tokens / 1000.0) * rate_in + (output_tokens / 1000.0) * rate_out, 6)


def percentile(values: list[float], fraction: float) -> float | None:
    """Linear-interpolated percentile. Returns None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 4)


def summarise(latencies: list[float], per_case_perf: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate latency and token statistics across cases."""
    clean = [v for v in latencies if isinstance(v, (int, float)) and v >= 0]
    stats: dict[str, Any] = {
        "samples": len(clean),
        "p50_seconds": percentile(clean, 0.50),
        "p95_seconds": percentile(clean, 0.95),
        "p99_seconds": percentile(clean, 0.99),
        "min_seconds": round(min(clean), 4) if clean else None,
        "max_seconds": round(max(clean), 4) if clean else None,
        "mean_seconds": round(sum(clean) / len(clean), 4) if clean else None,
        "measurement": "end-to-end wall clock per case, measured by the harness",
    }

    if len(clean) < 20:
        stats["sample_warning"] = (
            f"{len(clean)} sample(s): P95 and P99 are interpolated from too few "
            "points to be meaningful. Treat them as indicative only."
        )

    layer_totals: dict[str, float] = {}
    service_totals: dict[str, float] = {}
    tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    cost_total = 0.0
    cost_cases = 0
    unavailable: set[str] = set()
    sources: set[str] = set()

    for perf in per_case_perf:
        if not isinstance(perf, dict):
            continue
        sources.add(str(perf.get("source", "unknown")))
        for layer in perf.get("unavailable") or []:
            unavailable.add(str(layer))
        measured = perf.get("measured") or {}
        for layer, seconds in (measured.get("by_layer") or {}).items():
            layer_totals[layer] = round(layer_totals.get(layer, 0.0) + float(seconds), 4)
        for service, seconds in (measured.get("by_service") or {}).items():
            service_totals[service] = round(
                service_totals.get(service, 0.0) + float(seconds), 4
            )
        case_tokens = perf.get("tokens") or {}
        for key in tokens:
            try:
                tokens[key] += int(case_tokens.get(key, 0) or 0)
            except (TypeError, ValueError):
                continue
        if perf.get("cost_usd") is not None:
            cost_total += float(perf["cost_usd"])
            cost_cases += 1

    stats["by_layer_seconds"] = layer_totals
    stats["by_service_seconds"] = service_totals
    stats["layers_unavailable"] = sorted(unavailable)
    stats["sources"] = sorted(sources)
    stats["tokens"] = tokens if tokens["total_tokens"] or tokens["input_tokens"] else {}
    if cost_cases:
        stats["cost_usd_total"] = round(cost_total, 6)
        stats["cost_usd_per_turn_mean"] = round(cost_total / cost_cases, 6)
        stats["cost_basis"] = "derived from operator-supplied token rates"
    else:
        stats["cost_basis"] = (
            "unavailable: no token rates supplied (EVAL_PRICE_PER_1K_INPUT_USD / "
            "EVAL_PRICE_PER_1K_OUTPUT_USD) or no token data captured"
        )
    return stats
