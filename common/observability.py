"""OpenTelemetry tracing for the agents, exported to AgentCore Observability.

Design constraint: this project avoids platform SDKs in the agents, so tracing
is wired here with vendor-neutral `opentelemetry-*` packages and configured
entirely by environment variable. The AWS-specific pieces -- the X-Ray ID
generator and propagator from aws-opentelemetry-distro -- are imported
defensively, so the same code traces correctly on AgentCore, degrades to plain
W3C tracing anywhere else, and is a silent no-op when tracing is switched off.

What it gives you, which is the point worth demonstrating:

  ONE TRACE ACROSS THE WHOLE CALL GRAPH. A disruption query crossing
  orchestrator -> Journey Support -> Journey Planner -> gateway -> two MCP
  servers appears as a single trace, not five disconnected ones. That works
  because the httpx instrumentation injects W3C `traceparent` on every outbound
  A2A and MCP call, and the Starlette instrumentation picks it up on the way in
  and parents the server span to the caller's span.

Enable it by setting AGENT_OBSERVABILITY_ENABLED=true. deploy.py sets that plus
the OTEL_* variables on every runtime. Locally it stays off by default, because
a demo laptop has nowhere to export to.

One-time account setup: CloudWatch Transaction Search must be enabled, and the
X-Ray trace-segment destination pointed at CloudWatch Logs, before traces show
up on the CloudWatch GenAI Observability page. deploy.py reports whether that
is done; it does not change it for you, because it is an account-wide setting.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any
from collections.abc import Iterator

logger = logging.getLogger(__name__)

_INITIALISED = False


def enabled() -> bool:
    return os.getenv("AGENT_OBSERVABILITY_ENABLED", "").lower() == "true"


def service_name(default: str = "train-travel-agent") -> str:
    """Resolve the service name shown in the GenAI Observability dashboard.

    OTEL_RESOURCE_ATTRIBUTES wins over the caller's suggestion, so a deployed
    runtime reports the name the platform knows it by (and which matches its
    log group) while a local run keeps the readable label passed by the server.
    Having these disagree is why deployed spans could not be tied back to a
    runtime.
    """
    attributes = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")
    for pair in attributes.split(","):
        key, _, value = pair.partition("=")
        if key.strip() == "service.name" and value.strip():
            return value.strip()
    return os.getenv("OTEL_SERVICE_NAME", default)


def _span_exporter() -> tuple[Any, str]:
    """Build the span exporter and describe where it sends.

    The OTLP SDK's default endpoint is http://localhost:4318, which assumes a
    collector sidecar. AgentCore runs no collector -- the documented path is to
    post straight to the regional CloudWatch/X-Ray OTLP endpoint, SigV4-signed.
    Normally ADOT's own distro sets that up, but only when the process starts
    under `opentelemetry-instrument`; this project deploys plain code with an
    entryPoint, so nothing wires it and every export silently fails against
    localhost. Hence doing it here, explicitly.

    Returns (exporter, human-readable destination).
    """
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    # 1. An explicit endpoint always wins -- a local collector, a third-party
    #    OTLP backend, or a test double. Keeps this vendor-neutral.
    explicit = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    if explicit:
        return OTLPSpanExporter(), f"configured endpoint {explicit}"

    # 2. On AWS, post directly to the regional X-Ray OTLP endpoint. This needs
    #    SigV4, which the plain exporter cannot do, so use ADOT's signing
    #    subclass. Requires CloudWatch Transaction Search to be enabled on the
    #    account; without it the endpoint rejects the spans.
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if region:
        suffix = "amazonaws.com.cn" if region.startswith("cn-") else "amazonaws.com"
        endpoint = f"https://xray.{region}.{suffix}/v1/traces"
        try:
            from botocore.session import Session

            from amazon.opentelemetry.distro.exporter.otlp.aws.traces.otlp_aws_span_exporter import (  # noqa: E501
                OTLPAwsSpanExporter,
            )

            return (
                OTLPAwsSpanExporter(
                    aws_region=region, session=Session(), endpoint=endpoint
                ),
                f"{endpoint} (SigV4)",
            )
        except ImportError as exc:
            logger.warning(
                "Cannot sign trace exports (%s). The X-Ray OTLP endpoint "
                "requires SigV4, so spans would be rejected. Install "
                "aws-opentelemetry-distro, or set "
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT to a collector you run.",
                exc,
            )

    # 3. Nothing to export to. Say so loudly rather than letting the exporter
    #    retry against localhost and drop every batch while startup still
    #    reports tracing as active.
    logger.warning(
        "Tracing is enabled but no OTLP destination is configured and no AWS "
        "region is set, so spans will be dropped. Set AWS_REGION, or "
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, or unset "
        "AGENT_OBSERVABILITY_ENABLED."
    )
    return OTLPSpanExporter(), "SDK default (http://localhost:4318) -- NOT REACHABLE"


def adot_already_configured() -> bool:
    """True when the ADOT launcher has already built the OTEL pipeline.

    When the auto-instrumentation directory is on PYTHONPATH, Python imports
    its sitecustomize at interpreter startup, which loads the AWS distro and
    configurator and every matching instrumentor -- including botocore (Bedrock
    model calls) and the distro's own agentic instrumentation for LangChain and
    MCP. That is the pipeline the Bedrock AgentCore Observability dashboard
    reads: it needs GenAI spans for sessions and token counts, which plain HTTP
    instrumentation cannot produce.

    If it ran, this module must NOT build a second provider. OpenTelemetry
    refuses to override an existing TracerProvider, so doing so would leave the
    richer ADOT pipeline in place but have us silently exporting nothing.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
    except ImportError:
        return False
    return isinstance(trace.get_tracer_provider(), SdkTracerProvider)


def init(agent_name: str | None = None) -> bool:
    """Set up tracing. Returns True if tracing is active.

    Safe to call more than once; subsequent calls are no-ops.
    """
    global _INITIALISED
    if _INITIALISED:
        return True
    if not enabled():
        return False

    # Preferred path on AgentCore: ADOT already owns the pipeline. Adopt it
    # rather than competing with it.
    if adot_already_configured():
        _INITIALISED = True
        resolved = service_name(agent_name or "train-travel-agent")
        logger.info("adopting the ADOT pipeline for service.name=%s", resolved)
        print(f"  [{resolved}] tracing: ACTIVE -> ADOT auto-instrumentation")
        return True

    try:
        from opentelemetry import propagate, trace
        from opentelemetry.propagators.composite import CompositePropagator
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )
    except ImportError as exc:
        logger.warning(
            "AGENT_OBSERVABILITY_ENABLED is true but OpenTelemetry is not "
            "installed (%s). Tracing disabled. Install aws-opentelemetry-distro.",
            exc,
        )
        return False

    # Deployed runtimes carry service.name in OTEL_RESOURCE_ATTRIBUTES, and it
    # must win: it is the name the platform knows this runtime by. The caller's
    # label is only the local fallback.
    resolved_name = service_name(agent_name or "train-travel-agent")

    # X-Ray-compatible trace IDs, so spans land correctly in CloudWatch /
    # Transaction Search. Optional: without the AWS distro we still produce
    # valid W3C traces, just with random IDs.
    id_generator = None
    try:
        from opentelemetry.sdk.extension.aws.trace import AwsXRayIdGenerator

        id_generator = AwsXRayIdGenerator()
    except ImportError:
        logger.info("AWS X-Ray id generator unavailable; using default trace ids.")

    resource = Resource.create({"service.name": resolved_name})
    provider = (
        TracerProvider(resource=resource, id_generator=id_generator)
        if id_generator
        else TracerProvider(resource=resource)
    )
    # The OTLP protocol and headers come from OTEL_* environment variables; the
    # DESTINATION is resolved here, because the SDK's localhost default is
    # wrong on AgentCore and fails silently.
    exporter, destination = _span_exporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Accept BOTH W3C traceparent and X-Ray headers on the way in, and emit
    # both on the way out. Mixed-propagator meshes are the usual reason a
    # multi-agent trace arrives as several disconnected fragments.
    propagators: list[Any] = [TraceContextTextMapPropagator()]
    try:
        from opentelemetry.propagators.aws import AwsXRayPropagator

        propagators.append(AwsXRayPropagator())
    except ImportError:
        pass
    propagate.set_global_textmap(CompositePropagator(propagators))

    _instrument_httpx()
    _INITIALISED = True
    logger.info("tracing active for service.name=%s", resolved_name)
    # Printing the destination matters: "tracing: ACTIVE" on its own was true
    # while every span was being dropped against localhost:4318.
    print(f"  [{resolved_name}] tracing: ACTIVE -> {destination}")
    return True


def _instrument_httpx() -> None:
    """Trace outbound HTTP and inject W3C trace context.

    This is what stitches the A2A hops and the gateway MCP calls into one
    trace: every httpx request the agents make carries `traceparent`.
    """
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not instrument httpx: %s", exc)


def instrument_app(app: Any) -> Any:
    """Trace inbound requests on a Starlette app, continuing the caller's trace.

    Call this on the A2A and HTTP server apps. A no-op when tracing is off.
    """
    if not enabled():
        return app
    if adot_already_configured():
        # ADOT auto-instruments Starlette already; a second instrument_app
        # produces duplicate server spans.
        return app
    try:
        from opentelemetry.instrumentation.starlette import StarletteInstrumentor

        StarletteInstrumentor.instrument_app(app)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not instrument the Starlette app: %s", exc)
    return app


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Record an explicit span, e.g. around an agent turn or a delegation.

    Auto-instrumentation captures the HTTP hops; these spans give the trace the
    semantic shape a reviewer actually wants to read -- which agent ran, which
    peer it delegated to, which tool answered.
    """
    if not _INITIALISED:
        yield None
        return

    from opentelemetry import trace

    tracer = trace.get_tracer("train-travel-agent")
    with tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        try:
            yield current
        except Exception as exc:
            current.record_exception(exc)
            from opentelemetry.trace import Status, StatusCode

            current.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def otel_environment(agent_name: str, log_group: str) -> dict[str, str]:
    """The OTEL_* variables a runtime needs. Used by deploy.py.

    Kept here, next to the code that consumes them, so the two cannot drift.

    PYTHONPATH is what actually turns the pipeline on. These OTEL_PYTHON_DISTRO
    and OTEL_PYTHON_CONFIGURATOR values only take effect if something runs the
    OpenTelemetry launcher, and a direct-code deploy has no shell wrapper to
    run `opentelemetry-instrument`. That CLI does nothing more than prepend the
    auto_instrumentation directory to PYTHONPATH and re-exec, so putting the
    directory there ourselves is equivalent: Python imports its sitecustomize
    at interpreter startup, which loads the AWS distro, the AWS configurator
    and every matching instrumentor -- botocore for Bedrock model calls, and
    the distro's own LangChain and MCP agentic instrumentation that the Bedrock
    AgentCore Observability dashboard needs for sessions and token counts.
    Without it the libraries ship in the package and sit unused.

    Deliberately NOT set: OTEL_EXPORTER_OTLP_TRACES_HEADERS. Setting
    x-aws-log-group on traces redirects spans from the default aws/spans log
    group into our own, which additionally requires a CloudWatch Logs resource
    policy letting xray.amazonaws.com call logs:PutLogEvents on that group.
    Without that policy X-Ray cannot deliver and the spans are silently lost --
    so the default destination is the safer choice.

    Also deliberately NOT set: OTEL_EXPORTER_OTLP_TRACES_ENDPOINT. The AWS
    distro derives the regional X-Ray endpoint itself, and only via setdefault,
    so hardcoding one here would override it.

    OTLP LOG export is switched OFF. The distro turns it on by default when
    agent observability is enabled, but it is redundant here: AgentCore already
    captures stdout and stderr into the runtime log group, and the Bedrock
    AgentCore Observability dashboard is driven by SPANS -- sessions, token
    counts and the agent list all come from trace data, not log records. Left
    on, it ships a second copy of the same logs to the CloudWatch Logs OTLP
    endpoint, which also rejects every batch with 'code: 400 ... The specified
    log stream does not exist' once per second unless a fixed stream is
    pre-created. Nothing downstream wants that copy, so there is nothing to
    pre-create.
    """
    return {
        "AGENT_OBSERVABILITY_ENABLED": "true",
        "OTEL_PYTHON_DISTRO": "aws_distro",
        "OTEL_PYTHON_CONFIGURATOR": "aws_configurator",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
        "OTEL_TRACES_EXPORTER": "otlp",
        "OTEL_LOGS_EXPORTER": "none",
        "OTEL_RESOURCE_ATTRIBUTES": (
            f"service.name={agent_name},aws.log.group.names={log_group}"
        ),
    }


# Where the OpenTelemetry launcher's sitecustomize lives inside the deployed
# package. Prepending this to PYTHONPATH is what `opentelemetry-instrument`
# does, and is how the ADOT pipeline gets switched on without a shell wrapper.
AUTO_INSTRUMENTATION_PATH = (
    "/var/task/opentelemetry/instrumentation/auto_instrumentation"
)
