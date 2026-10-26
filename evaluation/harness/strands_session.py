"""Build a `strands_evals` Session from what the harness observed.

Why this exists
---------------
Eight of the nine evaluators used by the Strands runner declare
`evaluation_level = EvaluationLevel.TRACE_LEVEL`, and the base `Evaluator`
rejects anything other than a `Session` for `actual_trajectory`:

    Trace parsing requires actual_trajectory to be a Session object, got list.

So returning `{"output": ..., "trajectory": ["toolA"]}` from the task function
works for the deterministic evaluators but fails every semantic one. The task
function must hand back a real Session.

The harness runs outside the agent processes and cannot see their internal
turns, so the Session is SYNTHESISED from the single real turn it does observe:
the prompt it sent and the answer it got back. Tool spans are added only for
tool calls that were actually evidenced -- see trajectory.py, which recovers
tool-call counts per tool server locally and never invents a tool name.

What that means for the scores, stated plainly: the semantic evaluators judge a
faithful single-turn transcript, not the agent's full internal reasoning. They
are therefore evaluating the system's OUTPUT behaviour, which is what this
dataset asserts on anyway. A multi-turn or tool-argument-level judgement would
need in-process instrumentation, which is out of scope here.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from strands_evals.types.trace import (
    AgentInvocationSpan,
    Session,
    SpanInfo,
    SpanType,
    ToolCall,
    ToolConfig,
    ToolExecutionSpan,
    ToolResult,
    Trace,
)

TOOL_RESULT_PLACEHOLDER = (
    "(tool result not captured: the harness observes the system from outside "
    "the agent process)"
)


def build(
    prompt: str,
    answer: str,
    session_id: str,
    duration_seconds: float = 0.0,
    observed_tools: list[str] | None = None,
    available_tools: list[str] | None = None,
    system_prompt: str | None = None,
) -> Session:
    """Assemble a single-trace Session for one evaluated turn."""
    end = datetime.now(UTC)
    start = end - timedelta(seconds=max(duration_seconds, 0.0))

    trace_id = f"eval-{session_id}"
    agent_span_id = "agent-0"

    spans: list[Any] = [
        AgentInvocationSpan(
            span_info=SpanInfo(
                session_id=session_id,
                trace_id=trace_id,
                span_id=agent_span_id,
                start_time=start,
                end_time=end,
            ),
            span_type=SpanType.AGENT_INVOCATION,
            user_prompt=prompt,
            agent_response=answer,
            available_tools=[ToolConfig(name=name) for name in (available_tools or [])],
            system_prompt=system_prompt,
        )
    ]

    for index, tool_name in enumerate(observed_tools or []):
        spans.append(
            ToolExecutionSpan(
                span_info=SpanInfo(
                    session_id=session_id,
                    trace_id=trace_id,
                    span_id=f"tool-{index}",
                    parent_span_id=agent_span_id,
                    start_time=start,
                    end_time=end,
                ),
                span_type=SpanType.TOOL_EXECUTION,
                tool_call=ToolCall(
                    name=tool_name, arguments={}, tool_call_id=f"call-{index}"
                ),
                tool_result=ToolResult(
                    content=TOOL_RESULT_PLACEHOLDER, tool_call_id=f"call-{index}"
                ),
                agent_span_id=agent_span_id,
            )
        )

    return Session(
        session_id=session_id,
        traces=[Trace(trace_id=trace_id, session_id=session_id, spans=spans)],
    )
