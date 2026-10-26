"""Shared LangGraph agent construction.

One place to build the model client and the ReAct loop, so all three agents
share identical model configuration and only differ in tools and prompt.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_aws import ChatBedrockConverse  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # noqa: E402
from langchain_core.runnables import Runnable  # noqa: E402
from langchain_core.tools import BaseTool  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402

from common.config import AWS_REGION, model_id  # noqa: E402

# Deterministic output matters more than variety for a scripted demo.
TEMPERATURE = 0.0
MAX_TOKENS = 2048


def build_llm(temperature: float = TEMPERATURE) -> ChatBedrockConverse:
    """Bedrock Converse client, with the model resolved at call time.

    The model id comes from common/config_store.py, so an operator can
    repoint every agent by changing one Parameter Store value -- no redeploy,
    no new runtime version.

    It must be a us.* cross-Region inference profile: the current Sonnet
    models have no In-Region endpoint, so an un-prefixed
    anthropic.claude-sonnet-* id fails with a validation error.
    """
    return ChatBedrockConverse(
        model_id=model_id(),
        region_name=AWS_REGION,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
    )


def model_for_turn(tools: list[BaseTool]) -> Callable[[Any, Any], Runnable]:
    """Build the dynamic-model callable that create_react_agent will invoke.

    Exposed (rather than nested inside build_agent) so the tool binding is
    directly testable. The failure it guards against is silent: no exception,
    no warning, just an agent that narrates tool calls instead of making them.
    """

    def resolve(_state: Any, _runtime: Any) -> Runnable:
        # bind_tools IS REQUIRED HERE. create_react_agent binds the tools for
        # you only on the static path; for a dynamic (callable) model it uses
        # whatever the callable returns, unchanged. LangGraph's own contract:
        # "Ensure returned models have appropriate tools bound via
        # .bind_tools(). Bound tools must be a subset of those specified in the
        # tools parameter."
        #
        # Omitting it does not fail loudly. The model simply never receives the
        # tool schemas, and because the system prompt still lists the tools by
        # name, it imitates the provider's function-call syntax as ordinary
        # prose -- the agent appears to "call" a tool while nothing executes.
        llm = build_llm()
        return llm.bind_tools(tools) if tools else llm

    return resolve


def build_agent(
    name: str,
    tools: list[BaseTool],
    system_prompt: str,
) -> CompiledStateGraph:
    """Build a ReAct agent over the given tools.

    The model is passed as a CALLABLE, not an instance. create_react_agent
    accepts Callable[[State, Runtime], BaseChatModel] and invokes it on every
    agent invocation, so the model id is re-resolved from Parameter Store each
    turn (behind config_store's TTL cache). Changing the parameter repoints a
    running container without rebuilding the graph or redeploying.

    Passing a constructed client instead would bake the model into the
    compiled graph, and since the graph is built once per container, a model
    change would need a new deployment.

    No LangGraph checkpointer, deliberately. A checkpointer is LangGraph's own
    conversation-state store: it snapshots the graph's message list per
    thread_id so a later invocation resumes mid-conversation. We do not use one
    because AgentCore Memory already owns conversation state -- events are the
    short-term store, and the orchestrator loads them into `history` at the
    start of each turn. Running both would give two stores that drift apart,
    and only one of them survives a session boundary.

    Sub-agents get neither: they receive self-contained instructions and hold
    no conversation of their own.
    """
    return create_react_agent(
        model_for_turn(tools),
        tools,
        prompt=system_prompt,
        name=name,
    )


async def run_agent(
    agent: CompiledStateGraph,
    prompt: str,
    history: list[BaseMessage] | None = None,
    extra_context: str = "",
) -> str:
    """Invoke an agent with a single user turn and return its final text.

    `history` is prior turns of the same conversation (short-term memory).
    `extra_context` is long-term memory, prepended to the user turn as fenced,
    self-labelling data rather than merged into the system prompt -- so the
    model sees a clear boundary between its standing instructions and
    recalled material it should treat with suspicion.
    """
    messages: list[BaseMessage] = list(history or [])
    content = f"{extra_context}\n{prompt}" if extra_context else prompt
    messages.append(HumanMessage(content=content))

    result = await agent.ainvoke({"messages": messages})
    return final_text(result["messages"])


def final_text(messages: list[BaseMessage]) -> str:
    """Extract the last assistant text from a LangGraph message list.

    Bedrock Converse returns content as a list of typed blocks, so a plain
    str(content) would leak Python repr into the answer.
    """
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        content = message.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type", "text") == "text"
            ]
            joined = "\n".join(p for p in parts if p).strip()
            if joined:
                return joined
    return "(the agent produced no text response)"


def describe_tools(tools: list[BaseTool]) -> str:
    """Render the tool catalogue for inclusion in a system prompt."""
    return "\n".join(f"  - {tool.name}: {(tool.description or '').strip().splitlines()[0]}"
                     for tool in tools)
