"""Standalone evaluation subsystem for the Train Travel Agent.

Runs entirely outside the application. Nothing under this package is imported
by the agents, the MCP servers or the deployment packager, and nothing here
modifies the core application: the harness drives the system through its real
public interfaces only -- POST /invocations for the orchestrator, A2A for the
sub-agents.

Dependencies are installed into a SEPARATE virtual environment from
`evaluation/requirements-eval.txt`. They are deliberately kept out of the
project's own `requirements.txt`, which the packager vendors into a deployment
zip that is already 58 MB against a 250 MB limit. An LLM-as-judge evaluator has
no business shipping to production.
"""
