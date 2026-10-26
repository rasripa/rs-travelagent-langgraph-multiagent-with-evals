"""Provision the Train Travel Agent demo on Amazon Bedrock AgentCore.

Authorization is IAM/SigV4 end to end. There is no Cognito user pool, no OAuth
client, and no secret written to disk anywhere. Each agent's identity is its
execution role, and that same role ARN is what the gateway's Cedar policies
match on as AgentCore::IamEntity.

What it creates, in dependency order:

  1. IAM         8 roles: one per agent (4), one shared by the MCP runtimes,
                 one for the gateway, one for the knowledge base, one for
                 memory
  2. package     one arm64 code zip, uploaded to S3
  3. runtimes    3 MCP-protocol runtimes (the mock MCP servers)
  4. policy      a policy engine, created before the gateway that references it
  5. gateway     1 gateway, AWS_IAM inbound, policy engine in LOG_ONLY first
  6. targets     3 mcpServer targets + 1 knowledge-base connector target
  7. policies    3 Cedar policies keyed on the per-agent role ARNs -- one per
                 agent that HAS a grant, which is every agent except the router
  8. agents      3 A2A-protocol runtimes + 1 HTTP-protocol orchestrator
  9. env         deploy/agentcore/.env (no secrets -- just ARNs and URLs)
 10. enforce     flip the policy engine to ENFORCE, then verify the filtering

Five things worth knowing about why it is shaped this way:

  * ONE ROLE PER SUB-AGENT IS MANDATORY. With IAM auth the Cedar principal is
    the caller's role. Sharing one execution role between Journey Planner,
    Journey Support and Policy Advisor would make them indistinguishable to
    Cedar and the whole per-agent tool filtering story would collapse.

  * THE ROUTER HOLDS NO GATEWAY TOOLS, AND SO HOLDS NO GATEWAY PERMISSION.
    Tier 1 classifies intent and delegates; the knowledge base it used to own
    moved to the Policy Advisor sub-agent. Its grant in AGENT_TOOL_GRANTS is
    therefore empty, which has three consequences this script has to honour:
    no Cedar policy is created for it (default-deny already says "no tools",
    and a permit with an empty action list is not expressible), no
    InvokeGateway in its execution role (an IAM grant for tools it cannot see
    would contradict the design), and `verify` expects it to see 0 tools --
    which is a PASS, not a failure.

  * mcpServer targets need an explicit iamCredentialProvider. For MCP server
    and OpenAPI targets, IAM outbound authorization requires
    credentialProviderConfigurations to carry an iamCredentialProvider with
    service="bedrock-agentcore" -- credentialProviderType alone is not enough.
    (Lambda, API Gateway and Smithy targets are the opposite: they take
    GATEWAY_IAM_ROLE with no iamCredentialProvider.)

  * Sub-agent runtimes are created, then UPDATED with their own invocation URL.
    A2A clients route by the agent card's `url` field, so a card advertising
    0.0.0.0:9000 breaks every delegation. The URL derives from the runtime ARN,
    which does not exist until after creation -- hence two phases.

  * The policy engine starts in LOG_ONLY so a Cedar mistake cannot silently
    empty an agent's tool catalogue. Step 14 flips it to ENFORCE and then
    checks that each agent's tools/list really did shrink.

Usage:
    python deploy/agentcore/deploy.py preflight     # check access first
    python deploy/agentcore/deploy.py deploy        # all steps, idempotent
    python deploy/agentcore/deploy.py deploy --step gateway
    python deploy/agentcore/deploy.py deploy --from-step policies
    python deploy/agentcore/deploy.py verify        # re-check Cedar filtering
    python deploy/agentcore/deploy.py teardown      # delete everything
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import boto3  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402

from common import config_store  # noqa: E402
from common.config import AWS_REGION, KB_DIR, embed_model_id, model_id  # noqa: E402
from common.observability import (  # noqa: E402
    AUTO_INSTRUMENTATION_PATH,
    otel_environment,
)

PREFIX = "travel-demo"
# Policy engines, policies and runtimes accept [A-Za-z][A-Za-z0-9_]* only --
# no hyphens. Gateways and gateway targets do allow them. Two conventions in
# one API, so keep both prefixes explicit rather than remembering which is
# which at each call site.
PREFIX_U = PREFIX.replace("-", "_")
ENV_PATH = Path(__file__).parent / ".env"
RENDERED_POLICY_PATH = Path(__file__).parent / "policies.rendered.cedar"
POLICY_TEMPLATE = ROOT / "deploy" / "gateway" / "policies.cedar"
STATE_PATH = Path(__file__).parent / ".deploy-state.json"

GATEWAY_NAME = f"{PREFIX}-gateway"
POLICY_ENGINE_NAME = f"{PREFIX_U}_policy_engine"
KB_NAME = f"{PREFIX}-rail-policy-kb"
KB_TARGET = "RailPolicyKB"
# Memory names share the underscore-only rule: [a-zA-Z][a-zA-Z0-9_]{0,47}
MEMORY_NAME = f"{PREFIX_U}_memory"
MEMORY_EVENT_EXPIRY_DAYS = int(os.getenv("TRAVEL_MEMORY_EXPIRY_DAYS", "30"))

# Role names. The four agent roles appear verbatim in the Cedar policies.
ROLE_MCP = f"{PREFIX}-mcp-runtime-role"
ROLE_GATEWAY = f"{PREFIX}-gateway-role"
ROLE_KB = f"{PREFIX}-kb-role"
ROLE_MEMORY = f"{PREFIX}-memory-role"
ROLE_PLANNER = f"{PREFIX}-journey-planner-role"
ROLE_SUPPORT = f"{PREFIX}-journey-support-role"
ROLE_POLICY_ADVISOR = f"{PREFIX}-policy-advisor-role"
ROLE_ORCHESTRATOR = f"{PREFIX}-orchestrator-role"

AGENT_ROLES = {
    "journey_planner": ROLE_PLANNER,
    "journey_support": ROLE_SUPPORT,
    "policy_advisor": ROLE_POLICY_ADVISOR,
    "train_travel_agent": ROLE_ORCHESTRATOR,
}

# Every role this script creates, in one place so that adding a role cannot
# leave teardown behind -- an orphaned role is invisible until the next deploy
# reuses it with stale permissions.
ALL_ROLES = (
    ROLE_MCP,
    ROLE_GATEWAY,
    ROLE_KB,
    ROLE_MEMORY,
    ROLE_PLANNER,
    ROLE_SUPPORT,
    ROLE_POLICY_ADVISOR,
    ROLE_ORCHESTRATOR,
)

MCP_SERVERS = {
    "traveller_profile": {
        "target": "TravellerProfile",
        "entry": "mcp_servers/traveller_profile/server.py",
    },
    "journey_search": {
        "target": "JourneySearch",
        "entry": "mcp_servers/journey_search/server.py",
    },
    "ticket_operations": {
        "target": "TicketOperations",
        "entry": "mcp_servers/ticket_operations/server.py",
    },
}

SUB_AGENTS = {
    "journey_planner": "agents/journey_planner/server.py",
    "journey_support": "agents/journey_support/server.py",
    "policy_advisor": "agents/policy_advisor/server.py",
}
ORCHESTRATOR = ("train_travel_agent", "agents/train_travel_agent/server.py")

# Agents that get a Cedar policy, in the order their permit statements appear
# in deploy/gateway/policies.cedar. train_travel_agent is deliberately absent:
# its grant in AGENT_TOOL_GRANTS is empty, and Cedar has no way to say "permit
# nothing" -- an action list must be non-empty. Default-deny already expresses
# it, so the router simply gets no policy. See do_policies.
POLICY_AGENTS = ("journey_planner", "journey_support", "policy_advisor")

session = boto3.Session(region_name=AWS_REGION)
agentcore = session.client("bedrock-agentcore-control")
bedrock_agent = session.client("bedrock-agent")
iam = session.client("iam")
s3 = session.client("s3")
sts = session.client("sts")
xray = session.client("xray")
ssm = session.client("ssm")
logs_client = session.client("logs")


# ---------------------------------------------------------------- state
def load_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def step(label: str) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")


def info(message: str) -> None:
    print(f"  {message}")


def cedar_principal(role_arn: str) -> str:
    """Convert an IAM role ARN to the Cedar IamEntity id.

    arn:aws:iam::123:role/Foo  ->  arn:aws:sts::123:assumed-role/Foo

    AgentCore builds the entity id from the caller's ARN with the session name
    removed, which is what makes `principal ==` stable across invocations.
    """
    account = role_arn.split(":")[4]
    role_name = role_arn.split("/")[-1]
    return f"arn:aws:sts::{account}:assumed-role/{role_name}"


def validate_resource_names() -> None:
    """Check every name we will create against the API's own regex.

    The patterns are read from the botocore service models, so they cannot
    drift from the services. This exists because a bad name otherwise fails six
    steps into a deploy, after runtimes have already been provisioned -- and the
    two naming conventions in this one API are easy to mix up: gateways and
    targets allow hyphens, while policy engines, policies and runtimes do not.

    IAM role names are checked too, against the iam model rather than the
    agentcore one. They are the only names here carrying a different alphabet
    again (word characters plus +=,.@-, max 64), and they are load-bearing:
    a role name is the Cedar principal.
    """
    import re

    import botocore.session

    botocore_session = botocore.session.get_session()
    control = botocore_session.get_service_model("bedrock-agentcore-control")
    iam_model = botocore_session.get_service_model("iam")

    def pattern_for(
        model: Any, operation: str, field: str
    ) -> tuple[str | None, int | None]:
        meta = model.operation_model(operation).input_shape.members[field].metadata
        return meta.get("pattern"), meta.get("max")

    candidates: list[tuple[Any, str, str, str, str]] = [
        (control, "CreateGateway", "name", GATEWAY_NAME, "gateway"),
        (control, "CreatePolicyEngine", "name", POLICY_ENGINE_NAME, "policy engine"),
        (control, "CreateGatewayTarget", "name", KB_TARGET, "KB target"),
    ]
    for spec in MCP_SERVERS.values():
        candidates.append(
            (control, "CreateGatewayTarget", "name", spec["target"], "MCP target")
        )
    for name in list(MCP_SERVERS) + list(SUB_AGENTS) + [ORCHESTRATOR[0]]:
        candidates.append(
            (
                control,
                "CreateAgentRuntime",
                "agentRuntimeName",
                f"{PREFIX_U}_{name}",
                "runtime",
            )
        )
    # POLICY_AGENTS, not every agent: the router gets no policy, so there is no
    # travel_demo_train_travel_agent_tools name to validate.
    for agent in POLICY_AGENTS:
        candidates.append(
            (
                control,
                "CreatePolicy",
                "name",
                f"{PREFIX_U}_{agent}_tools",
                "Cedar policy",
            )
        )
    candidates.append((control, "CreateMemory", "name", MEMORY_NAME, "memory"))
    for role in ALL_ROLES:
        candidates.append((iam_model, "CreateRole", "RoleName", role, "IAM role"))

    failures = []
    for model, operation, field, value, label in candidates:
        pattern, maximum = pattern_for(model, operation, field)
        if pattern and not re.fullmatch(pattern, value):
            failures.append(f"{label} {value!r} does not match {pattern}")
        if maximum and len(value) > maximum:
            failures.append(
                f"{label} {value!r} is {len(value)} chars, max {maximum}"
            )

    if failures:
        raise SystemExit(
            "  resource names would be rejected by the API:\n"
            + "".join(f"    - {f}\n" for f in failures)
        )
    info(f"names   all {len(candidates)} resource names satisfy the API patterns")


# ---------------------------------------------------------------- 0. preflight
def do_preflight(state: dict[str, Any]) -> None:
    step("preflight")
    validate_resource_names()
    identity = sts.get_caller_identity()
    state["account_id"] = identity["Account"]
    state["deployer_arn"] = identity["Arn"]
    info(f"account {identity['Account']}  region {AWS_REGION}")
    info(f"caller  {identity['Arn']}")

    runtime = session.client("bedrock-runtime")
    try:
        runtime.converse(
            modelId=model_id(),
            messages=[{"role": "user", "content": [{"text": "reply with: ok"}]}],
            inferenceConfig={"maxTokens": 8, "temperature": 0.0},
        )
        info(f"model   {model_id()}  INVOCABLE")
    except ClientError as exc:
        raise SystemExit(
            f"\n  model {model_id()} is NOT invocable "
            f"({exc.response['Error']['Code']}).\n"
            f"  {exc.response['Error']['Message']}\n\n"
            f"  Enable model access in the Bedrock console for {AWS_REGION}, or "
            f"set TRAVEL_MODEL_ID to a model you do have."
        ) from exc

    try:
        runtime.invoke_model(
            modelId=embed_model_id(), body=json.dumps({"inputText": "ok"})
        )
        info(f"embed   {embed_model_id()}  INVOCABLE")
    except ClientError as exc:
        info(
            f"embed   {embed_model_id()} NOT invocable "
            f"({exc.response['Error']['Code']}) -- the knowledge base falls "
            f"back to lexical retrieval, which still works"
        )

    bucket = f"{PREFIX}-code-{identity['Account']}-{AWS_REGION}"
    state["bucket"] = bucket
    try:
        s3.head_bucket(Bucket=bucket)
        info(f"bucket  s3://{bucket} (exists)")
    except ClientError:
        info(f"bucket  s3://{bucket} (will be created)")

    # Transaction Search is an account-wide setting and a prerequisite for
    # traces appearing on the CloudWatch GenAI Observability page. We report on
    # it rather than change it, because flipping an account-wide observability
    # setting is not this script's call to make.
    try:
        config = xray.get_trace_segment_destination()
        destination = config.get("Destination", "UNKNOWN")
        status = config.get("Status", "UNKNOWN")
        if destination == "CloudWatchLogs" and status == "ACTIVE":
            info("xray    trace segment destination = CloudWatchLogs (ACTIVE)")
        else:
            info(
                f"xray    trace destination is {destination}/{status}, not "
                f"CloudWatchLogs/ACTIVE"
            )
            info(
                "        -> traces will still be emitted, but will NOT appear "
                "on the GenAI Observability page."
            )
            info(
                "        -> enable it with: aws xray "
                "update-trace-segment-destination --destination CloudWatchLogs"
            )
    except ClientError as exc:
        info(f"xray    could not read trace destination ({exc.response['Error']['Code']})")

    save_state(state)


# ---------------------------------------------------------------- 1. IAM
def _trust_policy(deployer_arn: str) -> dict[str, Any]:
    """AgentCore assumes these roles. The deployer is also allowed to assume
    them, purely so `deploy.py verify` can prove the Cedar filtering by calling
    the gateway as each agent. Remove that statement if you do not want a human
    able to act as an agent.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AgentCoreAssume",
                "Effect": "Allow",
                "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                "Action": "sts:AssumeRole",
            },
            {
                "Sid": "DeployerAssumeForVerification",
                "Effect": "Allow",
                "Principal": {"AWS": deployer_arn},
                "Action": "sts:AssumeRole",
            },
        ],
    }


def _ensure_role(name: str, trust: dict, policy: dict) -> str:
    created = False
    try:
        arn = iam.create_role(
            RoleName=name,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description=f"Train Travel Agent demo ({name})",
        )["Role"]["Arn"]
        info(f"created role {name}")
        created = True
    except iam.exceptions.EntityAlreadyExistsException:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
        iam.update_assume_role_policy(
            RoleName=name, PolicyDocument=json.dumps(trust)
        )
        info(f"reusing role {name}")

    iam.put_role_policy(
        RoleName=name,
        PolicyName=f"{name}-inline",
        PolicyDocument=json.dumps(policy),
    )
    if created:
        # IAM is eventually consistent, and AgentCore rejects a role it cannot
        # yet assume with an error that does not say why.
        info("  waiting 12s for IAM propagation")
        time.sleep(12)
    return arn


def _base_statements(
    bucket: str, *, dynamic_config: bool = True
) -> list[dict[str, Any]]:
    """Statements every role in this demo needs.

    dynamic_config=False for the gateway and memory SERVICE roles: they are
    assumed by AgentCore itself, not by our code, so they never call Parameter
    Store. Passed explicitly rather than sliced off the front of the list, so
    reordering these statements cannot silently drop the wrong grant.
    """
    account = boto3.client("sts").get_caller_identity()["Account"]
    statements: list[dict[str, Any]] = []
    if dynamic_config:
        statements.append(
            {
                # Dynamic configuration. Read-only, scoped to this demo's path.
                "Sid": "ReadDynamicConfig",
                "Effect": "Allow",
                "Action": ["ssm:GetParameter", "ssm:GetParameters"],
                "Resource": (
                    f"arn:aws:ssm:{AWS_REGION}:{account}:parameter"
                    f"{config_store.prefix()}/*"
                ),
            }
        )
    return statements + [
        {
            "Sid": "ReadCodePackage",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:ListBucket"],
            "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
        },
        {
            "Sid": "Observability",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogStreams",
                "cloudwatch:PutMetricData",
                "xray:PutTraceSegments",
                "xray:PutTelemetryRecords",
            ],
            "Resource": "*",
        },
    ]


def _bedrock_statement(account: str) -> dict[str, Any]:
    return {
        "Sid": "InvokeModels",
        "Effect": "Allow",
        "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
        "Resource": [
            "arn:aws:bedrock:*::foundation-model/*",
            f"arn:aws:bedrock:*:{account}:inference-profile/*",
        ],
    }


def _gateway_role_policy(
    account: str,
    bucket: str,
    kb_arn: str | None = None,
    policy_engine_arn: str | None = None,
    gateway_arn: str | None = None,
) -> dict[str, Any]:
    """Permission policy for the gateway execution role.

    Built in one place because two steps attach it: `iam` creates it with
    wildcards, and `gateway` re-attaches it scoped to the real policy engine
    and gateway ARNs once those exist.

    The three Policy-in-AgentCore actions are mandatory for a gateway with a
    policy engine attached. Without them, CreateGateway fails on
    GetPolicyEngine, and -- worse -- if a policy engine is attached to an
    existing gateway, every tool invocation is denied by default even with
    correct permit policies.

    Note what is deliberately absent: bedrock-agentcore:InvokeGateway. That
    belongs to the CALLER, not to the role the gateway assumes.
    """
    region_account = f"arn:aws:bedrock-agentcore:{AWS_REGION}:{account}"
    engine = policy_engine_arn or f"{region_account}:policy-engine/*"
    gateway = gateway_arn or f"{region_account}:gateway/*"
    knowledge_base = kb_arn or f"arn:aws:bedrock:{AWS_REGION}:{account}:knowledge-base/*"

    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "InvokeMcpRuntimes",
                "Effect": "Allow",
                "Action": "bedrock-agentcore:InvokeAgentRuntime",
                "Resource": f"{region_account}:runtime/*",
            },
            {
                "Sid": "PolicyEngineConfiguration",
                "Effect": "Allow",
                "Action": ["bedrock-agentcore:GetPolicyEngine"],
                "Resource": [engine],
            },
            {
                # AuthorizeAction evaluates Cedar for a tool call;
                # PartiallyAuthorizeActions is what filters tools/list.
                "Sid": "PolicyEngineAuthorization",
                "Effect": "Allow",
                "Action": [
                    "bedrock-agentcore:AuthorizeAction",
                    "bedrock-agentcore:PartiallyAuthorizeActions",
                ],
                "Resource": [engine, gateway],
            },
            {
                "Sid": "ValidateKnowledgeBase",
                "Effect": "Allow",
                "Action": "bedrock:GetKnowledgeBase",
                "Resource": knowledge_base,
            },
            {
                "Sid": "RetrieveFromKnowledgeBase",
                "Effect": "Allow",
                "Action": "bedrock:Retrieve",
                "Resource": knowledge_base,
            },
            {
                # Not resource-scopeable, per the AgentCore docs. The Cedar
                # denial of AgenticRetrieveStream is the compensating control.
                "Sid": "AgenticRetrieveStream",
                "Effect": "Allow",
                "Action": "bedrock:AgenticRetrieveStream",
                "Resource": "*",
            },
            *_base_statements(bucket, dynamic_config=False),
        ],
    }


def do_iam(state: dict[str, Any]) -> None:
    step("1. IAM roles (one identity per agent)")
    account, bucket = state["account_id"], state["bucket"]
    trust = _trust_policy(state["deployer_arn"])

    gateway_wildcard = f"arn:aws:bedrock-agentcore:{AWS_REGION}:{account}:gateway/*"
    runtime_wildcard = f"arn:aws:bedrock-agentcore:{AWS_REGION}:{account}:runtime/*"

    # MCP server runtimes: no model access, no gateway access. They only read
    # fixtures off disk and answer tool calls.
    state["role_mcp"] = _ensure_role(
        ROLE_MCP,
        trust,
        {"Version": "2012-10-17", "Statement": _base_statements(bucket)},
    )

    # Gateway service role: signs SigV4 to the MCP runtimes AND retrieves from
    # the managed knowledge base, both on the caller's behalf. Its permissions
    # are the upper bound of what any authorized caller can reach through the
    # gateway, so it gets nothing else.
    #
    # Note what is NOT here: bedrock-agentcore:InvokeGateway. Per the AgentCore
    # docs that permission belongs to the CALLER invoking the gateway, not to
    # the role the gateway assumes. An earlier version of this script wrongly
    # granted it here.
    state["role_gateway"] = _ensure_role(
        ROLE_GATEWAY,
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                    # Confused-deputy protection: only this account's gateways
                    # may assume this role.
                    "Condition": {
                        "StringEquals": {"aws:SourceAccount": account},
                        "ArnLike": {
                            "aws:SourceArn": (
                                f"arn:aws:bedrock-agentcore:{AWS_REGION}:"
                                f"{account}:gateway/*"
                            )
                        },
                    },
                }
            ],
        },
        # Wildcards now; step 7 re-attaches this scoped to the real policy
        # engine and gateway ARNs once they exist.
        _gateway_role_policy(account, bucket),
    )

    # Managed knowledge base service role. A MANAGED knowledge base needs no
    # vector store of its own, so this role is minimal.
    state["role_kb"] = _ensure_role(
        ROLE_KB,
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                    "Condition": {"StringEquals": {"aws:SourceAccount": account}},
                }
            ],
        },
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "EmbedAndRetrieve",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel",
                        "bedrock:Rerank",
                        "bedrock:GetInferenceProfile",
                    ],
                    "Resource": [
                        "arn:aws:bedrock:*::foundation-model/*",
                        f"arn:aws:bedrock:*:{account}:inference-profile/*",
                    ],
                }
            ],
        },
    )

    # Sub-agents: model access plus InvokeGateway, scoped to a wildcard now and
    # tightened to the exact gateway ARN in step 8.
    #
    # All three get the same shape. There is deliberately no peer-to-peer
    # statement any more: sub-agents do not call each other, so none of them
    # needs InvokeAgentRuntime. Journey Support used to carry an
    # InvokePlannerForPeerA2A grant for the peer hop that has since been
    # removed from its agent code; leaving the IAM grant behind would have kept
    # a second, undocumented topology reachable.
    #
    # policy_advisor needs InvokeGateway for the same reason the other two
    # do -- it reaches the knowledge base through the gateway, as the
    # RailPolicyKB connector target, and Cedar grants it exactly
    # RailPolicyKB___Retrieve.
    for key, role_name in (
        ("journey_planner", ROLE_PLANNER),
        ("journey_support", ROLE_SUPPORT),
        ("policy_advisor", ROLE_POLICY_ADVISOR),
    ):
        statements = [
            _bedrock_statement(account),
            {
                "Sid": "InvokeGateway",
                "Effect": "Allow",
                "Action": "bedrock-agentcore:InvokeGateway",
                "Resource": gateway_wildcard,
            },
            *_base_statements(bucket),
        ]
        state[f"role_{key}"] = _ensure_role(
            role_name, trust, {"Version": "2012-10-17", "Statement": statements}
        )

    # Memory service role. AgentCore assumes this to run the extraction
    # pipeline that turns events into summaries and preferences.
    state["role_memory"] = _ensure_role(
        ROLE_MEMORY,
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                    "Condition": {"StringEquals": {"aws:SourceAccount": account}},
                }
            ],
        },
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "ExtractionModels",
                    "Effect": "Allow",
                    "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                    "Resource": [
                        "arn:aws:bedrock:*::foundation-model/*",
                        f"arn:aws:bedrock:*:{account}:inference-profile/*",
                    ],
                },
                *_base_statements(bucket, dynamic_config=False),
            ],
        },
    )

    # Orchestrator: model access, the right to invoke the three sub-agents, and
    # the memory data plane. Note what it does NOT get: InvokeGateway. The
    # router holds no gateway tools at all (AGENT_TOOL_GRANTS is empty for it),
    # so granting it gateway access would contradict the design -- an IAM grant
    # for a tool plane it is not permitted to see anything on. The knowledge
    # base it used to reach directly now belongs to the policy_advisor
    # sub-agent, and the router gets there by delegating.
    state["role_train_travel_agent"] = _ensure_role(
        ROLE_ORCHESTRATOR,
        trust,
        {
            "Version": "2012-10-17",
            "Statement": [
                _bedrock_statement(account),
                {
                    "Sid": "InvokeSubAgents",
                    "Effect": "Allow",
                    # GetAgentCard is a SEPARATE action, and an A2A client
                    # fetches the card before it can send anything -- the
                    # client routes by card.url. Without it the delegation
                    # fails as a 403 on
                    # /invocations/.well-known/agent-card.json, which looks
                    # like an invoke-permission problem but is not.
                    #
                    # GetAgentCard is now doubly load-bearing: the router
                    # DISCOVERS its sub-agents from their cards at startup
                    # (common/a2a_discovery.py) rather than from hardcoded
                    # delegation tools, so without it the router comes up with
                    # no sub-agents at all and simply says it cannot help.
                    "Action": [
                        "bedrock-agentcore:InvokeAgentRuntime",
                        "bedrock-agentcore:GetAgentCard",
                    ],
                    "Resource": runtime_wildcard,
                },
                {
                    # Memory data plane. Only the orchestrator gets these --
                    # the sub-agents hold no conversation of their own.
                    "Sid": "MemoryDataPlane",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock-agentcore:CreateEvent",
                        "bedrock-agentcore:GetEvent",
                        "bedrock-agentcore:ListEvents",
                        "bedrock-agentcore:ListActors",
                        "bedrock-agentcore:ListSessions",
                        "bedrock-agentcore:RetrieveMemoryRecords",
                        "bedrock-agentcore:GetMemoryRecord",
                        "bedrock-agentcore:ListMemoryRecords",
                        "bedrock-agentcore:GetMemory",
                    ],
                    "Resource": (
                        f"arn:aws:bedrock-agentcore:{AWS_REGION}:{account}:memory/*"
                    ),
                },
                *_base_statements(bucket),
            ],
        },
    )

    info("")
    info("Cedar principals these roles will present:")
    for key in AGENT_ROLES:
        suffix = "" if key in POLICY_AGENTS else "   (no policy -- holds no tools)"
        info(f"  {key}: {cedar_principal(state[f'role_{key}'])}{suffix}")
    save_state(state)


# ---------------------------------------------------------------- 2. package
def do_package(state: dict[str, Any]) -> None:
    step("2. package and upload code")
    from deploy.agentcore import package

    zip_path = package.build()
    bucket, key = state["bucket"], "code/travel_agent_deploy.zip"

    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        kwargs: dict[str, Any] = {"Bucket": bucket}
        if AWS_REGION != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": AWS_REGION}
        s3.create_bucket(**kwargs)
        s3.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        s3.put_bucket_encryption(
            Bucket=bucket,
            ServerSideEncryptionConfiguration={
                "Rules": [
                    {"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}
                ]
            },
        )
        info(f"created s3://{bucket} (private, encrypted)")

    info(f"uploading to s3://{bucket}/{key}")
    s3.upload_file(str(zip_path), bucket, key)
    state["code_bucket"], state["code_key"] = bucket, key
    save_state(state)


# ---------------------------------------------------------------- runtimes
def runtime_invocation_url(arn: str, *, with_qualifier: bool) -> str:
    """Invocation URL for a runtime.

    with_qualifier=True  -> gateway target endpoints (MCP servers)
    with_qualifier=False -> A2A base URL. A2ACardResolver appends
        /.well-known/agent-card.json to this, so it must end in a path
        segment and carry no query string.
    """
    encoded = urllib.parse.quote(arn, safe="")
    base = (
        f"https://bedrock-agentcore.{AWS_REGION}.amazonaws.com/"
        f"runtimes/{encoded}/invocations"
    )
    return f"{base}?qualifier=DEFAULT" if with_qualifier else f"{base}/"


def _find_runtime(name: str) -> dict[str, Any] | None:
    for page in agentcore.get_paginator("list_agent_runtimes").paginate():
        for runtime in page.get("agentRuntimes", []):
            if runtime.get("agentRuntimeName") == name:
                return runtime
    return None


def _wait_runtime_ready(runtime_id: str, name: str, timeout: int = 600) -> None:
    deadline, last = time.time() + timeout, None
    while time.time() < deadline:
        described = agentcore.get_agent_runtime(agentRuntimeId=runtime_id)
        status = described.get("status")
        if status != last:
            info(f"  {name}: {status}")
            last = status
        if status == "READY":
            return
        if status in {"CREATE_FAILED", "UPDATE_FAILED", "DELETE_FAILED"}:
            # Surface the service's own reason. Without it this reads as a
            # generic failure and sends you to CloudWatch for an error the
            # control plane already told us.
            reason = described.get("failureReason") or "(no failureReason given)"
            raise RuntimeError(
                f"{name} reached {status}\n"
                f"  reason: {reason}\n"
                f"  entryPoint: "
                f"{described.get('agentRuntimeArtifact', {}).get('codeConfiguration', {}).get('entryPoint')}\n"
                f"  If the reason mentions the entrypoint, rebuild the package "
                f"(deploy.py deploy --step package) -- packaging now asserts "
                f"every entrypoint is present.\n"
                f"  Otherwise check "
                f"/aws/bedrock-agentcore/runtimes/{name} in CloudWatch Logs."
            )
        time.sleep(10)
    raise RuntimeError(f"{name} did not become READY within {timeout}s")


# Environment each runtime's code REQUIRES at startup. Checked before any
# runtime is created, because a missing variable crashes the container before
# it binds a port -- and AgentCore then reports "Runtime initialization time
# exceeded", which points at cold-start latency rather than the real cause.
REQUIRED_ENV = {
    # Bootstrap settings only. Every runtime needs these before it can read
    # Parameter Store at all.
    "_all": {"TRAVEL_CONFIG_SOURCE", "TRAVEL_CONFIG_PREFIX"},
}

# Config keys each runtime resolves AT USE TIME from Parameter Store. Checked
# as parameters, not environment variables -- see _check_required_config.
REQUIRED_CONFIG = {
    "traveller_profile": set(),
    "journey_search": set(),
    "ticket_operations": set(),
    "journey_planner": {"model_id", "gateway_url"},
    "journey_support": {"model_id", "gateway_url"},
    # The knowledge base is reached through the gateway like any other tool
    # plane, so this agent needs gateway_url even though its one tool is a
    # connector target rather than an MCP server.
    "policy_advisor": {"model_id", "gateway_url"},
    # No gateway_url: the router holds no gateway tools, so it never resolves
    # one. What it does need is the registry plus every sub-agent's A2A URL --
    # it discovers each agent from its card at startup, and an unresolvable URL
    # means that agent is silently skipped and becomes unroutable.
    "train_travel_agent": {
        "model_id",
        "sub_agents",
        "journey_planner_a2a_url",
        "journey_support_a2a_url",
        "policy_advisor_a2a_url",
    },
}


def _check_required_env(name: str, env: dict[str, str]) -> None:
    """Bootstrap env vars must be present before the runtime can read SSM."""
    missing = sorted(k for k in REQUIRED_ENV["_all"] if not env.get(k))
    if missing:
        raise RuntimeError(
            f"refusing to create runtime {name!r}: missing bootstrap "
            f"environment {', '.join(missing)}. Without these the container "
            f"cannot reach Parameter Store and will crash at startup, which "
            f"AgentCore reports as 'Runtime initialization time exceeded'."
        )


def _check_required_config() -> None:
    """Assert every key the runtimes resolve at use time is actually in SSM.

    Queried against Parameter Store directly rather than through
    config_store.get(), because the deploy shell may have the equivalent
    environment variables exported from a previous run -- and env wins in the
    resolution order, so a local export would mask a parameter that is missing
    in AWS and let a broken deployment report success.

    This is the check that the gateway-URL outage needed: a runtime whose
    config is absent starts, crashes on first resolve, and is reported as a
    cold-start timeout.
    """
    wanted = sorted(set().union(*REQUIRED_CONFIG.values()))
    names = {config_store.parameter_name(k): k for k in wanted}
    found: set[str] = set()
    batch = list(names)
    for i in range(0, len(batch), 10):  # GetParameters takes at most 10
        page = batch[i : i + 10]
        response = ssm.get_parameters(Names=page)
        found.update(p["Name"] for p in response["Parameters"])

    gaps = {
        agent: sorted(
            k for k in keys if config_store.parameter_name(k) not in found
        )
        for agent, keys in REQUIRED_CONFIG.items()
    }
    gaps = {a: k for a, k in gaps.items() if k}
    if gaps:
        detail = "; ".join(f"{a} needs {', '.join(k)}" for a, k in gaps.items())
        raise RuntimeError(
            f"configuration incomplete in Parameter Store under "
            f"{config_store.prefix()}: {detail}. These are resolved at use "
            f"time, so the runtimes exist but will fail on their first "
            f"invocation. Re-run:  --step parameters"
        )
    info(f"all {len(wanted)} dynamic config keys present in Parameter Store")


def runtime_log_group(runtime_id: str | None, full_name: str) -> str:
    """The log group AgentCore actually writes a runtime's logs to.

    The real name is /aws/bedrock-agentcore/runtimes/<runtimeId>-<ENDPOINT>,
    where runtimeId is full_name plus a service-assigned suffix. Guessing
    /aws/bedrock-agentcore/runtimes/<full_name> produces a group that does not
    exist, and aws.log.group.names is what CloudWatch uses to index telemetry
    under the Bedrock AgentCore Observability dashboard -- so a wrong value
    means spans still reach aws/spans but the dashboard reports zero agents.

    runtime_id is None only when the runtime is being created for the very
    first time; the caller then does one corrective update once the service
    has assigned the id.
    """
    base = runtime_id or full_name
    return f"/aws/bedrock-agentcore/runtimes/{base}-DEFAULT"


def _ensure_runtime(
    state: dict[str, Any],
    name: str,
    entry_point: str,
    protocol: str,
    role_arn: str,
    env: dict[str, str],
) -> tuple[str, str]:
    """Create or update a runtime. Returns (runtime_id, arn).

    No authorizerConfiguration is passed, so the runtime uses its default IAM
    (SigV4) inbound authorization. Callers sign with their execution role.
    """
    full_name = f"{PREFIX_U}_{name}"

    # Resolve the existing runtime FIRST, because its id determines the real
    # log group name, which has to go into the OTEL resource attributes.
    existing = _find_runtime(full_name)

    # A runtime left in CREATE_FAILED cannot be updated into health -- it has
    # no working version to update. Delete it and create cleanly, so a re-run
    # after a failed deploy actually recovers.
    if existing and existing.get("status") in {"CREATE_FAILED", "DELETE_FAILED"}:
        info(f"{full_name} is {existing['status']}; deleting before recreate")
        try:
            agentcore.delete_agent_runtime(
                agentRuntimeId=existing["agentRuntimeId"]
            )
        except ClientError as exc:
            info(f"  delete returned {exc.response['Error']['Code']}, continuing")
        for _ in range(30):
            time.sleep(5)
            if _find_runtime(full_name) is None:
                break
        existing = None

    artifact = {
        "codeConfiguration": {
            "code": {
                "s3": {"bucket": state["code_bucket"], "prefix": state["code_key"]}
            },
            "runtime": "PYTHON_3_12",
            "entryPoint": [entry_point],
        }
    }

    def build_env(runtime_id_hint: str | None) -> dict[str, str]:
        # The dynamic values (model id, gateway URL, A2A URLs, memory id) are
        # NOT baked in here -- the runtime reads them from Parameter Store at
        # use time, so changing one repoints every agent without a redeploy.
        # Only the bootstrap settings, which must be known before the store can
        # be read, live in the environment.
        return {
            "TRAVEL_RUN_MODE": "agentcore",
            "AWS_REGION": AWS_REGION,
            "TRAVEL_CONFIG_SOURCE": "ssm",
            "TRAVEL_CONFIG_PREFIX": config_store.prefix(),
            # The auto_instrumentation directory FIRST: that is what makes the
            # interpreter load the ADOT launcher at startup. See
            # observability.AUTO_INSTRUMENTATION_PATH.
            "PYTHONPATH": f"{AUTO_INSTRUMENTATION_PATH}:/var/task",
            # Tracing. The log group must be the one AgentCore really writes
            # to -- see runtime_log_group.
            **otel_environment(
                agent_name=full_name,
                log_group=runtime_log_group(runtime_id_hint, full_name),
            ),
            **env,
        }

    common_env = build_env(existing["agentRuntimeId"] if existing else None)
    # Checked against the merged block, which is what actually reaches the
    # container -- not against the per-agent overlay, which legitimately omits
    # the bootstrap settings.
    _check_required_env(full_name, common_env)

    if existing:
        runtime_id = existing["agentRuntimeId"]
        info(f"updating {full_name} ({protocol})")
        agentcore.update_agent_runtime(
            agentRuntimeId=runtime_id,
            agentRuntimeArtifact=artifact,
            roleArn=role_arn,
            networkConfiguration={"networkMode": "PUBLIC"},
            protocolConfiguration={"serverProtocol": protocol},
            environmentVariables=common_env,
        )
    else:
        info(f"creating {full_name} ({protocol})")
        runtime_id = agentcore.create_agent_runtime(
            agentRuntimeName=full_name,
            agentRuntimeArtifact=artifact,
            roleArn=role_arn,
            networkConfiguration={"networkMode": "PUBLIC"},
            protocolConfiguration={"serverProtocol": protocol},
            environmentVariables=common_env,
            description=f"Train Travel Agent demo: {name}",
        )["agentRuntimeId"]

        # The service assigns the id, and the id determines the log group, so
        # a brand-new runtime's OTEL attributes were necessarily built with a
        # placeholder. Correct them now, otherwise its telemetry is never
        # indexed under the AgentCore Observability dashboard. Same
        # chicken-and-egg as the A2A card URL.
        _wait_runtime_ready(runtime_id, full_name)
        info(f"  correcting log group to {runtime_log_group(runtime_id, full_name)}")
        agentcore.update_agent_runtime(
            agentRuntimeId=runtime_id,
            agentRuntimeArtifact=artifact,
            roleArn=role_arn,
            networkConfiguration={"networkMode": "PUBLIC"},
            protocolConfiguration={"serverProtocol": protocol},
            environmentVariables=build_env(runtime_id),
        )

    _wait_runtime_ready(runtime_id, full_name)
    arn = agentcore.get_agent_runtime(agentRuntimeId=runtime_id)["agentRuntimeArn"]
    return runtime_id, arn


# ---------------------------------------------------------------- 3. MCP runtimes
def do_mcp_runtimes(state: dict[str, Any]) -> None:
    step("3. MCP server runtimes")
    state.setdefault("mcp_runtimes", {})
    for key, spec in MCP_SERVERS.items():
        _, arn = _ensure_runtime(
            state, key, spec["entry"], "MCP", state["role_mcp"], {}
        )
        state["mcp_runtimes"][key] = {
            "arn": arn,
            "url": runtime_invocation_url(arn, with_qualifier=True),
        }
        save_state(state)


# ---------------------------------------------------------------- 6. policy engine
def do_policy_engine(state: dict[str, Any]) -> None:
    step("6. policy engine")
    if state.get("policy_engine_arn"):
        info("reusing policy engine from state")
        return
    for page in agentcore.get_paginator("list_policy_engines").paginate():
        for engine in page.get("policyEngines", []):
            if engine.get("name") == POLICY_ENGINE_NAME:
                state["policy_engine_arn"] = engine["policyEngineArn"]
                state["policy_engine_id"] = engine["policyEngineId"]
                info(f"reusing policy engine {engine['policyEngineId']}")
                save_state(state)
                return

    response = agentcore.create_policy_engine(
        name=POLICY_ENGINE_NAME,
        description="Tool authorization for the Train Travel Agent demo",
    )
    state["policy_engine_arn"] = response["policyEngineArn"]
    state["policy_engine_id"] = response["policyEngineId"]
    info(f"created policy engine {response['policyEngineId']}")
    save_state(state)


# ---------------------------------------------------------------- 4. knowledge base
def do_knowledge_base(state: dict[str, Any]) -> None:
    """Create a MANAGED knowledge base and ingest the policy documents.

    MANAGED means Bedrock owns the vector store -- there is no OpenSearch
    collection or Aurora cluster to provision, scale or pay for while idle. It
    is also a hard requirement: the AgentCore Gateway knowledge-base connector
    supports managed knowledge bases only.

    Documents are ingested INLINE through a CUSTOM data source, so the five
    markdown files go straight from this repo into the knowledge base with no
    S3 staging bucket for content.
    """
    step("4. managed knowledge base")

    kb_id = state.get("kb_id")
    if not kb_id:
        for page in bedrock_agent.get_paginator("list_knowledge_bases").paginate():
            for summary in page.get("knowledgeBaseSummaries", []):
                if summary.get("name") == KB_NAME:
                    kb_id = summary["knowledgeBaseId"]
                    info(f"reusing knowledge base {kb_id}")
                    break

    if not kb_id:
        info("creating managed knowledge base (no vector store to provision)")
        kb_id = bedrock_agent.create_knowledge_base(
            name=KB_NAME,
            description="Rail fare rules, railcards, Delay Repay, validity, accessibility",
            roleArn=state["role_kb"],
            knowledgeBaseConfiguration={
                "type": "MANAGED",
                "managedKnowledgeBaseConfiguration": {
                    "embeddingModelType": "MANAGED",
                },
            },
        )["knowledgeBase"]["knowledgeBaseId"]
        info(f"created knowledge base {kb_id}")

    state["kb_id"] = kb_id
    state["kb_arn"] = (
        f"arn:aws:bedrock:{AWS_REGION}:{state['account_id']}:knowledge-base/{kb_id}"
    )

    deadline = time.time() + 300
    while time.time() < deadline:
        status = bedrock_agent.get_knowledge_base(knowledgeBaseId=kb_id)[
            "knowledgeBase"
        ]["status"]
        if status == "ACTIVE":
            break
        if status in {"FAILED", "DELETING"}:
            raise RuntimeError(f"knowledge base {kb_id} is {status}")
        info(f"  knowledge base status: {status}")
        time.sleep(10)

    # Custom data source. For a MANAGED knowledge base this is NOT
    # dataSourceConfiguration.type = "CUSTOM" -- that is the self-managed form
    # and it is rejected here. A managed KB takes
    # type = "MANAGED_KNOWLEDGE_BASE_CONNECTOR" with the connector itself
    # declared in connectorParameters.
    #
    # Custom (rather than S3) because direct ingestion needs no sync: documents
    # persist until deleted, and because WE assign the document id, re-running
    # this step updates each document in place instead of creating duplicates.
    # An S3 data source is the right choice for bulk corpora on a schedule, but
    # its sync can overwrite or remove directly-ingested documents, so the two
    # do not mix well. Five files that version with this repo want direct
    # ingestion.
    ds_id = state.get("kb_data_source_id")
    if not ds_id:
        for page in bedrock_agent.get_paginator("list_data_sources").paginate(
            knowledgeBaseId=kb_id
        ):
            for summary in page.get("dataSourceSummaries", []):
                if summary.get("name") == "policy-documents":
                    ds_id = summary["dataSourceId"]
    if not ds_id:
        ds_id = bedrock_agent.create_data_source(
            knowledgeBaseId=kb_id,
            name="policy-documents",
            description="Markdown policy documents ingested inline from the repo",
            dataSourceConfiguration={
                "type": "MANAGED_KNOWLEDGE_BASE_CONNECTOR",
                "managedKnowledgeBaseConnectorConfiguration": {
                    "connectorParameters": {
                        "type": "CUSTOM",
                        "version": "1",
                        # Document-level ACLs off: every policy document is
                        # readable by anyone who may call Retrieve at all.
                        # Per-document access control would need userContext
                        # passed from the application on every call.
                        "aclEnabled": False,
                    }
                },
            },
        )["dataSource"]["dataSourceId"]
        info(f"created MANAGED_KNOWLEDGE_BASE_CONNECTOR/CUSTOM data source {ds_id}")
    else:
        info(f"reusing data source {ds_id}")
    state["kb_data_source_id"] = ds_id

    # Inline ingestion caps at 6 MB per document; these are a few KB each.
    # A single call accepts at most 10 documents, so batch if the corpus grows.
    paths = sorted(KB_DIR.glob("*.md"))
    if len(paths) > 10:
        raise RuntimeError(
            f"{len(paths)} documents, but IngestKnowledgeBaseDocuments accepts "
            f"10 per call. Batch the ingestion."
        )

    documents = [
        {
            "content": {
                "dataSourceType": "CUSTOM",
                "custom": {
                    # Stable id: re-running this step UPDATES the document
                    # rather than duplicating it.
                    "customDocumentIdentifier": {"id": path.name},
                    "sourceType": "IN_LINE",
                    "inlineContent": {
                        "type": "TEXT",
                        "textContent": {"data": path.read_text(encoding="utf-8")},
                    },
                },
            }
        }
        for path in paths
    ]

    info(f"ingesting {len(documents)} policy documents inline (no S3 staging)")
    bedrock_agent.ingest_knowledge_base_documents(
        knowledgeBaseId=kb_id, dataSourceId=ds_id, documents=documents
    )
    for path in paths:
        size_kb = path.stat().st_size / 1024
        info(f"    {path.name}  ({size_kb:.1f} KB)")

    # Wait for indexing. A Retrieve against a knowledge base whose documents
    # are still INDEXING returns nothing, which on stage looks like the agent
    # inventing "I could not find that policy".
    info("waiting for documents to finish indexing")
    deadline = time.time() + 300
    while time.time() < deadline:
        statuses: dict[str, int] = {}
        for page in bedrock_agent.get_paginator(
            "list_knowledge_base_documents"
        ).paginate(knowledgeBaseId=kb_id, dataSourceId=ds_id):
            for doc in page.get("documentDetails", []):
                statuses[doc["status"]] = statuses.get(doc["status"], 0) + 1
        summary = ", ".join(f"{count} {status}" for status, count in sorted(statuses.items()))
        if statuses and set(statuses) == {"INDEXED"}:
            info(f"  all documents INDEXED ({summary})")
            break
        failed = [s for s in statuses if "FAIL" in s]
        if failed:
            raise RuntimeError(
                f"document ingestion failed: {summary}. Inspect with "
                f"list-knowledge-base-documents --knowledge-base-id {kb_id}"
            )
        info(f"  {summary or 'no documents reported yet'}")
        time.sleep(10)
    else:
        info("  timed out waiting for INDEXED; check before demoing")

    # Scope the gateway role's bedrock permissions to this knowledge base.
    for sid in ("ValidateKnowledgeBase", "RetrieveFromKnowledgeBase"):
        _retarget_statement(ROLE_GATEWAY, sid, state["kb_arn"])
    info("scoped the gateway role's bedrock:Retrieve to this knowledge base")
    save_state(state)


# ---------------------------------------------------------------- 5. memory
def do_memory(state: dict[str, Any]) -> None:
    """Create the Memory resource with both long-term strategies.

    One resource carries both layers. Short-term needs no strategy -- events
    are stored as written, and eventExpiryDuration is their retention. The two
    strategies are what extract long-term records from those events in the
    background.

    Only the orchestrator uses this. The sub-agents are stateless workers that
    receive self-contained instructions; giving them memory would create
    several writers on one conversation.
    """
    step("5. memory (short-term events + long-term strategies)")

    existing = None
    try:
        for page in agentcore.get_paginator("list_memories").paginate():
            for summary in page.get("memories", []):
                if summary.get("id", "").startswith(MEMORY_NAME) or \
                        summary.get("name") == MEMORY_NAME:
                    existing = summary
    except Exception as exc:  # noqa: BLE001
        info(f"could not list memories ({type(exc).__name__}); will try create")

    if existing:
        state["memory_id"] = existing.get("id") or existing.get("memoryId")
        info(f"reusing memory {state['memory_id']}")
    else:
        info("creating memory with summary + userPreference strategies")
        response = agentcore.create_memory(
            name=MEMORY_NAME,
            description="Train Travel Agent conversation and traveller memory",
            memoryExecutionRoleArn=state["role_memory"],
            # Short-term retention. Required; 3-365 days.
            eventExpiryDuration=MEMORY_EVENT_EXPIRY_DAYS,
            memoryStrategies=[
                {
                    "summaryMemoryStrategy": {
                        "name": "conversation_summary",
                        "description": "Rolling summary of each conversation",
                        "namespaces": ["travellers/{actorId}/summaries/{sessionId}"],
                    }
                },
                {
                    "userPreferenceMemoryStrategy": {
                        "name": "traveller_preferences",
                        "description": "Durable travel preferences per traveller",
                        "namespaces": ["travellers/{actorId}/preferences"],
                    }
                },
            ],
        )
        memory = response.get("memory", response)
        state["memory_id"] = memory.get("id") or memory.get("memoryId")
        info(f"created memory {state['memory_id']}")

    # Wait for ACTIVE: events written against a CREATING memory are rejected.
    deadline = time.time() + 300
    while time.time() < deadline:
        described = agentcore.get_memory(memoryId=state["memory_id"])
        current = described.get("memory", described)
        status = current.get("status")
        if status == "ACTIVE":
            strategies = current.get("strategies") or current.get("memoryStrategies") or []
            info(f"memory ACTIVE with {len(strategies)} strategies:")
            for s in strategies:
                info(f"    {s.get('name')} ({s.get('type', s.get('memoryStrategyType','?'))})"
                     f"  ns={s.get('namespaces')}")
            break
        if status == "FAILED":
            raise RuntimeError(
                f"memory {state['memory_id']} FAILED: "
                f"{current.get('failureReason', '(no reason given)')}"
            )
        info(f"  memory status: {status}")
        time.sleep(10)

    info("")
    info("extraction is ASYNCHRONOUS -- long-term records appear a short while")
    info("after events are written, not on the same turn. Seed a prior session")
    info("before demoing personalisation:  python scripts/seed_memory.py")
    save_state(state)


def publish_parameter(key: str, value: str | None) -> str | None:
    """Write one config value to Parameter Store. Idempotent."""
    if not value:
        return None
    name = config_store.parameter_name(key)
    ssm.put_parameter(
        Name=name,
        Value=value,
        Type="String",
        Overwrite=True,
        Description=f"Train Travel Agent demo: {key}",
    )
    info(f"{name} = {value[:88]}")
    return name


# ---------------------------------------------------------------- 10. parameters
def do_parameters(state: dict[str, Any]) -> None:
    """Publish the dynamic configuration that is known before the agents exist.

    Runs BEFORE the agent runtimes. Not because a runtime reads these at
    startup -- it does not; gateway_url is resolved when tools are loaded and
    model_id on every turn -- but because a runtime whose configuration is
    absent starts healthy and fails on its FIRST invocation, which is the
    worst time to find out. Publishing first means the completeness assertion
    at the end of the agents step is meaningful.

    The three A2A URLs cannot be published here -- they derive from runtime
    ARNs that do not exist until creation -- so do_agent_runtimes publishes
    those as it goes, and performs the completeness check at the end.
    """
    step("10. publish dynamic config to Parameter Store")

    # The sub-agent REGISTRY. This is the one value that has to change to add a
    # Tier-2 agent: the router reads it at startup, fetches a card per key, and
    # synthesises a delegation tool per card. Published from the built-in
    # default (or an explicit TRAVEL_SUB_AGENTS export) and deliberately NOT
    # from config_store.get(), which would read the CURRENT parameter back when
    # the deploy shell has TRAVEL_CONFIG_SOURCE=ssm set -- so a newly added
    # agent would never reach the registry it is meant to be added to.
    registry_env, _, registry_default = config_store.SPEC["sub_agents"]
    registry = os.getenv(registry_env) or registry_default or ""

    # A registry that disagrees with what this script deploys is the one drift
    # that fails quietly: an unlisted agent is never routed to, and a listed
    # agent that does not exist is skipped with a warning buried in the
    # router's cold-start logs.
    registered = [k.strip() for k in registry.split(",") if k.strip()]
    if registered != list(SUB_AGENTS):
        info(
            f"WARNING registry {registered} does not match the sub-agents this "
            f"script deploys {list(SUB_AGENTS)}"
        )
        info(
            f"        -> the router routes by the registry, so set "
            f"{registry_env} or fix the default in common/config_store.py"
        )

    written = [
        name
        for name in (
            publish_parameter("model_id", model_id()),
            publish_parameter("embed_model_id", embed_model_id()),
            publish_parameter("gateway_url", state.get("gateway_url")),
            publish_parameter("memory_id", state.get("memory_id")),
            publish_parameter("sub_agents", registry),
        )
        if name
    ]
    state["parameters"] = written

    info("")
    info("read at use time, so to repoint a running deployment:")
    info(f"  aws ssm put-parameter --name {config_store.parameter_name('model_id')} \\")
    info("      --value us.anthropic.claude-haiku-5-5 --type String --overwrite")
    info(f"  (takes effect within {config_store.ttl_seconds()}s, no redeploy)")
    save_state(state)


def _runtime_resources(arns: list[str]) -> list[str]:
    """Runtime ARNs plus their endpoint children.

    InvokeAgentRuntime is authorized against the ENDPOINT-qualified ARN
    (runtime/<id>/runtime-endpoint/<qualifier>), not the bare runtime ARN, so a
    policy listing only the runtime denies the call. Verified with
    simulate-principal-policy: the bare ARN evaluates to `allowed` while
    .../runtime-endpoint/DEFAULT evaluates to `implicitDeny`, and the live call
    returns 403.

    Both forms are listed: the endpoint child is what the invoke needs, and the
    bare ARN keeps any management-style action on the runtime itself working.
    """
    resources: list[str] = []
    for arn in arns:
        resources.extend([arn, f"{arn}/runtime-endpoint/*"])
    return resources


def _retarget_statement(role_name: str, sid: str, resources: Any) -> None:
    """Re-scope one statement's Resource in a role's inline policy.

    Used to tighten the wildcard grants from step 1 down to exact ARNs once
    those resources exist.

    Raises if the Sid is not there, rather than writing the document back
    untouched. A silent miss is the worst outcome available here: the role
    keeps the step-1 WILDCARD -- InvokeGateway on gateway/*, or
    InvokeAgentRuntime on runtime/* -- and nothing reports that the narrowing
    never happened. This is what caught the stale peer-A2A retarget when the
    statement it referred to was removed from the role.
    """
    document = iam.get_role_policy(
        RoleName=role_name, PolicyName=f"{role_name}-inline"
    )["PolicyDocument"]
    policy = json.loads(document) if isinstance(document, str) else document
    matched = False
    for statement in policy["Statement"]:
        if statement.get("Sid") == sid:
            statement["Resource"] = resources
            matched = True
    if not matched:
        raise RuntimeError(
            f"role {role_name} has no statement with Sid {sid!r}, so it could "
            f"not be narrowed and still carries whatever step 1 gave it. "
            f"Present Sids: "
            f"{sorted(s.get('Sid', '(none)') for s in policy['Statement'])}"
        )
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=f"{role_name}-inline",
        PolicyDocument=json.dumps(policy),
    )


# ---------------------------------------------------------------- 7. gateway
def do_gateway(state: dict[str, Any]) -> None:
    step("7. gateway (AWS_IAM inbound)")

    # Re-attach the gateway role policy, now scoped to the real policy engine.
    # This must happen BEFORE create_gateway: the service assumes this role and
    # calls GetPolicyEngine during creation, and without the three
    # Policy-in-AgentCore actions it fails with "Access denied while calling
    # GetPolicyEngine". Doing it here (rather than only in step 2) means
    # --from-step gateway recovers without re-running the iam step.
    info("scoping the gateway role to the policy engine")
    iam.put_role_policy(
        RoleName=ROLE_GATEWAY,
        PolicyName=f"{ROLE_GATEWAY}-inline",
        PolicyDocument=json.dumps(
            _gateway_role_policy(
                state["account_id"],
                state["bucket"],
                kb_arn=state.get("kb_arn"),
                policy_engine_arn=state["policy_engine_arn"],
            )
        ),
    )
    info("waiting 10s for the IAM change to propagate")
    time.sleep(10)

    existing = None
    for page in agentcore.get_paginator("list_gateways").paginate():
        for gateway in page.get("items", []):
            if gateway.get("name") == GATEWAY_NAME:
                existing = gateway

    # LOG_ONLY first: a Cedar mistake should show up in the logs, not as a
    # silently empty tool catalogue minutes before a demo.
    policy_config = {"arn": state["policy_engine_arn"], "mode": "LOG_ONLY"}
    common = {
        "name": GATEWAY_NAME,
        "roleArn": state["role_gateway"],
        "protocolType": "MCP",
        # AWS_IAM gives Cedar an AgentCore::IamEntity principal built from the
        # caller's role. No authorizerConfiguration is needed or accepted.
        "authorizerType": "AWS_IAM",
        "policyEngineConfiguration": policy_config,
    }

    if existing:
        gateway_id = existing["gatewayId"]
        info(f"updating gateway {gateway_id}")
        agentcore.update_gateway(gatewayIdentifier=gateway_id, **common)
    else:
        info("creating gateway")
        gateway_id = agentcore.create_gateway(
            description="Single tool plane for the Train Travel Agent demo",
            **common,
        )["gatewayId"]

    described = agentcore.get_gateway(gatewayIdentifier=gateway_id)
    state["gateway_id"] = gateway_id
    state["gateway_arn"] = described["gatewayArn"]
    state["gateway_url"] = described["gatewayUrl"]
    info(f"gateway url {state['gateway_url']}")
    info("policy engine mode: LOG_ONLY (step 14 flips it to ENFORCE)")

    # Tighten the sub-agent roles from gateway/* to this exact gateway. The
    # orchestrator is NOT in this loop: it has no InvokeGateway statement to
    # narrow, because the router holds no gateway tools.
    for key in SUB_AGENTS:
        _retarget_statement(AGENT_ROLES[key], "InvokeGateway", state["gateway_arn"])
    info(
        f"scoped InvokeGateway on the {len(SUB_AGENTS)} sub-agent roles to "
        f"this gateway (the orchestrator has none)"
    )

    # And narrow the gateway role's own authorization statement from gateway/*
    # to this gateway, now that its ARN is known.
    iam.put_role_policy(
        RoleName=ROLE_GATEWAY,
        PolicyName=f"{ROLE_GATEWAY}-inline",
        PolicyDocument=json.dumps(
            _gateway_role_policy(
                state["account_id"],
                state["bucket"],
                kb_arn=state.get("kb_arn"),
                policy_engine_arn=state["policy_engine_arn"],
                gateway_arn=state["gateway_arn"],
            )
        ),
    )
    info("scoped the gateway role's Cedar authorization to this gateway")
    save_state(state)


# ---------------------------------------------------------------- 8. targets
def do_targets(state: dict[str, Any]) -> None:
    step("8. gateway targets (SigV4 outbound)")
    existing = {}
    for page in agentcore.get_paginator("list_gateway_targets").paginate(
        gatewayIdentifier=state["gateway_id"]
    ):
        for target in page.get("items", []):
            existing[target.get("name")] = target["targetId"]

    def upsert(name: str, config: dict, credentials: list[dict], description: str) -> None:
        if name in existing:
            info(f"updating target {name}")
            agentcore.update_gateway_target(
                gatewayIdentifier=state["gateway_id"],
                targetId=existing[name],
                name=name,
                targetConfiguration=config,
                credentialProviderConfigurations=credentials,
            )
            target_id = existing[name]
        else:
            info(f"creating target {name}")
            target_id = agentcore.create_gateway_target(
                gatewayIdentifier=state["gateway_id"],
                name=name,
                targetConfiguration=config,
                credentialProviderConfigurations=credentials,
                description=description,
            )["targetId"]
        _wait_target_ready(state["gateway_id"], target_id, name)

    for key, spec in MCP_SERVERS.items():
        # mcpServer and openApiSchema targets REQUIRE the iamCredentialProvider
        # for SigV4 signing; credentialProviderType alone is not sufficient.
        # (Lambda/API Gateway/Smithy targets are the reverse -- they must omit
        # it.) AgentCore Runtime verifies SigV4 natively, so the tool hop needs
        # no OAuth client and no API key.
        upsert(
            spec["target"],
            {
                "mcp": {
                    "mcpServer": {
                        "endpoint": state["mcp_runtimes"][key]["url"],
                        "listingMode": "DEFAULT",
                    }
                }
            },
            [
                {
                    "credentialProviderType": "GATEWAY_IAM_ROLE",
                    "credentialProvider": {
                        "iamCredentialProvider": {
                            "service": "bedrock-agentcore",
                            "region": AWS_REGION,
                        }
                    },
                }
            ],
            f"MCP server: {key}",
        )

    # The managed knowledge base, via the built-in connector. Connector targets
    # take a PLAIN GATEWAY_IAM_ROLE with no iamCredentialProvider -- the
    # gateway signs these as the `bedrock` service itself.
    #
    # Both connector tools are configured here, but Cedar grants the
    # orchestrator only Retrieve. AgenticRetrieveStream stays configured and
    # invisible, which is the point: the administrator binds capability, the
    # policy decides who sees it.
    upsert(
        KB_TARGET,
        {
            "mcp": {
                "connector": {
                    "source": {"connectorId": "bedrock-knowledge-bases"},
                    "configurations": [
                        {
                            "name": "Retrieve",
                            "parameterValues": {"knowledgeBaseId": state["kb_id"]},
                        },
                        {
                            "name": "AgenticRetrieveStream",
                            "parameterValues": {
                                "retrievers": [
                                    {
                                        "description": "Rail fare, railcard, compensation and accessibility policy",
                                        "configuration": {
                                            "knowledgeBase": {
                                                "knowledgeBaseId": state["kb_id"]
                                            }
                                        },
                                    }
                                ],
                                "agenticRetrieveConfiguration": {
                                    "foundationModelType": "MANAGED",
                                    "rerankingModelType": "MANAGED",
                                },
                            },
                        },
                    ],
                }
            }
        },
        [{"credentialProviderType": "GATEWAY_IAM_ROLE"}],
        "Bedrock Managed Knowledge Base: rail policy",
    )

    info("tools appear as TargetName___toolName, e.g. RailPolicyKB___Retrieve")
    save_state(state)


def _wait_target_ready(
    gateway_id: str, target_id: str, name: str, timeout: int = 180
) -> None:
    """Targets validate asynchronously (~30s), including a GetKnowledgeBase
    check on each bound knowledge base. A FAILED status carries a reason."""
    deadline, last = time.time() + timeout, None
    while time.time() < deadline:
        described = agentcore.get_gateway_target(
            gatewayIdentifier=gateway_id, targetId=target_id
        )
        status = described.get("status")
        if status != last:
            info(f"  {name}: {status}")
            last = status
        if status == "READY":
            return
        if status in {"FAILED", "UPDATE_FAILED"}:
            reasons = described.get("statusReasons") or [described]
            raise RuntimeError(f"target {name} is {status}: {reasons}")
        time.sleep(6)
    raise RuntimeError(f"target {name} did not become READY within {timeout}s")


# ---------------------------------------------------------------- 9. policies
def _split_policy_statements(text: str) -> list[str]:
    statements, current = [], []
    for line in text.splitlines():
        if line.strip().startswith("//"):
            continue
        current.append(line)
        if line.strip().endswith(";"):
            statement = "\n".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
    return statements


def do_policies(state: dict[str, Any]) -> None:
    step("9. Cedar policies (IamEntity principals, per-tool grants)")
    import re

    from common import policy_shim

    principals = {
        key: cedar_principal(state[f"role_{key}"]) for key in AGENT_ROLES
    }
    for key, principal in principals.items():
        info(f"{key}: {principal}")

    template_text = POLICY_TEMPLATE.read_text(encoding="utf-8")
    rendered = (
        template_text.replace("{{GATEWAY_ARN}}", state["gateway_arn"])
        .replace("{{PLANNER_PRINCIPAL}}", principals["journey_planner"])
        .replace("{{SUPPORT_PRINCIPAL}}", principals["journey_support"])
        .replace(
            "{{POLICY_ADVISOR_PRINCIPAL}}", principals["policy_advisor"]
        )
        # The knowledge-base statement is bound to policy_advisor, which is
        # the agent that now owns that tool. Both placeholder spellings are
        # substituted with the same principal so this works whether or not the
        # template has been renamed yet -- see the NOTE printed below.
        .replace("{{ORCHESTRATOR_PRINCIPAL}}", principals["policy_advisor"])
    )
    RENDERED_POLICY_PATH.write_text(rendered, encoding="utf-8")
    info(f"rendered -> {RENDERED_POLICY_PATH.relative_to(ROOT)}")

    if "{{ORCHESTRATOR_PRINCIPAL}}" in template_text:
        info("")
        info(
            "NOTE the template still names the knowledge-base statement's "
            "placeholder {{ORCHESTRATOR_PRINCIPAL}}, from when the router held"
        )
        info(
            "     the KB tool itself. It is bound to the policy_advisor "
            "role here, which is where that grant now lives -- the"
        )
        info(
            "     substitution is right, but the placeholder name and the "
            "prose around it in deploy/gateway/policies.cedar are stale."
        )

    statements = _split_policy_statements(rendered)
    if len(statements) != len(POLICY_AGENTS):
        raise RuntimeError(
            f"expected {len(POLICY_AGENTS)} permit statements in the template "
            f"(one per agent in POLICY_AGENTS), found {len(statements)}"
        )

    # The template is the renderer; AGENT_TOOL_GRANTS is the authority. Check
    # them against each other before anything is created, because the two
    # drifting apart is silent in both directions: the local shim would grant
    # what the deployed Cedar policy does not, and the demo would behave
    # differently in the two run modes with nothing to point at.
    for agent, statement in zip(POLICY_AGENTS, statements, strict=True):
        if principals[agent] not in statement:
            raise RuntimeError(
                f"statement {POLICY_AGENTS.index(agent) + 1} does not name "
                f"{agent}'s principal {principals[agent]}. The template's "
                f"permit statements must appear in POLICY_AGENTS order."
            )
        actions = set(re.findall(r'AgentCore::Action::"([^"]+)"', statement))
        granted = set(policy_shim.granted_qualified_tools(agent))
        if actions != granted:
            raise RuntimeError(
                f"{agent}: the Cedar template grants {sorted(actions)} but "
                f"AGENT_TOOL_GRANTS says {sorted(granted)}. Reconcile "
                f"deploy/gateway/policies.cedar with common/config.py -- the "
                f"local shim and the gateway must grant the same thing."
            )
    info(f"grants  all {len(POLICY_AGENTS)} policies agree with AGENT_TOOL_GRANTS")

    # Agents with an empty grant get NO policy at all. Cedar cannot express
    # "permit nothing" -- an action list must be non-empty, and a permit with
    # an empty list is a validation failure, which would leave the whole engine
    # with a CREATE_FAILED policy. Default-deny already means zero tools, so
    # the absence of a policy IS the grant. The router is the case this exists
    # for: Tier 1 classifies and delegates, and holds no tools of its own.
    for agent in AGENT_ROLES:
        if agent not in POLICY_AGENTS:
            info("")
            info(
                f"no policy for {agent}: its grant is empty, and Cedar has no "
                f"way to write a permit with no actions."
            )
            info(
                "              Cedar is default-deny, so no policy means no "
                "tools -- which is exactly the intent. `verify` expects 0."
            )

    existing = {}
    for page in agentcore.get_paginator("list_policies").paginate(
        policyEngineId=state["policy_engine_id"]
    ):
        for policy in page.get("policies", []):
            existing[policy.get("name")] = policy

    # A previous deploy may have left a policy behind for an agent that no
    # longer has a grant. The router's old RailPolicyKB___Retrieve policy is
    # exactly that: it would keep granting the tool the new topology says the
    # router must not see, and `verify` would read 1 tool where it expects 0.
    # Default-deny only holds if nothing is permitting.
    for agent in AGENT_ROLES:
        if agent in POLICY_AGENTS:
            continue
        name = f"{PREFIX_U}_{agent}_tools"
        stale = existing.pop(name, None)
        if stale:
            info(f"deleting stale policy {name} ({agent} now holds no tools)")
            agentcore.delete_policy(
                policyEngineId=state["policy_engine_id"],
                policyId=stale["policyId"],
            )

    created: list[tuple[str, str]] = []
    for agent, statement in zip(POLICY_AGENTS, statements, strict=True):
        name = f"{PREFIX_U}_{agent}_tools"
        definition = {"cedar": {"statement": statement}}
        current = existing.get(name)

        # A CREATE_FAILED policy cannot be updated into health -- there is no
        # good version to update. Delete and recreate.
        if current and current.get("status") in {"CREATE_FAILED", "UPDATE_FAILED"}:
            info(f"{name} is {current['status']}; deleting before recreate")
            agentcore.delete_policy(
                policyEngineId=state["policy_engine_id"],
                policyId=current["policyId"],
            )
            time.sleep(3)
            current = None

        if current:
            info(f"updating policy {name}")
            agentcore.update_policy(
                policyEngineId=state["policy_engine_id"],
                policyId=current["policyId"],
                definition=definition,
            )
            created.append((name, current["policyId"]))
        else:
            info(f"creating policy {name}")
            response = agentcore.create_policy(
                policyEngineId=state["policy_engine_id"],
                name=name,
                definition=definition,
                description=f"Tool grant for {agent}",
                validationMode="FAIL_ON_ANY_FINDINGS",
                enforcementMode="ACTIVE",
            )
            policy = response.get("policy", response)
            created.append((name, policy.get("policyId", name)))

    _wait_policies_active(state["policy_engine_id"], created)

    info("")
    info("the shared-target asymmetry these policies enforce:")
    for line in policy_shim.describe_shared_asymmetry():
        info(f"  {line}")
    save_state(state)


def _wait_policies_active(engine_id: str, policies: list[tuple[str, str]]) -> None:
    """Poll each policy to ACTIVE, and fail loudly with the real reason.

    CreatePolicy returns before validation completes. A policy that fails
    validation lands in CREATE_FAILED asynchronously, which means NO policy is
    active -- and under default-deny that shows up much later as "every agent
    sees zero tools", with nothing to point at. Checking here converts that
    into an immediate, specific error.
    """
    info("")
    info("waiting for policy validation")
    deadline = time.time() + 180
    pending = dict(policies)
    failures: list[str] = []

    while pending and time.time() < deadline:
        for name, policy_id in list(pending.items()):
            try:
                described = agentcore.get_policy(
                    policyEngineId=engine_id, policyId=policy_id
                )
            except ClientError:
                continue
            policy = described.get("policy", described)
            status = policy.get("status")
            if status in {"ACTIVE", "AVAILABLE", "CREATED"}:
                info(f"  {name}: {status}")
                pending.pop(name)
            elif status and "FAILED" in status:
                reasons = policy.get("statusReasons") or [
                    policy.get("failureReason", "(no reason given)")
                ]
                failures.append(f"{name} [{status}]: {'; '.join(map(str, reasons))}")
                pending.pop(name)
        if pending:
            time.sleep(5)

    if pending:
        failures.append(
            f"still pending after 180s: {', '.join(pending)}"
        )

    if failures:
        raise RuntimeError(
            "Cedar policy validation failed:\n"
            + "".join(f"    - {f}\n" for f in failures)
            + "\n  The engine validates every action against the gateway's real\n"
            "  tools, parsing each name as Target___tool. Common causes:\n"
            "    * a meta-action like 'initialize' or 'tools/list' -- these are\n"
            "      NOT valid actions; the handshake is permitted implicitly\n"
            "    * a target name that does not exist on the gateway\n"
            "    * a tool name that does not exist on that target\n"
            "  A failed policy means NO policy is active, and default-deny then\n"
            "  shows up as every agent seeing zero tools."
        )
    info("all policies ACTIVE")


# ---------------------------------------------------------------- 11. agents
def do_agent_runtimes(state: dict[str, Any]) -> None:
    step("11. agent runtimes")
    state.setdefault("agent_runtimes", {})

    for agent, entry in SUB_AGENTS.items():
        # No gateway URL, no peer URL, no memory id in the environment. Those
        # are resolved from Parameter Store at use time, and an environment
        # variable would WIN over the store -- silently pinning the runtime to
        # whatever was true on deploy day and making the parameter inert.
        _, arn = _ensure_runtime(
            state, agent, entry, "A2A", state[f"role_{agent}"], {}
        )
        url = runtime_invocation_url(arn, with_qualifier=False)

        # Second phase. The agent card must advertise THIS url, and the url is
        # derived from the arn, which did not exist until the create above.
        # Without this the card says 0.0.0.0:9000 and every delegation fails,
        # because A2A clients route by card.url.
        #
        # This one stays in the environment deliberately: it is an agent's own
        # identity, needed before it can serve its card, and it changes only
        # when the runtime itself is replaced. Nothing else reads it.
        info(f"  setting {agent.upper()}_A2A_URL for its own agent card")
        _ensure_runtime(
            state,
            agent,
            entry,
            "A2A",
            state[f"role_{agent}"],
            {f"{agent.upper()}_A2A_URL": url},
        )

        state["agent_runtimes"][agent] = {"arn": arn, "url": url}
        # Publish for the CALLERS. do_parameters could not: the ARN this url
        # comes from did not exist when it ran.
        publish_parameter(f"{agent}_a2a_url", url)
        save_state(state)

    name, entry = ORCHESTRATOR
    _, arn = _ensure_runtime(
        state, name, entry, "HTTP", state[f"role_{name}"], {}
    )
    state["agent_runtimes"][name] = {
        "arn": arn,
        "url": runtime_invocation_url(arn, with_qualifier=True),
    }

    # Tighten InvokeAgentRuntime from runtime/* to the exact sub-agents. The
    # endpoint children must be included -- see _runtime_resources.
    #
    # Built from SUB_AGENTS, so a new sub-agent is covered the moment it is
    # added to that dict -- there is no second list to keep in step. There is
    # no longer any peer-to-peer narrowing to do: the Journey Support role's
    # InvokePlannerForPeerA2A statement went away with the peer hop itself.
    sub_arns = [state["agent_runtimes"][a]["arn"] for a in SUB_AGENTS]
    _retarget_statement(
        ROLE_ORCHESTRATOR, "InvokeSubAgents", _runtime_resources(sub_arns)
    )
    info(
        f"scoped InvokeAgentRuntime + GetAgentCard to the exact "
        f"{len(sub_arns)} sub-agent runtimes"
    )

    state["parameters"] = sorted(
        set(state.get("parameters", []))
        | {
            config_store.parameter_name(f"{a}_a2a_url")
            for a in SUB_AGENTS
        }
    )
    save_state(state)

    # Now that the A2A urls are published, the picture should be complete.
    info("")
    _check_required_config()


# ---------------------------------------------------------------- 12. env
def do_env(state: dict[str, Any]) -> None:
    step("12. write deploy/agentcore/.env")
    runtimes = state.get("agent_runtimes", {})
    lines = [
        "# Generated by deploy/agentcore/deploy.py",
        "# Authorization is IAM/SigV4, so there are NO secrets in this file.",
        "#   source deploy/agentcore/.env",
        "",
        "# This file is for LOCAL CLIENTS (invoke.py, seed_memory.py, the",
        "# frontend proxy, demo.py) -- not for the deployed runtimes, which",
        "# carry only bootstrap settings and read the rest from Parameter",
        "# Store at use time.",
        "#",
        "# Note the resolution order: an exported variable BEATS the store. So",
        "# while these exports are sourced, a local process is pinned to the",
        "# values below. That is what you want for a client talking to a known",
        "# deployment -- but it also means that to see a put-parameter take",
        "# effect locally you must unset the matching variable first:",
        "#   unset TRAVEL_MODEL_ID TRAVEL_GATEWAY_URL",
        "# The runtimes in AWS are unaffected: they never had these set.",
        "",
        "export TRAVEL_RUN_MODE=agentcore",
        f"export AWS_REGION={AWS_REGION}",
        "export TRAVEL_CONFIG_SOURCE=ssm",
        f"export TRAVEL_CONFIG_PREFIX={config_store.prefix()}",
        f"export TRAVEL_GATEWAY_URL={state['gateway_url']}",
        f"export TRAVEL_GATEWAY_ARN={state['gateway_arn']}",
        "",
    ]
    for agent in SUB_AGENTS:
        if agent in runtimes:
            lines.append(f"export {agent.upper()}_A2A_URL={runtimes[agent]['url']}")
    lines += [
        "",
        f"export TRAVEL_ORCHESTRATOR_ARN="
        f"{runtimes.get('train_travel_agent', {}).get('arn', '')}",
        f"export TRAVEL_KB_ID={state.get('kb_id', '')}",
        f"export TRAVEL_MEMORY_ID={state.get('memory_id', '')}",
        "",
        "# Observability: traces land on the CloudWatch GenAI Observability page.",
        "# Each runtime already carries these; set them locally only if you want",
        "# a local agent to export traces too.",
        "# export AGENT_OBSERVABILITY_ENABLED=true",
        "",
        "# Per-agent execution roles -- these ARNs are the Cedar principals.",
    ]
    for key in AGENT_ROLES:
        if state.get(f"role_{key}"):
            lines.append(f"# {key}: {state[f'role_{key}']}")
    lines.append("")
    ENV_PATH.write_text("\n".join(lines), encoding="utf-8")
    info(f"wrote {ENV_PATH.relative_to(ROOT)} (no credentials in it)")


# ---------------------------------------------------------------- 13. enforce
def do_enforce(state: dict[str, Any]) -> None:
    step("13. flip the policy engine to ENFORCE")
    agentcore.update_gateway(
        gatewayIdentifier=state["gateway_id"],
        name=GATEWAY_NAME,
        roleArn=state["role_gateway"],
        protocolType="MCP",
        authorizerType="AWS_IAM",
        policyEngineConfiguration={
            "arn": state["policy_engine_arn"],
            "mode": "ENFORCE",
        },
    )
    info("policy engine mode: ENFORCE")
    save_state(state)
    do_verify(state)


# ---------------------------------------------------------------- verify
# What each identity must see in tools/list once Cedar is enforcing. Keyed on
# every agent, including the one expected to see nothing.
EXPECTED_TOOLS = {
    "journey_planner": {
        "JourneySearch___search_journeys",
        "JourneySearch___get_fares",
        "JourneySearch___book_ticket",
        "TravellerProfile___get_traveller_profile",
        "TravellerProfile___list_saved_journeys",
    },
    "journey_support": {
        "TicketOperations___get_booking",
        "TicketOperations___change_or_cancel_ticket",
        "TicketOperations___claim_delay_repay",
        "TravellerProfile___get_railcard_and_loyalty",
    },
    # The knowledge base moved off the router and onto this Tier-2 agent. One
    # tool: Retrieve. AgenticRetrieveStream is configured on the same target
    # and stays invisible.
    "policy_advisor": {"RailPolicyKB___Retrieve"},
    # ZERO, and that is a PASS. The router classifies intent and delegates; it
    # holds no gateway tools, has no Cedar policy, and has no InvokeGateway in
    # its role -- so the gateway may well refuse the handshake outright rather
    # than return an empty catalogue. do_verify accepts either outcome: both
    # mean nothing is reachable.
    "train_travel_agent": set(),
}

SHARED_TARGET = "TravellerProfile"


def _verify_observability() -> None:
    """Each runtime must declare a log group that actually exists.

    aws.log.group.names is how CloudWatch indexes telemetry under the Bedrock
    AgentCore Observability dashboard. Point it at a log group that is not
    there and the spans still arrive in aws/spans -- so tracing looks healthy
    from every angle except the dashboard, which reports zero agents. Nothing
    errors, which is exactly why this needs an explicit check.
    """
    info("")
    info("observability: does each runtime declare a real log group?")
    real = set()
    paginator = logs_client.get_paginator("describe_log_groups")
    for page in paginator.paginate(
        logGroupNamePrefix="/aws/bedrock-agentcore/runtimes/"
    ):
        real.update(g["logGroupName"] for g in page["logGroups"])

    bad = 0
    for key in list(MCP_SERVERS) + list(SUB_AGENTS) + [ORCHESTRATOR[0]]:
        full_name = f"{PREFIX_U}_{key}"
        found = _find_runtime(full_name)
        if not found:
            info(f"  {full_name}: NOT DEPLOYED")
            continue
        declared = None
        described = agentcore.get_agent_runtime(
            agentRuntimeId=found["agentRuntimeId"]
        )
        for pair in (
            described.get("environmentVariables", {})
            .get("OTEL_RESOURCE_ATTRIBUTES", "")
            .split(",")
        ):
            k, _, v = pair.partition("=")
            if k.strip() == "aws.log.group.names":
                declared = v.strip()
        expected = runtime_log_group(found["agentRuntimeId"], full_name)
        if declared == expected and expected in real:
            info(f"  {full_name}: OK")
        else:
            bad += 1
            info(f"  {full_name}: WRONG")
            info(f"      declares {declared!r}")
            info(f"      expected {expected!r} (exists: {expected in real})")
    if bad:
        info("")
        info(
            f"  {bad} runtime(s) declare a log group that does not match. Their "
            f"spans will reach aws/spans but AgentCore Observability will show "
            f"0 agents. Re-run:  --step agents  (and --step mcp-runtimes)"
        )
    else:
        info("  all runtimes declare their real log group")


def do_verify(state: dict[str, Any]) -> None:
    """Prove Cedar really is filtering tools/list per agent, per TOOL.

    Assumes each agent's execution role and calls the gateway as that role, so
    this exercises the identical identity path the deployed agent uses.

    Four identities, four different catalogues from one gateway endpoint:

        journey_planner     5 tools   JourneySearch x3 + TravellerProfile x2
        journey_support     4 tools   TicketOperations x3 + TravellerProfile x1
        policy_advisor   1 tool    RailPolicyKB___Retrieve
        train_travel_agent  0 tools   the router -- it delegates, it does not
                                      call tools. Zero is the expected result,
                                      not a failure.
    """
    step("verify: does each agent see only its own tools?")
    info("expected: " + ", ".join(
        f"{agent} {len(tools)}" for agent, tools in EXPECTED_TOOLS.items()
    ))
    import asyncio

    from langchain_mcp_adapters.client import MultiServerMCPClient

    from common.aws_sigv4 import SigV4HttpxAuth

    async def catalogue(agent: str) -> set[str]:
        assumed = sts.assume_role(
            RoleArn=state[f"role_{agent}"],
            RoleSessionName=f"verify-{agent}"[:64],
        )["Credentials"]
        from botocore.credentials import Credentials

        signer = SigV4HttpxAuth(
            service="bedrock-agentcore",
            region=AWS_REGION,
            credentials=Credentials(
                access_key=assumed["AccessKeyId"],
                secret_key=assumed["SecretAccessKey"],
                token=assumed["SessionToken"],
            ),
        )
        client = MultiServerMCPClient(
            {
                "gateway": {
                    "url": state["gateway_url"],
                    "transport": "streamable_http",
                    "auth": signer,
                    "timeout": 60.0,
                }
            }
        )
        return {tool.name for tool in await client.get_tools(server_name="gateway")}

    failures = []
    observed: dict[str, set[str]] = {}
    for agent, expected in EXPECTED_TOOLS.items():
        try:
            names = asyncio.run(catalogue(agent))
        except Exception as exc:  # noqa: BLE001
            # An agent expected to hold NO tools is allowed to fail here. With
            # no InvokeGateway in its role, the gateway refuses the call before
            # Cedar is ever consulted, so "cannot list" and "lists nothing" are
            # the same answer: zero tools reachable. Treating this as an error
            # would make the correct deployment look broken.
            if not expected:
                info(f"{agent}: 0 tools -- the gateway refused the call outright")
                info(f"    ({type(exc).__name__}: no InvokeGateway in its role)")
                observed[agent] = set()
                continue
            failures.append(f"{agent}: could not list tools ({exc})")
            continue

        observed[agent] = names
        info(f"{agent}: {len(names)} tools")
        for name in sorted(names):
            info(f"    {name}")
        if not names and not expected:
            info("    (none, as expected -- it holds no gateway tools)")
        if names != expected:
            failures.append(
                f"{agent}: unexpected {sorted(names - expected) or 'none'}, "
                f"missing {sorted(expected - names) or 'none'}"
            )

    # The headline: the shared target, seen two different ways.
    if {"journey_planner", "journey_support"} <= observed.keys():
        prefix = f"{SHARED_TARGET}___"
        planner = {n for n in observed["journey_planner"] if n.startswith(prefix)}
        support = {n for n in observed["journey_support"] if n.startswith(prefix)}
        print(f"\n  {'=' * 64}")
        print(f"  SHARED TARGET {SHARED_TARGET} -- one endpoint, two catalogues")
        print(f"  {'=' * 64}")
        print(f"  journey_planner sees {len(planner)} tool(s):")
        for n in sorted(planner):
            print(f"      {n.removeprefix(prefix)}")
        print(f"  journey_support  sees {len(support)} tool(s):")
        for n in sorted(support):
            print(f"      {n.removeprefix(prefix)}")
        hidden_from_support = sorted(planner - support)
        if hidden_from_support:
            print(
                "  Cedar hides from Journey Support: "
                + ", ".join(n.removeprefix(prefix) for n in hidden_from_support)
            )

    if failures:
        print("\n  VERIFICATION FAILED:")
        for failure in failures:
            print(f"    - {failure}")
        print(
            "\n  If every agent sees everything, the policy engine is probably "
            "still LOG_ONLY.\n  If an agent that should HAVE tools sees none, "
            "the Cedar principal ARNs do not\n  match what the gateway observed"
            " -- check the assumed-role form in\n  policies.rendered.cedar. "
            "(train_travel_agent seeing none is correct.)\n"
            "  Re-run: deploy.py deploy --step policies"
        )
        raise SystemExit(1)

    print(
        "\n  Cedar is filtering tools/list correctly, per tool, not just per "
        "server."
    )
    _verify_observability()


# ---------------------------------------------------------------- teardown
def do_teardown(state: dict[str, Any]) -> None:
    step("teardown")
    if input(
        "This deletes the gateway, targets, policies, all 7 runtimes, the\n"
        "managed knowledge base, the memory, the 8 IAM roles and the code bucket.\n"
        "Type 'delete' to confirm: "
    ).strip() != "delete":
        print("aborted, nothing deleted.")
        return

    def attempt(label: str, fn) -> None:
        try:
            fn()
            info(f"deleted {label}")
        except Exception as exc:  # noqa: BLE001
            info(f"skipped {label} ({type(exc).__name__})")

    if state.get("gateway_id"):
        for page in agentcore.get_paginator("list_gateway_targets").paginate(
            gatewayIdentifier=state["gateway_id"]
        ):
            for target in page.get("items", []):
                attempt(
                    f"target {target.get('name')}",
                    lambda t=target: agentcore.delete_gateway_target(
                        gatewayIdentifier=state["gateway_id"], targetId=t["targetId"]
                    ),
                )
        attempt(
            "gateway",
            lambda: agentcore.delete_gateway(gatewayIdentifier=state["gateway_id"]),
        )

    if state.get("policy_engine_id"):
        for page in agentcore.get_paginator("list_policies").paginate(
            policyEngineId=state["policy_engine_id"]
        ):
            for policy in page.get("policies", []):
                attempt(
                    f"policy {policy.get('name')}",
                    lambda p=policy: agentcore.delete_policy(
                        policyEngineId=state["policy_engine_id"],
                        policyId=p["policyId"],
                    ),
                )
        attempt(
            "policy engine",
            lambda: agentcore.delete_policy_engine(
                policyEngineId=state["policy_engine_id"]
            ),
        )

    for name in list(MCP_SERVERS) + list(SUB_AGENTS) + [ORCHESTRATOR[0]]:
        full = f"{PREFIX_U}_{name}"
        runtime = _find_runtime(full)
        if runtime:
            attempt(
                f"runtime {full}",
                lambda r=runtime: agentcore.delete_agent_runtime(
                    agentRuntimeId=r["agentRuntimeId"]
                ),
            )

    if state.get("parameters"):
        names = state["parameters"]
        for i in range(0, len(names), 10):
            attempt(
                f"{len(names[i:i + 10])} ssm parameter(s)",
                lambda b=names[i:i + 10]: ssm.delete_parameters(Names=b),
            )

    if state.get("memory_id"):
        attempt(
            "memory",
            lambda: agentcore.delete_memory(memoryId=state["memory_id"]),
        )

    if state.get("kb_id"):
        if state.get("kb_data_source_id"):
            attempt(
                "kb data source",
                lambda: bedrock_agent.delete_data_source(
                    knowledgeBaseId=state["kb_id"],
                    dataSourceId=state["kb_data_source_id"],
                ),
            )
        attempt(
            "managed knowledge base",
            lambda: bedrock_agent.delete_knowledge_base(
                knowledgeBaseId=state["kb_id"]
            ),
        )

    for role in ALL_ROLES:
        attempt(
            f"role policy for {role}",
            lambda r=role: iam.delete_role_policy(
                RoleName=r, PolicyName=f"{r}-inline"
            ),
        )
        attempt(f"role {role}", lambda r=role: iam.delete_role(RoleName=r))

    if state.get("code_bucket"):
        def empty_and_delete() -> None:
            for page in s3.get_paginator("list_object_versions").paginate(
                Bucket=state["code_bucket"]
            ):
                objects = [
                    {"Key": o["Key"], "VersionId": o["VersionId"]}
                    for o in page.get("Versions", []) + page.get("DeleteMarkers", [])
                ]
                if objects:
                    s3.delete_objects(
                        Bucket=state["code_bucket"], Delete={"Objects": objects}
                    )
            s3.delete_bucket(Bucket=state["code_bucket"])

        attempt("code bucket", empty_and_delete)

    for path in (STATE_PATH, ENV_PATH, RENDERED_POLICY_PATH):
        if path.exists():
            path.unlink()
            info(f"removed {path.name}")


# ---------------------------------------------------------------- driver
STEPS = [
    ("preflight", do_preflight),
    ("iam", do_iam),
    ("package", do_package),
    ("mcp-runtimes", do_mcp_runtimes),
    ("knowledge-base", do_knowledge_base),
    ("memory", do_memory),
    ("policy-engine", do_policy_engine),
    ("gateway", do_gateway),
    ("targets", do_targets),
    ("policies", do_policies),
    ("parameters", do_parameters),
    ("agents", do_agent_runtimes),
    ("env", do_env),
    ("enforce", do_enforce),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["preflight", "deploy", "verify", "teardown", "steps"]
    )
    parser.add_argument("--step", help="run a single deploy step by name")
    parser.add_argument("--from-step", help="resume from this step onwards")
    args = parser.parse_args()

    if args.command == "steps":
        print("deploy steps, in order:")
        for name, _ in STEPS:
            print(f"  {name}")
        return 0

    state = load_state()
    if args.command == "preflight":
        do_preflight(state)
        return 0
    if args.command == "verify":
        do_verify(state)
        return 0
    if args.command == "teardown":
        do_teardown(state)
        return 0

    selected = STEPS
    names = [n for n, _ in STEPS]
    if args.step:
        if args.step not in names:
            print(f"unknown step {args.step!r}", file=sys.stderr)
            return 2
        selected = [(n, f) for n, f in STEPS if n == args.step]
    elif args.from_step:
        if args.from_step not in names:
            print(f"unknown step {args.from_step!r}", file=sys.stderr)
            return 2
        selected = STEPS[names.index(args.from_step):]

    started = time.time()
    for _name, fn in selected:
        fn(state)
    print(f"\n{'=' * 70}\ndone in {time.time() - started:.0f}s")
    if not args.step:
        print(
            "\nnext:\n"
            "  source deploy/agentcore/.env\n"
            '  python deploy/agentcore/invoke.py --beat 3\n'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
