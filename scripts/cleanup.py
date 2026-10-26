"""Delete every AWS resource this demo creates. Discovery-based, not state-based.

Why this exists alongside `deploy.py teardown`: teardown reads
.deploy-state.json and deletes what it finds there. If that file is lost,
truncated, or a deploy failed partway through before writing it, teardown
silently misses resources and you keep paying for them. This script instead
DISCOVERS resources by naming convention, so it cleans up correctly even with
no state file at all.

It is also more thorough: it removes CloudWatch log groups and orphaned
workload identities, which teardown leaves behind.

SAFETY
  * Dry run by DEFAULT. It prints what it would delete and exits.
  * Deleting requires --yes AND typing the confirmation phrase.
  * It only ever touches resources whose names begin with the demo prefixes
    (travel-demo / travel_demo). Unrelated resources in the account are listed
    as skipped so you can see it left them alone.

    python scripts/cleanup.py                 # dry run -- shows what it finds
    python scripts/cleanup.py --yes           # actually delete (prompts)
    python scripts/cleanup.py --yes --local   # also remove local artefacts
    python scripts/cleanup.py --local-only    # local artefacts only, no AWS
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import boto3  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402

from common.config import AWS_REGION  # noqa: E402

# Everything this demo creates starts with one of these. Nothing else is
# touched -- the account may hold unrelated gateways and policy engines.
PREFIX_HYPHEN = "travel-demo"
PREFIX_UNDER = "travel_demo"
LOG_GROUP_PREFIX = f"/aws/bedrock-agentcore/runtimes/{PREFIX_UNDER}"

CONFIRM_PHRASE = "delete travel demo"

# Local files and directories, removed only with --local / --local-only.
LOCAL_ARTEFACTS = [
    "deploy/agentcore/.deploy-state.json",
    "deploy/agentcore/.env",
    "deploy/agentcore/policies.rendered.cedar",
    "knowledgebase/.kb_index.json",
    ".build",
    ".run",
]

session = boto3.Session(region_name=AWS_REGION)
agentcore = session.client("bedrock-agentcore-control")
bedrock_agent = session.client("bedrock-agent")
iam = session.client("iam")
s3 = session.client("s3")
sts = session.client("sts")
logs = session.client("logs")
ssm = session.client("ssm")

FOUND: list[tuple[str, str]] = []
DELETED: list[str] = []
FAILED: list[str] = []
SKIPPED: list[str] = []


def ours(name: str | None) -> bool:
    return bool(name) and (
        name.startswith(PREFIX_HYPHEN) or name.startswith(PREFIX_UNDER)
    )


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def note(text: str) -> None:
    print(f"  {text}")


def plan(kind: str, identifier: str) -> None:
    FOUND.append((kind, identifier))
    note(f"{kind:22} {identifier}")


def execute(label: str, fn: Callable[[], Any], live: bool) -> None:
    """Run a deletion, or report it in dry-run."""
    if not live:
        return
    try:
        fn()
        DELETED.append(label)
        note(f"    deleted {label}")
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in {"ResourceNotFoundException", "NoSuchEntity", "NoSuchBucket",
                    "ResourceNotFoundFault", "NotFoundException"}:
            note(f"    already gone: {label}")
            return
        FAILED.append(f"{label}: {code}")
        note(f"    FAILED {label}: {code} -- {exc.response['Error']['Message'][:120]}")
    except Exception as exc:  # noqa: BLE001
        FAILED.append(f"{label}: {type(exc).__name__}")
        note(f"    FAILED {label}: {type(exc).__name__}: {exc}")


def paginate(client: Any, operation: str, key: str, **kwargs: Any) -> list[dict]:
    """Collect every page of a list operation, tolerating absent pagination."""
    items: list[dict] = []
    try:
        paginator = client.get_paginator(operation)
        for page in paginator.paginate(**kwargs):
            items.extend(page.get(key, []))
    except Exception:  # noqa: BLE001
        try:
            method = getattr(client, operation)
            items.extend(method(**kwargs).get(key, []))
        except Exception as exc:  # noqa: BLE001
            note(f"    could not list via {operation}: {type(exc).__name__}: {exc}")
    return items


# ---------------------------------------------------------------- gateways
def clean_gateways(live: bool) -> None:
    banner("gateways and gateway targets")
    gateways = paginate(agentcore, "list_gateways", "items")
    mine = [g for g in gateways if ours(g.get("name"))]
    for g in gateways:
        if not ours(g.get("name")):
            SKIPPED.append(f"gateway {g.get('name')}")

    if not mine:
        note("none found")
        return

    for gateway in mine:
        gid = gateway["gatewayId"]
        plan("gateway", f"{gateway.get('name')} ({gid})")

        # Targets must go before the gateway that holds them.
        targets = paginate(
            agentcore, "list_gateway_targets", "items", gatewayIdentifier=gid
        )
        for target in targets:
            plan("  gateway target", target.get("name", target["targetId"]))
            execute(
                f"target {target.get('name')}",
                lambda t=target, g=gid: agentcore.delete_gateway_target(
                    gatewayIdentifier=g, targetId=t["targetId"]
                ),
                live,
            )
        if targets and live:
            time.sleep(5)

        execute(
            f"gateway {gateway.get('name')}",
            lambda g=gid: agentcore.delete_gateway(gatewayIdentifier=g),
            live,
        )


# ---------------------------------------------------------------- policies
def clean_policy_engines(live: bool) -> None:
    banner("policy engines and Cedar policies")
    engines = paginate(agentcore, "list_policy_engines", "policyEngines")
    mine = [e for e in engines if ours(e.get("name"))]
    for e in engines:
        if not ours(e.get("name")):
            SKIPPED.append(f"policy engine {e.get('name')}")

    if not mine:
        note("none found")
        return

    for engine in mine:
        eid = engine["policyEngineId"]
        plan("policy engine", f"{engine.get('name')} ({eid})")

        for policy in paginate(
            agentcore, "list_policies", "policies", policyEngineId=eid
        ):
            plan("  cedar policy", policy.get("name", policy["policyId"]))
            execute(
                f"policy {policy.get('name')}",
                lambda p=policy, e=eid: agentcore.delete_policy(
                    policyEngineId=e, policyId=p["policyId"]
                ),
                live,
            )
        execute(
            f"policy engine {engine.get('name')}",
            lambda e=eid: agentcore.delete_policy_engine(policyEngineId=e),
            live,
        )


# ---------------------------------------------------------------- runtimes
def clean_runtimes(live: bool) -> None:
    banner("agent runtimes")
    runtimes = paginate(agentcore, "list_agent_runtimes", "agentRuntimes")
    mine = [r for r in runtimes if ours(r.get("agentRuntimeName"))]
    for r in runtimes:
        if not ours(r.get("agentRuntimeName")):
            SKIPPED.append(f"runtime {r.get('agentRuntimeName')}")

    if not mine:
        note("none found")
        return

    for runtime in mine:
        rid = runtime["agentRuntimeId"]
        plan("runtime", f"{runtime['agentRuntimeName']} [{runtime.get('status')}]")

        # Non-DEFAULT endpoints block runtime deletion.
        for endpoint in paginate(
            agentcore, "list_agent_runtime_endpoints", "runtimeEndpoints",
            agentRuntimeId=rid,
        ):
            if endpoint.get("name") == "DEFAULT":
                continue
            plan("  runtime endpoint", endpoint.get("name", ""))
            execute(
                f"endpoint {endpoint.get('name')}",
                lambda e=endpoint, r=rid: agentcore.delete_agent_runtime_endpoint(
                    agentRuntimeId=r, endpointName=e["name"]
                ),
                live,
            )

        execute(
            f"runtime {runtime['agentRuntimeName']}",
            lambda r=rid: agentcore.delete_agent_runtime(agentRuntimeId=r),
            live,
        )


# ---------------------------------------------------------------- memory
def clean_memories(live: bool) -> None:
    banner("memory")
    # ListMemories returns arn/id/status with NO name field, so match on the id
    # prefix -- memory ids are derived from the name.
    memories = paginate(agentcore, "list_memories", "memories")
    mine = [m for m in memories if ours(m.get("id"))]
    for m in memories:
        if not ours(m.get("id")):
            SKIPPED.append(f"memory {m.get('id')}")

    if not mine:
        note("none found")
        return

    for memory in mine:
        mid = memory["id"]
        plan("memory", f"{mid} [{memory.get('status')}]")
        note("    (events and extracted records go with it)")
        execute(
            f"memory {mid}",
            lambda m=mid: agentcore.delete_memory(memoryId=m),
            live,
        )


# ---------------------------------------------------------------- knowledge base
def clean_knowledge_bases(live: bool) -> None:
    banner("knowledge bases")
    kbs = paginate(bedrock_agent, "list_knowledge_bases", "knowledgeBaseSummaries")
    mine = [k for k in kbs if ours(k.get("name"))]
    for k in kbs:
        if not ours(k.get("name")):
            SKIPPED.append(f"knowledge base {k.get('name')}")

    if not mine:
        note("none found")
        return

    for kb in mine:
        kid = kb["knowledgeBaseId"]
        plan("knowledge base", f"{kb.get('name')} ({kid}) [{kb.get('status')}]")

        for ds in paginate(
            bedrock_agent, "list_data_sources", "dataSourceSummaries",
            knowledgeBaseId=kid,
        ):
            plan("  data source", ds.get("name", ds["dataSourceId"]))
            execute(
                f"data source {ds.get('name')}",
                lambda d=ds, k=kid: bedrock_agent.delete_data_source(
                    knowledgeBaseId=k, dataSourceId=d["dataSourceId"]
                ),
                live,
            )
        if live:
            time.sleep(5)
        execute(
            f"knowledge base {kb.get('name')}",
            lambda k=kid: bedrock_agent.delete_knowledge_base(knowledgeBaseId=k),
            live,
        )


# ---------------------------------------------------------------- workload identities
def clean_workload_identities(live: bool) -> None:
    banner("workload identities")
    # Runtimes create these implicitly. They are normally removed with the
    # runtime, so anything left here is an orphan from a failed delete.
    identities = paginate(
        agentcore, "list_workload_identities", "workloadIdentities"
    )
    mine = [w for w in identities if ours(w.get("name"))]
    if not mine:
        note("none found (expected -- they go with their runtime)")
        return
    for identity in mine:
        plan("workload identity", identity["name"])
        execute(
            f"workload identity {identity['name']}",
            lambda w=identity: agentcore.delete_workload_identity(name=w["name"]),
            live,
        )


# ---------------------------------------------------------------- logs
def clean_log_groups(live: bool) -> None:
    banner("CloudWatch log groups")
    groups = paginate(
        logs, "describe_log_groups", "logGroups",
        logGroupNamePrefix=LOG_GROUP_PREFIX,
    )
    if not groups:
        note("none found")
        return
    for group in groups:
        name = group["logGroupName"]
        plan("log group", name)
        execute(
            f"log group {name}",
            lambda n=name: logs.delete_log_group(logGroupName=n),
            live,
        )


# ---------------------------------------------------------------- SSM
def clean_parameters(live: bool) -> None:
    banner("Parameter Store")
    path = f"/{PREFIX_HYPHEN}"
    names: list[str] = []
    try:
        for page in ssm.get_paginator("get_parameters_by_path").paginate(
            Path=path, Recursive=True
        ):
            names.extend(p["Name"] for p in page.get("Parameters", []))
    except ClientError as exc:
        note(f"could not list {path}/*: {exc.response['Error']['Code']}")
        return

    if not names:
        note(f"none found under {path}/")
        return
    for name in names:
        plan("ssm parameter", name)
    # delete_parameters takes up to 10 names per call.
    for i in range(0, len(names), 10):
        batch = names[i:i + 10]
        execute(
            f"{len(batch)} parameter(s)",
            lambda b=batch: ssm.delete_parameters(Names=b),
            live,
        )


# ---------------------------------------------------------------- IAM
def clean_roles(live: bool) -> None:
    banner("IAM roles")
    roles = paginate(iam, "list_roles", "Roles")
    mine = [r for r in roles if ours(r["RoleName"])]
    if not mine:
        note("none found")
        return

    for role in mine:
        name = role["RoleName"]
        plan("iam role", name)

        # A role cannot be deleted while policies remain attached.
        for policy_name in iam.list_role_policies(RoleName=name).get(
            "PolicyNames", []
        ):
            execute(
                f"inline policy {name}/{policy_name}",
                lambda n=name, p=policy_name: iam.delete_role_policy(
                    RoleName=n, PolicyName=p
                ),
                live,
            )
        for attached in iam.list_attached_role_policies(RoleName=name).get(
            "AttachedPolicies", []
        ):
            execute(
                f"detach {name}/{attached['PolicyName']}",
                lambda n=name, a=attached: iam.detach_role_policy(
                    RoleName=n, PolicyArn=a["PolicyArn"]
                ),
                live,
            )
        for profile in iam.list_instance_profiles_for_role(RoleName=name).get(
            "InstanceProfiles", []
        ):
            execute(
                f"remove {name} from {profile['InstanceProfileName']}",
                lambda n=name, p=profile: iam.remove_role_from_instance_profile(
                    InstanceProfileName=p["InstanceProfileName"], RoleName=n
                ),
                live,
            )
        execute(name, lambda n=name: iam.delete_role(RoleName=n), live)


# ---------------------------------------------------------------- S3
def clean_buckets(live: bool) -> None:
    banner("S3 code bucket")
    try:
        buckets = s3.list_buckets().get("Buckets", [])
    except ClientError as exc:
        note(f"could not list buckets: {exc.response['Error']['Code']}")
        return

    mine = [b for b in buckets if b["Name"].startswith(f"{PREFIX_HYPHEN}-code-")]
    if not mine:
        note("none found")
        return

    for bucket in mine:
        name = bucket["Name"]
        plan("s3 bucket", name)

        def empty_and_delete(n: str = name) -> None:
            # Delete object versions and markers, not just current objects --
            # a versioned bucket will refuse deletion otherwise.
            for page in s3.get_paginator("list_object_versions").paginate(Bucket=n):
                objects = [
                    {"Key": o["Key"], "VersionId": o["VersionId"]}
                    for o in page.get("Versions", []) + page.get("DeleteMarkers", [])
                ]
                for i in range(0, len(objects), 1000):
                    s3.delete_objects(Bucket=n, Delete={"Objects": objects[i:i + 1000]})
            s3.delete_bucket(Bucket=n)

        execute(f"bucket {name}", empty_and_delete, live)


# ---------------------------------------------------------------- local
def clean_local(live: bool) -> None:
    banner("local artefacts")
    for relative in LOCAL_ARTEFACTS:
        path = ROOT / relative
        if not path.exists():
            continue
        plan("local", relative)
        if not live:
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            DELETED.append(f"local {relative}")
            note(f"    removed {relative}")
        except Exception as exc:  # noqa: BLE001
            FAILED.append(f"local {relative}: {type(exc).__name__}")
            note(f"    FAILED {relative}: {exc}")


# ---------------------------------------------------------------- driver
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true",
                        help="actually delete (prompts for confirmation)")
    parser.add_argument("--local", action="store_true",
                        help="also remove local state, .env, build and run dirs")
    parser.add_argument("--local-only", action="store_true",
                        help="remove local artefacts only, touch no AWS resources")
    args = parser.parse_args()

    if args.local_only:
        clean_local(live=args.yes)
        if not args.yes:
            print("\nDRY RUN. Re-run with --yes --local-only to remove these.")
        return 0

    try:
        identity = sts.get_caller_identity()
    except Exception as exc:  # noqa: BLE001
        print(f"No usable AWS credentials: {exc}", file=sys.stderr)
        return 1

    live = args.yes
    print(f"account {identity['Account']}   region {AWS_REGION}")
    print(f"caller  {identity['Arn']}")
    print(f"matching resources named {PREFIX_HYPHEN}* or {PREFIX_UNDER}*")
    print(f"\nmode: {'DELETE' if live else 'DRY RUN (nothing will be deleted)'}")

    if live:
        print(
            "\nThis permanently deletes the gateway and its targets, Cedar\n"
            "policies and policy engine, all six agent runtimes, the memory\n"
            "(including every stored conversation and extracted record), the\n"
            "managed knowledge base, IAM roles, log groups and the code bucket."
        )
        answer = input(f"\nType '{CONFIRM_PHRASE}' to proceed: ").strip()
        if answer != CONFIRM_PHRASE:
            print("aborted, nothing deleted.")
            return 1

    # Dependency order: holders before held, consumers before producers.
    clean_gateways(live)          # targets, then gateways
    clean_policy_engines(live)    # policies, then engines
    clean_runtimes(live)          # endpoints, then runtimes
    clean_memories(live)
    clean_knowledge_bases(live)   # data sources, then KBs
    clean_workload_identities(live)
    clean_log_groups(live)
    clean_parameters(live)
    clean_roles(live)             # after everything that might assume them
    clean_buckets(live)
    if args.local:
        clean_local(live)

    banner("summary")
    print(f"  resources matched : {len(FOUND)}")
    if live:
        print(f"  deleted           : {len(DELETED)}")
        print(f"  failed            : {len(FAILED)}")
    print(f"  left alone (not ours): {len(SKIPPED)}")
    for item in SKIPPED:
        note(f"    kept {item}")

    if FAILED:
        print("\n  FAILURES:")
        for failure in FAILED:
            note(f"    - {failure}")
        print(
            "\n  Deletion is partly asynchronous and some resources refuse\n"
            "  removal while a dependant still exists. Re-running usually\n"
            "  clears the remainder."
        )
        return 1

    if not live:
        print(
            "\n  DRY RUN -- nothing was deleted."
            "\n  Re-run with --yes to delete, or --yes --local to also clear"
            "\n  local state and build artefacts."
        )
    elif not FOUND:
        print("\n  Nothing to do; the account is already clean.")
    else:
        print("\n  Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
