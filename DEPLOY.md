# Deploying and testing on AgentCore

End-to-end runbook. Steps 1–3 are the ones that catch problems; 4 onwards is
the actual demo.

---

## 1. Credentials and model access

```bash
aws login --profile rs-gen-ai
export AWS_PROFILE=rs-gen-ai
export AWS_REGION=us-west-2
```

```bash
python deploy/agentcore/deploy.py preflight
```

This is the gate. It reports:

- account and caller ARN
- whether `us.anthropic.claude-sonnet-4-5` is **actually invocable** (it calls
  Converse, not just `ListFoundationModels`)
- whether Titan embeddings are invocable
- whether **CloudWatch Transaction Search** is enabled, which traces need

Do not proceed until the model line says `INVOCABLE`. If Transaction Search is
off, enable it once per account:

```bash
aws xray update-trace-segment-destination --destination CloudWatchLogs
```

---

## 2. Deploy

```bash
python deploy/agentcore/deploy.py deploy
```

Fourteen idempotent steps, roughly 20–30 minutes on a first run (six runtimes
provision serially, and the knowledge base indexes). Re-runnable without
cleanup; resume with `--from-step <name>` after a failure.

```
preflight  iam  package  mcp-runtimes  knowledge-base  memory
policy-engine  gateway  targets  policies  parameters  agents
env  enforce
```

`parameters` deliberately runs **before** `agents`: the runtimes resolve their
configuration from Parameter Store rather than from baked-in environment
variables, so the values have to exist before an agent is invoked. The two A2A
URLs are the exception — they derive from runtime ARNs that do not exist yet, so
`agents` publishes those as it goes and asserts the set is complete at the end.

What gets created:

| | |
|---|---|
| 7 AgentCore runtimes | 3 × MCP, 3 × A2A, 1 × HTTP |
| 1 gateway | `AWS_IAM` inbound, SigV4 outbound |
| 4 gateway targets | 3 × `mcpServer`, 1 × managed-KB connector |
| 1 managed knowledge base | 5 documents, ingested inline |
| 1 memory store | events + summary and user-preference strategies |
| 8 IAM roles | one per agent (4), plus MCP runtimes, gateway, KB and memory |
| 1 policy engine | 3 Cedar policies — one per Tier-2 agent. The router gets **none**: Cedar is default-deny, so the absence of a policy *is* its entitlement, and `do_policies` deletes any policy left over from when it held the knowledge base. |
| 7 SSM parameters | under `/travel-demo`, read at use time, including the `sub-agents` registry the router discovers from |

The final step flips enforcement on and immediately verifies the filtering.

### Repointing a running deployment

The runtimes carry only bootstrap settings in their environment
(`TRAVEL_CONFIG_SOURCE`, `TRAVEL_CONFIG_PREFIX`, region, tracing). Everything
that might need to change — model id, gateway URL, memory id, the A2A URLs — is
read from Parameter Store on a short TTL, so a change applies to all six
runtimes without a redeploy or a new runtime version:

```bash
aws ssm put-parameter --name /travel-demo/model-id \
    --value us.anthropic.claude-haiku-4-5-20251001-v1:0 \
    --type String --overwrite
```

Within the TTL (60s) the next invocation uses the new model. The orchestrator
reports its own resolved view, with the provenance of each value:

```bash
# local stack -- the orchestrator listens on 8090
curl -s "http://127.0.0.1:8090/config?refresh=1" | python -m json.tool
```

```json
{
  "source": "config: AWS Systems Manager Parameter Store under /travel-demo",
  "resolved": {
    "model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0   <- ssm",
    "gateway_url": "https://travel-demo-gateway-....amazonaws.com/mcp   <- ssm"
  }
}
```

`/config` is a local diagnostic: a deployed AgentCore HTTP runtime only routes
`POST /invocations` and `GET /ping`, so the endpoint is not reachable against
the deployment. To confirm a parameter there, read it back with
`aws ssm get-parameter` and watch the next invocation's behaviour.

Note the resolution order — **an environment variable beats the store**. That is
why the deployed runtimes do not set these: an export would silently pin them.
`deploy/agentcore/.env` *does* export them, for local client scripts talking to
a known deployment; unset the matching variable if you want a local process to
pick up a parameter change.

---

## 3. Verify the Cedar story

```bash
source deploy/agentcore/.env
python deploy/agentcore/deploy.py verify
```

This assumes **each agent's own execution role** and performs a real
`tools/list` as that principal, so you are seeing Cedar's actual decision. It
prints the shared-target comparison:

```
SHARED TARGET TravellerProfile -- one endpoint, two catalogues
  journey_planner sees 2 tool(s):  get_traveller_profile, list_saved_journeys
  journey_support  sees 1 tool(s):  get_railcard_and_loyalty
  Cedar hides from Journey Support: get_traveller_profile, list_saved_journeys
```

If every agent sees everything, enforcement is still `LOG_ONLY`. If an agent
sees nothing, the Cedar principal ARNs do not match what the gateway observed —
check the assumed-role form in `deploy/agentcore/policies.rendered.cedar`.

---

## 4. Test from the CLI

```bash
python deploy/agentcore/invoke.py --beat 3
python deploy/agentcore/invoke.py "can I take a bike on the 07:15?"
```

`--beat 1..5` runs the scripted beats and prints what to watch for.

---

## 5. Seed memory before demoing personalisation

Extraction is **asynchronous** — long-term records do not exist on the turn
that produced them, so a fresh deployment has empty memory and the
"it remembers me" beat shows nothing.

```bash
source deploy/agentcore/.env
python scripts/seed_memory.py          # writes two prior conversations
python scripts/seed_memory.py --check  # repeat until records appear
```

`--check` prints the preference records per traveller. Run it until non-empty.

## 6. Test in the browser

```bash
.venv/bin/python frontend/server.py
# open http://127.0.0.1:8000
```

The badge top-right reads **AgentCore Runtime** when it is talking to the
deployed agent, **local stack** when it is not. When memory is on you also get
a traveller picker and a dropdown of that traveller's **3 most recent
sessions**, so you can resume an earlier conversation. Each reply reports how
many prior turns were recalled and whether long-term memory was applied.

> The frontend binds **127.0.0.1 only** and has **no authentication**. That is
> deliberate for a presenter's laptop, and the reason it must stay on loopback:
> anyone who can reach the port invokes the agent with your AWS credentials. To
> show it from another machine, forward the port over SSH rather than widening
> the bind:
>
> ```bash
> ssh -L 8000:127.0.0.1:8000 <this-host>
> ```

---

## 7. Watch the traces

CloudWatch → **GenAI Observability**, filter by service name
`travel_demo_train_travel_agent`.

A disruption question should appear as **one trace** spanning orchestrator →
Journey Support → gateway → MCP server, with a second branch orchestrator →
Journey Planner where the router fans out. Sub-agents never appear calling each
other: the topology is strict hub-and-spoke and IAM forbids a peer hop. The
explicit spans to point at are `agent.*` and `a2a.delegate`.

Gateway throughput, latency and error metrics are in the
`AWS/Bedrock-AgentCore` namespace and are **not** on the GenAI Observability
page — build a small separate dashboard if you want them on screen.

Vended gateway logs can contain request and response bodies, which here means
traveller profile data. Scope access and retention on those log groups even
with synthetic data.

---

## Teardown

Two options.

**`cleanup.py` — discovery-based, recommended.** Finds resources by naming
convention, so it works even if `.deploy-state.json` is lost or a deploy failed
partway. It also removes CloudWatch log groups and orphaned workload
identities, which `teardown` leaves behind.

```bash
python scripts/cleanup.py                 # DRY RUN -- shows what it would delete
python scripts/cleanup.py --yes           # delete (prompts for a phrase)
python scripts/cleanup.py --yes --local   # also clear local state/build/run
python scripts/cleanup.py --local-only    # local only, touch no AWS
```

Dry run is the default. It prints every matched resource and, importantly, every
resource it is **leaving alone** — so you can confirm it will not touch
unrelated gateways, policy engines or knowledge bases in the same account.

Deleting the memory destroys every stored conversation and extracted record.

**`deploy.py teardown` — state-based.** Reads `.deploy-state.json`. Fine
immediately after a successful deploy, but it misses anything absent from the
state file.

```bash
python deploy/agentcore/deploy.py teardown
```

---

## Why the knowledge base documents are not in S3

A deliberate choice, and worth being able to defend on stage.

The knowledge base uses a **custom data source with direct inline ingestion**
(`IngestKnowledgeBaseDocuments`), not an S3 data source:

- **No sync step.** Directly-ingested documents persist until deleted. An S3
  data source needs a sync job, and that sync can overwrite or remove documents
  added directly — so the two approaches do not mix cleanly.
- **We own the document id.** Re-running the deploy step **updates each
  document in place** rather than creating duplicates, because the id is the
  filename.
- **The documents version with the repo.** Editing `knowledgebase/*.md` and
  re-running one step is the whole update path. No bucket to keep in step with
  git.
- **Five small markdown files.** Inline ingestion caps at 6 MB per document and
  10 documents per call; these are a few KB each.

Switch to S3 when any of these becomes true: the corpus is large or binary
(PDFs, images), it is maintained by someone other than whoever edits this repo,
or it needs a scheduled refresh. Larger files can also be staged to S3 and
ingested **by reference** (up to 50 MB for text) while keeping direct
ingestion, which is the usual middle ground.

Note the knowledge base must be `type: MANAGED` regardless — the AgentCore
Gateway knowledge-base connector supports managed knowledge bases only. And for
a managed KB the custom data source is
`type: "MANAGED_KNOWLEDGE_BASE_CONNECTOR"` with the connector declared in
`connectorParameters`, **not** the plain `type: "CUSTOM"` used by self-managed
knowledge bases.

---

---

## Re-running the evaluations after an agent change

The evaluation subsystem is independent of the application: its own venv, its
own pinned dependencies, and no imports into the agents it measures. Nothing in
`evaluation/` ships to a runtime — `requirements.txt` is what the packager
vendors, and the evaluators are deliberately not in it.

One-time setup:

```bash
cd evaluation && uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements-eval.txt && cd ..
```

The loop after changing an agent's prompt, tools or graph:

```bash
# 0. free gate -- validates dataset coverage, no Bedrock calls, exits non-zero
#    on the first unmet requirement. Cheaper than discovering it mid-sweep.
evaluation/.venv/bin/python -m evaluation.selfcheck

# 1. redeploy just what changed
.venv/bin/python deploy/agentcore/deploy.py deploy --step package
.venv/bin/python deploy/agentcore/deploy.py deploy --step agents

# 2. confirm the deployment is coherent before you measure it
source deploy/agentcore/.env
.venv/bin/python deploy/agentcore/deploy.py verify

# 3. score it against the DEPLOYED runtimes, 12 threads
AWS_PROFILE=<profile> PYTHONPATH=. evaluation/.venv/bin/python \
    evaluation/runners/strands_runner.py  --mode agentcore --concurrency 12
AWS_PROFILE=<profile> PYTHONPATH=. evaluation/.venv/bin/python \
    evaluation/runners/langsmith_runner.py --mode agentcore --concurrency 12

# 4. rebuild the comparison dashboard
.venv/bin/python evaluation/report/build_dashboard.py
open evaluation/report/dashboard.html
```

Scope it while iterating — both runners take the same flags:

```bash
--target policy_advisor        # only the agent you touched
--tier sub_agent               # sub-agents directly over A2A
--tier orchestrator            # the router end-to-end
--category responsible_ai      # the 8 safety probes only
--case-id orc-ra-injection-09  # a single case, repeatable
--limit 3                      # cheap smoke run
--concurrency 12               # 12 is the sweet spot; 4 is far slower, not 3x slower
```

**Use `--mode agentcore` when you care about the performance breakdown.** Local
mode has no span export, so only wall-clock is real and the harness will tell
you which layers it could not measure. In `agentcore` mode it reads
OpenTelemetry spans from the `aws/spans` log group, groups them by `traceId`,
and attributes time per layer using a union of intervals — plus token counts
from the `gen_ai.usage.*` span attributes.

Two things to know before you trust a number:

- **P95/P99 need samples.** On a short run the harness prints an explicit
  warning that they were interpolated from too few points. Believe the P50.
- **The judge is a model.** Run the same cases twice and scores move slightly.
  Treat a few hundredths of difference between the two SDKs as noise; treat a
  metric where one runner *fails* and the other passes, or any responsible-AI
  score below 1.00, as a finding.

If `verify` and the smoke suites pass but evaluation scores drop, the wiring is
fine and the behaviour changed — which is exactly the split the two layers of
testing exist to give you.

## If something breaks

| Symptom | Likely cause |
|---|---|
| Runtime stuck `CREATE_FAILED` | Read the `failureReason` the script now prints. If it mentions the entrypoint, rebuild: `--step package`. Otherwise `/aws/bedrock-agentcore/runtimes/<name>` in CloudWatch Logs. |
| `ValidationException` on a name | Policy engines, policies and runtimes allow `[A-Za-z][A-Za-z0-9_]*` only — no hyphens, max 48 chars. Gateways and targets *do* allow hyphens. `preflight` checks all 16 names against the API's own patterns. |
| `configuration incomplete in Parameter Store` | The `agents` step asserts every key the runtimes resolve at use time is actually present. Re-run `--step parameters`, then `--step agents`. The check queries SSM directly, so a local export of the same variable will not mask a gap. |
| `RuntimeClientError ... Received error (400)` | The agent rejected the request and **AgentCore discards the response body**, so the reason never reaches the caller. Almost always a missing or unknown `actor_id`: it is required whenever memory is enabled, because a conversation has to belong to someone. Every rejection is now logged — check `/aws/bedrock-agentcore/runtimes/travel_demo_train_travel_agent-*`. |
| Agent *narrates* tool calls instead of making them (`<function_calls>` in the reply) | The tools were never bound to the model. `create_react_agent` binds them for you only when given a static model; with a **callable** model it uses whatever the callable returns, so the callable must do `model.bind_tools(tools)` itself. Fails silently: HTTP 200, no warning, and because the system prompt still lists the tools the model imitates the provider's call syntax as prose. Guarded by `test_tools_are_bound_to_the_model`. |
| Delegation fails with `403` on `/invocations/.well-known/agent-card.json` | `bedrock-agentcore:GetAgentCard` is a **separate action** from `InvokeAgentRuntime`, and an A2A client reads the card before it can send anything. Grant both to any role that delegates. Reads like an invoke-permission problem but is not. |
| `AccessDenied` invoking a sub-agent despite an exact-ARN grant | `InvokeAgentRuntime` authorizes against the endpoint-qualified ARN `runtime/<id>/runtime-endpoint/<qualifier>`, not the bare runtime ARN. Confirm with `aws iam simulate-principal-policy` against both forms. |
| `Runtime initialization time exceeded` | Usually not latency. Check `/aws/bedrock-agentcore/runtimes/<name>` for a crash in the first second — a missing bootstrap variable or an unresolvable config value looks like a cold-start timeout from the outside. |
| `Access denied while calling GetPolicyEngine` on `CreateGateway` | The gateway execution role needs `bedrock-agentcore:GetPolicyEngine`, `AuthorizeAction` and `PartiallyAuthorizeActions`. Step 7 attaches these before creating the gateway; re-run `--from-step gateway`. |
| `Insufficient permissions to call gateway` on `CreatePolicy` | This one is on **your** identity, not a role. Policy-in-AgentCore calls `InvokeGateway` during `CreatePolicy` to validate that the actions named in a Cedar statement exist on the target gateway. The deploying principal needs `bedrock-agentcore:InvokeGateway`. |
| Every tool call denied despite permit policies | The gateway role is missing `AuthorizeAction` / `PartiallyAuthorizeActions`. Same fix as the `GetPolicyEngine` row — the symptom differs only in *when* it bites. |
| Agent answers "I could not find that policy" | Documents still indexing. The deploy step waits for `INDEXED`; re-run `--step knowledge-base` to check. |
| Delegation fails, card resolved fine | Agent card advertising `0.0.0.0:9000`. Re-run `--step agents` so the second phase sets the runtime URL. |
| `403` on a gateway call | Caller lacks `bedrock-agentcore:InvokeGateway`, or Cedar denies it. `verify` distinguishes the two. |
| Every agent sees every tool | Policy engine still `LOG_ONLY`. Run `--step enforce`. |
| No traces in GenAI Observability | Three separate causes. (1) Transaction Search not enabled, or the X-Ray destination not CloudWatch Logs — `preflight` reports both. (2) No OTLP endpoint configured: the SDK default is `http://localhost:4318` and AgentCore runs **no collector**, so every batch is dropped against a refused connection while startup still reports tracing active. The code now resolves `https://xray.<region>.amazonaws.com/v1/traces` and signs with SigV4; startup prints the destination, so check that line. (3) Setting `OTEL_EXPORTER_OTLP_TRACES_HEADERS` redirects spans to your own log group, which also needs a CloudWatch Logs resource policy letting `xray.amazonaws.com` call `logs:PutLogEvents` — without it they are silently lost. We leave that unset and use the default `aws/spans`. |
