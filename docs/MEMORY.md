# Memory

The orchestrator uses AgentCore Memory for both layers. One `Memory` resource
carries them; the sub-agents have none.

| Layer | Mechanism | Lives |
|---|---|---|
| Short-term | Events against `(actorId, sessionId)` | `eventExpiryDuration` days, default 30 |
| Long-term — summarization | `summaryMemoryStrategy` | `travellers/{actorId}/summaries/{sessionId}` |
| Long-term — personalization | `userPreferenceMemoryStrategy` | `travellers/{actorId}/preferences` |

Short-term needs **no strategy** — events are stored as written. The strategies
are what extract long-term records from those events, in the background.

## Request shape

```json
POST /invocations
{ "prompt": "...", "actor_id": "TRV-1001", "session_id": "..." }
```

`session_id` is taken from AgentCore's own
`X-Amzn-Bedrock-AgentCore-Runtime-Session-Id` header when present, so a runtime
session and a memory session are the same thing. The body is the fallback.

Two read-only side endpoints support the UI:

```
GET /actors                      -> travellers this deployment knows
GET /sessions?actor_id=TRV-1001  -> 3 most recent sessions, newest first
```

## Per turn

1. `resolve_actor(body["actor_id"])` — validate
2. `load_turns(actor, session)` — `ListEvents` → LangChain messages
3. `memory_context_block(actor, prompt)` — `RetrieveMemoryRecords` on both namespaces
4. `run_agent(..., history=..., extra_context=...)`
5. `save_turn(...)` — `CreateEvent`, which also triggers extraction

The response reports `turns_recalled` and `long_term_recalled` so the UI can
show what memory actually contributed.

---

## actorId: what this build does, and what production needs

### Today

`actor_id` arrives as a **request parameter**, constrained two ways:

1. **It must be a traveller in the fixtures.** `resolve_actor()` rejects
   anything else, so the parameter cannot be used to probe or create
   namespaces.
2. **Reaching the orchestrator requires SigV4** with
   `bedrock-agentcore:InvokeAgentRuntime` on that runtime. The caller is an
   authenticated AWS principal, not the public.

So the trust boundary is: *an authenticated principal may act as any of the
three synthetic travellers.*

### What that does not give you

**Separation between travellers.** Any principal that can invoke the
orchestrator can pass any valid `actor_id` and read that traveller's
preferences and conversation summaries. With three fictional travellers that is
a non-issue. With real customer data it is a privacy incident.

Say this out loud if asked. The gap is in the identity model, not the memory
wiring — the namespaces, retrieval and fencing are all production-shaped.

### Production: derive actorId from a verified token

The change is contained, because everything downstream already treats
`resolve_actor()`'s return value as authoritative.

**1. Runtime authorizer.** Switch the orchestrator runtime from default IAM to
`CUSTOM_JWT`, and allowlist the `Authorization` header so the container can
read the token:

```python
agentcore.create_agent_runtime(
    ...,
    authorizerConfiguration={
        "customJWTAuthorizer": {
            "discoveryUrl": "https://<issuer>/.well-known/openid-configuration",
            "allowedClients": ["<app-client-id>"],
        }
    },
    requestHeaderConfiguration={"requestHeaderAllowlist": ["Authorization"]},
)
```

**2. Replace `resolve_actor()`.** Read the verified claim; ignore the body:

```python
def resolve_actor(request) -> str:
    claims = verified_claims(request.headers["Authorization"])  # verify signature!
    actor = claims.get("custom:traveller_id") or claims["sub"]
    if actor not in known_actors():
        raise UnknownActor(...)
    return actor
```

The signature **must** be verified against the issuer's JWKS. AgentCore's
authorizer validates the token at the edge, but if your code reads claims it
should not re-trust an unverified decode — `jwt_claim()` in `deploy.py` is
explicitly labelled unsafe-for-authz for this reason.

**3. Drop `actor_id` from the request schema entirely.** Leaving it accepted
but ignored is worse than removing it: it reads as supported.

**4. Keep the namespaces.** `travellers/{actorId}/...` is unchanged — only the
provenance of `actorId` changes.

### What this costs

The orchestrator moves from IAM to JWT inbound, so the "IAM end to end" story
no longer holds for the front door. The gateway and MCP hops stay SigV4. The
middle option is AgentCore's OBO token exchange, which carries the user's
identity alongside the agent's while keeping SigV4 on the invocation.

---

## Extraction is asynchronous

`CreateEvent` triggers a background job. **Long-term records do not exist on the
turn that produced them.** A fresh deployment has empty memory, so a
"it remembers me" demo shows nothing.

```bash
source deploy/agentcore/.env
python scripts/seed_memory.py          # write two prior conversations
python scripts/seed_memory.py --check  # has extraction produced records?
```

`--check` lists the preference records per traveller. Run it until non-empty
before presenting. `seed_memory.py` writes events directly, so it costs no
model tokens.

`extractionMode: "SKIP"` on `CreateEvent` suppresses extraction for turns that
do not warrant it — useful for cost control on chatty sessions.

## Retrieved memory is untrusted input

Memory content is model-generated text from earlier sessions, injected into a
prompt. That is a prompt-injection surface, and the AgentCore documentation
flags it explicitly.

Two defences, both in place:

- `memory_context_block()` wraps records in a fenced block that **states its own
  trust level**, rather than relying on the system prompt to remember.
- The orchestrator's prompt has a `Remembered context` section: memory is data
  never instruction, never a source of fact about prices, times or
  entitlements, may be stale, and must not be recited back.

It is attached to the **user turn**, not merged into the system prompt, so the
model sees a clear boundary between standing instructions and recalled
material.

## No memory for sub-agents

Deliberate. They are stateless workers receiving self-contained instructions.
Memory would give one conversation several writers and invite them to act on
stale context instead of the orchestrator's brief. Only the orchestrator role
carries the memory data-plane permissions.

## No LangGraph checkpointer

A **checkpointer** is LangGraph's own conversation-state store: it snapshots the
graph's message list per `thread_id` so a later invocation resumes
mid-conversation, and it is what enables `interrupt()`/resume for
human-in-the-loop.

We do not use one, because AgentCore Memory already owns conversation state.
Running both gives two stores that drift, and only one survives a session
boundary. `load_turns()` → `history` is the same capability sourced from the
store that also feeds extraction.

A checkpointer would earn its place if you needed to pause mid-graph awaiting
human approval — Memory records turns, not graph position.
