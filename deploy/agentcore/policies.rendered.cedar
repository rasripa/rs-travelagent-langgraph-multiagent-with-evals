// Cedar policies for the Train Travel Agent gateway (IAM authorization).
//
// This is a TEMPLATE. deploy/agentcore/deploy.py substitutes the placeholders
// once the gateway and the per-agent execution roles exist, and writes the
// result to deploy/agentcore/policies.rendered.cedar.
//
//   arn:aws:bedrock-agentcore:us-west-2:588792541783:gateway/travel-demo-gateway-ehcezfksij              the gateway these policies govern
//   arn:aws:sts::588792541783:assumed-role/travel-demo-journey-planner-role        arn:aws:sts::<account>:assumed-role/<planner-role>
//   arn:aws:sts::588792541783:assumed-role/travel-demo-journey-support-role        arn:aws:sts::<account>:assumed-role/<support-role>
//   arn:aws:sts::588792541783:assumed-role/travel-demo-policy-advisor-role arn:aws:sts::<account>:assumed-role/<policy-advisor-role>
//
// Model, per the AgentCore Policy documentation:
//   * Default deny. Only an explicit `permit` lets an action through.
//   * The gateway uses AWS_IAM inbound auth, so the principal is an
//     AgentCore::IamEntity. Its entity id is the caller's ARN with the SESSION
//     NAME REMOVED, giving arn:aws:sts::<account>:assumed-role/<role-name>.
//     That form is stable across invocations, so `principal ==` matches.
//   * Action is the tool name, gateway-prefixed as TargetName___toolName.
//   * Resource is the gateway itself.
//   * Do NOT list `initialize` or `tools/list` as actions. The policy engine
//     validates every action against the gateway's real tools, parsing each
//     name as Target___tool. A bare `initialize` is read as a TARGET name and
//     the whole policy is rejected:
//         Target 'initialize' does not exist in gateway '<id>'.
//         Available targets: JourneySearch, RailPolicyKB, ...
//     The policy lands in CREATE_FAILED, which means NO policy is active, which
//     under default-deny means every agent sees zero tools. The MCP handshake
//     and listing are permitted implicitly; only tool actions are policed.
//
// WHY EACH AGENT HAS ITS OWN EXECUTION ROLE: with IAM auth the role IS the
// identity Cedar evaluates. A shared execution role would make the agents
// indistinguishable and every grant below unenforceable. The role split is
// load-bearing, not tidiness.
//
// =========================================================================
// THE DEMO POINT: look at TravellerProfile in the two policies below.
//
// It is the SHARED target. Both sub-agents are permitted to reach it, through
// the same gateway endpoint, with the same MCP handshake. And yet:
//
//     Journey Planner  ->  2 tools   get_traveller_profile, list_saved_journeys
//     Journey Support   ->  1 tool    get_railcard_and_loyalty
//
// Cedar filters tools/list, so this is not "the call fails if you try it" --
// the denied tools are ABSENT from the catalogue. Journey Support's model never
// learns that get_traveller_profile exists. Run
// `deploy.py verify` to see the two catalogues printed side by side.
//
// The split has a reason, which is what makes it defensible rather than
// arbitrary: Support needs entitlement (what discount was paid, what loyalty
// tier applies), not identity. get_railcard_and_loyalty exists precisely so
// that servicing a booking does not require reading a traveller's contact
// details or journey history.
// =========================================================================

// ---------------------------------------------------------------------------
// Journey Planner: the pre-travel agent. 5 tools.
// JourneySearch in full (exclusive) + 2 of 3 TravellerProfile tools (shared).
// Denied TicketOperations entirely -- a planning agent must not be able to
// cancel a ticket or pay out compensation.
// Denied TravellerProfile___get_railcard_and_loyalty: redundant for it, since
// get_traveller_profile already returns the railcard.
// ---------------------------------------------------------------------------
permit(
    principal == AgentCore::IamEntity::"arn:aws:sts::588792541783:assumed-role/travel-demo-journey-planner-role",
    action in [
        AgentCore::Action::"JourneySearch___search_journeys",
        AgentCore::Action::"JourneySearch___get_fares",
        AgentCore::Action::"JourneySearch___book_ticket",
        AgentCore::Action::"TravellerProfile___get_traveller_profile",
        AgentCore::Action::"TravellerProfile___list_saved_journeys"
    ],
    resource == AgentCore::Gateway::"arn:aws:bedrock-agentcore:us-west-2:588792541783:gateway/travel-demo-gateway-ehcezfksij"
);

// ---------------------------------------------------------------------------
// Journey Support: the post-booking agent. 4 tools.
// TicketOperations in full (exclusive) + exactly 1 of 3 TravellerProfile tools.
// Denied JourneySearch entirely -- and that denial is load-bearing, not
// decorative. Support cannot find an alternative service during disruption, and
// it cannot ask a peer either: sub-agents never call each other. It reports
// that a search is needed and the ROUTER obtains it from the Journey Planner,
// so the capability boundary is visible in the call graph.
// ---------------------------------------------------------------------------
permit(
    principal == AgentCore::IamEntity::"arn:aws:sts::588792541783:assumed-role/travel-demo-journey-support-role",
    action in [
        AgentCore::Action::"TicketOperations___get_booking",
        AgentCore::Action::"TicketOperations___change_or_cancel_ticket",
        AgentCore::Action::"TicketOperations___claim_delay_repay",
        AgentCore::Action::"TravellerProfile___get_railcard_and_loyalty"
    ],
    resource == AgentCore::Gateway::"arn:aws:bedrock-agentcore:us-west-2:588792541783:gateway/travel-demo-gateway-ehcezfksij"
);

// ---------------------------------------------------------------------------
// Policy Advisor: 1 tool.
// The Bedrock Managed Knowledge Base connector, for questions about the rules.
//
// Granted Retrieve (a single hybrid search: fast, cheap, deterministic) and
// DENIED AgenticRetrieveStream, the multi-step planning variant the same
// connector also exposes. Two tools exist on the target; this agent sees one.
// A second illustration of per-tool control, this time to cap cost and keep a
// scripted demo deterministic rather than to protect data.
//
// Note what it is NOT granted: no TravellerProfile, no JourneySearch, no
// TicketOperations. It holds the rules and nothing else -- it cannot look up a
// traveller or act on a booking, so a policy question cannot become a data
// disclosure.
//
// THE ROUTER APPEARS NOWHERE IN THIS FILE, deliberately. Train Travel Agent
// holds no gateway tools at all: it classifies intent and delegates. Cedar is
// default-deny, so the absence of a permit statement IS its entitlement, and
// deploy.py deletes any policy left over from when it owned the knowledge base.
// ---------------------------------------------------------------------------
permit(
    principal == AgentCore::IamEntity::"arn:aws:sts::588792541783:assumed-role/travel-demo-policy-advisor-role",
    action in [
        AgentCore::Action::"RailPolicyKB___Retrieve"
    ],
    resource == AgentCore::Gateway::"arn:aws:bedrock-agentcore:us-west-2:588792541783:gateway/travel-demo-gateway-ehcezfksij"
);

// ---------------------------------------------------------------------------
// OPTIONAL extra guardrail, commented out by default.
//
// `forbid` always wins over `permit` in Cedar, so this is a hard ceiling that
// no future permit can widen.
//
// Before enabling it, understand how it interacts with tools/list. Listing is a
// META-action: the gateway treats a tool as listable if ANY circumstance would
// permit calling it. A parameter-conditional forbid like this one therefore
// leaves change_or_cancel_ticket VISIBLE in the catalogue, and the denial lands
// at call time instead. If you need the tool hidden outright, deny it
// unconditionally -- which is exactly what the three policies above do, and why
// the tools/list demo works at all.
//
// Note also that change_or_cancel_ticket takes no amount parameter in this
// demo, so this condition would never match as written -- it is here to show
// the shape of a context-conditional forbid, not as a working rule.
//
// IAM principals do NOT support tags, unlike OAuth principals. For broader
// matching use principal.id with `like`, for example:
//     principal.id like "arn:aws:sts::123456789012:assumed-role/*"
// ---------------------------------------------------------------------------
// forbid(
//     principal,
//     action == AgentCore::Action::"TicketOperations___change_or_cancel_ticket",
//     resource == AgentCore::Gateway::"arn:aws:bedrock-agentcore:us-west-2:588792541783:gateway/travel-demo-gateway-ehcezfksij"
// ) when {
//     context has input &&
//     context.input has refund_amount_gbp &&
//     context.input.refund_amount_gbp > 100
// };
