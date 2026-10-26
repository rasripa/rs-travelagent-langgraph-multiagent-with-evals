#!/usr/bin/env bash
# Start, stop and check the full local stack: 3 MCP servers + 3 A2A sub-agents
# + the orchestrator's HTTP endpoint.
#
# Every listener binds 127.0.0.1 only. These are developer-mode processes with
# no authentication of their own -- the authentication and tool-authorisation
# story lives at the AgentCore Gateway, which is not in the path locally. Do
# not bind these to 0.0.0.0 or expose them off-host; use the AgentCore
# deployment for anything reachable by someone else.
#
#   ./scripts/run_local.sh start    bring the stack up
#   ./scripts/run_local.sh status   show what is listening
#   ./scripts/run_local.sh logs     tail every log
#   ./scripts/run_local.sh stop     shut the stack down
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/.run/logs"
PID_DIR="$ROOT/.run/pids"

export PYTHONPATH="$ROOT"
export TRAVEL_RUN_MODE="${TRAVEL_RUN_MODE:-local}"
export PYTHONUNBUFFERED=1

# name:script:port
#
# ORDER IS LOAD-BEARING, and the startup phases below index into this array:
#   [0:3]  MCP servers    -- the sub-agents load their tool catalogue at
#                            startup, so these must be listening first
#   [3:4]  A2A sub-agents -- the orchestrator DISCOVERS these from their agent
#                            cards at startup (common/a2a_discovery.py), so a
#                            sub-agent that is not yet listening is simply
#                            skipped and becomes unroutable for that run
#   [7]    orchestrator   -- last, for the reason above
# A new sub-agent goes at the end of the A2A block, and the slice widths below
# move with it.
PROCS=(
  "traveller_profile:mcp_servers/traveller_profile/server.py:8101"
  "journey_search:mcp_servers/journey_search/server.py:8102"
  "ticket_operations:mcp_servers/ticket_operations/server.py:8103"
  "journey_planner:agents/journey_planner/server.py:9101"
  "journey_support:agents/journey_support/server.py:9102"
  "policy_advisor:agents/policy_advisor/server.py:9103"
  "train_travel_agent:agents/train_travel_agent/server.py:8090"
)

MCP_COUNT=3
A2A_COUNT=3

# Agents whose tool grant is summarised at the end of `start`. The router is in
# the list deliberately: its grant is EMPTY, and seeing that printed next to
# the others is the point -- Tier 1 classifies and delegates, and holds no
# gateway tools of its own.
GRANT_AGENTS=(journey_planner journey_support policy_advisor train_travel_agent)

if [[ ! -x "$PY" ]]; then
  echo "error: $PY not found. Create the environment first:" >&2
  echo "  uv venv --python 3.12 .venv && uv pip install -r requirements.txt" >&2
  exit 1
fi

port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

wait_for_port() {
  local port="$1" name="$2" tries=0
  until port_busy "$port"; do
    tries=$((tries + 1))
    if (( tries > 150 )); then
      echo "  FAILED  $name did not open port $port"
      echo "  ---- last 25 log lines ----"
      tail -n 25 "$LOG_DIR/$name.log" 2>/dev/null | sed 's/^/  /'
      return 1
    fi
    sleep 0.2
  done
  echo "  ready   $name  (127.0.0.1:$port)"
}

cmd_start() {
  mkdir -p "$LOG_DIR" "$PID_DIR"

  for entry in "${PROCS[@]}"; do
    IFS=: read -r name _ port <<<"$entry"
    if port_busy "$port"; then
      echo "error: port $port ($name) is already in use. Run './scripts/run_local.sh stop' first." >&2
      exit 1
    fi
  done

  # Build the KB index up front so the first query is not slowed by it, and so
  # a Bedrock problem surfaces here rather than mid-demo.
  echo "building knowledge base index..."
  "$PY" knowledgebase/indexer.py 2>&1 | sed 's/^/  /'

  echo
  echo "starting MCP servers..."
  for entry in "${PROCS[@]:0:$MCP_COUNT}"; do
    IFS=: read -r name script port <<<"$entry"
    nohup "$PY" "$script" >"$LOG_DIR/$name.log" 2>&1 &
    echo $! >"$PID_DIR/$name.pid"
  done
  for entry in "${PROCS[@]:0:$MCP_COUNT}"; do
    IFS=: read -r name _ port <<<"$entry"
    wait_for_port "$port" "$name" || exit 1
  done

  # Sub-agents must come up after the MCP servers: they load their tool
  # catalogue at startup.
  echo
  echo "starting A2A sub-agents..."
  for entry in "${PROCS[@]:$MCP_COUNT:$A2A_COUNT}"; do
    IFS=: read -r name script port <<<"$entry"
    nohup "$PY" "$script" >"$LOG_DIR/$name.log" 2>&1 &
    echo $! >"$PID_DIR/$name.pid"
  done
  for entry in "${PROCS[@]:$MCP_COUNT:$A2A_COUNT}"; do
    IFS=: read -r name _ port <<<"$entry"
    wait_for_port "$port" "$name" || exit 1
  done

  # The orchestrator goes last, and not just for tidiness: it fetches every
  # sub-agent's agent card at startup to build its delegation tools. A
  # sub-agent that is not listening yet is skipped with a warning, and the
  # orchestrator then cannot route to it for the whole of that run.
  echo
  echo "starting orchestrator..."
  IFS=: read -r name script port <<<"${PROCS[$((MCP_COUNT + A2A_COUNT))]}"
  nohup "$PY" "$script" >"$LOG_DIR/$name.log" 2>&1 &
  echo $! >"$PID_DIR/$name.pid"
  wait_for_port "$port" "$name" || exit 1

  echo
  echo "stack is up. tool grants in force:"
  "$PY" -c '
import sys
from common import policy_shim
for agent in sys.argv[1:]:
    grant = policy_shim.describe_grant(agent)
    print("  " + grant)
    if not policy_shim.granted_qualified_tools(agent):
        print("      ^ no gateway tools at all -- it delegates, it does not call tools")
print()
for line in policy_shim.describe_shared_asymmetry():
    print("  " + line)
' "${GRANT_AGENTS[@]}" 2>/dev/null

  echo
  echo "next:  .venv/bin/python scripts/demo.py"
  echo "       .venv/bin/python scripts/show_tool_access.py   <- the Cedar story"
  echo "logs:  ./scripts/run_local.sh logs"
}

cmd_stop() {
  local stopped=0
  for entry in "${PROCS[@]}"; do
    IFS=: read -r name _ port <<<"$entry"
    local pid_file="$PID_DIR/$name.pid"
    if [[ -f "$pid_file" ]]; then
      local pid; pid="$(cat "$pid_file")"
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null && stopped=$((stopped + 1))
      fi
      rm -f "$pid_file"
    fi
  done
  sleep 1
  # Anything still holding a port gets a second, firmer request.
  for entry in "${PROCS[@]}"; do
    IFS=: read -r name _ port <<<"$entry"
    if port_busy "$port"; then
      lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | xargs -r kill -9 2>/dev/null
    fi
  done
  echo "stopped $stopped process(es)."
}

cmd_status() {
  printf "%-22s %-6s %s\n" PROCESS PORT STATE
  for entry in "${PROCS[@]}"; do
    IFS=: read -r name _ port <<<"$entry"
    if port_busy "$port"; then
      printf "%-22s %-6s %s\n" "$name" "$port" "listening"
    else
      printf "%-22s %-6s %s\n" "$name" "$port" "down"
    fi
  done
}

cmd_logs() { tail -n 40 -f "$LOG_DIR"/*.log; }

case "${1:-start}" in
  start)  cmd_start ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  logs)   cmd_logs ;;
  restart) cmd_stop; echo; cmd_start ;;
  *) echo "usage: $0 {start|stop|restart|status|logs}" >&2; exit 2 ;;
esac
