#!/usr/bin/env bash
# Bring up the full local stack for evaluation, including the Rail Policy
# Advisor agent (agents/policy_advisor, port 9103).
#
# Why this wrapper exists
# -----------------------
# The evaluation dataset drives all four agents, so all four have to be
# listening -- and critically, the policy agent has to be listening BEFORE the
# orchestrator starts, because the router discovers its sub-agents from their
# agent cards at startup and never re-discovers. A policy agent started
# afterwards leaves the router with no delegation tool for policy, which the
# router logs as "UNREACHABLE -- not routable this run".
#
# Some revisions of scripts/run_local.sh start the policy agent themselves and
# some do not, so this script DETECTS which it is dealing with instead of
# assuming: if run_local.sh already manages port 9103, it simply delegates;
# otherwise it starts that one process first and then delegates. Either way
# scripts/run_local.sh is used unmodified -- changing it would be a change to
# the core application, which is out of scope for this subsystem.
#
# Every listener binds 127.0.0.1 only, matching run_local.sh. These are
# developer-mode processes with no authentication of their own; do not bind
# them to 0.0.0.0 or expose them off-host.
#
#   ./evaluation/scripts/eval_stack.sh start    the full stack, all 4 agents
#   ./evaluation/scripts/eval_stack.sh status   what is listening
#   ./evaluation/scripts/eval_stack.sh stop     shut everything down
#   ./evaluation/scripts/eval_stack.sh verify   probe all 4 agent endpoints
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/.run/logs"
PID_DIR="$ROOT/.run/pids"
RUN_LOCAL="$ROOT/scripts/run_local.sh"
POLICY_NAME="policy_advisor"
POLICY_SCRIPT="agents/policy_advisor/server.py"
POLICY_PORT="${PORT_POLICY_ADVISOR:-9103}"

export PYTHONPATH="$ROOT"
export TRAVEL_RUN_MODE="${TRAVEL_RUN_MODE:-local}"
export PYTHONUNBUFFERED=1

if [[ ! -x "$PY" ]]; then
  echo "error: $PY not found. Create the project environment first:" >&2
  echo "  uv venv --python 3.12 .venv && uv pip install -r requirements.txt" >&2
  exit 1
fi

port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

# True when scripts/run_local.sh already manages the policy agent's port.
run_local_owns_policy() {
  grep -qE "^[[:space:]]*\"${POLICY_NAME}:.*:${POLICY_PORT}\"" "$RUN_LOCAL" 2>/dev/null
}

wait_for_port() {
  local port="$1" name="$2" tries=0
  until port_busy "$port"; do
    tries=$((tries + 1))
    if (( tries > 300 )); then
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

  if run_local_owns_policy; then
    echo "scripts/run_local.sh manages $POLICY_NAME on $POLICY_PORT; delegating to it."
  elif port_busy "$POLICY_PORT"; then
    echo "note: port $POLICY_PORT is already in use; assuming the policy agent is up."
  else
    # The policy agent loads the knowledge-base index at startup. Build it
    # first so a Bedrock problem surfaces here rather than mid-run.
    echo "scripts/run_local.sh does NOT start $POLICY_NAME; starting it here first."
    echo "building knowledge base index..."
    "$PY" knowledgebase/indexer.py 2>&1 | sed 's/^/  /'
    echo
    echo "starting Rail Policy Advisor ($POLICY_SCRIPT)..."
    nohup "$PY" "$POLICY_SCRIPT" >"$LOG_DIR/$POLICY_NAME.log" 2>&1 &
    echo $! >"$PID_DIR/$POLICY_NAME.pid"
    wait_for_port "$POLICY_PORT" "$POLICY_NAME" || exit 1
  fi

  echo
  echo "handing off to ./scripts/run_local.sh start ..."
  echo
  "$RUN_LOCAL" start || exit 1
  echo
  cmd_verify
}

cmd_stop() {
  "$RUN_LOCAL" stop
  local pid_file="$PID_DIR/$POLICY_NAME.pid"
  if [[ -f "$pid_file" ]]; then
    local pid; pid="$(cat "$pid_file")"
    kill "$pid" 2>/dev/null
    rm -f "$pid_file"
  fi
  sleep 1
  if port_busy "$POLICY_PORT"; then
    lsof -nP -tiTCP:"$POLICY_PORT" -sTCP:LISTEN 2>/dev/null | xargs kill -9 2>/dev/null
    echo "policy agent stopped."
  fi
}

cmd_status() {
  printf "%-24s %-6s %s\n" PROCESS PORT STATE
  for entry in "traveller_profile:8101" "journey_search:8102" "ticket_operations:8103" \
               "journey_planner:9101" "journey_support:9102" "$POLICY_NAME:$POLICY_PORT" \
               "train_travel_agent:8090"; do
    IFS=: read -r name port <<<"$entry"
    if port_busy "$port"; then
      printf "%-24s %-6s %s\n" "$name" "$port" "listening"
    else
      printf "%-24s %-6s %s\n" "$name" "$port" "down"
    fi
  done
}

# Probe the endpoints the harness actually uses, so a broken agent card or a
# half-started agent is caught before a judged run spends money.
cmd_verify() {
  echo "probing the endpoints the harness uses..."
  "$PY" - <<'PYCODE'
import json, sys
import urllib.error
import urllib.request

ENDPOINTS = [
    ("train_travel_agent", "http://127.0.0.1:8090/ping"),
    ("journey_planner", "http://127.0.0.1:9101/.well-known/agent-card.json"),
    ("journey_support", "http://127.0.0.1:9102/.well-known/agent-card.json"),
    ("policy_advisor", "http://127.0.0.1:9103/.well-known/agent-card.json"),
]

bad = 0
for name, url in ENDPOINTS:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
            body = json.loads(response.read())
        label = body.get("name") or body.get("status") or "ok"
        print(f"  up    {name:<22} {label}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        bad += 1
        print(f"  DOWN  {name:<22} {type(exc).__name__}: {exc}")

if bad:
    print(f"\n{bad} endpoint(s) not reachable. Evaluation cases against them will fail.")
    sys.exit(1)
print("\nall four agents reachable.")
PYCODE
}

case "${1:-start}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  status)  cmd_status ;;
  verify)  cmd_verify ;;
  restart) cmd_stop; echo; cmd_start ;;
  *) echo "usage: $0 {start|stop|restart|status|verify}" >&2; exit 2 ;;
esac
