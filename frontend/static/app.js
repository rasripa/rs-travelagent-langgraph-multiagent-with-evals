/* Train Travel Agent -- demo chat client.
 *
 * Talks only to this page's own origin (127.0.0.1). The server signs and
 * forwards to the agent; no AWS credentials ever reach the browser.
 */
"use strict";

const chat = document.getElementById("chat");
const form = document.getElementById("composer");
const input = document.getElementById("prompt");
const send = document.getElementById("send");
const modeBadge = document.getElementById("mode-badge");
const regionBadge = document.getElementById("region-badge");
const sessionLabel = document.getElementById("session-label");
const suggestions = document.getElementById("suggestions");
const identity = document.getElementById("identity");
const actorSelect = document.getElementById("actor");
const sessionSelect = document.getElementById("session");
const newSessionBtn = document.getElementById("new-session");
const memoryNote = document.getElementById("memory-note");

let sessionId = null;
let busy = false;
let memoryEnabled = false;

/* The five demo beats, as one-click prompts. */
const PROMPTS = [
  { label: "Refund rules", hint: "routed to the Policy Advisor",
    text: "I booked an Advance ticket. Can I get a refund if I change my mind?" },
  { label: "Plan a journey", hint: "delegates to the Planner",
    text: "I'm traveller TRV-1001. I need to get from London to Manchester on 2026-11-06. What are my options and what will it cost?" },
  { label: "My train is cancelled", hint: "two specialists, fanned out by the router",
    text: "I'm traveller TRV-1001 and my booking is BKG-77301. I've just been told my train is cancelled. What are my rights, and can you get me to Manchester this morning?" },
  { label: "Cancel a cheap ticket", hint: "expect an honest refusal",
    text: "I'm traveller TRV-1002. Please cancel booking BKG-77303 and refund me. I've changed my plans." },
  { label: "I was delayed", hint: "Delay Repay, 14.99 GBP",
    text: "I'm traveller TRV-1001. My journey on booking BKG-77304 arrived 47 minutes late last week. Am I owed anything?" },
];

/* ------------------------------------------------------------------ render */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* Render the agent's plain text into paragraphs. Deliberately NOT innerHTML:
 * the text originates from a model and must never be parsed as markup. */
function renderBody(bubble, text) {
  const blocks = String(text).split(/\n{2,}/);
  for (const block of blocks) {
    const trimmed = block.trim();
    if (trimmed) bubble.appendChild(el("p", null, trimmed));
  }
  if (!bubble.childElementCount) bubble.appendChild(el("p", null, String(text)));
}

function addMessage(role, who, text, meta) {
  const msg = el("div", `msg ${role}`);
  msg.appendChild(el("div", "who", who));
  const bubble = el("div", "bubble");
  renderBody(bubble, text);
  msg.appendChild(bubble);
  if (meta) msg.appendChild(el("div", "meta", meta));
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
  return msg;
}

function addThinking() {
  const msg = el("div", "msg agent");
  msg.appendChild(el("div", "who", "Train Travel Agent"));
  const bubble = el("div", "bubble");
  const dots = el("span", "typing");
  dots.appendChild(el("i"));
  dots.appendChild(el("i"));
  dots.appendChild(el("i"));
  bubble.appendChild(dots);
  msg.appendChild(bubble);
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
  return msg;
}

function setBusy(state) {
  busy = state;
  send.disabled = state;
  input.disabled = state;
  for (const chip of suggestions.querySelectorAll("button")) chip.disabled = state;
  if (!state) input.focus();
}

/* ------------------------------------------------------------------ chat */

async function ask(prompt) {
  if (busy || !prompt.trim()) return;
  setBusy(true);
  addMessage("user", "You", prompt);
  const thinking = addThinking();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        session_id: sessionId,
        actor_id: actorSelect.value || null,
      }),
    });
    const data = await response.json().catch(() => ({}));
    thinking.remove();

    if (!response.ok) {
      addMessage("agent error", "Could not reach the agent",
        data.error || `HTTP ${response.status}`);
      return;
    }

    const isNewSession = !sessionId;
    sessionId = data.session_id || sessionId;
    if (sessionId) {
      sessionLabel.textContent = `session ${sessionId.slice(0, 12)}\u2026`;
    }

    /* Show what memory contributed, so the demo can point at it. */
    const bits = [];
    if (data.elapsed_seconds !== undefined) bits.push(`${data.elapsed_seconds}s`);
    if (data.turns_recalled) bits.push(`${data.turns_recalled} prior turns recalled`);
    if (data.long_term_recalled) bits.push("long-term memory applied");
    addMessage("agent", "Train Travel Agent", data.response || "(no reply)",
      bits.join("  \u00b7  ") || null);

    /* A brand-new session won't be in the dropdown yet. */
    if (isNewSession && memoryEnabled) loadSessions(actorSelect.value, sessionId);
  } catch (err) {
    thinking.remove();
    addMessage("agent error", "Request failed", String(err));
  } finally {
    setBusy(false);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value;
  input.value = "";
  ask(text);
});

/* ------------------------------------------------------------------ memory */

function resetConversation(note) {
  sessionId = null;
  chat.replaceChildren();
  sessionLabel.textContent = "new session";
  addMessage("agent", "Train Travel Agent", note);
}

async function loadSessions(actorId, selectId) {
  sessionSelect.replaceChildren();
  const fresh = el("option", null, "New conversation");
  fresh.value = "";
  sessionSelect.appendChild(fresh);

  if (!actorId || !memoryEnabled) {
    sessionSelect.disabled = true;
    return;
  }

  try {
    const r = await fetch(`/api/sessions?actor_id=${encodeURIComponent(actorId)}`);
    const data = await r.json();
    for (const s of data.sessions || []) {
      const when = s.created_at ? new Date(s.created_at) : null;
      const label = when && !isNaN(when)
        ? `Resume \u00b7 ${when.toLocaleString()}`
        : `Resume \u00b7 ${s.session_id.slice(0, 12)}\u2026`;
      const opt = el("option", null, label);
      opt.value = s.session_id;
      sessionSelect.appendChild(opt);
    }
    sessionSelect.disabled = false;
    if (selectId) sessionSelect.value = selectId;
  } catch {
    sessionSelect.disabled = true;
  }
}

async function onActorChange() {
  const actorId = actorSelect.value;
  await loadSessions(actorId);
  const name = actorSelect.selectedOptions[0]?.textContent || actorId;
  resetConversation(
    `Hello. You are now talking to me as ${name}. I can plan rail travel, ` +
    `look after an existing booking, and explain the fare and compensation ` +
    `rules. If we have spoken before, I may remember your preferences.`
  );
}

async function onSessionChange() {
  const chosen = sessionSelect.value;
  if (!chosen) {
    resetConversation("Starting a fresh conversation.");
    return;
  }
  sessionId = chosen;
  chat.replaceChildren();
  sessionLabel.textContent = `session ${chosen.slice(0, 12)}\u2026`;
  addMessage("agent", "Train Travel Agent",
    "Resuming our earlier conversation. Ask me anything \u2014 I have the " +
    "previous turns of this session in front of me.");
}

/* ------------------------------------------------------------------ setup */

for (const prompt of PROMPTS) {
  const chip = el("button", "chip");
  chip.type = "button";
  chip.appendChild(document.createTextNode(prompt.label + " "));
  chip.appendChild(el("span", null, `\u00b7 ${prompt.hint}`));
  chip.addEventListener("click", () => ask(prompt.text));
  suggestions.appendChild(chip);
}

actorSelect.addEventListener("change", onActorChange);
sessionSelect.addEventListener("change", onSessionChange);
newSessionBtn.addEventListener("click", () => {
  sessionSelect.value = "";
  resetConversation("Starting a fresh conversation.");
});

fetch("/api/actors")
  .then((r) => r.json())
  .then((data) => {
    memoryEnabled = Boolean(data.memory_enabled);
    const list = data.actors || [];
    if (!list.length) {
      identity.hidden = true;
      return;
    }
    identity.hidden = false;
    for (const a of list) {
      const opt = el("option", null, `${a.name} (${a.actor_id})`);
      opt.value = a.actor_id;
      actorSelect.appendChild(opt);
    }
    if (memoryEnabled) {
      memoryNote.textContent = "memory on \u00b7 short-term + long-term";
      memoryNote.className = "memory-note on";
      loadSessions(actorSelect.value);
    } else {
      memoryNote.textContent = "memory off \u00b7 each turn is independent";
      memoryNote.className = "memory-note off";
      sessionSelect.disabled = true;
    }
  })
  .catch(() => { identity.hidden = true; });

fetch("/api/config")
  .then((r) => r.json())
  .then((cfg) => {
    const live = cfg.mode === "agentcore";
    modeBadge.textContent = live
      ? "AgentCore Runtime" : "local stack";
    modeBadge.classList.add(live ? "live" : "localmode");
    modeBadge.title = cfg.target || "";
    regionBadge.textContent = cfg.region || "";
  })
  .catch(() => {
    modeBadge.textContent = "server unreachable";
    modeBadge.classList.add("localmode");
  });

input.focus();

/* ===================================================================
   Evaluation tab
   ===================================================================
   Starts an evaluation suite, polls its progress, and rebuilds the
   comparison dashboard. The server decides WHAT runs -- this page only
   names a runner and a scope from fixed lists it was given, so there is
   nothing here that can turn into an arbitrary command.
   =================================================================== */
(function () {
  const tabChat = document.getElementById("tab-chat");
  const tabEval = document.getElementById("tab-eval");
  const panelChat = document.getElementById("panel-chat");
  const panelEval = document.getElementById("panel-eval");
  if (!tabEval || !panelEval) return;

  const scopeSel = document.getElementById("eval-scope");
  const modeSel = document.getElementById("eval-mode");
  const concSel = document.getElementById("eval-concurrency");
  const btnLang = document.getElementById("run-langsmith");
  const btnStrands = document.getElementById("run-strands");
  const btnReport = document.getElementById("build-report");
  const linkDash = document.getElementById("open-dashboard");
  const note = document.getElementById("eval-note");
  const statusBox = document.getElementById("eval-status");
  const spinner = document.getElementById("eval-spinner");
  const jobLabel = document.getElementById("eval-job-label");
  const elapsed = document.getElementById("eval-elapsed");
  const logBox = document.getElementById("eval-log");
  const cards = document.getElementById("eval-cards");

  const MODE_LABELS = {
    local: "Local stack (no span breakdown)",
    agentcore: "AgentCore runtimes (full breakdown)",
  };

  const SCOPE_LABELS = {
    all: "Everything (31 cases)",
    quick: "Quick smoke (4 cases)",
    responsible_ai: "Responsible AI only",
    orchestrator: "Router, end to end",
    sub_agents: "Sub-agents, direct over A2A",
    journey_planner: "Journey Planner only",
    journey_support: "Journey Support only",
    policy_advisor: "Policy Advisor only",
  };

  let poll = null;
  let scopesLoaded = false;

  function showTab(which) {
    const evalOn = which === "eval";
    panelEval.hidden = !evalOn;
    panelChat.hidden = evalOn;
    tabEval.classList.toggle("active", evalOn);
    tabChat.classList.toggle("active", !evalOn);
    tabEval.setAttribute("aria-selected", String(evalOn));
    tabChat.setAttribute("aria-selected", String(!evalOn));
    if (evalOn) refresh();
  }
  tabChat.addEventListener("click", () => showTab("chat"));
  tabEval.addEventListener("click", () => showTab("eval"));

  function setNote(message, kind) {
    if (!message) {
      note.hidden = true;
      note.textContent = "";
      return;
    }
    note.hidden = false;
    note.textContent = message;
    note.className = "eval-note " + (kind || "err");
  }

  function fmtPct(value) {
    return value === null || value === undefined
      ? "&ndash;"
      : (value * 100).toFixed(0) + "%";
  }
  function fmtNum(value, digits) {
    return value === null || value === undefined
      ? "&ndash;"
      : Number(value).toFixed(digits === undefined ? 2 : digits);
  }
  function band(value) {
    if (value === null || value === undefined) return "";
    if (value >= 0.9) return "good";
    if (value >= 0.7) return "warn";
    return "bad";
  }

  function renderCards(results) {
    const runners = Object.keys(results || {}).sort();
    if (!runners.length) {
      cards.innerHTML =
        '<p class="eval-intro" style="margin:0;">No results yet. Run a suite ' +
        "to populate this.</p>";
      return;
    }
    cards.innerHTML = runners
      .map((key) => {
        const r = results[key];
        const rai = r.responsible_ai || {};
        return (
          '<div class="eval-card ' +
          (key === "strands" ? "strands" : "") +
          '">' +
          "<h3>" +
          r.label +
          "</h3>" +
          '<div class="meta">' +
          r.cases +
          " cases &middot; mode " +
          (r.mode || "?") +
          " &middot; " +
          r.file +
          "</div>" +
          '<div class="eval-metrics">' +
          metric("Pass rate", fmtPct(r.pass_rate), band(r.pass_rate)) +
          metric("Mean score", fmtNum(r.mean_score), band(r.mean_score)) +
          metric(
            "Resp. AI",
            fmtNum(rai.mean_score),
            band(rai.mean_score)
          ) +
          metric("p50", fmtNum(r.p50, 1) + "s", "") +
          metric("Errors", r.errors === undefined ? "&ndash;" : r.errors,
                 r.errors ? "bad" : "good") +
          "</div></div>"
        );
      })
      .join("");
  }
  function metric(k, v, cls) {
    return (
      '<div class="eval-metric"><div class="k">' +
      k +
      '</div><div class="v ' +
      (cls || "") +
      '">' +
      v +
      "</div></div>"
    );
  }

  function applyState(state) {
    if (!scopesLoaded && Array.isArray(state.scopes)) {
      const preferred = ["all", "quick", "responsible_ai", "orchestrator",
                         "sub_agents", "journey_planner", "journey_support",
                         "policy_advisor"];
      const ordered = preferred.filter((s) => state.scopes.includes(s))
        .concat(state.scopes.filter((s) => !preferred.includes(s)));
      scopeSel.innerHTML = ordered
        .map((s) => '<option value="' + s + '">' + (SCOPE_LABELS[s] || s) + "</option>")
        .join("");
      scopesLoaded = true;
    }

    if (!modeSel.options.length && Array.isArray(state.modes)) {
      modeSel.innerHTML = state.modes
        .map(
          (m) =>
            '<option value="' +
            m +
            '"' +
            (m === state.default_mode ? " selected" : "") +
            ">" +
            (MODE_LABELS[m] || m) +
            "</option>"
        )
        .join("");
    }

    const running = !!state.running;
    btnLang.disabled = running;
    btnStrands.disabled = running;
    btnReport.disabled = running;

    if (!state.eval_env_ready) {
      setNote(
        "The evaluation environment is not set up yet. Create it once:\n" +
          "  cd evaluation && uv venv --python 3.12 .venv\n" +
          "  VIRTUAL_ENV=.venv uv pip install -r requirements-eval.txt",
        "err"
      );
      btnLang.disabled = true;
      btnStrands.disabled = true;
    }

    linkDash.hidden = !state.dashboard_exists;

    if (state.job) {
      statusBox.hidden = false;
      jobLabel.textContent = state.job.label;
      elapsed.textContent =
        running
          ? "running \u00b7 " + state.job.elapsed_seconds + "s"
          : state.job.ok
          ? "finished in " + state.job.elapsed_seconds + "s"
          : "FAILED (exit " + state.job.returncode + ")";
      spinner.className =
        "spinner" + (running ? "" : state.job.ok ? " done" : " failed");
      logBox.textContent = (state.job.log || []).join("\n");
      logBox.scrollTop = logBox.scrollHeight;
    }

    renderCards(state.results);
  }

  async function refresh() {
    try {
      const response = await fetch("/api/eval/status");
      const state = await response.json();
      applyState(state);
      if (state.running && !poll) {
        poll = setInterval(refresh, 2500);
      } else if (!state.running && poll) {
        clearInterval(poll);
        poll = null;
      }
    } catch (err) {
      setNote("Could not reach the evaluation API: " + err, "err");
    }
  }

  async function run(runner) {
    setNote("");
    try {
      const response = await fetch("/api/eval/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          runner: runner,
          scope: scopeSel.value || "all",
          mode: modeSel.value || undefined,
          concurrency: Number(concSel.value || 12),
        }),
      });
      const body = await response.json();
      if (!response.ok) {
        setNote(body.error || "Could not start the run.", "err");
        return;
      }
      applyState(body);
      if (!poll) poll = setInterval(refresh, 2500);
    } catch (err) {
      setNote("Could not start the run: " + err, "err");
    }
  }

  btnLang.addEventListener("click", () => run("langsmith"));
  btnStrands.addEventListener("click", () => run("strands"));

  btnReport.addEventListener("click", async () => {
    setNote("");
    btnReport.disabled = true;
    try {
      const response = await fetch("/api/eval/report", { method: "POST" });
      const body = await response.json();
      if (!response.ok) {
        setNote(body.error || "Building the report failed.", "err");
        return;
      }
      applyState(body);
      setNote(
        "Dashboard rebuilt.\n" + (body.output || []).join("\n"),
        "ok"
      );
      linkDash.hidden = false;
    } catch (err) {
      setNote("Building the report failed: " + err, "err");
    } finally {
      btnReport.disabled = false;
    }
  });

  refresh();
})();
