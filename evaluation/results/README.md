# Committed evaluation runs

These result files are kept deliberately, so the scores and the judge reasoning
can be read **without running anything or spending anything**. Each file is a
complete record of one sweep: every case, the agent's actual output, each
judge's score and its written reason, and the performance breakdown.

`report/build_dashboard.py` reads only the **latest file per runner**, so the
older ones here are history rather than inputs.

## The runs

| File | Runner | Target | Assertions | Pass | Mean | Resp. AI | `journey_planner` | Wall clock |
|---|---|---|---|---|---|---|---|---|
| `langsmith-20261025T192715Z.json` | LangSmith | local | 152/167 | 91.0% | 0.91 | 0.897 | 0.794 | 230s |
| `langsmith-20261025T202243Z.json` | LangSmith | local | 138/166 | 83.1% | 0.86 | 0.897 | 0.735 | 214s |
| `langsmith-20261025T213353Z.json` | LangSmith | **agentcore** | 128/167 | 76.6% | 0.82 | 0.897 | 0.471 | 219s |
| `langsmith-20261025T220032Z.json` | LangSmith | **agentcore** | 134/167 | 80.2% | 0.83 | 0.897 | 0.471 | 161s |
| `strands-20261025T194433Z.json` | Strands | local | 136/167 | 81.4% | 0.81 | 0.897 | 0.647 | 882s |
| `strands-20261025T203923Z.json` | Strands | local | 140/167 | 83.8% | 0.83 | 0.931 | 0.706 | 850s |
| `strands-20261025T214954Z.json` | Strands | **agentcore** | 131/167 | 78.4% | 0.78 | 0.828 | 0.529 | 960s |
| `strands-20261025T221615Z.json` | Strands | **agentcore** | 129/167 | 77.2% | 0.77 | 0.897 | 0.559 | 703s |

**Quote the `agentcore` rows.** They are the only ones measured against the
deployed runtimes, and the only ones carrying a per-component span breakdown —
a `local` run reports wall-clock and names the layers it could not measure.

The two screenshots in `docs/images/` show the agentcore pairs:
`eval-screen.png` shows the later pair (80% / 77%), `eval-dashboard.png` the
earlier one (77% / 78%).

## What these files are good for

- **Judge reasoning.** Each case carries the judge's written justification, not
  just a number. This is where you confirm a failure is real — see
  `jp-accessible-options-04`, failed by Strands for returning one service when
  `fixtures/services.json` contains exactly one on that route.
- **Judge disagreement.** The same output scored by two independently
  implemented suites. On `jp-fare-railcard-01` and `jp-accessible-options-04`,
  LangSmith scores 1.0 and Strands 0.0.
- **Run-to-run variance.** LangSmith moved 76.6% → 80.2% on *identical* code
  and the same target. A single sweep is not a baseline; quote a range.

## Two findings visible in this table

**Deployed scores are lower than local across the board** — 76.6–80.2% against
AgentCore where the local stack gave 81.4–91.0%. Report which target was
scored; a local figure flatters the system.

**`journey_planner` degrades specifically when deployed.** Mean pass rate falls
**0.213** between local and AgentCore (0.65–0.79 down to 0.47–0.56). No other
agent behaves like this — `policy_advisor` loses 0.056, the router 0.060, and
`journey_support` actually *improves* by 0.051. The per-case causes are
diagnosed in `AGENT.md` §7.1, which also records why those failures are **left
in place deliberately**. The local/deployed gap itself is not yet explained and
is worth investigating before anyone concludes the agent is simply weak.

**Responsible AI is the stable signal** — exactly 0.897 in six of these eight
runs, across both judges and both targets (the outliers are 0.828 and 0.931,
both Strands). If an RAI score moves, suspect a real behaviour change before
suspecting the judge.
