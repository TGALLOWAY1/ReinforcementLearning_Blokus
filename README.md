# MCTS Laboratory — 4-player Blokus AI

A lab for building, training, and evaluating strong Monte-Carlo-Tree-Search
agents for 4-player Blokus, with a web demo you can play against.

**Read this first:** [AUDIT_REPORT.md](AUDIT_REPORT.md) — the July 2026 audit
that simplified this repo, fixed a structural MCTS bug (opponents in the tree
were modeled as *cooperating* with the root player), and replaced the training
workflow. Everything below reflects the post-audit state.

## What this repo does

- **Game engine** (`engine/`) — complete Blokus rules: legal move generation,
  corner-contact adjacency, scoring with all-pieces/monomino bonuses, pass and
  endgame handling. Numpy grid + bitboard acceleration.
- **Agents** (`agents/`, `mcts/`) — `random`, `heuristic` (no search), and the
  canonical `MCTSAgent`: UCT with **maxⁿ per-player backups** (each player in
  the tree optimizes its own reward), heuristic move ordering, configurable
  rollout policies, and optional experimental layers (RAVE, progressive
  widening, opponent modeling — off by default).
- **Arena** (`analytics/tournament/`) — seeded, seat-randomized 4-player
  tournaments with TrueSkill/Elo tracking and Wilson confidence intervals.
- **Training pipeline** (`training/`, driven by `mcts_lab` CLIs) — self-play
  data generation, evaluation-weight fitting (regression + TD), a fixed
  benchmark pool, and a two-stage statistically-gated champion promotion.
  The GitHub Actions workflow (`.github/workflows/nightly-mcts-training.yml`)
  is **frozen** (manual dispatch only) until the measurement gates in
  `docs/agent-strength-rebuild/DIAGNOSTIC_ASSESSMENT_2026-09-06.md` pass.
- **Web demo** (`frontend/` + `webapi/`) — React UI playing against the
  champion in-browser via Pyodide (`scripts/build_browser_core.sh` bundles the
  Python core), plus a FastAPI backend for research/analysis views and for
  natively served agents (Pentobi, the human-protocol opponent — see below).

## Quick start

```bash
pip install -r requirements.txt

# 1. Sanity check the lab (~1 min)
python -m mcts_lab.checks

# 2. Run a benchmark arena (leaderboard with confidence intervals)
python -m mcts_lab.eval --agents champion,baseline --games 10 --seeds 20260620,20260621

# 3. Generate self-play data (snapshot corpus for weight fitting)
python -m mcts_lab.self_play --games 24

# 4. Train: produce candidate agent configs from the data
python -m mcts_lab.train

# 5. Promote: screen + confirm a candidate against the champion, gated
python -m mcts_lab.promote --candidate training/artifacts/candidates/<artifact>.json
```

`--thinking-ms 50` on `eval`/`promote` gives fast smoke runs; drop the flag
for full-strength (500 ms/move) evaluation.

TD trajectories and move-policy targets are collected separately, from the
**registry champion at a teacher budget** (stronger search than the champion's
own — labels should come from a stronger teacher; see
`mcts/search_profiles.py` for the fast/balanced/strong/teacher profiles):

```bash
python -m training.td_selfplay --num-games 20        # → data/td_trajectories.csv
python -m training.policy_selfplay --num-games 20    # → data/policy_targets.csv
python -m training.diagnostics.search_quality        # depth/stability across budgets
```

### Run the web demo

```bash
bash scripts/build_browser_core.sh   # bundle engine+mcts+agents for Pyodide
python run_server.py                 # FastAPI backend on :8000
cd frontend && npm install && npm run dev
```

### Play against Pentobi (the human protocol, milestones M4/M5)

The strongest opponent the lab can serve is Pentobi's engine (GPL-3), run
natively by the backend with a 10 s cap per AI move
(`docs/agent-strength-rebuild/DECISIONS.md` D-024). Build it once per machine
(`docs/agent-strength-rebuild/PENTOBI.md`), start the backend and frontend as
above, open the Play page and use the **Play Pentobi** card: pick the level and
your colour (rotate seats between games). Every game is logged to
`data/human_games/` (`GAME_LOG_DIR` overrides). Summarise the record against
the pre-registered gate (Pentobi takes first place in ≥ 70% of 20 seat-rotated
games, a tie for first counting against it, and no agent loss attributable to
a fallback, timeout or error) with:

```bash
python -m mcts_lab.human_games
```

Serving parity and latency evidence: `python scripts/m4_serving_gate.py`
(report in `training/reports/m4_serving_gate/`).

## The improvement loop

```
current champion ──(mcts_lab.self_play)──> snapshot corpus
        │                                        │
        │                              (mcts_lab.train)
        │                                        ▼
        │                              candidate artifact
        │                                        │
        └───(mcts_lab.promote: screen ≥20 games → confirm ≥60 games,
             fixed seeds, fixed benchmark pool, Wilson-CI + TrueSkill gates)
                                                 │
                                    pass ──> new champion
                                    (training/state/champion.json,
                                     old champion checkpointed as a
                                     future benchmark opponent)
```

- **Champion registry:** `training/state/champion.json` is the single source
  of truth (version, params, rating, full lineage). Promoted champions are
  checkpointed under `training/state/checkpoints/` and join the evaluation
  pool, so strength ratchets and can't silently cycle.
- **Statistical guardrails:** promotion requires beating the champion
  head-to-head, positive Elo *and* TrueSkill deltas, rank #1 vs the fixed pool
  (heuristic, random, two fixed MCTS anchors), ≥2 fixed seeds, and holding all
  of that over a 60+ game confirmation run. Elo movement without a promotion
  is measurement noise, and the reports label it as such.
- **Variance-reduced sequential screen** (`--sequential-eval`, the nightly
  default): instead of a fixed 20-game screen swamped by ±72 Elo re-rating noise,
  a seat-balanced **SPRT** on the *paired* champion-vs-candidate outcome stops as
  soon as the result is conclusive — a real edge is detectable in far fewer games,
  a no-op candidate is rejected fast, and a candidate promotes only if the SPRT
  accepts *and* it clears the conservative gate on those games
  (`training/evaluation/sequential.py`; see AUDIT_REPORT.md §7).
- **Nightly automation (frozen since 2026-07-12):** the GitHub Actions workflow
  can be dispatched manually to resume from committed state, run the same loop
  (`training.nightly_run`), commit results, and email a summary. Reports land in
  `training/status.md` and `training/reports/`. It is not scheduled: see the
  2026-09 assessment for why its measurements could not detect improvement.

## Reproducing benchmark results

All evaluation is seed-deterministic. To reproduce a leaderboard:

```bash
python -m mcts_lab.eval --agents champion,baseline --games 20 \
    --seeds 20260620,20260621 --out results.json
```

Same seeds + same agent configs ⇒ identical games. Arena outputs (per-game
JSONL + pooled summary) are written under `training/state/selfplay_runs/` and
historical gauntlet runs live in `arena_runs/`.

## Current best agent status

> **2026-09-06 assessment** (`docs/agent-strength-rebuild/DIAGNOSTIC_ASSESSMENT_2026-09-06.md`):
> until that date the engine's piece set was not the Blokus set (duplicate S/Z
> tetromino, no Z-pentomino). It is now the standard 21-piece set, pinned by
> `tests/test_piece_set_standard.py` and an independent rules-based cross-check
> (`tests/test_reference_movegen.py`). **All datasets, weight files, candidate
> artifacts, ratings and experiment results produced before this fix are
> invalid for the standard game.** The bullets below are historical.

- The long-standing `gen0` champion (random rollouts, depth-5 cutoff, RAVE on
  a broken reward scheme) was diagnosed as *weaker than the no-search
  heuristic agent* — see AUDIT_REPORT.md §1. The corrected strong baseline —
  **greedy-sample rollouts (sample 12 moves/step, pick best by fast
  heuristic), rollout cutoff 12, heuristic move ordering, plain UCT with maxⁿ
  backups** — passed the promotion *screen* decisively at the same search
  budget: **16–4 head-to-head vs the champion, ΔElo +228, Δμ +23.4** over 20
  games / 2 fixed seeds (AUDIT_REPORT.md §3.3). The 60-game confirmation (and
  the actual registry update) completes via the nightly workflow or
  `python -m mcts_lab.promote --candidate training/artifacts/candidates/baseline_mcts_20260701T202756Z.json`.
- Absolute Elo numbers before this fix (~1200 plateau) are measurement noise
  around an unchanging agent and are not comparable to post-fix ratings.
- Next strength milestones, in order: retune evaluation weights on fresh
  self-play from the fixed search; retest RAVE/minimax/progressive-widening on
  top of maxⁿ; learn a rollout/move-ordering policy from MCTS visit counts.
- **Update (2026-07-06):** after gen140 the loop re-plateaued because the
  candidate roster had collapsed onto the champion (a byte-identical `baseline`
  clone; a `policy` that self-distilled back into the fixed heuristic) and the
  20-game screen was too noisy to detect a real gain. Fixed by self-retiring the
  clone, leading the roster with genuinely-different search candidates
  (`mcts_sweep`, `progressive_widening`), and adding the sequential SPRT screen
  above (AUDIT_REPORT.md §7). Highest-leverage remaining work: a richer move
  policy and an off-Actions, parallelised daily driver.
- **Update (2026-07-07, AUDIT_REPORT.md §8):** a search-signal audit found the
  search was effectively **depth-1** for most of the game (branching factor >
  iteration budget), UCB exploration was ~100× under-weighted against the raw
  reward scale, the evaluator's hard clamp returned **exactly 0.0 for most
  positions** (no signal at all), rollouts spent ~92% of wall-clock enumerating
  moves nobody used, and both self-play collectors were generating training
  data with the **gen0 cold-start config** instead of the promoted champion.
  Fixed: tanh eval squash (order-preserving, signal restored), sampled rollout
  movegen (~6× faster iterations), deterministic-only transposition caching,
  learned-leaf rewards scaled onto the shared magnitude, teacher-profile data
  collection from the registry champion, and a `rich_leaf` candidate that
  finally deploys the 45-feature TD value model at leaves. A full reward
  normalisation was tried and **reverted after measurement** (72.9%→29.2%
  champion win rate at its pinned exploration constant — see §8.2).
  `training/diagnostics/search_quality.py` measures search depth/label
  stability across budgets.

## What files matter

| Path | What it is |
|---|---|
| `engine/` | Blokus rules engine (board, move generator, pieces, scoring) |
| `mcts/mcts_agent.py` | The canonical MCTS agent (maxⁿ UCT + options) |
| `mcts/state_evaluator.py` | Fast state evaluation (phase-dependent weights) |
| `agents/` | Random/heuristic agents + champion loading |
| `analytics/tournament/arena_runner.py` | Arena harness (seeded, seat-randomized) |
| `analytics/tournament/gauntlet.py` | Pooled stats + conservative promotion decision |
| `mcts_lab/` | **The canonical CLI workflow** (checks/eval/self_play/train/promote) |
| `training/` | Durable pipeline internals + state (`training/state/`) |
| `training/state/champion.json` | Champion registry (version, params, lineage) |
| `.github/workflows/nightly-mcts-training.yml` | Nightly training automation |
| `webapi/`, `frontend/`, `browser_python/worker_bridge.py` | Web demo |
| `scripts/` | Arena CLI, champion gauntlet, Pyodide bundle build |
| `tests/` | Engine/agent/pipeline tests (`python -m pytest tests/ -q`) |

Docs index: [docs/README.md](docs/README.md). Historical layer-by-layer
experiment narratives predate the maxⁿ fix — their strength conclusions are
obsolete (see AUDIT_REPORT.md §6).
