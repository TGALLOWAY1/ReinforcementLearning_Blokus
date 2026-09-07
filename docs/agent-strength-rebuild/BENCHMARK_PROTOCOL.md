# Benchmark Protocol

**Protocol version:** `protocol_v3` (2026-09-07) for all M1+ work; `rescue_v2` (2026-07-12) is retained below for reading the July experiments. Any change to pool, seeds, budgets, seat policy,
or scoring mode requires a version bump recorded here and referenced in every experiment entry.
Raw match outcomes and matchup matrices are the primary evidence; every rating is a summary.

**Version history:**
- `protocol_v3` (2026-09-07) — implemented in `training/evaluation/protocol_v3.py`; CLI
  `python -m mcts_lab.calibrate {clone,discrimination}`. Differences from `rescue_v2`, each
  motivated by a finding of `DIAGNOSTIC_ASSESSMENT_2026-09-06.md`: **fresh run seed per run**
  (derived from label + launch time, recorded in `report.json`; pass `--seed` to replay) instead
  of two fixed seeds replayed every run (B5); **iteration-pinned single-worker search at the
  serving budget** (250 iterations for the champion) instead of a 100 ms override (B4);
  **all 24 seat permutations** (game index mod 24; a cyclic round-robin keeps who-moves-after-whom fixed and was measured to bias a champion-vs-clone contrast — see EXP-014a) so seats AND successor relations balance over any multiple of 24 games; **paired per-game score difference with a sign-flip permutation
  test and pre-registered n** as the primary statistic; fixed external anchors (deterministic
  `greedy`, `random`, Pentobi levels — see `PENTOBI.md`) instead of a pool that copied the
  champion (M8); games played in a process pool (results independent of worker count).
  Harness gates before any strength claim: **clone calibration** (equivalence test: the 90% CI
  of the champion-vs-clone paired difference within ±4 points, p > 0.05, no errored games,
  n ≥ 240 = 10 cycles of the permutations; with a paired-difference sd of 15-18 points this
  passes an unbiased harness ≥ 93% of the time and passes a 4-point bias ≤ 5%) and
  **discrimination** (pre-registered known-different pairs separate at p < 0.01, n ≥ 120;
  power ≈ 0.99 for a 10-point gap, ≈ 0.6 for 5 points). Floors are hard-coded; a smaller
  `--games` is an exploratory run and the CLI says so. Games that error or truncate are
  excluded from statistics and any exclusion fails a gate (`excluded_games` in report.json).
- `rescue_v2` (2026-07-12) — scoring mode is **`SCORING_MODE_STANDARD`** (coverage + 15
  all-pieces + 5 monomino-last), now the engine/arena default (D-002 implemented). All
  pre-`rescue_v2` results — including every rating in `training/state/ratings.sqlite` and the
  gen140 champion numbers — are **house-scored and not directly comparable**; standard-scored
  baselines are re-anchored by the first `rescue_v2` evaluation runs.
- `rescue_v1` (2026-07-12) — initial codification; house scoring (historical engine default).

## Fixed conditions (rescue_v2)

| Condition | Value | Source |
|---|---|---|
| Opponent pool | `benchmark_v2`: heuristic, random, `baseline_mcts_fast` (100 ms-equiv), `baseline_mcts_strong` (500 ms-equiv), `best_historical` checkpoint | `training/evaluation/benchmark_pool.py` |
| Seeds | `20260620, 20260621` (nightly convention); pool defaults `20260101, 20260202` where the code requires them | `mcts_lab/eval.py`, `benchmark_pool.py` |
| Seat policy | `round_robin` (required — cancels first-move advantage deterministically; `randomized` allowed only for exploratory runs, never gates) | `analytics/tournament/arena_runner.py` |
| Move budget | **Iteration-deterministic** (`deterministic_time_budget` + `iterations_per_ms`), never wall-clock, for anything feeding a gate | `mcts/search_profiles.py`, repo convention |
| Scoring mode | `SCORING_MODE_STANDARD` (engine, arena `RunConfig`, and TD self-play default; house is explicit opt-in via `RunConfig.scoring_mode`) | D-002, implemented 2026-07-12 |
| Minimum games | screen ≥ 20; confirmation ≥ 60; SPRT cap 160 paired games/candidate | `promotion_gate.py`, `sequential.py` |
| Statistics | Wilson 95% CI on win rate; TrueSkill (PlackettLuce) μ/σ, conservative = μ−3σ; Wald SPRT α=β=0.05; paired permutation test | existing implementations |
| Hardware note | record runner class (GitHub-hosted 2-core vs local) in every experiment entry | — |

## Required metrics per evaluation (master prompt §5)

First-place rate; average finishing position; final normalized score; score margin; placement
distribution; per-seat performance; per-opponent-composition performance; head-to-head /
mixed-table matrix; CI or equivalent uncertainty; runtime per move; simulations per move;
(once a model exists) inference time; search throughput; memory; TrueSkill summary (+ Elo as a
legacy secondary).

Most of these are already emitted by `analytics/tournament/arena_runner.py` summaries and
`mcts_lab/eval.py`; gaps (placement distribution, per-seat breakdown surfaced per run) are
Phase 3/4 deliverables of the node-stats/benchmark tooling.

## Canonical commands

```bash
# Sanity gate before any benchmark
python -m mcts_lab.checks

# Standard evaluation (pooled, seeded)
python -m mcts_lab.eval --agents champion,baseline --games 10 --seeds 20260620,20260621

# Candidate promotion (two-stage, never bypass)
python -m mcts_lab.promote --candidate training/artifacts/candidates/<artifact>.json

# Search-quality diagnostic (Phase 4 starting point)
python -m training.diagnostics.search_quality   # see training/diagnostics/
```

## Rules

1. **No gate decision from a single rating number.** Promotion and phase gates cite game counts,
   CIs/SPRT verdicts, and the matchup matrix.
2. **Never change benchmark conditions mid-experiment.** A changed condition = new protocol
   version = results not comparable.
3. **Benchmark agents are immutable.** Pool members are pinned configs/checkpoints; replacing
   one is a protocol version bump.
4. **Seat balance is mandatory** for gates (round_robin), and per-seat results must be reported.
5. **Determinism:** fixed seeds, iteration budgets, single-thread search for gate games
   (root-parallel allowed for data generation, not for gate evaluation unless separately
   validated).
6. **Reproducibility block** (per `EXPERIMENT_LOG.md` template) is required for every run used
   as evidence: commit, seeds, configs, hardware, game counts, artifacts.
7. **Era hygiene:** results from before the maxⁿ fix (`DEBUGGED_BACKPROP_EPOCH_RUN_ID =
   20260701T204805Z`) are never mixed into current comparisons; post-Phase-2 (standard scoring)
   results are likewise never mixed with house-scored ones.
