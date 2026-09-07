# MCTS Laboratory — Audit Report

Date: 2026-07-01
Scope: full-repo audit, dead-code cleanup, training-plateau diagnosis, and a
canonical training/evaluation pipeline.

> Status: this report is written up-front as the cleanup plan and updated as
> each stage lands. Sections marked ✅ are implemented in this branch.

---

## 1. Executive summary

The repo accumulated ~10 experimental "layers" of MCTS features, three
generations of training loops (RL/PPO era, layer-sweep era, nightly-approaches
era), a generated-but-committed browser copy of the whole Python core, ~40MB of
stale run artifacts and story images, and 186 docs files. Meanwhile the actual
agent has been stuck at ~1200 Elo for 137 nightly generations.

The plateau is **not** noise in one place — it has three compounding causes,
in order of importance:

1. **A structural MCTS bug: opponents in the tree are modeled as cooperating
   with the root player.** All rewards are backpropagated from the root
   player's perspective (`mcts/mcts_agent.py`), and *every* node — including
   opponent nodes — selects children by **maximizing** that root-perspective
   value (`select_child`, `max(children, key=ucb)`). The search therefore
   explores fantasy lines in which opponents deliberately help us. Deeper
   search adds almost no strength, which is why every layer experiment
   measured ≈ noise.
2. **The champion is a crippled configuration and has never been replaced.**
   `training/state/champion.json` is still the `gen0` cold-start seeded from
   `config/key_findings_best_params.json`: *random* rollouts, rollout cutoff
   depth 5, `iterations_per_ms: 0.5` (1/20th of the calibrated 10.0), and
   RAVE k=1000 (whose statistics are corrupted by bug #1). Its TrueSkill μ≈22
   vs ≈44 for the plain no-search heuristic agent: the champion is *weaker
   than its own rollout heuristic*. Zero promotions in 137 generations.
3. **The promotion gate is statistically incompatible with the evaluation
   budget.** The gate requires ≥40 games, 2 seeds and 6 conservative criteria,
   but the nightly budget split across 4 candidate approaches yields 9–15
   games per candidate. No candidate can ever pass; Elo history
   (1015 → 1224 swings with an unchanged champion) is pure measurement noise
   from 25–50 game samples.

Everything else (engine rules, scoring, pass/turn handling, seat rotation,
state persistence) checked out correct.

---

## 2. What was removed ✅

| Area | Removed | Why |
|---|---|---|
| `archive/` (19MB) | entire directory | Stale run artifacts + archived FastMCTS agent (already rejected by the arena at runtime; git history preserves it) |
| `browser_python/` | entire directory (regenerated at build time) | Generated copy of `engine/ mcts/ agents/` made by `scripts/build_browser_core.sh`; committing it duplicated the whole core. Now gitignored and rebuilt on demand. |
| `arena_runs/` | all but the two champion-gauntlet runs | Historical experiment outputs; the webapi `/api/arena-runs` listing still works on what remains |
| `docs/story_images/` (20MB), `docs/showcase_game/`, `docs/_archived-2026-05/`, `docs/superpowers/`, stale top-level docs (TD_*, roadmap, audits, duplicated `architecture/`, `frontend/`, `engine/`, …) | | Stale/duplicated docs; the numbered `docs/0X-*` tree plus a small set of current docs remains |
| `league/` | entire module | Only referenced by an exploratory benchmark; the live rating stack is `analytics/tournament/` + `training/ratings_db.py` |
| `analytics/baseline/`, `analytics/heatmap/` | modules + their broken tests | Orphaned analysis code; `analytics/tests/test_metrics_*.py` imported functions that no longer exist |
| `benchmarks/` | entire directory | RL-era and league-era perf experiments with committed results |
| ~25 dead scripts | `parameter_sweep.py`, `arena_tuning.py`, `ab_rich_leaf.py`, `generate_shap_importance.py`, `generate_frontier_video.py`, `generate_mcts_heatmap_video.py`, `generate_detailed_analysis.py`, `benchmark_env.py`, `benchmark_vecenv.py`, `load_test_engine_service.py`, layer runners, etc. | One-off experiments; none referenced by CI, webapi, frontend, or the canonical workflow |
| ~28 layer arena configs | `arena_config_layer3..10_*.json`, overnight/one-off configs | Layer-sweep era; the experiments are concluded (verdict: noise) and the canonical configs remain |
| `config/agents/*.yaml` | PPO-era configs | RL era is gone |
| Root clutter | `FEATURES.md` (44KB), `TODO.md`, `tasks/`, `KEY_FINDINGS.md`, `CODE_QUALITY_AUDIT_NOTES.md`, `arena_visuals/`, `prompts/` | Superseded by this report + rewritten README; KEY_FINDINGS' "best config" was the source of the crippled champion |
| Obsolete tests | `test_frontier_video_renderer.py`, `test_legal_move_count_plot.py`, `analytics/tests/*`, layer-specific tests for removed features | Validated deleted/obsolete behavior |

**Kept intact:** `engine/` (canonical rules engine), `mcts/` (bug-fixed), `agents/`,
`analytics/tournament|logging|metrics|aggregate|winprob`, `training/` (pruned),
`webapi/` + `api-runtime/` + `vercel.json` (web demo backend), `frontend/`,
`schemas/`, `scripts/{arena,champion_gauntlet,champion_loop,champion_arena,build_browser_core}`,
core configs and `data/` registry files.

### Final repo structure

```
engine/            Blokus rules engine (board, movegen, pieces, scoring)
mcts/              Canonical MCTSAgent (maxⁿ UCT) + evaluators
agents/            random / heuristic / champion-loading agents
analytics/         tournament harness (arena, gauntlet, Elo/TrueSkill),
                   game logging, per-move metrics, winprob features
mcts_lab/          ★ canonical CLI workflow: checks / eval / self_play / train / promote
training/          durable pipeline internals + state/ (champion.json, ratings,
                   history) + approaches/ + evaluation/ (gates, benchmark pool)
scripts/           arena CLI, champion gauntlet/loop, build_browser_core.sh
webapi/ api-runtime/  FastAPI backend (research profile / Vercel deploy profile)
frontend/          React web demo (Pyodide in-browser play)
browser_python/    worker_bridge.py only (rest generated by build script)
config/ data/      champion param seeds, champion registry, calibrated weights
tests/             engine / MCTS / pipeline / webapi tests (~650 passing)
docs/              slim numbered docs tree (banner: pre-audit conclusions invalid)
.github/workflows/ nightly-mcts-training.yml (every 6h, resumes from state)
```

---

## 3. Bugs and risks found

### 3.1 Opponent-cooperation bug in MCTS selection (critical) ✅ fixed
- `mcts/mcts_agent.py:1673` — rollout reward computed only from the root
  player's perspective.
- `mcts/mcts_agent.py:2064` — the same scalar is added to every node up the
  tree (`node.update(reward)`).
- `mcts/mcts_agent.py:323` — `select_child` always takes
  `max(children, key=ucb)`, i.e. opponent nodes pick the move that is best
  *for the root player*.

**Fix (maxⁿ):** rollouts/leaf evaluations now return a per-player reward
vector; each node accumulates the reward of the player who made the move into
it, so UCB selection at any node maximizes the acting player's own value.
Implicit-minimax backup simplifies to `max` (each node's children Q are
already from the acting player's perspective).

### 3.2 RAVE / progressive-history reward pollution (high) ✅ fixed
RAVE tables and the progressive-history table were updated with the
root-perspective reward at *all* nodes, so opponent move statistics were
credited with the root's outcomes. History is now keyed per player and RAVE
updates apply the acting player's reward.

### 3.3 Crippled, never-replaced champion (critical) ✅ diagnosed; replacement screened, confirmation pending
`gen0` champion = random rollouts + depth-5 cutoff + 0.5 iter/ms + RAVE
(see §1.2). The corrected strong baseline (greedy-sample rollouts, cutoff 12,
move ordering, RAVE/minimax off, on the fixed maxⁿ search) was run through the
real promotion **screen** (fixed seeds 20260620/20260621, uniform 50 ms
budget, fixed benchmark pool) and passed every gate, twice (the seeded replay
reproduced it exactly):

```
[screen] games=20 seeds=2 win%=47.5 CI=[0.28,0.68] vs champion 16-4
         ΔElo +228.0  TrueSkill Δμ +23.35
[screen] PROMOTE baseline: beats champion (16-4), Δμ +23.35, ΔElo +228.0
```

A 47.5% win rate in 4-player games (chance = 25%) and a 16–4 head-to-head at
the *same* search budget as the incumbent. The 60-game confirmation stage was
intentionally stopped early (session time); per the gate rules the champion
registry is untouched until confirmation passes. The nightly workflow (now
two-stage, 2 candidates) completes the gated promotion automatically, or run:

```
python -m mcts_lab.promote \
    --candidate training/artifacts/candidates/baseline_mcts_20260701T202756Z.json \
    --thinking-ms 50   # ~2.5h total
```

### 3.4 Promotion gate vs budget mismatch (high) ✅ fixed
Gate demanded ≥40 games/candidate while the nightly budget delivers 9–15.
The pipeline now evaluates **one candidate at a time** with a two-stage gate:
a cheap screen (≥20 games) and a confirmation run (≥60 games, Wilson 95% CI
lower bound above the champion) only for the screen winner. The nightly
workflow runs 2 approaches instead of 4.

### 3.5 Elo bookkeeping inflates 4-player results (medium) — documented
`analytics/tournament/elo.py` treats each 4-player game as 6 independent
pairwise updates at full K. Ratings are comparable *within* a run but the
absolute numbers are inflated; the canonical pipeline reports head-to-head
win-rate with confidence intervals as the primary promotion metric and
TrueSkill (which handles multiplayer natively) as secondary. Not changed —
changing K would invalidate the existing ratings DB; noted as future work.

### 3.6 Full-heuristic rollouts are ~10x too expensive (high) ✅ fixed
The `"heuristic"` rollout policy scores **every legal move at every rollout
step** — measured at ~2s per rollout from the opening position (vs ~190ms for
random). This is why the layer experiments retreated to random rollouts +
shallow cutoffs, and why the nightly "baseline rescue" candidate (full
heuristic rollouts) could only afford 9–15 evaluation games. Added a
`greedy_sample` rollout policy (sample K=12 legal moves, play the best by the
fast move heuristic): ~260ms full rollout, ~17ms with cutoff 12 — near-
heuristic quality at close to random-rollout cost.

### 3.7 Evaluation wall-clock sinks in the benchmark pool (high) ✅ fixed
Two more places quietly burned the evaluation budget (found while running the
first real promotion gauntlet):
- The fixed MCTS benchmark anchors hardcoded `rollout_policy="heuristic"` with
  no cutoff — the exact ~2s-per-rollout pathology of §3.6 — so anchor games
  alone could consume the whole budget (measured ~11–30+ min/game). Anchors
  now use `greedy_sample` + cutoff 12 (~1s/move).
- Uniform thinking-time overrides kept the champion's `num_workers: 2` root
  parallelization, which forks worker processes on **every move**; at
  tens-of-iterations budgets the spawn overhead dominates while the split
  trees are weaker. Overridden evaluations now force single-process search
  (faster *and* stronger for the incumbent, so never biased pro-candidate).

### 3.8 Non-bugs verified
- Engine rules (first-move corner, corner-adjacency, no edge-contact),
  scoring (+15 all-pieces bonus, +5 monomino-last), pass/turn order and
  termination are correct (`engine/board.py`).
- TD-learned weights *do* flow into evaluated candidates
  (`training/selfplay_core.py:build_td_candidate`); the weights simply can't
  beat a broken search.
- Arena seat rotation and seed control are sound
  (`analytics/tournament/arena_runner.py`).

---

## 4. Why training was stuck at ~1200 Elo

Putting §3 together, the loop was: *a search that can't exploit depth*
(bug 3.1/3.2) *wrapped in a champion config chosen for the wrong reasons*
(3.3 — "KEY_FINDINGS best" was selected from noise-level layer experiments,
which we now know were noise **because** of 3.1) *behind a gate that no
candidate could pass* (3.4) *measured with samples too small to detect
anything* (3.5, 25–50 games ⇒ ±80–100 Elo noise). The TD/regression
"learning" was real but was fitting evaluation weights for a search whose
tree statistics were corrupted — optimizing a flawed objective. Elo movement
across 137 generations was measurement drift of one unchanged agent.

---

## 5. What replaced it ✅

Canonical pipeline (see README for usage):

```
python -m mcts_lab.checks       # sanity: rules, determinism, agent loading
python -m mcts_lab.eval         # seeded benchmark arena, CIs, JSON output
python -m mcts_lab.self_play    # self-play data generation (TD trajectories)
python -m mcts_lab.train        # fit evaluation weights from self-play data
python -m mcts_lab.promote      # candidate vs champion gauntlet + gated promotion
```

- **Champion registry:** `training/state/champion.json` (+ lineage) remains
  the single source of truth; `mcts_lab.promote` is the only writer.
- **Two-stage promotion gate:** screen (≥20 games) → confirm (≥60 games,
  Wilson 95% lower bound of candidate-vs-champion win share > 0.5·expected,
  ≥2 seeds, TrueSkill μ delta > 0). Guards against promoting noise.
- **Fixed benchmark pool:** random, heuristic, fast-MCTS anchor, strong-MCTS
  anchor — used for absolute tracking across generations.
- **Regression positions:** `tests/test_tactical_positions.py` pins known
  tactical answers so engine/search regressions fail fast.

---

## 6. Recommended next steps

1. Let the nightly workflow run 1–2 weeks with the fixed search; expect the
   first genuine promotions. Watch `training/reports/status.md`.
2. Increase nightly game budget (or move evaluation to a beefier runner);
   the gate is honest now, so more games = faster promotion cycles.
3. Re-run the old layer experiments (RAVE, minimax backup, cutoff) on top of
   the fixed maxⁿ search before re-enabling any of them — all prior verdicts
   are invalid.
4. Once eval-weight learning saturates, the highest-leverage next feature is
   a stronger rollout/move-ordering policy learned from MCTS visit counts
   (supervised, cheap), then progressive widening at high branching factors.
   **Implemented (2026-07):** a learned move policy (`mcts/move_policy.py`)
   distilled from the MCTS root visit distribution
   (`training/policy_selfplay.py` → `training/policy_learning.py`) and used as a
   PUCT prior + rollout/ordering policy (`policy_prior_enabled`), surfaced as the
   `policy` candidate approach. This turns the pipeline into an expert-iteration
   loop: search generates the policy target, the policy sharpens the next search.
   Off by default; default policy == the fixed move heuristic.
5. Consider migrating Elo to placement-based updates (§3.5) with a one-time
   ratings reset.

---

## 7. Follow-up (2026-07-06): the *second* plateau — candidate collapse + measurement noise ✅

The maxⁿ fix worked: `baseline_mcts` was promoted to **gen140** (+94.7 Elo, 48–12
head-to-head). But Elo then went flat again for ~16 generations (gen140→156, a
16-generation promotion drought). This is a *different* failure from §4, with two
compounding causes:

### 7.1 The candidate generators collapsed onto the champion (root cause)
Two of the three nightly candidates became **no-ops by construction**:

- **`baseline` is now a byte-for-byte clone of the champion.** Every value in
  `STRONG_SEARCH_OVERRIDES` (greedy_sample, cutoff 12, RAVE off, minimax 0.0,
  workers 1, iters/ms 0.5) already *is* the gen140 champion — because gen140 was
  promoted *from* this approach. Building the candidate reproduces the champion
  exactly (verified: zero differing params), so it can never strictly beat the
  champion; its ~47–50% is a coin flip.
- **`policy` self-distills back into the fixed heuristic.** The learned move policy
  is a 4-weight log-linear model over the *same* four move features, **seeded at the
  fixed heuristic** and trained on the *champion's own* visit counts. The trained
  artifact (`training/state/policy_weights.json`) is `[1.01, 1.99, 0.44, 0.79]` vs the
  heuristic default `[1, 2, 0.5, 1]` with piece biases ≤ 0.14 — i.e. behaviourally the
  champion. A 4-parameter student, seeded at and regularised toward its teacher and
  evaluated at the same budget, cannot exceed the search that generated it.
- `heuristic_tune` (evaluator re-fit) is the only candidate with real degrees of
  freedom, and it kept landing *below* the champion (~45%).

Net: the loop was comparing the champion against near-copies of itself. **Fix:**
`baseline` now **self-retires** when identical to the champion
(`training/approaches/baseline_mcts.py`); the nightly roster now leads with candidates
that are genuinely *different* — `mcts_sweep` (tuned exploration constant),
`progressive_widening` (`training/approaches/progressive_widening.py`, focus search on
top moves — the audit's own §6.4 lever, re-measured post-maxⁿ-fix), and
`heuristic_tune`. `policy` is held out of the default roster until its model is made
richer than the fixed heuristic it currently reproduces (§7.3).

### 7.2 The screen was too noisy to detect a real gain (compounding cause)
The fixed 20-game screen re-rated the champion from a **fresh** Elo tracker
(1200 start, K=32, 6 pairwise updates/4-player game) over ~20 games at 100 ms/move.
A *fixed* agent's rating swings **±72 Elo** run-to-run on that basis (the reported
"noise floor"), which swamps any incremental gain — so no candidate could ever clear
the gate even if it were genuinely better. **Fix — variance-reduced sequential
evaluation** (`training/evaluation/sequential.py`, `--sequential-eval`), which reuses
infrastructure already in the repo:

- **Paired** champion-vs-candidate comparison (they are co-present in every game;
  agent RNG is keyed by name, not seat), so shared seat/opponent/opening variance
  cancels — the paired within-game record and `analytics/tournament/statistics.py`
  (`bootstrap_score_ci`, `paired_permutation_test`) already existed, just unused.
- **Seat-balanced** `round_robin` rotation (was hardcoded `randomized`).
- **SPRT** (Wald sequential test) on the paired W/D/L stream: decisive candidates —
  clearly better *or* clearly not — resolve in far fewer games; only genuinely
  borderline ones cost many. A candidate promotes only if the SPRT accepts H1 **and**
  it still clears the conservative gate on those same games (the SPRT games double as
  the confirmation battery — no separate 60-game run). This is standard computer-chess
  engine-testing practice and is what makes *effective daily training with fewer
  games* possible. The old fixed-N screen remains available (flag off).

### 7.3 Still open (next highest-leverage) — see §8 for the 2026-07-07 follow-up
- **Richer move policy.** The 4-feature linear policy is fundamentally too weak to
  beat the search it distils. Give it more features (phase × occupancy interactions,
  finer corner/blocking counts, opponent-mobility deltas), decouple it from the
  heuristic seed, and **collect visit targets from a stronger teacher** (higher
  sim-count games) than the student runs — teacher > student is the point of
  expert iteration.
- **Off-Actions daily driver.** The GitHub runner (2 cores, 350-min cap, forced to
  100 ms) is the binding budget constraint. Parallelise the arena across cores and run
  the sequential loop at full strength on a dedicated box; leave Actions as the
  report/commit layer. `--sprt-elo1` can then be lowered to chase smaller gains.
- **Exploration in self-play.** Add root temperature / Dirichlet noise and a
  population of past checkpoints so the learning signal stops being "imitate the
  champion".

---

## 8. Follow-up (2026-07-07): search-signal audit — why training labels were weak ✅

Focus: "training results are still poor — is the search too shallow, the
feature set too limited, or is something mislearning?" Answer: **the search
was numerically blind and an order of magnitude more expensive than it needed
to be, and the training data was generated by the wrong agent.** Measured,
fixed, and instrumented in this round:

### 8.1 Measured: the search was effectively depth-1 (diagnostic added)

`python -m training.diagnostics.search_quality` (new) runs any agent config
across an iteration-budget ladder on seeded probe positions and reports tree
depth, ms/iteration, move-choice stability, and agreement with a
larger-budget reference. Baseline (gen140 champion, pre-fix code,
`training/reports/search_quality_baseline.json`):

| ply | branching | 50 it | 250 it (champion) | 1000 it |
|---|---|---|---|---|
| 4  | 214 | depth 1 | depth 2 | depth 2 |
| 16 | 471 | depth 1 | depth 1 | depth 2 |
| 32 | 131 | depth 1 | depth 2 | depth 2 |
| 52 | 20  | depth 3, agree 0.00 | depth 6, agree 0.50 | depth 8 (ref) |

Early/mid game the branching factor exceeds the whole iteration budget: the
root never finishes expanding, every child gets ≤1 visit, and "most visited"
degenerates to "first expanded" (= the move-ordering heuristic). Late game,
where the tree *can* deepen, the champion budget's move choice agreed with
the reference only 50% of the time. Visit counts and outcomes collected at
this budget are what the policy/TD trainers were imitating.

### 8.2 Fixed: four search-signal defects (mcts core)

1. **Reward-scale mismatch (measured, then partially REVERTED after
   measurement).** Rollout/cutoff rewards are O(100) (score delta + 100 win
   bonus; static eval ×100) while `exploration_constant=1.414` contributes
   O(1–5) — UCB is effectively greedy after the mandatory first-visit sweep,
   and the learned-leaf path returned win probabilities in [0,1], 100× too
   small next to sibling paths. Normalising every reward onto [0,1] was
   implemented and **measured to regress the champion 72.9% → 29.2% win
   rate** at its *pinned* exploration constant (24 fixed-seed games each vs
   the heuristic/random pool): stored configs are implicitly calibrated to
   the legacy near-greedy balance, and the extra exploration at bf ≥ budget
   turns move choice into single-rollout noise. The scale was restored; the
   one genuine defect kept from this line of work is the learned-leaf path,
   now scaled ×100 to match its siblings. Re-tuning exploration on a clean
   scale belongs to the gated `mcts_sweep` approach, not a code default.
2. **Evaluator clamp destroyed the signal (high confidence, measured).** The
   champion's learned `state_eval_weights` sum *negative* on typical
   early/mid boards and `BlokusStateEvaluator.evaluate` hard-clamped to
   [0,1]: **every cutoff evaluation returned exactly 0.0** (measured:
   best-Q = 0.000, root visit entropy = uniform), so for most of the game the
   search optimized nothing. The clamp is now an order-preserving tanh
   squash.
3. **Rollouts burned 92% of wall-clock enumerating moves nobody used
   (measured via cProfile).** Every `greedy_sample` rollout step enumerated
   *all* legal moves (~14 ms, ~2,500 legality checks) to uniformly sample 12.
   New `LegalMoveGenerator.sample_legal_moves` stops after K legal moves in
   shuffled order (empty result still authoritative for pass detection).
   Iterations measured 60 → 9.5 ms at ply 8 (~6×): the same wall-clock now
   buys ~6× the search.
4. **Transposition table froze stochastic rollouts (high confidence).** A
   rollout's reward was cached by board hash and replayed on every revisit —
   one random sample masquerading as a Monte-Carlo average. The TT now only
   memoises deterministic results (leaf evals, depth-0 static eval).

Also: named search profiles (`mcts/search_profiles.py`:
fast/balanced/strong/teacher; teacher = 1200 iterations + progressive
widening) replace ad-hoc budget arithmetic. A root-move tie-break by mean
reward was tried and reverted: with bf ≥ budget every root child has one
visit, and the single-rollout Q tie-break measured strictly noisier than
falling back to the move-ordering heuristic's top choice.

### 8.3 Fixed: training data came from the wrong agent (high confidence)

- `training/td_selfplay.py` and `training/policy_selfplay.py` seeded their
  "champion" seat from `BASE_CHAMPION_PARAMS` — the **gen0 cold-start
  config** (random rollouts, RAVE on, cutoff 5) — not the promoted gen140
  champion. The TD value model and the distilled move policy were learning
  from a teacher that the audit had already diagnosed as weaker than the
  no-search heuristic. Both collectors now build their roster from the
  **registry champion + a search profile** (default `teacher`), via
  `training/teacher_roster.py`.
- TD trajectories recorded **all four seats**, including the random agent's:
  V(s) was learning the value of states *under random play*. Champion seats
  only are captured now (`--capture-all-seats` restores the old behaviour).
- Label provenance (`generator@profile:budget`, e.g. `gen140@teacher:1200`)
  is stamped into each row's `agent_version`.
- The nightly `run_approaches` mode never regenerates self-play data —
  `heuristic_tune` has been re-fitting the same 44,832 snapshot rows (from
  pre-gen140, i.e. broken-search games) every night. Fresh collection is a
  CLI away now, but wiring it into the nightly loop is **still open**.

### 8.4 Fixed: the 45→8 feature projection bottleneck

The TD model trains 45 rich features but the deployed agent consumed only
the 8-feature projection — discarding the most predictive signals
(score margin, rank; see `docs/` rich-feature analysis). The existing
`RichLeafEvaluator` (45-feature linear value at leaves, ~1 ms/leaf on the
"score" subset) was wired into the agent but **nothing ever enabled it**,
and it had a train/serve mismatch: trained on full vectors, served with 14
features zeroed. Now:

- `TDConfig.feature_subset` trains on exactly the masked view served at
  leaves; the artifact records it; the evaluator defaults its serving subset
  from the artifact.
- New `rich_leaf` candidate approach (leads `DEFAULT_APPROACHES`): champion
  clone with `rich_leaf_eval_enabled` on a subset-matched artifact. Leaf
  evaluation replaces rollouts, so it is also ~10× cheaper per iteration.
  Promotion stays gated — nothing changes until it wins the SPRT screen.

### 8.5 Evaluation results (fixed seeds 20260620/20260621)

All runs: the **gen140 champion at its stored budget** (250 iterations /
500 ms deterministic) vs the fixed non-MCTS pool (two heuristic agents +
random), 24 games over the two fixed seeds. The champion *config* is
identical across rows — only the engine *code* differs — so this isolates
the effect of the code changes on the same agent.

| Change set | Champion win% | 95% CI | TS μ | wall-clock |
|---|---|---|---|---|
| **Before** (pre-branch `main`) | 72.9% | [0.53, 0.87] | 51.4 | 5281 s |
| Full reward normalisation to [0,1] (**reverted**) | 29.2% | [0.15, 0.49] | 29.4 | 2737 s |
| **After** (kept fixes: tanh squash, sampled rollout movegen, TT fix, learned-leaf scaling) | **89.6%** | **[0.72, 0.97]** | **52.8** | 2696 s |

Reading:

- The kept fixes lift the champion **72.9% → 89.6%** against the same pool
  at the same iteration budget — a real strength gain, driven by the tanh
  squash restoring the exploitation signal the hard clamp had zeroed
  (§8.2.2). Wall-clock also roughly halved (the sampled rollout movegen,
  §8.2.3), so the same budget is cheaper *and* stronger.
- The normalisation experiment is shown for the record: it dropped the same
  agent to **29.2%** (below the plain heuristic), which is why it was
  reverted. The before/after harness is exactly what caught it.
- Absolute win-rate here is inflated by the weak pool (no MCTS opponent);
  the number that matters is the **within-column delta at a fixed config**,
  which is clean because every row replays the same fixed seeds.

**Candidate screen (`rich_leaf` vs champion, same seeds/budget, 24 games,
4-agent arena `[champion, rich_leaf, heuristic, random]`):**

| Agent | win% | 95% CI | TS μ |
|---|---|---|---|
| champion | 45.8% | [0.28, 0.65] | 40.4 |
| `rich_leaf` candidate | 41.7% | [0.24, 0.61] | 41.6 |
| heuristic | 12.5% | [0.04, 0.31] | 24.6 |
| random | 0.0% | [0.00, 0.14] | −5.8 |

The `rich_leaf` candidate (45-feature TD value at leaves, trained on the
fresh 732-row teacher-budget corpus) is **statistically indistinguishable
from the champion** on this screen — the two strong agents split the wins
(CIs overlap almost entirely; candidate marginally ahead on TrueSkill μ,
marginally behind on win-rate). It does **not** clear the "strictly beats
the champion" bar, so it would **not** promote — the correct gated outcome.
The champion registry is untouched regardless; promotion requires the SPRT
screen + conservative gate in the nightly pipeline, not this smoke screen.

Takeaway: the rich-feature leaf value now *reaches* the search and plays at
champion strength at a fraction of the per-iteration cost (leaf eval ~1 ms
vs a full rollout), but it is not yet a clear improvement on 24 games. The
next step is a larger SPRT screen on more teacher-budget data (§8.6), not a
promotion on this evidence.

### 8.6 Remaining risks / next steps

1. **Champion eval weights were fit against the broken search** — after these
   fixes, re-collect trajectories (`training.td_selfplay`, teacher profile)
   and let `heuristic_tune`/`td`/`rich_leaf` re-fit through the gate.
2. **Nightly loop still doesn't regenerate data** (§8.3) — add a collection
   step to `run_approaches` or a scheduled `mcts_lab.self_play` run.
3. **`exploration_constant` is untuned on the new scale.** 1.414 on [0,1]
   rewards with small Q-gaps may over-explore; `mcts_sweep` (C grid) is the
   measurement vehicle. Do not hand-pick.
4. **Elo-delta promotion gate keys off a fresh K=32 tracker** (±90 Elo noise
   on a frozen agent, §7.2 / measured in `latest.json.trajectory`) — the
   SPRT path avoids it; migrating the fixed-N gate to TrueSkill-only is
   still worth doing.
5. **Layer-experiment verdicts (RAVE, minimax, NST, opponent modeling) are
   invalid again** — they were measured on the greedy-selection,
   zero-signal-eval search. Re-measure before re-enabling anything.

---

## 9. 2026-09 diagnostic assessment (supersedes the conclusions above where they conflict)

An independent diagnostic assessment against the goal "reliably beat a competent human" is in
`docs/agent-strength-rebuild/DIAGNOSTIC_ASSESSMENT_2026-09-06.md`. Headline corrections to this
report: the engine catalogue was not the Blokus set (duplicate S/Z tetromino, no Z-pentomino —
fixed 2026-09-06, every prior dataset/artifact/rating invalid); every promotion decision was
made at 25-50 simulations per move where search collapses to the move-ordering heuristic; nightly
runs replayed identical seeded games; Elo was carried across the maxⁿ-fix boundary; the served
champion is the gen0 configuration; no human has ever played any agent. The §3.8 "non-bugs
verified" claims about scoring and rules are withdrawn.

