# Blokus Agent: Diagnostic Assessment and Path Forward

**Date:** 2026-09-06. **Repo state audited:** `main` at `a22cd1d` (2026-07-17) plus the unmerged PR #208 branch (EXP-013).
**Goal this assessment is judged against:** an agent that reliably beats a competent human Blokus player (≥ 70% over 20 seat-rotated games, no loss attributable to an illegal move, timeout, or crash).

**How this was produced.** Thirteen audit lanes (engine, independent move-generation cross-check, measurement gate, ratings forensics, human interface, search code, search measurements, endgame tactics, value and policy learning, automation, specification drift, live yardstick arena) ran as separate agents with scratch scripts; nothing in the repo was modified. The planned adversarial cross-verification stage did not complete (session limits), so I re-verified every blocker myself by reading the cited lines or re-running the cited script. Each finding carries a tag: **[re-verified]** = I reproduced it; **[lane]** = one auditor verified it with cited evidence I spot-checked; **[unverified]** = labelled as such. Scratch scripts and raw outputs are under the session scratchpad (`scratchpad/<lane>/`).

**Bottom line.** The rules engine has never implemented Blokus (20 distinct pieces, not 21). No human has ever played any agent. The agent a human can play is a different, weaker, never-validated configuration running a two-month-old code bundle at 12-32 simulations per move. Every training-era comparison was made at 25-50 simulations per move, where the "search" is the fixed move-ordering heuristic, on two fixed seeds that replayed identical games night after night, with an Elo carried across incompatible eras. The training cycles were measuring noise around an agent whose configuration changed exactly once in 179 generations, and that one change was hand-written. Rescue-era experiments improved the discipline but rested on 4-24 games, including two "decisions" made on four distinct games.

---

## 1. Findings

Severity: **blocker** = makes it impossible to know whether the agent is improving, or caps strength regardless of training; **major** = materially limits strength against a human or the reliability of measurement; **minor** = worth noting only.

### Blockers

**B1. The engine does not implement Blokus: 20 distinct pieces, 88 squares.** [re-verified] *(line numbers as of `a22cd1d`; fixed on branch `fix/standard-piece-set`)*
`engine/pieces.py:318-321` defines piece 9 "Tetromino S" `[[0,1,1],[1,1,0]]` and piece 10 "Tetromino Z" `[[1,1,0],[0,1,1]]`. Blokus pieces are free polyominoes and the engine itself generates reflections (`engine/pieces.py:368-401`), so ids 9 and 10 are the same piece. The Z-pentomino is absent (`pieces.py:323-354` lists F, I, L, N, P, T, U, V, W, X, Y only). Totals: 20 distinct shapes, 88 squares (standard: 21, 89); maximum score 108 (standard 109; the value 108 is what winners score in the experiment logs). The frontend ships the same set (`frontend/src/constants/gameConstants.ts:58-80`). Present since the first commit (`d604462`, 2025-11-30). Two internal sanity numbers coincidentally match the real game (91 fixed orientations, 58 opening placements per corner) because the duplicate S-tetromino contributes exactly what the missing Z-pentomino would; only an independent reference caught it (now `tests/test_reference_movegen.py`). `training/rich_features.py` even documented the set as "non-standard: 1/1/2/6/11" in a normaliser footnote (July 2026) without anyone treating it as a bug. Comments and docs assert 89 squares (`mcts/state_evaluator.py:45`, `engine/telemetry.py:88`, `docs/02-architecture/DATA_MODEL.md:19`). No test asserts 21 distinct shapes (`tests/test_piece_shapes_match.py:31-34` checks count only).
*Mechanism:* everything the lab has ever measured is about a variant; a competent human plays with a Z-pentomino the engine cannot represent and one S/Z tetromino the agent expects to place twice. Fixing it changes piece ids and inventories, invalidating every id-keyed dataset, weight file, and benchmark number. Secondary: 3.8-4.2% of legal moves are exact cell-duplicates (piece 9 vs 10), splitting visits between identical children.

**B2. No human has ever played any agent, and there is no proxy for human strength.** [lane, spot-checked]
No artifact, log, database record, manifest, or commit shows a human game; the browser demo persists nothing and its "Save Game" export omits player types (`frontend/src/store/gameStore.ts:447-460`); `agents/gameplay_human.py` is empty; `training/human_estimate.py:19-24` states the 1200/1500/1700 anchors are assumed with "no measured human-vs-MCTS Elo anchor". `training/state/latest.json` nonetheless reports `human_target_elo 1700` and `estimated_days_to_target 72.4`. MASTER_PLAN Phase 11 (human evaluation) never started.
*Mechanism:* the goal metric has never been attempted, so no decision in the project's history was connected to it.

**B3. The agent a human can play is not the trained agent, runs stale code, and barely searches.** [re-verified: bundle date, scoring default, dispatch, registry flag; lane: Pyodide measurements]

- Serving resolves `data/champion_registry.json` v2 (`agents/champion.py:148-159`): random rollouts, cutoff 5, RAVE k=1000, `minimax_backup_alpha 0.25`, 2 root workers, no move ordering. Its record says "PROVISIONAL (not gauntlet-validated)", `gauntlet_run_path null`, `seeds []`, TrueSkill "the prior", yet `gate_pass: true`; the loader checks only that field plus non-null win rate (`agents/champion.py:122-133`). This is the gen0 config that gen140 beat 48-12. The nightly workflow never passes `--promote-registry` (`training/nightly_run.py:1082-1086`; 0 occurrences in the workflow file).
- `frontend/public/blokus_core.zip` was last rebuilt in `0a85b89` (2026-07-01). The bundle defaults to house scoring (`engine/game.py:66` inside the zip: `SCORING_MODE_HOUSE`), has no monomino bonus, no root reward baseline (D-014), no `sample_legal_moves`, and none of the value-model or policy modules. Pyodide loads only numpy, so the D-016 configuration (sklearn ridge leaves) cannot run in the browser at all.
- The bridge always sets `time_limit = budget_ms/1000` (bundle `worker_bridge.py:156`) and `MCTSAgent` dispatches on `time_limit` before `iterations` (`mcts/mcts_agent.py:988-992`), so the 250-iteration contract is discarded. Measured under Pyodide 0.29.3: "Play the Champion" runs 12-32 iterations per move; the default "Start Game" tiers build `MCTSAgent(iterations=500000)` with the default `heuristic` rollout policy, complete exactly 1 iteration in 3.0-3.6 s, and play the last legal move every turn; "Challenge Champion" is the only mode that searches, at a fixed 30 s per AI move.
- Scoring divergence: default demo modes are house-scored; the in-game rules text describes house bonuses; only "Play the Champion" is standard, and that standard lacks the monomino rule.
*Mechanism:* lab improvements cannot reach a human; anecdotal demo results are uninformative; the served agent is the weakest config in the repo.

**B4. Every evaluation and promotion ran at 25-50 simulations per move against a branching factor of 200-450, where the search is the move-ordering heuristic.** [re-verified: conversion; lane: tree measurements]
Budget conversion is `iterations = round(iterations_per_ms × thinking_time_ms)` (`analytics/tournament/arena_runner.py:483-489`); champion and candidates carry `iterations_per_ms 0.5`. The nightly override was 100 ms (workflow line 164) = 50 iterations; the promotion screen used 50 ms = 25. With standard expansion every root child is visited once before any revisit (`mcts_agent.py:1706-1719`) and ties resolve to the first-expanded child, i.e. the ordering heuristic's top move (`mcts_agent.py:426-443`). Measured: at ply 8 (444 legal moves) and ply 24 (83), at 25, 50 and 250 iterations, every root child has exactly one visit and depth is 1; only at ply 48 (13 legal) does a child accumulate visits. Branching factor on a seeded game: 58 / 444 / 233 / 83 / 22 at plies 0 / 8 / 16 / 24 / 32.

| Context | thinking_time_ms | iterations/move |
|---|---|---|
| Nightly evaluation override (all 41 post-fix runs) | 100 | 50 |
| Promotion screen (AUDIT_REPORT §3.3) | 50 | 25 |
| Pool anchors under the same override | 100 | 100 (both "fast" and "strong") |
| Training champion contract | 500 | 250 |
| Browser "Play the Champion" (measured) | 500 wall-clock | 12-32 |
| Browser default tiers (measured) | 200-900 wall-clock | 1 |
| D-016 teacher config | pinned | 500 (8-29 s per move) |

*Mechanism:* evaluator, RAVE, backup, prior and policy changes cannot express themselves as move differences at these budgets; the gate compared the same heuristic policy against itself. At the 250-iteration serving budget the same collapse holds for the first third of the game.

**B5. Nightly runs replayed identical games; the Elo series is convergence, not evidence.** [re-verified: seed and RNG code; lane: forensics]
Seeds are fixed (`training/evaluation/benchmark_pool.py:25`; the workflow always passes 20260620/20260621), per-game and per-agent seeds are deterministic hashes (`arena_runner.py:285-291`), and search is iteration-pinned, so an unchanged candidate replays the same games. A candidate byte-identical to the champion (`baseline`, gens 141-158, before the self-retire guard landed in `8255bc0` on 2026-07-06) recorded exactly 20 games, 9-10 head-to-head, ΔElo −81.9, Δμ −6.22 in 18 consecutive runs. Per-game champion Elo sign sequences match 100% across 13 consecutive runs (gens 142-154); 15 of 41 post-fix runs are pure replays; champion σ shrank 8.33 → 5.02 on repeated data, and `training/status.md` reported "+4.11 Elo/run improvement rate".
*Mechanism:* repeated identical samples manufacture false precision and a fake trend; a real change cannot be separated from tracker convergence.

**B6. The champion changed once in 179 generations, by hand; Elo is a slot rating carried across incompatible eras; no learned component ever entered the champion.** [re-verified: `nightly_run.py:799-804`, `:943`; lane]
Promotions: gen0 (seed) and gen140 (2026-07-02, "corrected weak-champion search settings", `learning_method: null`). The Elo tracker is seeded from the all-time latest DB row with no era filter (`training/nightly_run.py:943`, `training/ratings_db.py:341`), and promotion resets only TrueSkill (`nightly_run.py:799`) while Elo is inherited (`:804`). The reported 1388.55 is one walk over 6,290 games of which 2,196 predate the maxⁿ fix. The post-promotion +198 Elo rise (gen140→154) is zero-sum re-calibration against the stale pre-fix `heuristic` rating (−427). Post-fix candidate outcomes: 107 rows, mean ΔElo −63 (sd 56), 61 SPRT screens all "inconclusive", 0 accepted. TrueSkill ranks an untrained anchor with twice the iterations (`baseline_mcts_strong`, μ 55.06) above the champion (54.39): strength is budget-bound, not training-bound. Nightly TD weights were static (07-01→07-02: zero change) or drifting by < 0.05; `policy_weights.json` is `[1.01, 1.99, 0.44, 0.79]` versus the hand heuristic `[1, 2, 0.5, 1]`.
*Mechanism:* "did each generation get stronger?" has a hard answer: no, it could not have, because the agent did not change; and if it had, the rating could not have shown it.

**B7. The SPRT screen could not decide within its budget by construction.** [lane; defaults and workflow flags re-read]
Workflow: `--sprt-elo1 40`, `--sprt-max-games 160`, 315 min split over 3 candidates at 100 ms/move; observed 48-56 paired games per candidate, LLR −1.5..−1.8 against bounds ±2.94, every verdict "inconclusive → hold" (`training/evaluation/sequential.py:307-308`). Monte Carlo of the repo's own `sprt_decision` (elo1 = 40, α = β = 0.05, min 24, block 4): with a 48-game cap the probability of any decision is < 0.5% even for a true +70 Elo candidate; at the nominal 160 cap a true +40 candidate is accepted 14% of the time and a clone rejected 15%. The code's own docstring says elo1 = 50 needs ~285/~130 games.
*Mechanism:* under this configuration no candidate could ever promote regardless of true strength.

**B8. Rescue-era gate decisions rest on tiny or replicated samples.** [re-verified: EXP-012 replication; lane: EXP-006b seats, p-value census]

- EXP-012 and EXP-013 (the decisions that paused the learned move policy) contain **4 distinct games each**: all four agents are deterministic (value-model leaves, no rollouts), round-robin seat rotation has period 4, and both seed directories are byte-identical (`training/reports/experiments/search_scaling/exp012_mlp_prior/seed_*/*/games.jsonl`). The logged p = 0.048 reproduces only on the 20 duplicated rows (p = 0.145 on one seed's 10 rows; n = 4 independent). EXP-013: p = 0.0011 duplicated, 0.033 on 10 rows, 4 distinct.
- EXP-006b (the Phase 4 "GATE PASS", it500 − it150 = +14.08, p = 0.013) is 12 games with seats 2/2/4/4 per agent, not balanced; EXPERIMENT_LOG reports 34 p-values, 16 < 0.05, and among the roughly 15 non-anchor contrasts this single p = 0.013 is what chance produces at α = 0.05.
- In the same harness two byte-identical agents differ "significantly" (EXP-012: prior_a vs prior_b −11.2 pts, p = 0.0017 over 10 games); seat position alone flips first-place rate 60% vs 20%.
*Mechanism:* D-016 was adopted as "strongest validated" and the policy track was paused on evidence indistinguishable from seat noise; the rescue's decision tree inherits that uncertainty.

### Major

**M1. The served v2 configuration plays near-random moves in the opening and midgame and carries a perspective bug.** [re-verified: minimax repro; lane: root-parallel tie measurement]
Without move ordering and with 250 iterations split 125 per root worker, whenever the branching factor exceeds 125 (28 of 72 plies in a heuristic self-play game) each worker expands the same first 125 moves once, every merged move ties at 2 visits, and the merge returns the last generator-enumerated move: at a bf-401 position the chosen move had heuristic rank 248/401 (`mcts/mcts_agent.py:363`, `mcts/parallel.py:298-319`). Separately, `blended_q` (`mcts_agent.py:172-184`) mixes a node's own average (the mover's reward) with `minimax_value` = max over children's Q, which is the *next* player's reward (`:2360-2371`); with α = 0.25 a synthetic root prefers the child that is good for the opponent (`scratchpad/search-code-audit/repro_minimax_perspective.py`, "BUG CONFIRMED"). gen140 and D-016 use α = 0 and are unaffected; the served v2 and `config/challenge_champion_config.json` use 0.25.

**M2. Reward scales inside one tree are incoherent, and exploration is mis-sized for both.** [re-verified: probe; lane: rollout-noise and tree-shape measurements]
Cutoff leaves return `state_evaluator × 100` ≈ 38-43 for every player (spread 2-5); terminal rollouts return root-baselined score delta plus 100 (sole winner) or 10 (tie): measured 0.0 / 10.0 / 112.0 (`mcts_agent.py:1835-1848`, `:2018-2031`). UCT exploration at N = 250, n = 1 is 3.3, larger than the entire midgame Q range (3.4-4.8) so visits stay flat (ply 24: top visits 9, 6, 6, 5, 5 of 83), while in the endgame the +100 bonus makes one child take 209/250 visits. Per-rollout std (0.5-0.7) equals or exceeds the top-2 Q gap at plies 16-24 with 1-8 visits per child. The [0,1] normalisation was tried and reverted (AUDIT §8.2) without re-tuning C; the pinned C = 1.414 is a legacy near-greedy calibration.

**M3. No learned component has produced strength beyond the hand heuristic plus rollouts.** [lane; reports re-read]
Model leaves vs rollout leaves at equal 500 iterations: +2.4 pts, p = 0.56 (EXP-006a); v2 vs v1 evaluator: +2.25, p = 0.60 (EXP-007); non-linear models add nothing over ridge (gate 1 failed, R² ≈ 0.26, pairwise ordering ceiling ≈ 0.68); v1 ridge scores R² = −0.38 on teacher positions; the MLP move scorer fails its own overfit gate (0.78 vs bar 0.80 on 200 training decisions) and as a prior lowers score by 24 pts. Cause measured in the teacher data: sibling Q gaps under model leaves are 0.1-2 points (median top-2 gap 0.10) against a model MAE of 5.6-7.3 points and an exploration term of 3.5, so 500-iteration visits are near-uniform (median max target 0.20; 31.5% of decisions have no child above 0.10) and the distilled "policy" is mostly "which 44 moves did ordering + progressive widening expand first". The prior term at c = 1.5, P = 0.23 is 3.9, dominant over those Q gaps, which is why adding it hurt.

**M4. Endgame: no solver, and the strongest validated config throws away a two-ply win.** [lane; positions saved]
There is no exact endgame search anywhere. On a hand-built position (bf 8) where blocking the leader's only corner wins 118-115, D-016 chose the greedy piece 0/6 at 200 and 500 iterations and at 2,000 iterations gave the winning block 1 visit versus 1,982 (its leaf predicts own final score with no win or rank term; `mcts/value_model_evaluator.py:71-85`). gen140 (rollouts with the win bonus) found it 3/3 at 25-1,000 iterations, but only because bf was 8; the ordering heuristic ranks the block last in both test positions, so at bf > budget it is never played. Rollout reward gives zero credit for denying an opponent unless it flips the sole winner (`mcts_agent.py:2018-2029`).

**M5. Stale incremental frontier corrupts the evaluator's two heaviest features and creates train/serve skew.** [lane; code re-read]
`Board.update_frontier_after_move` runs only for the placing player (`engine/board.py:327-379`, `:562`), so cells occupied by other colours stay in a player's frontier set; `Board.copy` preserves them, `Board.from_dict` rebuilds clean ones (`:706-721`). Legality is unaffected (superset, occupied cells rejected: 0 missing cells in 834 checks). But `accessible_corners` (weight 0.243) and `opponent_avg_mobility` (−0.3) read `len(board.get_frontier())` directly (`mcts/state_evaluator.py:150-168`). At ply ≥ 50 the reported frontier is 18.6 versus 10.6 true (43% phantom). Value and policy models train on clean `from_dict` boards and are evaluated on live boards in search; 17 of 45 rich features shift; static-evaluator player ordering flips in 66-77% of late positions. Effect on the v2 ridge is small (0.27 pts), on the champion's static evaluator large.

**M6. Nothing validated is fast enough to serve.** [lane; instrumented arena]
gen140 at 250 iterations: 4-5 s per move mean, 12 s max, single-threaded (9-33 ms per iteration). D-016 at 500 iterations: 8-29 s per move; 75% of its time is rich-feature extraction that enumerates all legal moves for mobility (`training/rich_features.py:204,252`), and in a 500 ms wall-clock budget it completes 9-18 iterations. The `search_profiles.py` docstring assumes 4-10 ms per iteration, i.e. at best 50-125 iterations in 500 ms, never 250.

**M7. The "heuristic" yardstick is a temperature-1 softmax sampler, and the served champion is indistinguishable from it.** [re-verified: code; lane: live arena]
`agents/heuristic_agent.py:54-66` samples from softmax over size + 2×new corners − 1.5×edge + 0.5×center; measured P(argmax) is 0.09-0.13 in the opening (18-40 effective choices), 0.44 overall; it fails simple endgame picks (0/2 on a 5-cell pocket). Live yardstick arena run for this audit (8 round-robin games over seeds 20260620/20260621, iteration-pinned at 250, one worker; `scratchpad/strength-yardstick/audit_yardstick/report.json`): first place gen140 69%, served v2 25%, heuristic 6%, random 0%; mean scores 89.4 / 72.5 / 76.0 / 50.8; paired score differences: v2 − heuristic = −3.5 pts (p = 0.42), gen140 − heuristic = +13.4 (p = 0.017), gen140 − v2 = +16.9 (p = 0.041). At n = 8 the served agent is statistically indistinguishable from the softmax sampler and is beaten by the training champion; per-move cost at 250 iterations was 4-5 s mean, 12 s max. No greedy-deterministic, most-corners, or one-ply-lookahead baseline exists.

**M8. The measurement protocol moved constantly while reports drew one trend; anchors are not fixed.** [lane; workflow history re-read]
Nine protocol changes in 21 days (games per arena 12→100→20→10→5; 100 ms override introduced 06-29; roster rotated six times; SPRT added 07-06 with elo1 = 70 in code and 40 in the workflow; cron 03:00 → every 6 h → removed). Pool anchors are deep copies of the *current* champion params (`benchmark_pool.py:34-66`) and under the override both become identical 100-iteration clones (champion gets 50); `best_historical` never played a game (`head_to_head.py:139-151` caps arenas at four opponents); the fixed-N screen used `randomized` seats on fixed seeds giving a generation-invariant seat imbalance (champion seat 4 in 2/20 games, candidate 6/20); Elo and TrueSkill disagree on the top-3 and on 13 of 64 recent movements.

**M9. Seven diagnosis documents in 18 days, each naming a different root cause, none re-measured under a stable protocol.** [lane; documents re-read]
06-26 "loop never closes"; 06-29 "evaluation starvation"; 07-01 "opponent-cooperation backprop bug"; 07-06 "candidate collapse + noisy screen"; 07-07 "search numerically blind, wrong teacher"; 07-12 "candidate generation exhausted"; 07-12..14 "evaluator is the ceiling"; 07-19 "representation is the axis / policy fails as guide". Each was declared fixed and superseded. AUDIT_REPORT §3.8 claimed scoring verified while the monomino bonus was missing; §3.6's "~17 ms per rollout" is now 7.3 ms; README says training "runs nightly ... every 6 hours" (frozen 07-12), that the strong baseline's confirmation is "pending" (promoted 07-02), and that the demo plays "the champion" (it serves v2).

**M10. Standard-rules mismatch persists at the human-facing entry points.** [re-verified]
Engine, arena, teacher and TD generators default to standard scoring now, but the served bundle defaults to house (`worker_bridge.py:36` in the zip) and lacks the monomino bonus; the rules modal describes house scoring.

### Minor

- Root-parallel workers bypass `select_action`, so adaptive depth and several constructor params are silently dropped (`mcts/parallel.py:198-221`); v2's "adaptive rollout depth" never happens in workers.
- Pass children share the parent's Zobrist hash (`mcts/zobrist.py:88-90`); harmless today, latent for side-to-move-aware evaluators.
- Evaluator leaf paths have no terminal short-circuit: game-over nodes get a model guess instead of the exact score (`mcts_agent.py:1760-1766`).
- Tree-parallel virtual loss is applied to the parent, not the selected child, and is negligible at the O(100) reward scale; unused by any config.
- D-016's joblib artifact was pickled with scikit-learn 1.9.0; this environment runs 1.6.1 (version warning on every load).
- `mcts_lab.node_stats` builds an agent without move ordering or evaluator weights, so it cannot inspect the real champion tree.
- An illegal agent move in the arena is converted into a pass and counted, never a fault (`arena_runner.py:863-868`); the webapi converts a timeout into a pass; the timeout test mocks `wait_for` and the no-legal-moves test body is `pass`.
- No board-symmetry augmentation anywhere; value data is 60-118 games, policy data 18 games.
- Bookkeeping: generations 2-44 absent from `ratings.sqlite`; `sum(games_today)` 8,534 vs `total_games` 6,290; 15 stale `mcts_*` agents rated every run with zero games.
- Orphaned state: `.worktrees/challenge-champion-v1` at a pre-fix commit (2026-04-24); PR #208 (EXP-013 results and the "pause policy" decision) and PR #175 open; the 72.9% → 29.2% normalisation regression exists only as a commit message and a comment.
- Test suite is self-referential for rules: all differential tests derive expectations from `engine/pieces.py`, so a catalogue error is invisible to CI.

### Verified sound (so nobody re-audits these)

- **Placement legality and pass detection are correct for the engine's piece set.** Two independently written reference generators agreed with the production movegen on 595 positions / 82,903 moves and on 273 positions / 31,620 placements (0 engine-only, 0 reference-only); 15,452 rule-violating placements were all rejected; `sample_legal_moves` (the rollout path) returned only legal moves in 42,610 samples; `BLOKUS_DIFF_GAMES=8` harness passes.
- **maxⁿ backup is mover-credited and select_child maximises the player to move** (`mcts_agent.py:2325-2330`, `:346`); root-parallel merge preserves it.
- **Iteration-pinned search is deterministic** (same seed → identical move and root visit vector, 6/6; D-016 is seed-independent).
- **No illegal moves or crashes** in any arena game run during the audit (yardstick and safety games, `invalid_actions 0`).
- **Standard scoring is order-correct** given the piece set (`engine/board.py:575-594`).
- Gate, SPRT, TrueSkill and arena statistics unit tests pass (51/51); the teacher-record datasets validate.

---

## 2. Root-cause hypothesis

The project never possessed a measurement that could see agent strength, so every "generation" and every audit was fitting noise. Evaluation ran at 25-50 simulations per move, where the tree is depth one and the move is the fixed ordering heuristic's top choice (B4); it replayed two fixed seeds so unchanged agents produced identical results night after night (B5); it compared against a pool that copied the champion and against a rating carried across a semantics change (B6, M8); and its statistical gate could not reach a decision within its budget (B7). Under those conditions the champion configuration changed once in 179 generations, by hand, and every learned candidate measured below the plain heuristic search (B6, M3), while the reports projected a "days to human target" from an invented anchor (B2). Each audit then explained the flat curve with a new mechanism and moved on without re-measuring the previous one (M9). Meanwhile the goal itself was unreachable by construction: the human-facing agent is a different, weaker, unvalidated configuration on a stale bundle at 12-32 simulations (B3), and the game being trained is a 20-piece variant (B1).

**Evidence that supports it:** the clone-candidate series (identical config, −82 Elo, 18 identical runs); the 100% sign-agreement replays; one-visit-per-child trees at every historical budget; the single hand-authored promotion; the 61/61 inconclusive SPRT verdicts; the four-distinct-game experiments; the softmax yardstick that the served agent ties.

**Evidence that would refute it:** a seat-balanced, fresh-seed, iteration-pinned tournament of ≥ 100 games in which (i) a champion clone scores within noise of the champion and (ii) gen140, D-016 and the served v2 separate consistently at p < 0.01 at their serving budgets; or a recorded series of human games in which any existing configuration reaches ≥ 70%. Neither exists today.

---

## 3. Candidate paths forward

Effort is for one engineer with this machine (10 cores, no GPU). Probability is my estimate of reaching the stated goal (≥ 70% vs a competent human) within the effort window.

**(a) Fix measurement and fundamentals, then continue the AlphaZero-style loop.**
Effort: 3-6 months. Probability: ~15%. What it is: fix B1/B4/B5/B8, then resume teacher self-play → value/policy training → gated promotion. Why the low number: there is no neural network, no GPU, 18 teacher games of policy data, a confirmed ~0.68 ordering ceiling on the feature representation, a value leaf whose Q gaps are below its own error, and a search that costs 10-50 ms per iteration in Python so a 500-iteration teacher move takes 8-29 s. Nothing learned has ever added strength over hand heuristic + rollouts. Biggest risk: months of compute with no strength signal, again.

**(b) Strong hand-crafted heuristic + plain MCTS with a large simulation budget + endgame solver.**
Effort: 4-8 weeks. Probability: ~55%, conditional on a 20-50× engine speed-up. What it is: standard 21-piece rules; move ordering with corner mobility, opponent-corner denial, piece-size ordering and territory; progressive widening; a per-player reward on one coherent scale (rank/win-based, C re-tuned); exact search when the remaining move count is small; 5,000-20,000 simulations per move at a human-tolerable 3-8 s. This is how strong open-source Blokus engines work (Pentobi, which has beaten strong humans for years, is MCTS with hand-crafted priors and no learned network). Biggest risk: Python cannot deliver the simulation rate; the hot loops (movegen, rollouts, feature extraction) need numba or a C++/Rust core, and a competent human may still be stronger than a 10k-sim engine on the 4-player board. Second risk: "competent" is undefined until a human plays.

**(c) Hybrid: heuristic-guided MCTS baseline, network only where it demonstrably beats the heuristic.**
Effort: (b) + 4-8 weeks. Probability: ~55% for the goal (same as (b); the network is not on the critical path), with upside if data from a strong engine is used. What it is: build (b) first, then train a move-ordering or value model on games generated by the strong engine or by Pentobi, and admit it only when it passes an equal-wall-clock, seat-balanced, ≥ 200-game gate. Biggest risk: the network work distracts from finishing (b), which is the whole gain.

**(d) Repo-specific: anchor everything to an external reference engine and a real human protocol.**
Effort: 2-3 weeks before any strength work. Probability: raises (b)/(c) by making success measurable; on its own it does not produce strength. What it is: use Pentobi's GTP engine (`pentobi-gtp`, GPL, Classic 4-player supported, levels 1-9) as (i) the rules cross-check for the fixed piece set, (ii) the fixed external yardstick at several levels, replacing the softmax heuristic and the moving pool, and (iii) a stand-in for "competent human" until real games are logged; add game logging with player types to the web demo; run the calibration tests in §4 M1. Biggest risk: Pentobi's levels do not map cleanly onto human strength, so a human protocol is still required.

---

## 4. Recommendation

**Take path (b), instrumented by (d).** It is the only path whose every step is a measurable strength gain at serving budget, it does not depend on a learning signal that has never appeared, and it matches how existing strong Blokus programs are built. Path (a) optimises a proxy that has never correlated with strength here; (c) is (b) plus optional work.

Milestones with pre-registered gates. Every gate uses protocol v3: fresh seeds per run (derived from the run id), `round_robin` seats, iteration-pinned search at the serving budget, paired sign-flip permutation on score plus first-place rate, n stated in advance, no re-runs to "confirm" until the pre-registered n is reached.

| # | Milestone | Gate (pre-registered) | Stop-loss |
|---|---|---|---|
| M0 | Standard game: 21 distinct pieces, 89 squares; frontend catalogue updated; standard scoring in every entry point; bundle rebuilt | Independent-reference cross-check 0 disagreements on ≥ 300 positions (every ply of ≥ 7 seeded games); Pentobi cross-check 0 disagreements on ≥ 200 positions; 58 opening placements per corner with 2 Z-pentomino placements; all datasets/artifacts declared invalid | none (prerequisite) |
| M1 | A measurement that can see: protocol v3 + Pentobi anchors (e.g. levels 1, 3, 5) + deterministic greedy baseline | Calibration: champion clone vs champion over 100 games has \|Δscore\| < 3 pts and p > 0.3; discrimination: gen140 vs served v2 vs D-016 at 250 iterations separates with p < 0.01 over 100 games | If the clone test fails, fix the harness before anything else |
| M2 | Speed: numba or native core for movegen + rollouts (+ features) | ≥ 2,000 simulations/s from the opening position with greedy-sample rollouts (≈ 20× today), identical move choice on 100 seeded positions vs the Python reference, differential legality 0 disagreements | If < 800 sims/s after 3 weeks: adopt Pentobi's engine core (GPL) as the search and keep this repo as harness/UI |
| M3 | Strong plain MCTS: ordering with denial and mobility terms, PW, coherent rank-based reward with re-tuned C, exact endgame search when total remaining moves ≤ N | ≥ 65% first place and paired p < 0.01 vs the M3-start agent over 200 games; beats Pentobi level 3 (to be calibrated) ≥ 60% first place over 200 seat-rotated games at ≤ 5 s per move | If after 4 weeks with ≥ 5,000 sims/move it does not beat Pentobi level 2: the heuristic+MCTS approach is insufficient on this board → switch to (c) with Pentobi-generated training data |
| M4 | Serving parity: the gated champion is what a human plays (one registry, bundle rebuilt from the same commit, iteration-pinned in the browser or served from the backend) | Browser or backend agent reproduces the arena agent's move on 50 positions; per-move latency ≤ 8 s | none |
| M5 | Human protocol: logged games with player types, seats rotated, standard rules, 20 games vs you | ≥ 70% first place; 0 losses attributable to illegal move/timeout/crash | If < 50% after 20 games with M3 passed: raise the simulation budget and re-run once; if still < 50%, the target human is above what plain MCTS reaches here → (c) |

Estimated calendar: M0 one session, M1 one week, M2 two-three weeks, M3 three-four weeks, M4 one week, M5 one week of play.

> **M1 status (2026-09-07, branch `feat/m1-measurement`, PR stacked on #209):** protocol v3 built and both harness gates passed under pre-registered rules — clone calibration (240 games, +1.76 points, 90% CI [−0.35, +3.87]) and discrimination (120 games; gen140 and D-016 beat the served v2 settings by 8.2 and 14.7 points, p < 0.0001). Pentobi 30.3 is wired in as the external anchor (its rules agree with the engine on 130,000 placements). **Calibration result that changes the plan:** Pentobi level 3, at 10 ms per move, beats the repo's best configuration (D-016, 250 iterations, 8 s per move) by 7.6 points and the champion by 16; level 7 (1.2 s per move) beats them by 28-36. Levels 1 and 2 (8 ms per move, about 3 and 30 simulations) are not beaten either: D-016 ties level 1 and loses to level 2. Per the pre-registered rule the M3 target anchor is level 3 and the baseline anchor is the greedy agent. The gap is not only speed: level 3 uses roughly 90 simulations. This is the M2 stop-loss evidence arriving early. Decision now required (DECISIONS.md D-023): adopt Pentobi's GPL engine as the search core behind this repo's harness, UI and human protocol (shortest path to the goal), continue the home-grown core as research measured against Pentobi levels, or both. Recommendation: both.

> **M4 status (2026-09-08, branch `feat/m4-pentobi-serving`, stacked on #209/#210):** gate passed for the Pentobi agent under D-023 option (c) (accepted 2026-09-08) — see DECISIONS.md D-024; this row's "one registry" clause is superseded by D-024 §6 (the human-facing opponent is a logged Pentobi level). The backend serves agent type `pentobi` (the arena's own Pentobi agent, same level and seed); the web UI has a "Play Pentobi" card that runs the game over the REST routes with the frontend's orientation indices; every backend-served game is logged (`data/human_games/`) with player types, seats, levels, per-move times, fallbacks and forfeited AI turns; `python -m mcts_lab.human_games` scores the log against the M5 gate from the agent's side (Pentobi first place ≥ 70%, ties against the agent). Time control per D-019: hard 10 s cap with a deterministic fallback, no search on one-move positions, optional per-game budget stepping the level down; "dynamic" allocation across moves is Pentobi's own (weighting by move number, early abort), documented rather than re-invented. **Gate:** served agent reproduces the arena agent's move on 50/50 positions at levels 3 and 7 (0 mismatches); full-game whole-turn latency at level 7 stays under the 8 s gate (max 5.3 s, p99 4.5 s) with 0 fallbacks (`training/reports/m4_serving_gate/report.json`). Measured per-level move times (provenance in D-024): levels 1-7 fit comfortably (3 and 7 gated); level 8's one-game maximum exceeds the 8 s gate and it was not gated; level 9 does not fit the cap (mean 9.4 s, max 54 s). M5 can start.

---

## 5. What to stop doing

- Running the nightly loop, in any form, until M1's calibration gate passes. Every run before that produces noise that will be re-audited later.
- Reporting Elo trends, improvement rates, and "days to human target". The scale is internal, name-aliased, era-mixed and anchored to nothing.
- Evaluating anything at 25-50 iterations per move. If a comparison cannot be afforded at the serving budget, it is not worth running.
- Treating p < 0.05 at n = 10-24 games (or any replicated-game run) as a decision. Pre-register n and the test; run once.
- Linear refits of the same 8/45 features (`heuristic_tune`, `rich_leaf`, `td`), visit-count distillation from near-uniform 500-iteration distributions, and the 4-parameter policy. None has ever produced strength.
- Writing the next audit document before the previous audit's headline cause has been re-measured under a stable protocol.
- Hand-promoting or hand-authoring "champions" into the serving registry, and keeping three champions (registry v2, `champion.json` gen140, D-016).
- Comparing model leaves at equal iterations while they cost 3-6× more per iteration; compare at equal wall-clock or not at all.

---

## 6. First concrete step

**Restore the standard 21-piece set and make the piece set testable against an independent reference.** One session.

> **Status (2026-09-07):** done. The Pentobi cross-check ran on branch `feat/m1-measurement` (Pentobi 30.3 built GTP-only; 860 positions, 130,000 placements, 0 disagreements — see EXPERIMENT_LOG.md M0 closure note). Criteria 1-3 met (21 distinct shapes / 89 squares / 91 orientations; 519 positions incl. 89 zero-move, 0 disagreements, 58 openings per corner with 2 Z-pentomino placements; suite green; `mcts_lab.checks` 8/8); criterion 4 met via `scripts/pyodide_smoke.cjs` (bundle sha matches source; 4 seeded games, Z-pentomino placed by every colour). Also found and fixed on the way: `compute_piece_penalty` charged id 10 as a tetromino; schema versions bumped to v2 so old data is machine-distinguishable.

Tasks: replace piece 10 with the Z-pentomino `[[1,1,0],[0,1,0],[0,1,1]]` (5 squares) in `engine/pieces.py` and in `frontend/src/constants/gameConstants.ts`; fix the "89 squares" and piece-size tables everywhere they are derived from ids (`mcts/state_evaluator.py`, `engine/telemetry.py`, `training/rich_features.py`); write an independent rules-based reference generator as `tests/test_reference_movegen.py`; rebuild the browser bundle; add a one-line notice to `DATA_LINEAGE.md` that every dataset and weight file predating this change is invalid.

Acceptance criterion (all must hold):
1. `PieceGenerator.get_all_pieces()` yields 21 pieces whose canonical free-polyomino forms are pairwise distinct, totalling 89 squares and 91 fixed orientations (asserted by a new test).
2. The new reference test reports 0 engine-only and 0 reference-only placements on ≥ 300 positions sampled from seeded random and heuristic self-play, including ≥ 30 zero-move positions, and 58 opening placements per corner of which exactly 2 use the Z-pentomino.
3. `python -m pytest tests/ -q` passes except tests explicitly pinned to old ids, each of which is updated or deleted with a reason; `python -m mcts_lab.checks` passes.
4. The rebuilt `frontend/public/blokus_core.zip` contains the new `engine/pieces.py` (sha256 equal to source) and a scripted Pyodide smoke game completes 84 placements with a Z-pentomino placed by each seat at least once across 4 games.

---

## 7. Open questions for you

> **Answered 2026-09-06 (artifact comments, recorded as D-019/D-020):** Classic 4-player, you play one colour and agents play the other three; about 50 games of experience; a 10-second cap per AI move with a preference for dynamic allocation under an overall budget; native core (C++/Rust/numba) and Pentobi's GPL engine both allowed. Still open: the human-match interface (judgment: web frontend with a natively served agent, decided at M4), and PR #208 / registry v2 disposition (left untouched).

1. **Variant and seats.** Classic 4-player 20×20 with you in one seat and three AI seats (what the demo does), or you controlling two colours against two AI colours, or 2-player Duo? The answer changes the reward design (rank vs win) and the endgame solver size.
2. **Your strength.** Roughly how many games have you played, and against what (family, club, online, Pentobi)? If you have a Pentobi level you beat or lose to, that calibrates "competent" immediately.
3. **Time control.** What per-move latency will you tolerate from the AI (3 s, 8 s, 30 s)? This sets the simulation budget and decides whether a native core is mandatory.
4. **Hardware and licence.** May the search core be rewritten in C++/Rust or compiled with numba, and is using Pentobi's GPL engine (as yardstick, or as the core under a GPL fork) acceptable for this project?
5. **Interface for the human match.** The web demo (needs logging, standard rules, current bundle), a CLI with you entering moves from a physical board, or Pentobi's GUI driving this agent over GTP?
6. **Path decision.** Confirm path (b) with (d) instrumentation, or choose otherwise. If you want the learned-evaluator track kept alive, it should become a background data-generation task fed by the M3 engine, not the critical path.
7. **Disposition of PR #208 and the three champions.** Merge #208 for the record (its conclusions are now known to rest on 4 games), and retire the registry v2 entry so the demo stops serving a config the lab rejected?
