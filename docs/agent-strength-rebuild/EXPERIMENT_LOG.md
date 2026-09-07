# Experiment Log

Record experiments **when they run**, not afterward. Never overwrite an entry; corrections are
appended. Every entry uses the template below (from the governing master prompt §3).

```
Experiment ID:
Date:
Commit:
Hypothesis:
Independent variable:
Controlled variables:
Agents:
Game count:
Seat-balancing method:
Seeds:
Hardware:
Result:
Uncertainty:
Interpretation:
Decision:
Artifacts:
```

---

## EXP-016 — Pentobi level calibration under protocol v3

- **Experiment ID:** EXP-016
- **Date:** 2026-09-07 (pre-registered; launched after EXP-014/015)
- **Commit:** branch `feat/m1-measurement`
- **Hypothesis / question:** where do the gen140 champion config and the D-016 config (both
  250 pinned iterations, one worker) sit relative to Pentobi levels 3 and 7 (no book, one
  thread, seeded per game)? This calibrates the external yardstick; there is no pass/fail.
- **Table:** [gen140, d016_250, pentobi_l3, pentobi_l7]; protocol v3; 120 games (5 permutation
  cycles); 8 workers; fresh recorded seed.
- **Pre-registered reading:** report first-place rate, average rank, mean score and every
  paired score difference with p-values. The Pentobi level that the current best config beats
  at p < 0.01 (if any) becomes the M3 baseline anchor; the first level it loses to at p < 0.01
  becomes the M3 target anchor. If gen140 loses to level 3 decisively, M3's gate is re-stated
  against level 3 rather than level 3-5.
- **Reproduce:** commit 6356c3e, `python -m mcts_lab.calibrate pentobi --games 120 --workers 8
  --seed 1732817614 --label pentobi_exp016`
- **Result (120/120 games, 0 errors, 0 Pentobi mismatches):**
  | agent | 1st% | avg rank | mean score | s/move |
  |---|---|---|---|---|
  | pentobi_l7 | 91.2% | 1.07 | 99.0 | 1.20 |
  | pentobi_l3 | 7.5% | 2.16 | 78.8 | 0.01 |
  | d016_250 | 1.2% | 2.99 | 71.2 | 8.35 |
  | gen140 | 0.0% | 3.57 | 62.7 | 5.67 |
  Paired: gen140 − pentobi_l3 = **−16.1** [−18.0, −14.2]; d016_250 − pentobi_l3 = **−7.6**
  [−9.3, −5.9]; gen140 − pentobi_l7 = −36.3; d016_250 − pentobi_l7 = −27.8; d016_250 − gen140
  = +8.5 [+6.5, +10.4]; all p < 0.0001.
- **Reading (pre-registered rule applied):** the current best config loses decisively to the
  weakest level tested, so the **M3 target anchor is Pentobi level 3** (as pre-registered for
  this outcome) and the M3 baseline anchor is not yet located (levels 1-2 untested → EXP-016b).
  Pentobi level 3 spends ~10 ms per move (≈ 90 simulations scaled by move number) and still
  beats an 8-second, 250-iteration Python search by 7.6 points per game; level 7 (1.2 s/move)
  beats it by 28. The repo's search is not merely slow — per simulation it is far weaker than
  Pentobi's. This is the strongest evidence yet for the assessment's M2 stop-loss branch
  (adopt Pentobi's engine core) and it is a strategic decision for the user (see D-023).
- **Artifacts:** `training/reports/protocol_v3/pentobi_exp016/{report.json,games.jsonl,run_config.json}`

## EXP-016b — Pentobi levels 1 and 2 (locating the baseline anchor)

- **Experiment ID:** EXP-016b
- **Date:** 2026-09-07 (pre-registered; launched immediately after EXP-016)
- **Commit:** branch `feat/m1-measurement`
- **Question:** does the repo's best config (d016_250) or the champion (gen140) beat Pentobi at
  level 1 (~3 simulations) or level 2 (~30 simulations) at p < 0.01? Whichever level is beaten
  becomes the M3 baseline anchor; if neither, the baseline anchor is the deterministic greedy
  agent and M3 is measured purely against Pentobi levels from above.
- **Table:** [gen140, d016_250, pentobi_l1, pentobi_l2]; protocol v3; 120 games; 8 workers.
- **Reproduce:** `python -m mcts_lab.calibrate pentobi-low --games 120 --workers 8 --seed <seed>`
- **Result:** _pending_


## M0 closure note — Pentobi rules cross-check (2026-09-07)

Engine legal-move sets vs Pentobi 30.3 `all_legal` on every ply of 12 seeded games (6 random-agent,
6 heuristic-agent): **860 positions (100 with no legal move), 130,000 placements compared, 0
disagreements** (19 s). Together with the in-repo independent reference (519 positions) and
the earlier auditor references (1,103 positions), the corrected engine agrees with three
independent implementations of the rules. Test: `tests/test_pentobi_agent.py::test_engine_legal_moves_match_pentobi_all_legal`
(default 3 games; `BLOKUS_PENTOBI_GAMES=12` reproduces the numbers above).

## EXP-015 — M1 discrimination gate (protocol v3)

- **Experiment ID:** EXP-015
- **Date:** 2026-09-07 (pre-registered; launched after EXP-014 reports)
- **Commit:** branch `feat/m1-measurement` (on top of `fix/standard-piece-set`, standard piece set)
- **Hypothesis:** under protocol v3 the harness separates agents known to differ:
  the gen140 champion config and the D-016 config (both pinned to 250 iterations, one
  worker) beat the served registry-v2 config (250 iterations, one worker) on paired
  per-game score.
- **Independent variable:** agent configuration (4-seat table: gen140, serving_v2,
  d016_250, greedy).
- **Controlled variables:** protocol v3 (fresh run seed recorded in the report, all 24 seat
  permutations, standard scoring, iteration-pinned single-worker search, paired sign-flip
  permutation test with stat seed 20260907), 120 games (5 permutation cycles), 8 worker
  processes. Note: `serving_v2` is the registry-v2 search settings at 250 iterations in ONE
  worker; production runs 2 root workers × 125 iterations (same total, split trees).
- **Pre-registered decision rule:** PASS iff both pairs (gen140 > serving_v2) and
  (d016_250 > serving_v2) show a positive mean paired score difference with p < 0.01 over
  ≥ 120 games and no game errored. gen140 vs d016_250 is reported but NOT part of the gate
  (they may be equal). FAIL → the harness cannot see known differences at n = 120;
  investigate before any strength work (assessment §4 stop-loss).
- **Reproduce:** commit a394a79, `python -m mcts_lab.calibrate discrimination --games 120
  --workers 8 --seed 1712991924 --label discrimination_exp015`
- **Result (120/120 games, 0 errors): PASS.** Pre-registered pairs: gen140 − serving_v2 =
  +8.17 [90% CI +5.5, +10.8], p=0.0000; d016_250 − serving_v2 = +14.67 [90% CI +12.1, +17.3], p=0.0000.
  First place: d016_250 50.4%, gen140 35.4%, serving_v2 11.7%, greedy 2.5%; mean rank 1.68 /
  2.02 / 2.75 / 3.06. Per-move cost at 250 iterations under 8-way contention: d016 7.0 s,
  gen140 5.6 s, serving_v2 5.8 s.
- **Informative (not part of the gate):** at an EQUAL 250-iteration budget the D-016 config
  beats gen140 by 6.50 points (p = 0.0002; 90% CI +3.6 to +9.4) — the first
  sound-protocol evidence that the value-model leaf is stronger than rollouts at equal
  iterations (EXP-006a had parity at n = 20). And the served registry-v2 search settings are
  indistinguishable from the deterministic greedy baseline (+1.71 [90% CI -0.1, +3.5], p=0.1243):
  the configuration a human currently faces plays at one-ply-greedy level.
- **Interpretation:** the harness separates known-different agents by 8-15 points at p < 0.0001
  with n = 120; together with EXP-014b the M1 gates are met and protocol v3 is the measurement
  for all further work. D-016's edge over gen140 is a strength finding to carry into M3.
- **Artifacts:** `training/reports/protocol_v3/discrimination_exp015/{report.json,games.jsonl,run_config.json}`

## EXP-014b — M1 clone-calibration gate (protocol v3, all-permutation schedule)

- **Experiment ID:** EXP-014b
- **Date:** 2026-09-07 (pre-registered after the harness review; launched when EXP-014a finished)
- **Commit:** branch `feat/m1-measurement` (post-review harness: 24-permutation seat schedule,
  equivalence gate, incremental game records)
- **Hypothesis:** with seats AND successor relations balanced, a byte-identical clone of the
  champion is statistically equivalent to it.
- **Table:** [champion, champion_clone, greedy, random]; all MCTS at 250 pinned iterations, one
  worker; 240 games (10 cycles of the 24 permutations); 8 worker processes; fresh recorded seed.
- **Pre-registered decision rule:** PASS iff the 90% CI of the champion − clone paired score
  difference lies within ±4 points AND p > 0.05 AND no game errored/truncated AND n ≥ 240.
  Expected under an unbiased harness (sd ≈ 15-18): pass probability ≥ 93%; a 4-point bias passes
  ≤ 5%. FAIL → the harness still manufactures differences; diagnose with the per-permutation
  breakdown before any strength claim (assessment §4 stop-loss).
- **Reproduce:** commit fcbbf01, `python -m mcts_lab.calibrate clone --games 240 --workers 8
  --seed 707726340 --label clone_exp014b`
- **Result (240/240 games, 0 errors, 8 workers): PASS.** champion − clone = **+1.76 points**,
  90% CI **[−0.35, +3.87]**, p = 0.165. First place 46.5% / 38.8% / greedy 14.8% / random 0%;
  mean rank 1.73 / 1.82 / 2.28 / 3.80. Champion vs greedy +8.27 (p < 0.0001), clone vs greedy
  +6.51 (p < 0.0001), greedy vs random +22.1. Champion 5.9 s per move mean at 250 iterations
  under 8-way contention.
- **Caveat recorded:** the band was met with little margin (upper bound 3.87 vs 4.0), and the
  agent listed first scored higher in both clone runs (EXP-014a +0.78, EXP-014b +1.76; pooled
  340 games ≈ +1.5 ± 1.05). Not significant, but a residual bias of 1-2 points cannot be
  excluded. Consequence: any future strength claim that hinges on ≤ 2 points per game is not
  supported by this protocol; claims must clear the discrimination threshold (p < 0.01) with
  effects well above that band. If a later result sits in the 1-3 point range, re-run the clone
  gate with the agent order swapped before believing it.
- **Artifacts:** `training/reports/protocol_v3/clone_exp014b/{report.json,games.jsonl,run_config.json}`

## EXP-014a — clone contrast under the cyclic round-robin schedule (harness-bias measurement)

_Originally pre-registered as EXP-014 with the cyclic `round_robin` seat policy and a
"|Δ| < 3, p > 0.30" rule. The harness review (2026-09-07) showed that schedule keeps every
successor relation fixed (champion always followed by the clone, always preceded by random)
and that the rule fails an unbiased harness 30% of the time. The run was allowed to finish and
is recorded as a measurement of that confound; it is NOT the M1 gate._


- **Experiment ID:** EXP-014
- **Date:** 2026-09-07 (pre-registered before launch)
- **Commit:** branch `feat/m1-measurement` (on top of `fix/standard-piece-set`)
- **Hypothesis:** under protocol v3 an agent byte-identical to the champion, differing
  only in name (hence RNG stream and seat rotation phase), is statistically
  indistinguishable from the champion. The pre-rescue screen measured such a clone at
  −82 Elo / 9-10 head-to-head (assessment B5); this is the calibration test of the harness.
- **Independent variable:** none (champion vs its clone); 4-seat table
  [champion, champion_clone, greedy, random], all MCTS at 250 pinned iterations, one worker.
- **Controlled variables:** 100 games; cyclic round_robin seats; fresh run seed 1886604495.
- **Original decision rule (superseded, see above):** |mean paired score difference| < 3 AND
  p > 0.30 over ≥ 100 games.
- **Reproduce:** commit c12c73c, `python -m mcts_lab.calibrate clone --games 100 --workers 8
  --seed 1886604495`
- **Result (100/100 games, 8 workers, 61 min):** champion − clone = **+0.78 points, p = 0.68**
  (old rule: PASS). First place: champion 49.0%, clone 39.5%, greedy 11.5%, random 0%; mean
  rank 1.65 / 1.74 / 2.40 / 3.76. Champion vs greedy +10.24 (p < 0.0001); greedy vs random
  +19.6 (p < 0.0001). Champion at 250 pinned iterations: 6.1 s per move mean under 8-way
  contention. Paired-difference sd (for sizing EXP-014b): champion − clone **19.4** (se 1.94 at
  n = 100); champion − greedy 19.0; greedy − random 12.7; per-agent score sd 13.4 / 11.8 / 10.2 /
  7.0. Seat effect is real: champion mean score by seat 1→4 = 85.2 / 85.3 / 82.5 / 81.5. With
  sd 19.4 the EXP-014b design (n = 240, band ±4, 90% CI) passes an unbiased harness ≈ 88% of
  the time and a 4-point bias ≈ 5%. An early read at 22 games had shown the clone +7 points
  ahead — it regressed to +0.8 at 100, a reminder that n = 20 reads mean nothing.
- **Interpretation:** under the cyclic schedule no confound larger than ~3 points is visible at
  n = 100 (SE ≈ 1.5). That does not rescue the schedule — its successor relations are fixed by
  construction — but it bounds the effect. The M1 gate proper is EXP-014b (all permutations,
  equivalence rule, n = 240).
- **Artifacts:** `training/reports/protocol_v3/clone_exp014/{report.json,games.jsonl,run_config.json}`

## EXP-013 — Phase 6: prior-calibration fix (flattened MLP prior) vs baseline

- **Experiment ID:** EXP-013
- **Date:** 2026-07-16 (launched)
- **Commit:** PR #207 head (arena `policy_temperature` override)
- **Hypothesis:** EXP-012's loss is prior over-sharpness (diagnostic: top-move mass
  0.227 vs heuristic 0.071). Flattening the MLP prior with softmax temperature 3.0 —
  which matches the heuristic prior's entropy (0.978 vs 0.984) while KEEPING the
  EXP-011 ranking — recovers parity or better against the D-016 baseline.
- **Setup:** identical to EXP-012 (2×2 same-table, D-016 at 500 iters, seeds
  20260718/20260719, 20 games), single variable = `policy_temperature=3.0` on the
  prior agents (no retraining; artifact untouched, override applied at agent build).
- **Pre-registered decision rule:** flattened prior reaches parity or better (paired
  permutation p>0.05 for "worse", or positive) → prior-calibration confirmed as the
  EXP-012 cause; the scorer is salvageable and the next step is a proper c/temperature
  tuning + root-noise study. Still decisively worse → the prior's move CONTENT
  misleads search (not just its sharpness); policy work pauses per the master-plan
  gate and the encoding/target is the suspect.
- **Reproduce:** `python -m training.experiments.search_scaling
  --agents-json training/experiments/exp013_agents.json --games-per-seed 10
  --seeds 20260718,20260719 --deadline-minutes 420 --label exp013_softprior`
- **Result:** _pending_

## EXP-012 — Phase 6 search integration: MLP policy prior in the D-016 agent

- **Experiment ID:** EXP-012
- **Date:** 2026-07-16 (launched)
- **Commit:** PR #207 head (production wiring: `mcts/move_encoding.py`,
  `mcts/move_policy_mlp.py`, agent artifact dispatch, arena `policy_weights_path`)
- **Hypothesis:** the EXP-011 teacher-distilled MLP move policy, used as PUCT prior +
  move ordering (`policy_prior_enabled`, c=1.5) in the D-016 agent at the D-008 budget
  (500 iterations), beats the identical agent without the prior at the same table.
- **Setup:** 2×2 same-table (prior_500_a/b vs base_500_a/b), single variable = the
  prior; both sides D-016 exactly (minimal search + PW 2.0/0.5 + v1 ridge leaves,
  500 iterations pinned, num_workers 1). round_robin seats, seeds 20260718+20260719,
  10 games/seed = 20 games, protocol rescue_v2 via `search_scaling --agents-json`.
  Artifact: `training/artifacts/move_scorer/v2_mlp/move_policy_v2.json` (teacher-only,
  1,288 decisions, held-out reference 0.228/0.744 — gate-2 PASS config).
- **Pre-registered decision rule:** paired sign-flip permutation on per-game rank
  (prior-pair vs base-pair) + first-place split. Decisively positive → the scorer is
  validated in search; queue gate-C style adoption experiments + more teacher data.
  Null → the prior's ordering quality is insufficient at c=1.5; the next single
  variable is prior strength (c), NOT retraining. Negative → prior misleads search;
  investigate before any further policy work. No champion change either way.
- **Reproduce:** `python -m training.experiments.search_scaling
  --agents-json training/experiments/exp012_agents.json --games-per-seed 10
  --seeds 20260718,20260719 --deadline-minutes 420 --label exp012_mlp_prior`
- **Result — NEGATIVE (the prior HURTS search):** 20 games (seed 20260718 in one run;
  seed 20260719 re-run after a container restart — seeds pinned, so the re-run is
  identical). Prior agents 4 first-places / avg rank 2.80; base agents 16 / avg rank
  2.00. Paired per-game (prior-pair total − base-pair total) mean **−20.6 points**,
  exact sign-flip permutation **p=0.048** (two-sided) — decisively worse, not null.
- **Diagnostic (no games, 116 sampled teacher decisions) — mechanism identified:**
  the MLP prior is far SHARPER than the heuristic prior it replaced —
  normalized entropy 0.817 vs 0.984, top-move mass **0.227 vs 0.071** (3.2×). It was
  distilled from *completed* 500-iteration visit distributions (naturally peaked), so
  as a PUCT prior on a FRESH search it prematurely concentrates visits on the move a
  finished search would pick, starving the exploration that finds the actual best move
  (no root Dirichlet noise offsets this). The training-side win (EXP-011: 0.228/0.744
  ordering) is real; it just does not compose with an unexploring prior at c=1.5.
- **Decision (pre-registered "negative → investigate"):** the mechanism is a prior
  *calibration* problem, not a content problem — the single distinguishing follow-up
  is to FLATTEN the prior (artifact `temperature`, no retraining) toward the
  heuristic's entropy and re-run the same table (EXP-013). If a flattened prior
  reaches parity/positive, calibration is confirmed and the scorer is salvageable; if
  it still loses, the prior's move *content* misleads search and policy work pauses per
  the master-plan gate. Champion untouched; the MLP is NOT wired into any default
  config (opt-in via `policy_prior_enabled` + `policy_weights_path` only).
- **Artifacts:** `training/reports/experiments/search_scaling/exp012_mlp_prior/`
  (per-seed games + report.json).

## EXP-011 — Phase 6 build, step 2: shape-aware MLP move scorer (capacity remediation)

- **Experiment ID:** EXP-011
- **Date:** 2026-07-16 (launched)
- **Commit:** PR #207 head (`training/experiments/move_scorer_mlp.py`)
- **Hypothesis:** EXP-010's failure was capacity: a small MLP over a SHAPE-AWARE move
  encoding (piece one-hot + local board patch around the placement + the engineered
  scalars) can (1) nearly memorize 200 teacher decisions and (2) beat the fixed
  heuristic and legacy policy on held-out teacher decisions on BOTH tie-aware top-1
  and pairwise ordering.
- **Encoding (move_encoding_v1):** 9×9 patch centered on the placement centroid,
  6 channels (own occupied, opponent occupied, off-board, new-piece cells, own
  frontier, opponent frontier) = 486, + 21 piece one-hot + the 10 `move_features_v2`
  scalars + phase = 518 inputs per candidate.
- **Model:** numpy MLP 518→64(tanh)→1 shared across candidates, listwise softmax CE
  over each decision's children (same objective as EXP-010), Adam, fixed seed —
  numpy-only inference (Pyodide-safe, D-006/D-017).
- **Controls:** identical data, game-level split seed 20260716, identical held-out
  sets and baselines as EXP-010 (tie-aware top-1 throughout).
- **Pre-registered decision rule:** overfit gate = tie-aware top-1 ≥ 0.80 on its own
  200 training decisions (the linear model managed 0.165) — below that, capacity is
  still insufficient and the encoding (not the trainer) is the next suspect. Held-out
  bars identical to EXP-010: beat BOTH baselines on top-1 (outside ~2 SE) AND
  pairwise. Both pass → production wiring (`move_policy_v2`) + search-integration
  experiment. Overfit passes but held-out fails → capacity is fine, data volume is
  binding → scale teacher decisions (more 500-iter games) before revisiting.
- **Reproduce:** `python -m training.experiments.move_scorer_mlp --split-seed 20260716`
- **Result — gate 1 PASS (after optimization-budget correction), gate 2 PASS in the
  teacher-only condition; bulk mixing REFUTED for policy distillation:**
  1. **Capacity confirmed:** at the pre-registered 300 epochs the tiny-set score was
     0.780 (bar 0.80, within 1 SE); the distinguishing check (tiny set only, no
     held-out contact) shows 64/128/256 hidden units at 1000–2000 epochs reach
     **0.930–0.945 top-1 / 0.966–0.981 pairwise** — the encoding+MLP memorizes; the
     shortfall was optimization budget, not capacity.
  2. **Pre-registered mixed-data condition FAILED gate 2** (held-out teacher: 0.151 /
     0.637 vs heuristic 0.140 / 0.591) — and transferred WORSE than baselines on bulk
     held-out. Attribution check (identical model/config, training data as the only
     variable): **teacher-only (1,003 decisions) → 0.228 / 0.744; mixed (6,337) →
     0.151 / 0.637.** The PW-50 bulk corpus actively poisons the scorer — its visit
     distributions (50 iterations, ⌈2√50⌉≈14 children) are a different, noisier
     policy than the 500-iteration teacher's. Consistent with EXP-009's finding on the
     value side: volume of the wrong distribution is negative signal, now confirmed
     on the policy side with a controlled pair.
  3. **Teacher-only clears the pre-registered gate-2 bars decisively and robustly:**
     top-1 0.228/0.196/0.218 across seeds 12345/777/20260717 (baselines 0.140/0.133;
     every run > 2.3 SE clear), pairwise 0.742–0.744 (baselines 0.591/0.596).
- **Decision:** shape-aware MLP + teacher-only distillation is the first Phase 6
  candidate to clear the pre-arena training bars → proceed to production wiring
  (`move_policy_v2`: encoding + numpy inference in `mcts/`, artifact format,
  masking/round-trip tests) and the search-integration experiment (PUCT prior +
  ordering at fixed budgets vs the D-016 baseline). Bulk data is EXCLUDED from
  policy-distillation training sets; `value_dataset_v2` remains valid for state-value
  work and as game records. More 500-iter teacher games are the highest-value data
  spend (only 1,003 training decisions produced this).
- **Artifacts:** `training/artifacts/move_scorer/v2_mlp/{report.json,mlp_weights.json}`
  (pre-registered mixed run), attribution/seed checks logged here (scratchpad runs,
  scripts inline in the log entry's reproduce block lineage).

## EXP-010 — Phase 6 build, gate 1–2: teacher-distilled move scorer vs heuristic/legacy baselines

- **Experiment ID:** EXP-010
- **Date:** 2026-07-16 (launched)
- **Commit:** post-#206 main (branch restarted at `6a8b9f4`); harness
  `training/experiments/move_scorer.py` (this PR)
- **Hypothesis:** a listwise log-linear move scorer distilled from TEACHER root visit
  distributions, over an extended move-varying feature set (`move_features_v2`),
  orders candidate moves on held-out strong-teacher decisions decisively better than
  (a) the fixed 4-feature heuristic and (b) the legacy self-distilled policy artifact.
- **Independent variables:** feature set (mf4 = the four `MOVE_FEATURE_NAMES` vs
  mf_v2 = those + 6 move-varying extensions) trained identically; baselines evaluated
  untrained on the same decisions.
- **Data:** decisions from `teacher_dataset_v1` (500-iter teachers) +
  `value_dataset_v2` (PW-50 rollout teachers), children = recorded `search` entries,
  target = `policy_target`. Game-level split, seed 20260716: 25% of teacher_dataset_v1
  games held out (primary eval, strong distribution); 25% of value_dataset_v2 games
  held out (secondary).
- **Gate order (master plan Phase 6):** (1) tiny-data overfit sanity — trained on 200
  decisions, must clearly beat the fixed heuristic ON those decisions (pipeline/optim
  sanity); (2) held-out generalization vs the bars below. No production wiring, no
  arena spend before both pass.
- **Pre-registered decision rule (primary = held-out teacher decisions):**
  mf_v2 must beat BOTH baselines on top-1 agreement AND within-decision pairwise
  ordering (outside binomial noise), and beat trained mf4 (feature-set control).
  mf_v2 ≤ baselines → listwise-linear-on-these-features is refuted; move to
  shape-aware encodings / NN. mf_v2 > baselines but ≈ mf4 → wire mf4 (simpler),
  extensions rejected.
- **Reproduce:** `python -m training.experiments.move_scorer --split-seed 20260716`
- **Result (gate 1 PASS; gate 2 FAIL):** held-out teacher decisions (n=285), under the
  tie-aware top-1 metric (review fix, PR #207: predictions of any tied-max child count
  as correct — 174/1,288 teacher decisions have tied visit maxima; the correction
  shifted all top-1 values ≈+0.01 and changed no conclusion):
  | scorer | top-1 | pairwise |
  |---|---|---|
  | fixed_heuristic | 0.140 | 0.591 |
  | legacy_policy | 0.133 | 0.596 |
  | mf4_trained | 0.147 | 0.629 |
  | mf_v2_trained | 0.140 | **0.632** |
  Distillation lifts pairwise ordering (+0.04 over both baselines) but top-1 does not
  move, and mf_v2 ≈ mf4 (extensions add ~nothing, third strike for hand-crafted
  feature extensions after EXP-008/009). Bars required BOTH metrics → **FAIL**.
- **Attribution — capacity, not data:** the tiny-overfit sanity run scores only 0.165
  top-1 ON ITS OWN 200 TRAINING DECISIONS, and full-train held-out performance equals
  train performance. A listwise linear model over cheap geometric move features cannot
  even fit the teacher's choices, let alone generalize better. The 500-iteration
  teacher argmax depends on shape/interaction structure these features cannot express.
- **Decision (per the pre-registered rule):** listwise-linear-on-engineered-features is
  refuted → proceed to the shape-aware / higher-capacity scorer (D-017 revisit
  condition; D-006 permits sklearn MLP with numpy-exportable inference). The +0.04
  pairwise gain is NOT wired into production — top-1 (what PUCT priors and ordering
  actually consume at the argmax) did not improve, and §20 forbids shipping a lateral
  move as progress.
- **Artifacts:** `training/artifacts/move_scorer/v1/report.json` (all conditions,
  trained weights, per-set feature names).

## EXP-009 — Phase 6 path 1: rich_blokus_v2 at state-carrying volume vs the 0.68 bar

- **Experiment ID:** EXP-009
- **Date:** 2026-07-16 (launched)
- **Commit:** PR #206 head (`value_model_v2.py` --bulk volume conditions;
  `data/value_dataset_v2` finalized at `a17e76b`)
- **Hypothesis:** with ~12.4k state-carrying training rows (teacher_dataset_v1 +
  value_dataset_v2, all 51 features extractable) the v2 contested-territory block lifts
  held-out pairwise ordering decisively above 0.68 — the EXP-008 negative was a data-volume
  artifact, not a representation failure.
- **Independent variables:** feature set (51 vs 45) at equal rows (`volume_full` vs
  `volume_45`), and data mix (`volume_plus_v1_45` adds the 17.4k feature-only v1 rows).
- **Controls:** identical held-out teacher games as EXP-008 (split seed 20260716, 25% of
  teacher_dataset_v1 games); same ridge family; v1 ridge baseline re-evaluated on the same
  split. Bulk rows are TRAINING-only (PW-50 play is a weaker distribution than the
  500-iteration evaluation games).
- **Pre-registered decision rule:** pre-arena bar = best condition's held-out pairwise
  decisively > 0.68 (vs mixed_45's 0.678). Bar cleared → gate C arena re-run
  (exp007_agents.json pattern). Bar failed → move-level candidate-scoring evaluator (full
  Phase 6 build); no arena spend either way until the bar clears (§20: more data alone is
  not an escape — this run isolates feature-set contribution at equal volume, which is the
  distinguishing experiment EXP-008 called for).
- **Reproduce:** `python -m training.experiments.value_model_v2 --bulk data/value_dataset_v2
  --out training/artifacts/value_models/v4 --split-seed 20260716`
- **Result (held-out teacher games; teacher 5,236 rows / bulk 28,832 rows / v1 17,408 rows):**
  | condition | R² | pairwise |
  |---|---|---|
  | volume_full (teacher+bulk, 51 feats, ~33k rows) | 0.212 | 0.655 |
  | volume_45 (same rows, 45 feats) | 0.211 | 0.651 |
  | volume_plus_v1_45 (+17.4k v1 rows, ~50k) | 0.220 | 0.655 |
  | mixed_45 (teacher+v1 only — EXP-008 best) | 0.234 | **0.678** |
  | v1_ridge_baseline | −0.384 | 0.658 |
- **Verdict: NEGATIVE — bar not met.** Two findings:
  1. **The representation hypothesis is refuted, not data-starved:** at 6.5× the
     state-carrying volume, the v2 feature block still adds only +0.004 pairwise
     (0.655 vs 0.651) — the same margin as EXP-008. The contested-territory features
     do not carry ordering signal a linear model can use.
  2. **The bulk corpus does not substitute for value_dataset_v1:** every volume
     condition UNDERPERFORMS mixed_45 (0.655 vs 0.678), and adding bulk rows to the
     mixed data hurts. Plausible cause: value_dataset_v2 games open with τ=1.0 visit
     sampling (first 24 decisions), so early-state final scores are noisier labels than
     v1's greedy-argmax games. Volume of the wrong distribution is not more signal.
- **Decision (per the pre-registered rule):** the feature-extension path is exhausted —
  proceed to the **move-level candidate-scoring evaluator** (master plan §13, full
  Phase 6 build). The 0.68 pre-arena bar stands; zero arena compute was spent.
  mixed_45 (v3 artifact) remains the best evaluator; the pairwise ceiling of the
  current state-feature representation is confirmed at ~0.68.
- **Artifacts:** `training/artifacts/value_models/v4/report.json` (all conditions),
  `v4/value_mixed_45.joblib` (winning condition retrained; equivalent to v3's).

## EXP-008 — Phase 6 probe: contested-territory feature block (rich_blokus_v2) — NEGATIVE

- **Experiment ID:** EXP-008
- **Date:** 2026-07-16
- **Commit:** the Phase 6 PR head (feature block implemented in `training/rich_features.py`)
- **Hypothesis:** six contested/exclusive-territory features (who can realize their
  remaining piece area) lift held-out pairwise ordering decisively above the 0.68 ceiling.
- **Independent variable:** feature set (45 vs 51), model family (ridge, HistGB), on
  identical teacher-game splits (split seed 20260716, 25% held out).
- **Result (held-out teacher games):**
  | condition | R² | pairwise |
  |---|---|---|
  | teacher_only ridge 51 | 0.150 | 0.619 |
  | teacher_only ridge 45 | 0.169 | 0.615 |
  | mixed_45 ridge (17k v1 rows + teacher) | 0.234 | **0.678** |
  | teacher_only HistGB 51 / 45 | 0.008 / −0.015 | 0.612 / 0.603 |
- **Interpretation:** **the feature block adds ~nothing at current data volume**
  (+0.004 pairwise, linear; HistGB is data-starved on 5.2k rows). The mixed condition
  still leads purely on volume — but v1 rows carry no states, so they CANNOT receive the
  new features. Representation and state-carrying data volume are intertwined constraints.
- **Decision:** the pre-arena acceptance bar did its job — **zero arena compute spent** on
  an unproven evaluator. Features stay in (versioned rich_blokus_v2, append-only, needed
  for the follow-up). Ranked next options: (1) **generate a state-carrying bulk corpus**
  (teacher-recorder format at the cheap PW-50 budget, ~2 min/game — 100+ games ≈ 3.5 h)
  so mixed-volume training can use v2 features, then re-test the bar; (2) if that fails,
  the move-level candidate-scoring evaluator (full Phase 6 build).
- **Artifacts:** `training/artifacts/value_models/v3/report.json` (45-vs-51 conditions).

## EXP-007 — Phase 8 GATE C: vm2 (teacher-trained) vs vm1 (ridge) at equal budget

- **Experiment ID:** EXP-007
- **Date:** 2026-07-15 (launched ~13:3x UTC)
- **Commit:** the gate-C PR head (adds `training/experiments/value_model_v2.py`)
- **Hypothesis (the loop's first turn):** the value model retrained on teacher-search
  self-play (v2_mixed: teacher_dataset_v1 + value_dataset_v1) beats the v1 ridge as a leaf
  evaluator at equal budget — demonstrating better data → better evaluator → stronger search.
- **Pre-check (training half, held-out TEACHER games, split seed 20260716, 14/4 games):**
  | model | R² | MAE (pts) | pairwise |
  |---|---|---|---|
  | v2_mixed | **0.234** | **7.27** | **0.678** |
  | v2_teacher_only | 0.169 | 7.59 | 0.615 |
  | v1_ridge | **−0.384** | 11.82 | 0.658 |
  **PASS** — and a finding in itself: v1 generalizes badly to the stronger teacher-play
  distribution (negative R²); retraining on teacher data fixes the calibration.
- **Independent variable:** leaf artifact per seat (vm2_500 = v2_mixed retrained on all
  games; vm1_500 = the frozen v1 ridge). Identical D-016 search config, 500 iterations.
- **Controlled variables:** one mixed table [vm2_500, vm1_500, heuristic, random];
  round_robin; seeds 20260620/20260621; 10 games/seed; standard scoring; stat seed 20260712.
- **Game count:** 20 (deadline 280 min). **Hardware:** session container.
- **Result:** 20/20 games in 162 min.
  | agent | 1st% | avg rank | TS μ (σ) |
  |---|---|---|---|
  | vm2_500 | **57.5%** | **1.35** | 42.32 (7.82) |
  | vm1_500 | 42.5% | 1.50 | 42.88 (7.85) |
  | heuristic | 0.0% | 2.70 | 20.58 (7.52) |
  | random | 0.0% | 3.70 | −4.56 (7.62) |
  Primary test: vm2 − vm1 = **+2.25 pts, p=0.596**. Both crush the anchors (p < 0.0001).
- **Uncertainty:** 20 paired games; the first-place/rank direction favors vm2 consistently,
  but the score-diff test is far from significance.
- **Interpretation:** **the loop turns, but the per-generation gain is small.** Training half
  demonstrated the mechanism decisively (v1 miscalibrated on teacher play, v2 fixes it);
  arena half shows no regression and a positive trend — yet ordering quality (what argmax
  move choice actually uses) barely moved (pairwise 0.658 → 0.678), consistent with the
  EXP-004 finding that the 45-feature representation caps at ~0.68. One generation of
  18 teacher games cannot push past a feature ceiling.
- **Decision:** **Gate C: PARTIAL — MORE EVIDENCE REQUIRED.** Per §20 (no default escapes:
  not "more games", not "more data" first), the isolated bottleneck is the REPRESENTATION —
  the next distinguishing work is the Phase 6 upgrade (richer feature set or move-level
  candidate scoring), after which gate C re-runs with a real ordering-quality delta to
  detect. v2_mixed is retained as the current best evaluator artifact (no regression,
  better calibrated); NOT promoted anywhere.
- **Artifacts:** `training/reports/experiments/search_scaling/exp007_vm2_vs_vm1/`,
  `training/artifacts/value_models/v2/`.

## EXP-006a — Phase 4 confirmation 1: direct same-table model-500 vs rollout-500

- **Experiment ID:** EXP-006a
- **Date:** 2026-07-14 (launched ~00:4x UTC)
- **Commit:** the EXP-006 PR head (harness gains `--agents-json` mixed-table mode)
- **Hypothesis:** at equal budget (500 iterations, identical PW/search config), the
  ridge-model leaf agent beats the rollout leaf agent head-to-head in the SAME games —
  the clean version of gate-3 criterion (a), no cross-condition caveats.
- **Independent variable:** leaf source per seat (vm500 = `value_model_path` ridge artifact;
  rollout500 = greedy_sample rollouts, cutoff 12 — exact EXP-002 config).
- **Controlled variables:** one mixed table [vm500, rollout500, heuristic, random];
  round_robin; seeds 20260620/20260621; 10 games/seed; standard scoring; stat seed 20260712.
- **Game count:** 20 (deadline 260 min). **Hardware:** session container (concurrent with
  EXP-006b — iteration budgets, so contention affects wall-clock only).
- **Result:** 20/20 games in 180.7 min.
  | agent | 1st% | avg rank | TS μ (σ) |
  |---|---|---|---|
  | vm500 | 47.5% | **1.45** | 42.50 (7.80) |
  | rollout500 | 42.5% | 1.60 | 39.39 (7.82) |
  | heuristic | 10.0% | 2.50 | 21.89 (7.56) |
  | random | 0.0% | 3.70 | −2.92 (7.64) |
  **vm500 − rollout500 = +2.40 pts, p=0.556** (primary test). Both crush the anchors
  (vm +23.8 vs heuristic p<0.0001; rollout +21.4 p=0.0002).
- **Uncertainty:** 20 paired games; the primary comparison is far from significant.
- **Interpretation:** **parity at equal budget, slight positive trend for model leaves** —
  NOT the "decisively better at 500" the EXP-005 cross-condition anchor margins suggested
  (that comparison overstated; the anchor faces different opposition per condition). The
  honest criterion-(a) verdict: model leaves EQUAL rollout leaves in strength at 500 iters
  today, while being deterministic, TT-cacheable, and — unlike rollouts — improvable
  through the training loop.
- **Decision:** criterion (a) as originally worded ("beats rollouts at equal budget") is
  NOT met at n=20; parity is. The architectural argument for model leaves survives on
  improvability, not present superiority. Verdict rolls into the combined Phase 4 update
  with EXP-006b.
- **Artifacts:** `training/reports/experiments/search_scaling/exp006a_vm_vs_rollout/`.

## EXP-006b — Phase 4 confirmation 2: model-leaf saturation ladder (150/500/1500)

- **Experiment ID:** EXP-006b
- **Date:** 2026-07-14 (launched ~00:4x UTC)
- **Commit:** the EXP-006 PR head
- **Hypothesis:** model-leaf scaling continues past 500 (1500 > 500), or saturates —
  locating the knee sets the teacher budget (D-008).
- **Independent variable:** iteration budget (150/500/1500), all with PW + ridge leaves.
- **Controlled variables:** heuristic anchor; round_robin; seeds 20260620/20260621;
  6 games/seed; standard scoring; stat seed 20260712. (Fewer games — the 1500 rung costs
  ~2× an entire EXP-005 game; deadline 330 min, partials analyzable via --reanalyze.)
- **Game count:** 12 requested (deadline-capped).
- **Result:** 12/12 games in 247.8 min.
  | agent | 1st% | avg rank | TS μ (σ) |
  |---|---|---|---|
  | mcts_it150 | 8.3% | 2.50 | 23.96 (7.83) |
  | mcts_it500 | 50.0% | **1.50** | 35.77 (8.00) |
  | mcts_it1500 | 41.7% | 1.67 | 35.55 (8.01) |
  | heuristic | 0.0% | 3.75 | 5.18 (7.83) |
  Paired: **it500 − it150 = +14.08 pts, p=0.013** (first conventionally significant budget
  pair of the investigation); it1500 − it150 = +10.50 (p=0.063); it1500 ≈ it500 (+3.58 for
  500, p=0.54). All rungs beat the anchor (p ≤ 0.0013).
- **Uncertainty:** 12 games — yet the 150→500 step clears p<0.05; direction consistent
  across both 006b pairs and with EXP-005's monotonic ranks.
- **Interpretation:** **positive scaling confirmed with significance over 150→500; knee at
  ~500** (1500 adds nothing at this evaluator quality — the saturation point is where
  evaluator improvements, not budget, buy strength next).
- **Decision:** combined with EXP-005 and EXP-006a → **Phase 4 gate PASS** for the
  PW + model-leaf configuration (addendum 4); **teacher budget = 500 iterations (D-008)**;
  PW + ridge-model leaves adopted as the experimental baseline config (D-016) on
  scaling + improvability grounds (rollouts: parity today, flat scaling, not trainable).
- **Artifacts:** `training/reports/experiments/search_scaling/pw_vm_b150_500_1500/`.

## EXP-005 — D-015 gate 3: acceptance ladder with the ridge value model as leaf evaluator

- **Experiment ID:** EXP-005
- **Date:** 2026-07-13 (launched ~19:5x UTC)
- **Commit:** the gate-3 PR head (adds `mcts/value_model_evaluator.py`, `value_model_path`
  plumbing through MCTSAgent/build_agent/parallel config, `--value-model` ladder flag)
- **Hypothesis (the decisive one):** the ridge value model (pairwise rank acc 0.682,
  in-tree Q-spread 4.6–6.3 pts) as the leaf evaluator (a) beats the EXP-002 rollout
  baseline at equal budgets and (b) produces positive scaling. PASS reopens the Phase 4
  gate path; FAIL closes the state-value-on-45-features family (D-015 consequence clause).
- **Independent variable (vs EXP-002/003):** budget agents' leaf source =
  `value_model_path=training/artifacts/value_models/v1/value_v1_ridge_baseline.joblib`
  (replaces rollouts via the rich-leaf slot; deterministic).
- **Controlled variables:** everything else identical to EXP-002/003 (PW pw_c=2.0 α=0.5,
  budgets 50/150/500, heuristic anchor, round_robin, seeds 20260620/20260621, 12 games/seed,
  standard scoring, stat seed 20260712) — results pool across the three conditions.
- **Agents:** mcts_it50/150/500 (PW + ridge-model leaves), heuristic.
- **Game count:** 24 (deadline 170 min). **Seat-balancing:** round_robin.
- **Hardware:** session container (single process, num_workers=1).
- **Result:** 24/24 games in 171 min (427 s/game — model leaves cost more than rollouts;
  4-player 45-feature extraction dominates).
  | agent | 1st% | Wilson 95% | avg rank | TS μ (σ) | vs heuristic (paired) |
  |---|---|---|---|---|---|
  | mcts_it50 | 16.0% | [0.06, 0.35] | 2.12 | 30.41 (7.57) | +15.04 (p=0.0001) |
  | mcts_it150 | 45.1% | [0.27, 0.64] | 1.83 | 37.00 (7.66) | +19.88 (p<0.0001) |
  | mcts_it500 | 38.9% | [0.22, 0.59] | 1.75 | 31.52 (7.60) | +20.75 (p<0.0001) |
  | heuristic | 0.0% | [0.00, 0.14] | 3.38 | 1.46 (7.50) | — |
  Budget pairs: it500−it50 = **+5.71 (p=0.054)**; it150−it50 = +4.83 (p=0.29);
  it150−it500 = +0.88 (p=0.86).
- **Uncertainty:** 24 games; the 10× pair is borderline-significant; cross-condition
  anchor-margin comparison shares seeds/protocol but the anchor faces different budget
  opponents per condition (directional, not exact).
- **Interpretation:** **the first positive scaling signal of the entire investigation.**
  (b) Scaling: avg rank improves monotonically with budget (2.12 → 1.83 → 1.75); the 10×
  pair reaches p=0.054; strength rises 50→150 then saturates by 500 at this evaluator
  quality. (a) vs rollout leaves: anchor margins exceed EXP-002 at every rung
  (+15.0/+19.9/+20.8 vs +14.75/+16.38/+11.04), decisively at 500 — and the anchor is
  shut out entirely (0% first place). A 0.68-pairwise evaluator already outperforms
  rollouts as the leaf source AND converts budget into strength at the low end.
- **Decision:** **Gate 3: PARTIAL — MORE EVIDENCE REQUIRED** (per the report vocabulary):
  direction validated, two confirmations outstanding before Phase 4 can be declared
  PASSED: (1) EXP-006a — a DIRECT same-table equal-budget test: [model-500, rollout-500
  (exact EXP-002 config), heuristic, random], round_robin, same seeds — the clean (a)
  criterion; (2) EXP-006b — a 150/500/1500 model-leaf rung to locate the saturation point
  and the teacher budget (D-008). The 45-feature family is NOT closed — D-015's
  consequence clause is superseded by this result; feature/representation upgrades become
  an optimization axis rather than a prerequisite.
- **Artifacts:** `training/reports/experiments/search_scaling/pw_vm_b50_150_500/report.json`
  (+ per-seed run dirs).

## EXP-004 — Evaluator track v1: model training + D-015 gates 1–2

- **Experiment ID:** EXP-004
- **Date:** 2026-07-13 (early UTC)
- **Commit:** the evaluator-v1 PR head (harness `training/experiments/value_model.py`)
- **Hypothesis:** a non-linear model (GBM/MLP) on the 45 `rich_blokus_v1` features beats the
  linear baseline at predicting standard final scores (D-015 gate 1), and its in-tree root
  Q-spread far exceeds Layer-6's measured 1.3-point flatness (gate 2).
- **Independent variable:** model family (ridge baseline vs HistGB vs MLP), same features,
  same data.
- **Controlled variables:** dataset `data/value_dataset_v1` (17 408 rows / 60 games,
  standard-scored PW-50 teacher self-play); GAME-level 80/20 held-out split (48/12 games,
  split seed 20260713); target = final_score/100; identical probe protocol to EXP-003's
  pre-probe (PW, 500 iterations, seeds/plies matched).
- **Result:**
  | model | held-out R² | MAE (pts) | pairwise rank accuracy |
  |---|---|---|---|
  | ridge (linear baseline) | **0.264** | **5.61** | **0.682** |
  | hist_gb | 0.246 | 5.66 | 0.680 |
  | mlp | 0.221 | 5.87 | 0.664 |
  **Gate 1: FAIL** — no non-linear lift on any metric. Gate 2 probe (hist_gb as leaf
  evaluator): root Q-spread **6.34 / 4.57** points at ply 8/24 (vs Layer-6's ~1.3), best-child
  share 23%, depth 5 — the bar "substantially exceed the flatness" is met, though top-3 Q at
  ply 24 remain within 0.06.
- **Uncertainty:** single split (12 held-out games); metric ordering consistent across all
  three metrics; gate-1 differences are small but uniformly non-positive for non-linear.
- **Interpretation:** **capacity is not the bottleneck — the 45-feature representation is.**
  The features carry genuine ordering signal (0.68 pairwise ≫ 0.5 chance) but saturate at
  R² ≈ 0.26 regardless of model family. This explains why historical `rich_leaf`/TD linear
  refits plateaued: they were already at the feature ceiling. Whether 0.68-pairwise value
  quality is enough to beat rollout leaves in actual play is exactly gate 3.
- **Decision:** run gate 3 (the fixed Phase 4 acceptance ladder) with the **ridge** model
  (best on every metric; trivially numpy-servable) before designing the Phase 6 move-level /
  richer-representation evaluator — the ladder verdict calibrates any future design either
  way. D-015 consequence clause is armed: if gate 3 fails, the state-value-on-45-features
  family is closed.
- **Artifacts:** `training/artifacts/value_models/v1/report.json`,
  `value_v1_hist_gb.joblib` (fitted artifact incl. dataset manifest + split + metrics).

## EXP-003 — Phase 4 remediation: value-signal isolation (PW + pure static-eval leaves)

- **Experiment ID:** EXP-003
- **Date:** 2026-07-12 (launched ~16:45 UTC)
- **Commit:** the EXP-003 PR head (base `859b163`; harness gains `--cutoff`)
- **Hypothesis (discriminating):** replacing noisy rollout values with deterministic
  static-eval leaves (`rollout_cutoff_depth: 0`) either (A) restores positive scaling —
  rollout NOISE was the blocker, confirming the Phase 5 direction — or (B) leaves the curve
  flat — the Layer-6 evaluator itself adds nothing beyond the ordering prior, exonerating
  search mechanics and pivoting the rescue to leaf-evaluation quality.
- **Independent variable (vs EXP-002):** `rollout_cutoff_depth: 0` on the budget agents.
  Everything else identical (PW pw_c=2.0 α=0.5, budgets 50/150/500, heuristic anchor,
  round_robin, seeds 20260620/20260621, 12 games/seed, standard scoring, stat seed).
- **Pre-probe:** cutoff-0 costs ~9–11 ms/it early/mid, ~3.4 ms/it late (2–4× cheaper than
  rollouts). **Static-eval Q is nearly flat across root moves**: top-3 children within
  ~0.2 pts, full spread ~1.3 on a ~47 scale; best-child visit share drops to 5–9% at
  bf-325/385 (vs 23% with rollout values); depth still grows with budget (3→4 midgame,
  4→8 at bf 7). Early lean toward outcome B, arena decides.
- **Agents:** mcts_it50, mcts_it150, mcts_it500 (all PW + cutoff 0), heuristic.
- **Game count:** 12 games/seed × 2 seeds = 24 (deadline 170 min; est. ~1 h).
- **Seat-balancing method:** round_robin. **Seeds:** 20260620, 20260621.
- **Hardware:** session container (single process, num_workers=1).
- **Result:** 24/24 games in 55.5 min (139 s/game). **Outcome B, decisively.**
  | agent | 1st% | Wilson 95% | avg rank | TS μ (σ) | vs heuristic (paired) |
  |---|---|---|---|---|---|
  | heuristic | 41.7% | [0.24, 0.61] | 2.12 | 30.83 (7.64) | — |
  | mcts_it50 | 37.5% | [0.21, 0.57] | 2.21 | 28.26 (7.59) | −0.58 (p=0.78) |
  | mcts_it150 | 12.5% | [0.04, 0.31] | 2.42 | 25.60 (7.52) | +0.21 (p=0.94) |
  | mcts_it500 | 8.3% | [0.02, 0.26] | 2.83 | 15.35 (7.46) | −2.00 (p=0.48) |
  Budget pairs all flat (p=0.46–0.75).
- **Uncertainty:** Wilson 95% CIs; 10k-permutation sign-flip tests; 24 games — but the
  cross-experiment contrast is the finding, and it is large: EXP-002 (identical except
  rollout leaves) had every rung +11..+16 pts over the anchor at p ≤ 0.0006; EXP-003's
  advantage is zero.
- **Interpretation:** **the Layer-6 static evaluator contributes nothing beyond the
  move-ordering prior** — replacing rollouts with static-eval leaves erased the ENTIRE
  strength margin over the heuristic anchor (consistent with the pre-probe's ~1.3-point Q
  spread on a ~47 scale). Conversely, the rollouts carry all of the current value signal:
  informative (EXP-002's anchor margins) but too noisy/saturating to convert extra visits
  into strength (flat scaling in EXP-001/002). Combined attribution across EXP-001/002/003:
  tree shape — fixed by PW; value signal — rollouts = all signal + noise-limited; static
  evaluator = uninformative. **The strength ceiling is leaf-evaluation quality.**
- **Decision:** Phase 4 gate: **FAIL — ATTRIBUTED** (stays open). Search mechanics are
  exonerated; the rescue pivots to Phase 5 (formal equal-time leaf-evaluation comparison is
  effectively done: rollouts ≻ static eval) and Phase 6 (build a leaf evaluator that actually
  discriminates — the candidate-scoring/value-vector model), then RE-RUN this scaling ladder
  as the acceptance test for any new evaluator. PW remains un-adopted until that re-run
  passes.
- **Artifacts:** `training/reports/experiments/search_scaling/pw_c0_b50_150_500/report.json`
  (+ per-seed run dirs).

## EXP-002 — Phase 4 remediation: progressive-widening ladder (one variable vs EXP-001)

- **Experiment ID:** EXP-002
- **Date:** 2026-07-12 (launched ~08:15 UTC)
- **Commit:** the EXP-002 PR head (base `d2ea3ff`; harness gains `--pw`)
- **Hypothesis:** progressive widening (pw_c=2.0, α=0.5) restores positive scaling —
  mcts_it500 > mcts_it150 > mcts_it50 — by concentrating visits on top-ordered moves
  instead of the one-visit-per-child sweep that made EXP-001 fail.
- **Independent variable (vs EXP-001):** `progressive_widening_enabled: true, pw_c: 2.0,
  pw_alpha: 0.5` on the three budget agents. EVERYTHING else identical to EXP-001
  (same budgets, seeds, seat policy, scoring, table composition, game counts, stat seed).
- **Pre-probe (mechanism check before spending arena time):** with PW at bf-325/385
  positions — 500 iters: 44 expanded children (vs all 325–385), best child 116 visits /
  23% share (vs 3 visits / 0.6%), depth 4 (vs 2); concentration grows with budget.
- **Agents:** mcts_it50, mcts_it150, mcts_it500 (all +PW), heuristic (anchor).
- **Game count:** 12 games/seed × 2 seeds = 24 (deadline 170 min).
- **Seat-balancing method:** round_robin. **Seeds:** 20260620, 20260621.
- **Hardware:** session container (single process, num_workers=1).
- **Result:** 24/24 games in 155.2 min (session container was suspended mid-run; all games
  completed).
  | agent | 1st% | Wilson 95% | avg rank | TS μ (σ) | EXP-001 comparison |
  |---|---|---|---|---|---|
  | mcts_it50 | 33.3% | [0.18, 0.53] | 2.00 | 34.83 (7.62) | ≈ unchanged |
  | mcts_it150 | 31.2% | [0.16, 0.51] | 2.04 | 33.24 (7.61) | ≈ unchanged |
  | mcts_it500 | 31.2% | [0.16, 0.51] | 2.38 | 24.22 (7.58) | **up from 22.2% / μ 14.5** |
  | heuristic | 4.2% | [0.01, 0.20] | 3.38 | 7.84 (7.48) | down from 8.3% |
  Paired permutation: it50−it150 = −1.62 (p=0.66); it50−it500 = +3.71 (p=0.42);
  it150−it500 = +5.33 (p=0.30); vs heuristic: it50 **+14.75 (p=0.0002)**, it150 **+16.38
  (p<0.0001)**, it500 **+11.04 (p=0.0006)** — every rung now beats the anchor decisively
  (EXP-001's it500 was p=0.22).
- **Uncertainty:** Wilson 95% CIs; 10k-permutation sign-flip tests; 24 games/condition —
  budget-vs-budget differences remain within noise; anchor comparisons are decisive.
- **Interpretation:** progressive widening **fixed the tree-shape pathology and raised
  absolute strength** (biggest gain exactly where EXP-001 was worst, the 500 rung; the
  pre-probe's concentration/depth predictions held), **but the scaling curve is still flat**:
  50 ≈ 150 ≈ 500, with it500 still trending slightly behind. With concentration no longer the
  bottleneck, the remaining suspect is the VALUE SIGNAL — greedy_sample rollout deltas /
  tanh×100 static-eval mix appears to add no reliable information beyond the move-ordering
  prior at these budgets, so extra visits refine noise.
- **Decision:** Phase 4 gate remains **FAIL/OPEN**. PW is necessary-but-insufficient;
  adopting PW into the minimal config is deferred until a scaling experiment passes (no
  config churn without a passed gate). Next distinguishing experiments (one variable each):
  (a) value-signal probe — same PW ladder with pure static-eval leaves (`rollout_cutoff_depth
  0`) to isolate rollout noise vs evaluator quality; (b) a 1500-iteration PW rung to test
  whether scaling appears beyond 500; (c) exploration-constant / visit-floor robustness.
- **Artifacts:** `training/reports/experiments/search_scaling/pw_b50_150_500/report.json`
  (+ per-seed run dirs), pre-probe numbers above.

## EXP-001 — Phase 4 search-scaling gate, primary ladder (50/150/500 + heuristic anchor)

- **Experiment ID:** EXP-001
- **Date:** 2026-07-12 (launched ~06:05 UTC)
- **Commit:** the Phase 4 PR head (contains `training/experiments/search_scaling.py`; base `b6b8c1f`)
- **Hypothesis:** with the Phase 3 reward-baseline fix in place, more search iterations
  produce stronger play: mcts_it500 > mcts_it150 > mcts_it50 (> heuristic) in first-place
  rate / avg rank / paired score difference.
- **Independent variable:** exact iteration budget (50, 150, 500), pinned via `iterations`
  param (thinking_time_ms None — build_agent cannot rewrite it).
- **Controlled variables:** identical minimal search config for all budget agents
  (greedy_sample K=12, cutoff 12, heuristic ordering, C=1.414, TT on, 1 worker, maxⁿ,
  root reward baseline); engine @ standard scoring; mixed 4-seat table; round_robin seats;
  max_turns 2500; scoring_mode standard.
- **Agents:** mcts_it50, mcts_it150, mcts_it500, heuristic (anchor).
- **Game count:** 12 games/seed × 2 seeds = 24 (deadline-capped at 170 min).
- **Seat-balancing method:** round_robin (rotates each game).
- **Seeds:** 20260620, 20260621 (game seeds derived via SHA256 per arena_runner).
- **Hardware:** session container (Linux, single process, num_workers=1).
- **Result:** 24/24 games completed in 109.5 min (274 s/game).
  | agent | 1st% | Wilson 95% | avg rank | TS μ (σ) |
  |---|---|---|---|---|
  | mcts_it50 | 34.7% | [0.19, 0.55] | 1.96 | 34.62 (7.61) |
  | mcts_it150 | 34.7% | [0.19, 0.55] | 2.00 | 35.69 (7.68) |
  | mcts_it500 | 22.2% | [0.10, 0.42] | 2.75 | 14.49 (7.51) |
  | heuristic | 8.3% | [0.02, 0.26] | 2.96 | 15.59 (7.44) |
  Paired permutation (score diff): it50−it150 = −1.79 (p=0.585); it50−it500 = **+4.08**
  (p=0.276); it150−it500 = **+5.88** (p=0.149); it50−heuristic = +7.58 (**p=0.0007**);
  it150−heuristic = +9.38 (**p=0.004**); it500−heuristic = +3.50 (p=0.220).
- **Uncertainty:** Wilson 95% CIs; 10k-permutation sign-flip tests; TrueSkill μ/σ; 24 paired
  games — the 500-worse-than-50/150 direction is a consistent point estimate, not
  individually significant.
- **Interpretation:** **no positive scaling over a 10× budget span; 500 iterations trends
  WORSE than 50/150.** Mechanistic probe (root statistics at bf 325/385 positions): at 50/150
  iterations every expanded child has exactly 1 visit → max-visits selection ties → the agent
  plays the move-ordering heuristic's first-expanded (top) move; at 500 iterations all ~325–385
  children expand and revisits (best child 3–7 visits) follow single-rollout noise, overriding
  the ordering. Below the branching factor the "search" is the ordering heuristic; above it,
  it follows noise. Search does beat the raw HeuristicAgent anchor at low budgets (the
  ordering heuristic + occasional rollout signal is stronger than HeuristicAgent's own move
  scoring), but compute does not convert to strength. Fully consistent with the training
  plateau: nightly evaluation budgets (50–250 iters) cannot express evaluator improvements.
- **Decision:** **Phase 4 gate FAIL** (`phases/PHASE_4_SEARCH_SCALING.md`). No learning-phase
  work (Phases 5–8) until scaling is repaired. Next distinguishing experiment: EXP-002 — the
  same ladder with progressive widening enabled (already implemented, off by default; the
  teacher profile uses pw_c=2.0, α=0.5), which restricts expansion so visits concentrate.
- **Artifacts:** `training/reports/experiments/search_scaling/b50_150_500/report.json`
  (+ per-seed run dirs with games.jsonl); mechanistic probe numbers in the phase report.

## EXP-000 — Rescue baseline snapshot (no new games played)

- **Experiment ID:** EXP-000
- **Date:** 2026-07-12
- **Commit:** `cabe2dd7738daca661798d422ee487179640e34f`
- **Hypothesis:** n/a — reference snapshot of the frozen system's last recorded performance.
- **Independent variable:** none.
- **Controlled variables:** all values read from committed durable state
  (`training/state/latest.json`, `training/status.md`, run `20260711T190001Z`).
- **Agents:** champion gen140 vs benchmark_v2 pool + candidates rich_leaf / heuristic_tune /
  mcts_sweep.
- **Game count:** cumulative 6 290 (generation 179); last run: 48–56 paired games/candidate.
- **Seat-balancing method:** round_robin (SPRT sequential screen).
- **Seeds:** 20260620, 20260621.
- **Hardware:** GitHub-hosted 2-core runner (nightly workflow).
- **Result:** champion gen140 Elo 1388.55, TrueSkill μ 54.39 σ 5.02 (conservative 39.33). Last
  run candidates all HELD, SPRT inconclusive, all negative vs champion: rich_leaf 56 games,
  39% win vs champ, ΔElo −60.1, Δμ −9.62; heuristic_tune 48 games, ΔElo −117.3, Δμ −9.71;
  mcts_sweep 48 games, ΔElo −108.4, Δμ −10.54. Best historical Elo 1418.1 (gap −29.5, within
  documented rating noise σ≈±42.5). 39 generations without promotion.
- **Uncertainty:** rating noise at nightly game counts documented at ±42–72 Elo; SPRT verdicts
  inconclusive (neither H0 nor H1).
- **Interpretation:** the plateau is real at the current approach family's effect size:
  candidates are consistently *weaker* than the champion, not undetectably better. Supports
  risk ranking #2 (candidate generation exhausted) and motivates Phases 4–8 rather than more
  nightly iterations.
- **Decision:** freeze (Phase 0); proceed per `MASTER_PLAN.md`.
- **Artifacts:** `training/state/latest.json` (`last_approach_comparison`),
  `training/status.md`, `training/reports/approach_comparison.md`, hashes in `DATA_LINEAGE.md`.
