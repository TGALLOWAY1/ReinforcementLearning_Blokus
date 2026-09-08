# Decision Records

Format per governing master prompt §21. Statuses: Proposed / Accepted / Superseded.

---

## D-001 — Repository strategy: stay in the current repo

- **Date:** 2026-07-12
- **Status:** Accepted
- **Context:** The master prompt allows a rewrite or a new agent-focused repo. The July 2026
  cleanup already reduced the lab to one engine and one MCTS implementation.
- **Options considered:** (a) new `agent-core` repo inheriting validated pieces; (b) stay here
  with `docs/agent-strength-rebuild/` as the governing layer.
- **Evidence:** single engine/search implementations with real differential + regression tests
  (`tests/test_move_generation_equivalence.py`, `test_legality_bitboard_equivalence.py`,
  `test_maxn_backprop.py`); UI coupling is one-directional (webapi/frontend import core, never
  the reverse); evaluation infra (TrueSkill/SPRT/gate) is the strongest subsystem and would have
  to be ported wholesale; no neural training stack exists yet to isolate.
- **Decision:** stay in `MCTS_Laboratory`. Legacy experiment surfaces are archived-in-place via
  `AUDIT_INVENTORY.md` classifications rather than physically moved.
- **Consequences:** less migration risk and immediate reuse of eval infra; discipline required
  to keep new datasets/models from silently mixing with legacy ones (see `DATA_LINEAGE.md`).
- **Revisit conditions:** Phase 6 model work makes the Pyodide/serving coupling or repo weight a
  concrete obstacle; or the archive surfaces materially slow CI/test cycles.
- **Related:** `MASTER_PLAN.md` §3, `AUDIT_INVENTORY.md`.

## D-002 — Target ruleset: standard Blokus scoring

- **Date:** 2026-07-12
- **Status:** Accepted (explicit user decision, session 1)
- **Context:** Engine default is the non-standard "house" mode (+5/controlled corner,
  +2/center square; `engine/game.py:25-38`). Standard Blokus is coverage + 15 all-pieces bonus
  + 5 monomino-last. The monomino bonus is **not implemented** in `Board.get_score`
  (`engine/board.py:562`), although `AUDIT_REPORT.md` §3.8 claims scoring was verified.
  The project goal is beating a skilled human, who plays standard rules.
- **Options considered:** standard (implement bonus, retarget); keep house; defer to Phase 2
  experiments.
- **Decision:** **Standard Blokus scoring** is the training/evaluation objective. Phase 2
  implements the monomino-last bonus and makes `SCORING_MODE_STANDARD` the default for all new
  evaluation. House mode remains available for historical comparability only.
- **Implemented:** 2026-07-12 (Phase 2 task 1) — `Board.player_last_piece` + the +5 bonus in
  `Board.get_score`; `BlokusGame`, arena `RunConfig`, TD self-play, browser worker, and API
  schema defaults flipped to standard; protocol bumped to `rescue_v2`. Tests:
  `tests/test_standard_scoring.py`.
- **Consequences:** historical arena results (house-scored) are not directly comparable to
  post-Phase-2 results; benchmark baselines must be re-anchored under standard scoring
  (recorded in `BENCHMARK_PROTOCOL.md` as a protocol version bump when it happens).
- **Revisit conditions:** none anticipated; a product decision to ship house-rules play would
  add a secondary objective, not replace this one.
- **Related:** Phase 2 in `MASTER_PLAN.md`; risk #3 in the Phase 1 report.

## D-003 — Phase 0 freeze method: drop the cron, keep workflow_dispatch

- **Date:** 2026-07-12
- **Status:** Accepted
- **Context:** `.github/workflows/nightly-mcts-training.yml` runs every 6 h with
  `contents: write`, appending corpora and refreshing ratings — exactly the uncontrolled data
  generation Phase 0 must stop.
- **Options considered:** delete the workflow; disable via repo UI (not reproducible in-code);
  remove `schedule:` and keep `workflow_dispatch`.
- **Decision:** remove the `schedule:` trigger, keep `workflow_dispatch` so deliberate,
  attended runs remain possible with the full input surface.
- **Consequences:** **GitHub reads cron schedules from the default branch — the freeze takes
  effect only when this branch merges to `main`.** Until then, 6-hourly runs continue and may
  append to corpora/ratings; the asset hashes in `DATA_LINEAGE.md` are pinned to the baseline
  commit, so later drift is detectable.
- **Revisit conditions:** the Phase 7+ pipeline earns back a scheduled run under the new gates.
- **Related:** `phases/PHASE_0_FREEZE.md`.

## D-004 — Branch naming: harness-designated branch

- **Date:** 2026-07-12
- **Status:** Accepted
- **Context:** The generic master prompt suggests `agent-rescue/phase-*` branches; the execution
  environment designates `claude/agent-rescue-master-plan-i9kvwf` and forbids pushing elsewhere.
- **Decision:** all rescue work develops on the designated branch (per session), merged to
  `main` via PR; phase identity is carried by commit messages and `phases/` reports instead of
  branch names.
- **Related:** session protocol in `CURRENT_STATUS.md`.

## D-012 — Property tests: hand-rolled seeded invariant loops, no hypothesis (for now)

- **Date:** 2026-07-12
- **Status:** Accepted
- **Context:** Phase 2 task 2 requires property/invariant tests. The master plan flagged
  `hypothesis` as a candidate tool; the repo has no property-testing framework and its existing
  invariant suites (`tests/test_frontier_basic.py`, `tests/test_bitboard_basic.py`) are
  hand-rolled loops over seeded random self-play.
- **Options considered:** adopt `hypothesis` (shrinking, broader input distribution, new dev
  dependency + CI cost); extend the existing hand-rolled seeded-trajectory pattern.
- **Decision:** hand-rolled seeded full-game invariant loops
  (`tests/test_engine_properties.py`), consistent with the repo's existing pattern; every ply
  of real games is checked, seeds are fixed, failures reproduce exactly.
- **Consequences:** no automatic input shrinking; coverage is bounded by the seeded
  trajectories rather than adversarial generation.
- **Revisit conditions:** an engine bug slips past these suites, or Phase 2 differential
  testing needs adversarial position generation the trajectories don't reach.
- **Related:** Phase 2 in `MASTER_PLAN.md`; `tests/test_engine_differential.py`.

## D-013 — Engine state/action schema v1; TD corpus keeps its existing columns

- **Date:** 2026-07-12
- **Status:** Accepted
- **Context:** Phase 2 task 3 versions the state/action formats and surfaces them in self-play
  records. `data/td_trajectories.csv` is an append-only corpus with a fixed header — adding a
  column would misalign every existing row against the header, violating the DATA_LINEAGE
  immutability rule.
- **Decision:** `STATE_SCHEMA_VERSION = "board_state_v1"` (`engine/board.py`,
  `Board.to_dict/from_dict`; frontiers/bitboards rebuilt from the grid on load) and
  `ACTION_SCHEMA_VERSION = "move_v1"` (`engine/move_generator.py`, `Move.to_dict/from_dict`,
  matching the shape already persisted by game_history/webapi/schemas). Arena game records
  stamp both plus `scoring_mode`. The TD corpus is NOT modified — its rows already carry
  `feature_set_version` + `agent_version` provenance; raw-state schema ids belong to the
  Phase 7 fresh-manifest datasets.
- **Consequences:** every new arena game is provenance-complete; the browser worker's
  frontend-orientation remap is explicitly outside `move_v1` (documented at the constant).
- **Revisit conditions:** Phase 7 dataset design (D-007) supersedes this for training data.
- **Related:** `phases/PHASE_2_TRUSTED_ENGINE.md`, D-007.

## D-014 — Rollout reward baseline: root-board deltas (fixing sibling-blind endgame values)

- **Date:** 2026-07-12
- **Status:** Accepted
- **Context:** Phase 3 node-level verification found both children of the endgame-pocket
  position at exactly Q = 0.0: `MCTSAgent._rollout` measured per-player rewards as score
  deltas from the expanded leaf's own board, subtracting the points banked by the move that
  created the leaf out of its own value. End-of-game rollouts (the only ones returning
  deltas — cutoff-hit rollouts return static evals) could not distinguish sibling moves whose
  difference was immediate banked points. The tactical regression test passed only via
  move-ordering tie-break.
- **Options considered:** root-board baseline (constant per search → child values reflect true
  final-score differences, magnitudes unchanged in scale); absolute final scores (equivalent
  argmax, larger magnitudes vs C=1.414 calibration); leave as-is and rely on static-eval
  cutoff (leaves endgames blind).
- **Evidence:** deterministic probe — legacy baseline Q 0.0/0.0 and 40/40 visits; root
  baseline Q 5.0/1.0 (exact final-score difference) with correct visit dominance. Pinned as
  an A/B regression test in `tests/test_minimal_search_semantics.py`. Bounded post-fix sanity
  arena showed no collapse.
- **Decision:** `rollout_reward_baseline="root"` is the default; "leaf" retained strictly for
  A/B experiments; value validated in the constructor; propagated to root-parallel workers.
- **Consequences:** rollout rewards change for every MCTS agent including the champion —
  absolute ratings drift across this boundary (era note in `DATA_LINEAGE.md`). Reward-scale
  mix (score-delta+bonus vs tanh×100 static eval) unchanged and flagged for Phase 4/5.
- **Revisit conditions:** Phase 4 scaling results; Phase 5 leaf-evaluation selection may
  replace rollout deltas entirely.
- **Related:** `phases/PHASE_3_MINIMAL_SEARCH.md`; AUDIT_REPORT §3.1 (the sibling maxⁿ fix).

## D-015 — Evaluator track v1 scope (data source, target, features, framework)

- **Date:** 2026-07-12
- **Status:** Accepted (v1 scope; D-005/D-006/D-007 remain open for the full Phase 6 design)
- **Context:** Phase 4 FAIL is attributed to the static evaluator (EXP-003: swapping rollout
  leaves for Layer-6 static-eval leaves erased the entire margin over the heuristic anchor;
  Q spread across root moves ~1.3 points on a ~47 scale). The rescue needs a leaf evaluator
  that discriminates. Constraints: no torch in deps; browser serving is numpy-only; the
  existing 45-feature TD refits (`rich_leaf` etc.) already failed as candidates — a v1 must
  change more than the fitting pass, and everything must run on fresh standard-scored data
  (legacy corpora are house-scored, DATA_LINEAGE).
- **Decision (v1):**
  - **Data:** fresh manifested dataset `data/value_dataset_v1/` (never the legacy CSV),
    generated by `training/experiments/value_dataset.py`: self-play among four identical
    PW+rollout-50 agents (EXP-002-validated config family; 50 ≈ 500 in strength there),
    standard scoring, per-ply capture, `rich_blokus_v1` features, full manifest (commit,
    agent configs, seeds, schema ids).
  - **Target:** per-player normalized standard final score (primary), final rank
    (secondary) — matches the existing TD-row label machinery so v1 needs no new plumbing.
  - **Model family:** sklearn (gradient boosting and/or small MLP) with numpy-exportable
    inference; MUST use a held-out split (Phase 1 risk #6).
  - **Gates for v1, in order:** (1) held-out predictive skill vs the linear baseline;
    (2) root Q-spread probe ≫ the measured 1.3-point Layer-6 flatness; (3) the fixed
    Phase 4 acceptance ladder (`--pw --cutoff 0` with the new evaluator) must beat the
    EXP-002 rollout baseline at equal budget AND show positive scaling.
- **Consequences:** if v1 fails gate (2) or (3), the state-value-on-45-features family is
  falsified and Phase 6's candidate-scoring (move-level) architecture becomes the next
  design, with D-005/D-006/D-007 decided in full then.
- **Revisit conditions:** any v1 gate result; torch/other framework only via a new decision.
- **Related:** `phases/PHASE_4_SEARCH_SCALING.md` addendum 2, EXP-003, `HANDOFF.md`.

## D-008 — Teacher search budget: 500 iterations (PW + model leaves)

- **Date:** 2026-07-14
- **Status:** Accepted
- **Context:** Phase 4 required an experimentally located teacher budget.
- **Evidence:** EXP-006b — 500 beats 150 by +14.08 pts (p=0.013); 1500 ≈ 500 (p=0.54):
  the scaling knee is ~500 at current (ridge, pairwise 0.682) evaluator quality.
- **Decision:** teacher self-play (Phase 7) runs at 500 iterations, PW pw_c=2.0 α=0.5,
  ridge-model leaves. ~10–13 s/move on this container (~20 min/4-seat game).
- **Revisit conditions:** any evaluator upgrade that moves the saturation knee (re-run the
  ladder); production budgets are a separate Phase 10 decision.
- **Related:** EXP-005/006b, `phases/PHASE_4_SEARCH_SCALING.md` addendum 4.

## D-016 — Experimental baseline search config: PW + ridge-model leaves

- **Date:** 2026-07-14
- **Status:** Accepted (experimental baseline; NOT a champion promotion)
- **Context:** Phase 4 passed for exactly this configuration; the pre-rescue rollout
  configuration never scaled (EXP-001/002) and is not improvable by training.
- **Evidence:** EXP-005 + EXP-006b scaling; EXP-006a parity vs rollouts at 500 (honest
  caveat: the model is not stronger today — it is EQUAL and trainable).
- **Decision:** the rescue's experimental baseline agent = minimal search + progressive
  widening (2.0/0.5) + `value_model_path` ridge leaves; iteration budgets per role
  (teacher 500). The production champion is UNCHANGED — promotion still goes through the
  Phase 9 gate only.
- **Consequences:** Phase 5 closes (model leaves selected; rollouts deprecated as the
  strength path, retained as a parity/regression baseline); Phase 7/8 proceed on this
  config; the saturation knee becomes the standing evaluator-quality metric.
- **Revisit conditions:** a future evaluator failing gate C, or a rollout hybrid beating
  model leaves in a controlled test.
- **Related:** D-008, D-014, D-015; `phases/PHASE_4_SEARCH_SCALING.md`.

---

## D-017 — Phase 6 move-level candidate scorer: architecture, target, and consumption slot

- **Date:** 2026-07-16
- **Status:** Accepted (scopes the Phase 6 build; production wiring gated on EXP-010)
- **Context:** EXP-008/EXP-009 confirmed the state-feature representation ceiling
  (~0.68 held-out pairwise) at any tested data volume; the pre-registered EXP-009 rule
  routes Phase 6 to the master plan §4-Phase-6 "legal-move candidate scoring"
  architecture. Required sub-decisions (D-005/6/7 reservations): framework, target,
  action representation — plus where the scorer plugs into search.
- **Decision:**
  - **Architecture/framework:** listwise log-linear softmax over engineered per-move
    features + per-piece bias — the exact structure of the existing
    `mcts/move_policy.py` (numpy-only, Pyodide-safe, per-node cheap), generalized to a
    versioned, extensible feature vector. No NN until this saturates.
  - **Target:** distillation of the TEACHER root visit distribution
    (`teacher_record_v2.policy_target`, already normalized and aligned to `search`) —
    per-child Q kept in the datasets as a future auxiliary target, not used in v1.
  - **Action representation:** `move_features_v2` — the four `MOVE_FEATURE_NAMES`
    unchanged (order-stable prefix), then append-only move-varying extensions
    (own-frontier consumption, true opponent-frontier occupation, opponent wall
    contact, own diagonal redundancy, phase×size interaction, edge fraction).
    State-constant features are excluded by construction: within-decision softmax and
    ordering are invariant to per-decision constant shifts.
  - **Consumption slot:** the existing `policy_prior` machinery (PUCT prior +
    rollout/ordering policy) — this REPLACES the legacy 4-feature artifact
    (`training/state/policy_weights.json`, top-1 0.53, self-distilled from heuristic
    play; AUDIT §7) rather than extending it. Legacy corpus `policy_targets.csv` is
    NOT used (Suspect: era/budget-mixed).
- **Gate order (master plan):** tiny-data overfit sanity → held-out generalization
  (EXP-010 bars) → masking/ordering/round-trip tests with the production wiring →
  search-integration experiment. No arena spend before the held-out bar clears.
- **Consequences:** training data comes exclusively from the validated teacher-record
  datasets (visits collected at real search budgets); the browser/Pyodide path stays
  numpy-only; the feature vector is versioned from day one.
- **Revisit conditions:** EXP-010 shows the extended features add nothing over the
  four (→ representation work moves to piece/shape-aware encodings or an NN); or
  listwise linear saturates below the legacy baseline.
- **Related:** D-005/6/7 (scoped v1), EXP-008/009/010; master plan §4 Phase 6.

---

## D-018 — Standard 21-piece Blokus set; all prior artifacts invalidated

- **Date:** 2026-09-06
- **Status:** Accepted
- **Context:** `DIAGNOSTIC_ASSESSMENT_2026-09-06.md` B1: the engine catalogue had two copies of
  the S/Z tetromino (ids 9, 10) and no Z-pentomino; 88 squares, not 89. The frontend shipped the
  same set. Internal sanity numbers (91 orientations, 58 openings/corner) matched by coincidence.
- **Decision:** piece id 10 = Pentomino Z `[[1,1,0],[0,1,0],[0,1,1]]`; ids 1-9 and 11-21 unchanged.
  The catalogue is pinned by `tests/test_piece_set_standard.py` (21 distinct free polyominoes,
  89 squares, 91 orientations), `tests/test_reference_movegen.py` (independent rules-based
  generator; 0 disagreements on every ply of seeded games + 58 openings per corner with exactly
  2 Z-pentomino placements) and `tests/test_frontend_piece_catalogue.py` (frontend == engine).
- **Consequences:** every dataset, weight file, candidate, rating and experiment predating this
  commit is invalid for the standard game (see `DATA_LINEAGE.md` notice). The state/action
  schema versions are bumped (`board_state_v2`, `move_v2`) so new records are machine-
  distinguishable from invalidated ones, and `teacher_selfplay --validate` reports the mismatch
  explicitly. `engine/advanced_metrics.compute_piece_penalty` now derives tiers from piece size.
  The browser bundle is rebuilt from this commit; because the previous bundle dated from
  2026-07-01, the rebuild also ships every engine/MCTS change since then (standard scoring
  default, monomino bonus, D-014, `sample_legal_moves`, value-model/policy modules) — verified
  with `scripts/pyodide_smoke.cjs`. No champion is promoted or demoted by this change.
- **Related:** master plan §4 Phase 2; `DIAGNOSTIC_ASSESSMENT_2026-09-06.md` §6 (first step).

## D-019 — Human-match parameters (user decisions)

- **Date:** 2026-09-06 (answers given as comments on the assessment artifact, timestamped
  2026-09-07 01:51–01:53 UTC)
- **Status:** Accepted
- **Variant / seats:** Classic 4-player 20×20; the human plays exactly one colour; agents play the
  other three.
- **Human strength reference:** the user has played roughly 50 games; treat "competent" as an
  experienced casual player, to be calibrated against Pentobi levels before the 20-game match.
- **Time control:** hard cap 10 s per AI move; preferred: a dynamic allocation (short on trivial
  moves, longer on critical ones) under an overall per-game budget. Search budgets for milestone
  M3+ are sized to this.
- **Interface:** not yet decided (deferred to milestone M4; judgment call: web frontend with a
  natively served agent, since the Pyodide path cannot deliver the budget).

## D-020 — Native search core and Pentobi are permitted

- **Date:** 2026-09-06 (user answer "Yes" as a comment on the assessment artifact,
  2026-09-07 01:53 UTC)
- **Status:** Accepted
- **Decision:** the search core may be rewritten in C++/Rust or compiled with numba, and
  Pentobi's GPL engine may be used as the external yardstick, rules cross-check, and (if the
  M2 stop-loss triggers) as the search core under a GPL fork.
- **Consequence:** milestone M2 (≥ 2,000 sims/s) is unblocked; licence review is only needed if
  Pentobi code is vendored rather than run as a separate GTP process.

---

## D-021 — Protocol v3 is the only evaluation protocol for strength claims

- **Date:** 2026-09-07
- **Status:** Accepted
- **Context:** assessment findings B4–B8, M7, M8: evaluations at 25–50 iterations, replayed
  fixed seeds, moving anchors, Elo carried across eras, decisions on 4–24 games.
- **Decision:** `training/evaluation/protocol_v3.py` (fresh recorded seed per run, round-robin
  seats, iteration-pinned single-worker search at serving budgets, paired sign-flip permutation
  statistics, pre-registered n) is the only protocol for strength claims from M1 on. Two
  harness gates must pass before any strength claim: clone calibration (EXP-014) and
  discrimination (EXP-015). The nightly workflow stays frozen until they pass and is rebuilt
  on this protocol if it ever returns.
- **Related:** `BENCHMARK_PROTOCOL.md` (v3 entry), `mcts_lab/calibrate.py`.

## D-022 — Pentobi levels are the external strength anchors

- **Date:** 2026-09-07
- **Status:** Accepted (permitted by D-020)
- **Decision:** `pentobi-gtp` 30.3 (GTP-only build, no book, one thread, seeded per game) is
  the fixed external yardstick, via arena agent type `pentobi` at explicit levels; the
  engine's legal-move generator is cross-checked against `all_legal` in the test suite.
  Level-to-strength calibration against the repo's agents is measured (EXP-016), never assumed.
  Build recipe and protocol facts: `PENTOBI.md`.

---

## D-023 — M3 anchors after EXP-016, and the search-core question

- **Date:** 2026-09-07; answered 2026-09-08
- **Status:** Accepted as option (c). The options and the recommendation (c) were put to the user
  in the M1 hand-off; the user's reply was "proceed with the next milestone" after merging M1,
  which is taken as accepting the recommendation (the next milestone under (c) being M4). If a
  different option was intended, D-024 is the record to revisit.
- **Context:** under protocol v3 (both gates passed) Pentobi level 3 at ~10 ms/move beats the
  repo's best configuration (D-016, 250 iterations, ~8 s/move) by 7.6 points per game and the
  gen140 champion by 16; level 7 (1.2 s/move) beats them by 28-36 (EXP-016). Levels 1-2 (8 ms/move,
  ~3 and ~30 simulations) are not beaten either: D-016 ties level 1 and loses to level 2 (EXP-016b).
  Per the pre-registered reading, Pentobi level 3 is the M3 target anchor and `greedy` the baseline.
- **Options:**
  (a) **Adopt Pentobi's engine as the search core** (GPL-3, permitted by D-020): the repo keeps
  its rules engine, harness, UI and human protocol; "the agent" becomes a Pentobi level chosen by
  the human protocol (M5) plus the dynamic time budget the user asked for. Shortest path to the
  goal; the home-grown MCTS becomes a research track measured against Pentobi levels.
  (b) **Build the native core (M2) and heuristics (M3) as planned**, with Pentobi level 3 as the
  first target: honest but long — the gap is not only speed (level 3 uses ~90 simulations).
  (c) Both: ship (a) for the human goal now; continue (b) as research with (a) as the yardstick.
- **Recommendation:** (c), unless the user's goal is specifically a self-built engine, in which
  case (b) with realistic timelines.
- **Related:** EXP-016, EXP-016b, assessment §4 M2 stop-loss.

---

## D-024 — M4: the human-facing agent is Pentobi served natively; what "dynamic time budget" means

- **Date:** 2026-09-08
- **Status:** Accepted (implements the serving half of D-023 option (c), accepted 2026-09-08; the
  home-grown core stays a research track measured against Pentobi levels)
- **Context:** the M1 calibration (EXP-016/016b) showed Pentobi level 3 already beats the repo's
  best search; the human goal (≥ 70% first place over 20 games) needs an agent that can be served
  under the D-019 time control today. The Pyodide path cannot run Pentobi (native binary) and
  cannot deliver the budget, so the web UI now has a second transport.
- **Decision:**
  1. **Agent type `pentobi`** is served by the FastAPI backend (`webapi/gameplay_agent_factory.py`,
     `PentobiGameplayAdapter`) and wraps the *same* `agents.pentobi_agent.PentobiAgent` the arena
     builds, with the same level and seed. Served-vs-arena parity is a test
     (`tests/test_pentobi_gameplay_adapter.py`, 50 positions) and a gate report
     (`scripts/m4_serving_gate.py` → `training/reports/m4_serving_gate/report.json`). What parity
     checks: the same `PentobiAgent` class with the same seed, one engine process driving all four
     seats, produces identical moves through the serving wrapper (watchdog thread, budget, legal-move
     list, orientation plumbing); it is not a check against an independent engine. Human-protocol
     games use fresh random per-seat seeds, which are logged.
  2. **Time control.** Pentobi's strength knob is its level (a fixed simulation count per move;
     classic counts 3 / 30 / 90 / 181 / 667 / 5,028 / 69,809 / 349,044 / 1,745,221 for levels
     1-9, weighted by move number and with early abort when the best move cannot change). Wall
     time is therefore an outcome of the level, not an input. The D-019 controls are implemented
     around it (`agents/time_budget.py`): a **hard 10 s per-move cap** enforced by a watchdog that
     kills the search and plays the deterministic greedy move (recorded as `fallback: timeout`);
     **no search when one legal move exists**; an **optional per-game budget** (not used by the
     UI) that steps the level down to a floor level as soon as the remaining budget can no longer
     fund one full-cap move (the cap itself is never shrunk, so the boundary never forces a
     timeout). The cap and level are validated server-side in both profiles
     (`webapi/deploy_validation.py::normalize_pentobi_agent_config`); the engine binary is never
     taken from a request. "Dynamic allocation" beyond that (Pentobi's own weighting by move
     number and early abort) is Pentobi's; the repo does not add a criticality model, because it
     would add an uncalibrated variable to the level that M5 must choose.
  3. **Measured per-move wall time** (this Mac, one thread; `training/reports/m4_serving_gate/
     level_timings.json`), mean / max: L1 8 / 57 ms, L2 8 / 72 ms, L3 10 / 86 ms and L7 1.2 / 4.9 s
     from the protocol-v3 runs EXP-016b/016 (2,100-2,500 moves each, MCTS opponents); L4 5 / 19 ms,
     L5 10 / 30 ms, L6 63 / 240 ms, L8 1.9 / 8.6 s and L9 9.4 s mean / 27 s p90 / 54 s max from
     one 4-seat self-play game each (71-73 moves, `scripts/pentobi_level_timings.py`). Reading:
     levels 3 and 7 pass the M4 latency gate (whole turn ≤ 8 s, gate report); levels 1-6 are far
     below it; **level 8 stayed under the 10 s cap in its one game but its maximum (8.6 s search,
     plus 0.1-1.0 s for move application/telemetry) exceeds the 8 s gate and it has not been
     gated**; level 9 does not fit the cap. If level 8 becomes an M5 candidate, run
     `scripts/m4_serving_gate.py --levels 8` first.
  4. **Every backend-served game is logged** (`webapi/game_log.py`, one JSON per game, rewritten
     atomically after every event) with seat → player type, each AI seat's level/seed/cap, every
     move with its wall time and any fallback, passes (an AI turn forfeited to the server's
     timeout or an exception is recorded as an attributable event), final scores/ranks/winner,
     and the repo commit and schema versions. Games played in the Pyodide worker (the default
     Play page) are not logged. Research profile logs to `data/human_games/` by default
     (`GAME_LOG_DIR` overrides; blank disables); deploy logs only when `GAME_LOG_DIR` is set.
     `python -m mcts_lab.human_games` scores the logs against the M5 gate from the agent's side
     (Pentobi first place ≥ 70%; a human tie for first counts against the agent; no agent loss
     with a fallback or forfeited turn).
  5. **Orientation indices.** The backend can speak the frontend's 0-7 rotation/flip indices
     (`GameConfig.orientation_space = "frontend"`, `engine/orientation_map.py`), translated
     exactly as the Pyodide bridge does; the browser transport uses this so human moves land on
     the intended cells.
  6. **Registry (D-009) stays open:** the human-facing opponent is the Pentobi level chosen by
     M5, recorded in the game logs and DECISIONS rather than in either champion registry. The
     repo's own champions remain research artefacts measured under protocol v3.
- **Consequences:** M5 can start: pick a level (the card in the game-setup modal), play with seats
  rotated, and read `python -m mcts_lab.human_games`. Level 9 is not a candidate under a 10 s cap
  on one thread; level 8 needs its own gate run first; multi-threaded search would break
  served-vs-arena determinism and is not enabled. The assessment's M4 wording "one registry" is
  superseded by §6 (the human-facing opponent is a logged Pentobi level, not a registry entry).
- **Related:** D-019, D-020, D-023, assessment §4 M4/M5.

---

## Open decisions (required before their phases)

| ID (reserved) | Topic | Needed by | Notes |
|---|---|---|---|
| D-005 | Multiplayer value target — **v1 scoped 2026-07-12**: per-player standard final score (normalized) primary, final rank secondary; picked to match the TD-row label machinery; full target study deferred until a discriminating model exists | Phase 6 | v1 record below (D-015 context); revisit with the first model's calibration results |
| D-006 | Model framework — **v1 scoped 2026-07-12**: sklearn (GBM/MLP) with numpy-exportable inference; torch not in deps, browser path numpy-only | Phase 6 | v1 record below (D-015 context) |
| D-007 | State/action encoding schema (versioned) — v1 uses existing `rich_blokus_v1` 45 features (already versioned); candidate-scoring action representation remains the Phase 6 direction | Phase 6/7 | v1 record below (D-015 context) |
| D-008 | Teacher search budget | Phase 7 | Output of the Phase 4 scaling study |
| D-009 | Single champion lineage: training state vs `data/champion_registry.json` serving registry | Phase 9 | Two sources of truth currently disagree (gen140 vs v2) |
| D-010 | Production latency target & difficulty modes | Phase 10 | Measure first |
| D-011 | Rating methodology confirmation (TrueSkill-primary, matrices as evidence) | Phase 9 | Existing practice, needs a formal record |
