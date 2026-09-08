# Data Lineage

Rules and inventory for every dataset and model artifact. Governing rules:

1. **Datasets are immutable once finalized.** New examples go to a new version or an
   append-only shard with explicit provenance — never silently into an old corpus.
2. **Never mix incompatible data**: different engines, encodings, value definitions, search
   semantics/configs, or model generations require separate, manifested datasets.
3. **Every Phase 7+ dataset carries a manifest** (schema version, engine version, generating
   commit, checkpoint hash, search config, seeds, game/position counts, validator result).
4. **New systems do not load legacy corpora by default.** The Phase 6–8 loop starts from fresh,
   manifested teacher data; the legacy CSVs below remain only for the legacy approaches.

## Era boundaries (already implemented in `training/reporting_era.py`)

> **2026-09-06 — standard piece set (D-018).** Until the `fix/standard-piece-set` change (commit SHA
> recorded in D-018 once merged) the engine
> catalogue was NOT the Blokus set: ids 9 and 10 were the same free tetromino (S/Z) and the
> Z-pentomino was absent (20 distinct shapes, 88 squares, max score 108). Piece id 10 is now
> the Z-pentomino (5 squares). **Every dataset, weight file, candidate artifact, benchmark
> number, rating row and experiment result produced before this change is INVALID for the
> standard game**: recorded inventories, `player_pieces_used`, legal-move sets, policy
> targets, feature values (piece areas, remaining-area counts) and final scores all assume
> the old catalogue. The state/action schema versions were bumped to `board_state_v2` /
> `move_v2` in the same change, so `teacher_selfplay --validate` reports a schema mismatch
> for every old record (and their recorded legal-action sets no longer regenerate).
> Nothing below this line is to be loaded by the rebuilt pipeline; keep the files for forensics
> only. Diagnosis: `DIAGNOSTIC_ASSESSMENT_2026-09-06.md` finding B1.

| Boundary | Run ID | Meaning |
|---|---|---|
| maxⁿ backprop fix | `DEBUGGED_BACKPROP_EPOCH_RUN_ID = 20260701T204805Z` (gen 139, commit `732bd9c`) | Anything earlier used the cooperating-opponents search — invalid for strength conclusions |
| Multi-agent framework | `MULTI_AGENT_EPOCH_RUN_ID = 20260626T055723Z` | Reporting-era cutoff for approach comparisons |
| Rescue baseline | commit `cabe2dd7738daca661798d422ee487179640e34f` (2026-07-12) | All hashes below pinned at this commit |
| Standard-scoring switch | Phase 2 task 1 commit (2026-07-12, protocol `rescue_v2`) | Arena results, ratings rows, and self-play `final_score` labels produced **before** this commit are house-scored; everything after defaults to standard scoring (+5 monomino-last implemented). Never mix across this boundary |
| Rollout-baseline fix (D-014) | Phase 3 commit (2026-07-12) | All MCTS agents (champion included) evaluate endgame rollouts with root-board reward baselines from this commit on; absolute ratings and self-play data generated before/after are not strength-comparable |

## Frozen-asset inventory (sha256 @ rescue baseline)

**Caveat:** until the Phase 0 workflow change merges to `main`, the 6-hourly nightly job may
keep appending to the CSVs and ratings DB on `main`. These hashes identify the rescue-baseline
snapshot; the baseline commit SHA is the authoritative pin (git history preserves every prior
state).

### Data corpora (`data/`)

| File | sha256 | Compatibility |
|---|---|---|
| `champion_snapshots.csv` (~1.5k rows, per-ply Layer-6/se_* features + outcome labels) | `ad5322ee34e7e2cb01a4fb28cd5614677814ca0125db8e1bfd71f0f1451ebcdd` | **Suspect** — era-mixed accumulation; post-fix rows not separable without join on run metadata; usable only for legacy `heuristic_tune` refits; all existing `final_score` labels are house-scored |
| `td_trajectories.csv` (~2.1k rows, 45 `f_*`/`nf_*` features, `agent_version`, `feature_set_version=rich_blokus_v1`) | `e46c81461f7f48254dc06d087b2be8cd957e571cf850a8a1ca9d501cddd69cad` | **Verified-tagged** — rows carry generating agent/budget tags; filterable by era; all existing reward/score labels are house-scored (rows appended post-`rescue_v2` are standard-scored) |
| `policy_targets.csv` (~75k rows, root visit counts, grouped by `decision_id`) | `e420bf0ffde2c62a7172536d5db3c45e9d80a87f2202cfabf93df0117af305e0` | **Suspect** — no era/agent column; collection spans budgets (old 40 ms near-random rows + teacher rows indistinguishable) |
| `champion_registry.json` (serving registry, `current_version: v2`) | `561a45fa4fb737fb424555faf4f729945f6878a501c7fb13781700d5c536f6c4` | **Verified** as serving state; lineage split from training champion (D-009) |
| `layer6_calibrated_weights.json` | `412231ffc0c49a26ac54089e7dd2812bc67d2b36eceedb47cd883ff3511207d3` | **Verified** artifact |
| `archive/champion_snapshots_pre_searchfix_gen140.csv.gz` | `40ad30eee3f8ceb18ba83cbb068e8f6e03d8f94092b2238e815fe780e4d58b12` | **Incompatible** — pre-maxⁿ-fix; forensic use only |
| `archive/README.md` | `08c3dd56618db7026af44a70c42b3a8d921b8b804c7f369bfecf2c3811d6e0eb` | — |

### Training state (`training/state/`)

| File | sha256 | Compatibility |
|---|---|---|
| `champion.json` (gen140, promoted 2026-07-02; Elo 1388.55, μ 54.39 σ 5.02) | `baf1ed68b163fc45c1dc04adf19725b87c3a9be24e0de590219b4f0c506eb26d` | **Verified** — post-fix gated promotion |
| `latest.json` (generation 179, 6 290 games, last_promoted 140) | `82ed0bac9f1428545a89fb01cb56e2673e322fc6e361b56a16587246568330cc` | **Verified** durable state |
| `checkpoints/champion_gen140.json` | `2b50f351e302cb19513003c457df54b15ed677a79b830232d83a3e2fb99437c1` | **Verified** — immutable historical anchor |
| `ratings.sqlite` (append-only, SCHEMA_VERSION 2) | `6e6a8f1d065d90a082af20076a130cae082f849786376f534674ac6a1e3d92c2` | **Verified** with era filtering; pre-fix rows excluded from trends by `reporting_era.py` |
| `policy_weights.json` (log-linear move policy, top-1 agreement 0.53) | `07a2ba90f738918ac3b5f2a29d683615d2026b0e9ca7b786df678e54622ef3dd` | **Suspect** for strength (self-distils toward fixed heuristic, AUDIT_REPORT §7); mechanically valid |
| `td_evaluator_weights.json` (45-feature TD, full subset, 732 rows) | `2860a0791d957f272a2c7413aea687c34d1d64f6e84a8b843373c4806d73a7cc` | **Unknown** strength value; post-fix data |
| `rich_leaf_weights.json` (TD, score subset, 2 120 rows) | `0dbdbd3f46341ed6199251417a2795ec33ec56b4b8e3e5ca985d02dbe4220b48` | **Unknown** strength value; its candidate lost to champion (−60 Elo, run 20260711T190001Z) |

### Rescue-era datasets (fresh, manifested)

| Dataset | sha256 | Status |
|---|---|---|
| `data/value_dataset_v1/trajectories.csv` (17 408 rows, 60 games, PW+rollout-50 self-play, standard scoring, `rich_blokus_v1` features, per-ply capture, seed 20260713) | `2ae265b3c65c39ee825dea27759f8e90670dc5e28c76f19fbb24c616d42fbd81` | **Verified & finalized** — immutable (generator refuses appends); manifest `4c3b1578…` records commit/agents/seeds/schema ids |
| `data/teacher_dataset_v1/records.jsonl.gz` (1 309 `teacher_record_v2` records, 18 games, 4× D-016 teachers @ D-008 budget 500, standard scoring, opening-phase visit sampling τ=1.0/24 decisions, seed 20260715; per-decision full state + visit counts + per-child Q + policy targets) | file `e241f78700ed38709a9b914bafb2dc142c98aefaa425c866390f3c9f742b834a`; content (decompressed) `5902b65d…` — identical to the shards-concat hash originally registered | **Verified & finalized** — engine-level validator PASSED (pre- and post-pack); 18/18 unique games; winners across all four seats. Packed 2026-07-16 (shards → one gzip; migration verified lossless by content-hash equality) |
| `data/value_dataset_v2/records.jsonl.gz` (7 208 `teacher_record_v2` records, 100 games, 4× PW-50 **rollout-leaf** teachers (`value_model: null` — EXP-002 cheap config family, same as value_dataset_v1 but state-carrying), standard scoring, visit sampling τ=1.0/24 decisions, seed 20260716; per-decision full state + visit counts + per-child Q + policy targets) | file `1df289ba55fd739f4512ea683d16a35963a6940ed4315b1db5eae9cb280767f1`; content (decompressed) `e8ee17f6…` — identical to the shards-concat hash originally registered | **Verified & finalized** — engine-level validator PASSED (pre- and post-pack); 100/100 unique games; winners across all four seats (38/24/31/19). EXP-009 verdict: as training data it *underperforms* value_dataset_v1 (τ-sampled openings → noisier labels). Packed 2026-07-16 |

**Packed-dataset format (2026-07-16 on):** teacher-record datasets are generated as
per-game shards (for `--resume`) and packed at finalization into a single
`records.jsonl.gz` (gzip mtime=0, byte-reproducible; shards deleted). The decompressed
bytes equal the in-order shard concatenation, so content hashes are stable across
layouts. Verify with `zcat records.jsonl.gz | sha256sum` against the content hash above.
Programmatic access: `training.experiments.teacher_selfplay.iter_dataset_records`
(handles both layouts) or `--validate DIR`.

### Not versioned

Per-game `training/state/selfplay_runs/*/games.jsonl` are gitignored (bulk). Consequence:
historical per-game records exist only for runs still on disk somewhere — treat as absent.
Phase 7 datasets must version (or manifest-hash) their game records.

## Compatibility status legend

- **Verified** — provenance and generating code understood; safe for its stated purpose.
- **Suspect** — usable with care; era or budget mixing possible; never feed the new loop.
- **Incompatible** — generated under known-wrong semantics (pre-maxⁿ fix); forensic only.
- **Unknown** — mechanically valid, strength value unmeasured.

## Phase 7 dataset requirements (forward-looking)

Every new self-play record: dataset schema version, engine version, game id, position index,
full state, current player, legal actions, visit counts, normalized policy target, root values,
final score + placement vectors, selected action, search config, model checkpoint hash, seed,
seat mapping, symmetry transform (if any). A `training/` dataset validator must check legality
of selected actions, policy/legal-action alignment, player-vector ordering, terminal-score
agreement with the engine, and manifest/record consistency before any training run consumes it.
