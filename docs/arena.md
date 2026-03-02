# Arena Experiments

This project includes a reproducible arena harness for running multi-agent Blokus experiments and writing run artifacts suitable for analysis and downstream modeling.

## Run Command

```bash
python scripts/arena.py --config scripts/arena_config.json
```

Useful overrides:

```bash
python scripts/arena.py \
  --config scripts/arena_config.json \
  --num-games 100 \
  --seed 12345 \
  --seat-policy randomized \
  --snapshots-enabled \
  --snapshot-plys "8,16,24,32,40,48,56,64" \
  --notes "fair-time benchmark"
```

Two benchmark configs are provided:

- `scripts/arena_config.json`: as-is benchmark (different per-agent `thinking_time_ms`)
- `scripts/arena_config_fair_time.json`: fair-time benchmark (equal `thinking_time_ms` for all agents)

Print resolved config without running:

```bash
python scripts/arena.py --config scripts/arena_config.json --print-config
```

## Config Schema

`scripts/arena.py` expects a JSON config with:

- `agents`: exactly 4 entries (`name`, `type`, optional `thinking_time_ms`, and `params`)
- `num_games`: number of games to run
- `seed`: global run seed
- `seat_policy`: `randomized` or `round_robin`
- `output_root`: root output folder (default `arena_runs`)
- `max_turns`: hard per-game turn cap
- `snapshots`: snapshot dataset options
  - `enabled`: whether to log snapshots
  - `strategy`: currently `fixed_ply`
  - `checkpoints`: list of ply checkpoints (e.g. `[8,16,24,...]`)
- `notes`: optional string

Supported `type` values:

- `random`
- `heuristic`
- `mcts`
- `fast_mcts`
- `gameplay_fast_mcts` (recommended for time-budget arena benchmarking)

For MCTS-style agents, useful `params` include:

- `deterministic_time_budget` (default `true`): when enabled, `thinking_time_ms` is converted to a deterministic iteration cap (`iterations_per_ms * thinking_time_ms`) for reproducible runs.
- `iterations_per_ms`: conversion rate used in deterministic mode.
- `iterations`: baseline max iterations.
- `exploration_constant`, `time_limit`, and other agent-specific knobs.
- learned-evaluator controls:
  - `learned_model_path`: `.pkl` model artifact path
  - `leaf_evaluation_enabled`
  - `progressive_bias_enabled`
  - `progressive_bias_weight`
  - `potential_shaping_enabled`
  - `potential_shaping_gamma`
  - `potential_shaping_weight`
  - `potential_mode`: `prob` or `logit`
  - `max_rollout_moves`

Legacy map-style agent configs (old `scripts/arena_config.json` shape) are still accepted and converted automatically.

## Output Layout

Each run writes to:

`arena_runs/YYYYMMDD_HHMMSS_<short_hash>/`

Run artifacts:

- `run_config.json`: resolved config used for the run
- `games.jsonl`: one JSON record per game
- `summary.json`: machine-readable aggregate summary
- `summary.md`: human-readable summary report
- `snapshots.parquet`: ML-ready snapshot dataset (if parquet engine available)
- `snapshots.csv`: always written fallback snapshot dataset

Run registry:

- `arena_runs/index.csv`: appended row per run

## Reproducibility

Reproducibility is enforced by:

- global `seed` in `RunConfig`
- deterministic per-game seed derivation:
  - `game_seed = stable_hash(run_seed, game_index)`
- deterministic per-agent per-game seed derivation
- deterministic seat assignment under both policies:
  - `round_robin`: cyclic seat rotation
  - `randomized`: deterministic shuffle driven by derived seed
- deterministic MCTS budgets by default (`deterministic_time_budget=true`)

Given the same config and seed, `games.jsonl` winners and final scores should match exactly.

## Reading the Summary

`summary.json` contains:

- `win_stats`: overall wins and win rate by agent
- `wins_by_seat`: seat-specific win rates
- `score_stats`: mean, median, std, p25, p75, min, max final scores
- `pairwise_matchups`: pair counts (`A beats B`, `B beats A`, `tie`)
- `time_sim_efficiency`: average move time, simulation throughput, and efficiency ratios
- `snapshots`: snapshot write status + row counts
- `snapshot_diagnostics` (if snapshots enabled): feature distributions, high-correlation pairs, and winner-lead-by-checkpoint diagnostics

`summary.md` renders the same core metrics in a concise report for quick inspection.

## Win-Probability Modeling

Train v1 calibrated baseline (pairwise logistic regression):

```bash
python scripts/train_winprob_v1.py \
  --snapshots arena_runs/<run_id> \
  --output-model models/winprob_logreg_v1.pkl \
  --output-report reports/winprob_logreg_v1.md
```

Train v2 nonlinear phase-aware baseline (gradient boosting):

```bash
python scripts/train_winprob_v2.py \
  --snapshots arena_runs/<run_id> \
  --output-model models/winprob_gbt_v2.pkl \
  --output-report reports/winprob_gbt_v2.md
```

Both scripts:

- build pairwise rows (`x = features_i - features_j`)
- split by `game_id` to avoid leakage
- report `log_loss`, `Brier`, and optional pairwise AUC
- persist model artifacts for MCTS integration

## Learned Evaluator in MCTS

Use learned evaluation options in an MCTS agent config:

```json
{
  "name": "mcts_learned_v1",
  "type": "mcts",
  "thinking_time_ms": 100,
  "params": {
    "learned_model_path": "models/winprob_logreg_v1.pkl",
    "leaf_evaluation_enabled": true,
    "progressive_bias_enabled": true,
    "progressive_bias_weight": 0.25,
    "potential_shaping_enabled": true,
    "potential_shaping_gamma": 1.0,
    "potential_shaping_weight": 1.0,
    "potential_mode": "prob"
  }
}
```

Recommended rollout for incremental risk:

1. enable `leaf_evaluation_enabled` only
2. then enable `progressive_bias_enabled`
3. then evaluate `potential_shaping_enabled`
