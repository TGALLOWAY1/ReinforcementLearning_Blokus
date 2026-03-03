# Arena Status Audit Report

## 1. What Exists and Works Today
The codebase currently contains a complete, robust Arena runner framework and ML pipeline for evaluating agent performance and generating datasets.

*   **Arena Runner:** `scripts/arena.py` (CLI entrypoint) and `analytics/tournament/arena_runner.py` (core logic).
    *   Supports N-game runs with configurable seeds, seat policies (like `randomized` or `round_robin`), and maximum turns.
    *   **Determinism:** Agents are instantiated with predictable, hashed seeds derived from the master run seed (`stable_hash_int`).
    *   Agent adapters exist for Random, Heuristic, MCTS, and FastMCTS variants, handling deterministic thinking time and iterations caps.
*   **Statistics and Output:** `analytics/tournament/arena_stats.py`.
    *   Generates a `summary.json` containing win rates, score statistics (mean, median, p25/p75), simulation efficiency, and pairwise matchups.
    *   Generates a human-readable `summary.md` Markdown report.
    *   Saves game-level records (presumably to `games.jsonl` read format).
*   **ML Logging / Datasets:**
    *   Snapshot dataset generation is fully functional via `analytics/winprob/features.py` and `dataset.py`.
    *   Creates telemetry rows (one per player per checkpoint ply), saving to `snapshots.csv` and `snapshots.parquet`.
*   **ML Training Scripts:**
    *   `scripts/train_winprob_v1.py`: Trains a pairwise logistic regression win-probability model.
    *   `scripts/train_winprob_v2.py`: Trains a phase-aware gradient boosting tree (GBT) model using XGBoost.

**How to run existing tests:**
```bash
python scripts/arena.py --config scripts/arena_config.json --num-games 10
python scripts/train_winprob_v2.py --snapshots arena_runs/<run_id>/snapshots.parquet
```

## 2. What is Partially Implemented
*   While MCTS parameters (exploration constant, use of transposition table, progressive bias, etc.) can be configured manually in `arena_config.json`, there is no programmatic sweep or "Tuning" object system for structured ablation.
*   The current `arena_runner.py` handles 4 distinct arbitrary agents, but we do not have an *isolated* equal-time tournament mode that specifically focuses on ablating MCTS parameters and reporting parameter importance.

## 3. What is Missing Entirely
*   **`MctsTuning` definitions:** A structured Python object to define different tunings (e.g., `c_uct`, `leaf_evaluation_enabled`) and generate candidates.
*   **Equal-Time Tuning Tournament Runner:** A specific script or harness to initialize 4+ identical MCTS agents equipped with *different tunings*, force strictly equal thinking time budgets, orchestrate games among them, and spit out tuning-specific ranking summaries.
*   **Parameter Importance / Ablation analysis report:** Statistics identifying the marginal contribution or performance delta of specific parameters (e.g., "turning on progressive bias improves win% by X%").

## 4. Confidence Level
*   **High Confidence:** I verified the CLI entrypoints and structure of the output objects manually by inspecting `arena_runner.py` and running the `arena.py` script. The determinism logic is explicitly implemented via `hashlib` seed derivations. ML training scripts (`v1` and `v2`) are fully present and use the `SNAPSHOT_FEATURE_COLUMNS`.
