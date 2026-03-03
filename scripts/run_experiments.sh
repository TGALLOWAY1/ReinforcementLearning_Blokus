#!/usr/bin/env bash
set -e

echo "Starting reliable tournament baselines..."
python3 scripts/arena_tuning.py --tunings-set ablation_leaf_eval --num-games 200 --thinking-time-ms 200 --seed 2026 --agent-backend fast_mcts --notes "Primary Clean Baseline N=200 200ms seed 2026"

echo "Starting sensitivity suites..."
python3 scripts/arena_tuning.py --tunings-set ablation_leaf_eval --num-games 200 --thinking-time-ms 50 --seed 2026 --agent-backend fast_mcts --notes "Sensitivity Time = 50ms"
python3 scripts/arena_tuning.py --tunings-set ablation_leaf_eval --num-games 200 --thinking-time-ms 400 --seed 2026 --agent-backend fast_mcts --notes "Sensitivity Time = 400ms"

echo "Done running experiments! Check arena_runs directory."
