#!/bin/bash
set -e

echo "Starting 50ms Benchmark..."
python3 scripts/arena_tuning.py --tunings-set adaptive_vs_best_fixed --num-games 100 --thinking-time-ms 50 --agent-backend fast_mcts --seeds 2026,2027,2028,2029,2030

echo "Starting 200ms Benchmark..."
python3 scripts/arena_tuning.py --tunings-set adaptive_vs_best_fixed --num-games 100 --thinking-time-ms 200 --agent-backend fast_mcts --seeds 2026,2027,2028,2029,2030

echo "Starting 400ms Benchmark..."
python3 scripts/arena_tuning.py --tunings-set adaptive_vs_best_fixed --num-games 100 --thinking-time-ms 400 --agent-backend fast_mcts --seeds 2026,2027,2028,2029,2030

echo "All Adaptive Benchmarks Complete!"
