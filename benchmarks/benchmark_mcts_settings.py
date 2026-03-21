"""
Benchmark MCTS agent settings to understand the throughput/quality tradeoffs.

Measures simulations/sec, time spent, and move stability across different
time budgets and exploration constants.

Usage:
    python benchmarks/benchmark_mcts_settings.py
    python benchmarks/benchmark_mcts_settings.py --budgets 50,100,500,1000
    python benchmarks/benchmark_mcts_settings.py --board-moves 8 --trials 5
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.fast_mcts_agent import FastMCTSAgent
from engine.move_generator import get_shared_generator
from tests.utils_game_states import generate_random_valid_state


def run_trial(agent, board, player, legal_moves, time_budget_ms):
    """Run a single MCTS trial and return stats."""
    result = agent.think(board, player, legal_moves, time_budget_ms)
    stats = result["stats"]
    move = result["move"]
    move_id = (
        f"{move.piece_id}-o{move.orientation}-r{move.anchor_row}-c{move.anchor_col}"
        if move else "None"
    )
    return {
        "move_id": move_id,
        "time_spent_ms": stats["timeSpentMs"],
        "simulations": stats["nodesEvaluated"],
        "sims_per_sec": (
            int(stats["nodesEvaluated"] / (stats["timeSpentMs"] / 1000.0))
            if stats["timeSpentMs"] > 0 else 0
        ),
        "top_moves": stats.get("topMoves", [])[:3],
    }


def benchmark_time_budgets(budgets_ms, board, player, legal_moves, trials, exploration_constant, seed):
    """Benchmark across different time budgets."""
    print(f"\n{'='*72}")
    print(f"  Time Budget Sweep  (exploration_constant={exploration_constant})")
    print(f"  Board: {len(legal_moves)} legal moves, {trials} trials per budget")
    print(f"{'='*72}")
    print(f"{'Budget (ms)':>12} {'Avg Sims':>10} {'Sims/sec':>10} {'Avg Time':>10} {'Move Agree':>12}")
    print(f"{'-'*12:>12} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*12:>12}")

    for budget_ms in budgets_ms:
        trial_results = []
        for t in range(trials):
            agent = FastMCTSAgent(
                iterations=999999,  # let time be the only constraint
                time_limit=budget_ms / 1000.0,
                exploration_constant=exploration_constant,
                seed=seed + t,
                enable_diagnostics=True,
            )
            trial_results.append(run_trial(agent, board, player, legal_moves, budget_ms))

        avg_sims = sum(r["simulations"] for r in trial_results) / trials
        avg_sps = sum(r["sims_per_sec"] for r in trial_results) / trials
        avg_time = sum(r["time_spent_ms"] for r in trial_results) / trials

        # Move agreement: fraction of trials that picked the same move as the majority
        move_counts = {}
        for r in trial_results:
            move_counts[r["move_id"]] = move_counts.get(r["move_id"], 0) + 1
        most_common_count = max(move_counts.values())
        agreement = most_common_count / trials

        print(f"{budget_ms:>12} {avg_sims:>10.0f} {avg_sps:>10.0f} {avg_time:>9.0f}ms {agreement:>11.0%}")


def benchmark_exploration_constants(constants, budget_ms, board, player, legal_moves, trials, seed):
    """Benchmark across different exploration constants at a fixed budget."""
    print(f"\n{'='*72}")
    print(f"  Exploration Constant Sweep  (budget={budget_ms}ms)")
    print(f"  Board: {len(legal_moves)} legal moves, {trials} trials per constant")
    print(f"{'='*72}")
    print(f"{'C':>12} {'Avg Sims':>10} {'Sims/sec':>10} {'Top Q':>10} {'Top Visits':>12} {'Move Agree':>12}")
    print(f"{'-'*12:>12} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*12:>12} {'-'*12:>12}")

    for c in constants:
        trial_results = []
        for t in range(trials):
            agent = FastMCTSAgent(
                iterations=999999,
                time_limit=budget_ms / 1000.0,
                exploration_constant=c,
                seed=seed + t,
                enable_diagnostics=True,
            )
            trial_results.append(run_trial(agent, board, player, legal_moves, budget_ms))

        avg_sims = sum(r["simulations"] for r in trial_results) / trials
        avg_sps = sum(r["sims_per_sec"] for r in trial_results) / trials

        # Best move's average Q-value and visits across trials
        top_qs = []
        top_visits = []
        for r in trial_results:
            if r["top_moves"]:
                top_qs.append(r["top_moves"][0].get("q_value", 0.0))
                top_visits.append(r["top_moves"][0].get("visits", 0))
        avg_top_q = sum(top_qs) / len(top_qs) if top_qs else 0.0
        avg_top_v = sum(top_visits) / len(top_visits) if top_visits else 0

        move_counts = {}
        for r in trial_results:
            move_counts[r["move_id"]] = move_counts.get(r["move_id"], 0) + 1
        most_common_count = max(move_counts.values())
        agreement = most_common_count / trials

        print(f"{c:>12.3f} {avg_sims:>10.0f} {avg_sps:>10.0f} {avg_top_q:>10.3f} {avg_top_v:>12.0f} {agreement:>11.0%}")


def benchmark_iteration_caps(caps, budget_ms, board, player, legal_moves, trials, exploration_constant, seed):
    """Benchmark with iteration caps to show iterations vs time interaction."""
    print(f"\n{'='*72}")
    print(f"  Iteration Cap Sweep  (budget={budget_ms}ms, C={exploration_constant})")
    print(f"  Board: {len(legal_moves)} legal moves, {trials} trials per cap")
    print(f"{'='*72}")
    print(f"{'Iter Cap':>12} {'Actual Sims':>12} {'Time (ms)':>10} {'Sims/sec':>10} {'Hit Cap?':>10}")
    print(f"{'-'*12:>12} {'-'*12:>12} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10}")

    for cap in caps:
        trial_results = []
        for t in range(trials):
            agent = FastMCTSAgent(
                iterations=cap,
                time_limit=budget_ms / 1000.0,
                exploration_constant=exploration_constant,
                seed=seed + t,
                enable_diagnostics=True,
            )
            trial_results.append(run_trial(agent, board, player, legal_moves, budget_ms))

        avg_sims = sum(r["simulations"] for r in trial_results) / trials
        avg_time = sum(r["time_spent_ms"] for r in trial_results) / trials
        avg_sps = sum(r["sims_per_sec"] for r in trial_results) / trials
        limited_by = "iters" if avg_sims >= cap * 0.95 else "time" if avg_time >= budget_ms * 0.9 else "neither"

        print(f"{cap:>12} {avg_sims:>12.0f} {avg_time:>9.0f}ms {avg_sps:>10.0f} {limited_by:>10}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MCTS settings.")
    parser.add_argument("--budgets", type=str, default="25,50,100,250,500,1000",
                        help="Comma-separated time budgets in ms")
    parser.add_argument("--exploration-constants", type=str, default="0.5,1.0,1.414,2.0,3.0",
                        help="Comma-separated exploration constants to test")
    parser.add_argument("--iteration-caps", type=str, default="50,100,500,1000,5000,10000",
                        help="Comma-separated iteration caps to test")
    parser.add_argument("--board-moves", type=int, default=8,
                        help="Number of random moves to build the test board (higher = fewer legal moves)")
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of trials per setting for averaging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-time", action="store_true", help="Skip time budget sweep")
    parser.add_argument("--skip-exploration", action="store_true", help="Skip exploration constant sweep")
    parser.add_argument("--skip-iterations", action="store_true", help="Skip iteration cap sweep")
    return parser.parse_args()


def main():
    args = parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]
    constants = [float(x) for x in args.exploration_constants.split(",")]
    iter_caps = [int(x) for x in args.iteration_caps.split(",")]

    print("Generating test board state...")
    board, player = generate_random_valid_state(args.board_moves, seed=args.seed)
    move_gen = get_shared_generator()
    legal_moves = move_gen.get_legal_moves(board, player)
    print(f"Board after {args.board_moves} moves: {len(legal_moves)} legal moves for player {player.name}")

    if not legal_moves:
        print("No legal moves available. Try a lower --board-moves value.")
        return

    if not args.skip_time:
        benchmark_time_budgets(budgets, board, player, legal_moves, args.trials, 1.414, args.seed)

    if not args.skip_exploration:
        mid_budget = budgets[len(budgets) // 2] if budgets else 250
        benchmark_exploration_constants(constants, mid_budget, board, player, legal_moves, args.trials, args.seed)

    if not args.skip_iterations:
        mid_budget = budgets[len(budgets) // 2] if budgets else 250
        benchmark_iteration_caps(iter_caps, mid_budget, board, player, legal_moves, args.trials, 1.414, args.seed)

    print(f"\n{'='*72}")
    print("  Summary")
    print(f"{'='*72}")
    print(f"Two knobs control how long the agent thinks:\n")
    print(f"  1. time_limit / time_budget_ms  - wall-clock cap (primary)")
    print(f"  2. iterations                   - simulation count cap (secondary)")
    print(f"\nThe agent stops at whichever limit is hit first.")
    print(f"For gameplay, set iterations high (e.g. 999999) and use time_budget_ms.")
    print(f"For reproducibility, fix iterations and set time_limit very high.\n")


if __name__ == "__main__":
    main()
