"""
Micro-benchmark script to compare naive vs frontier-based move generation performance.

This script generates representative board states and times all generator variants
to provide performance comparison data.

NOTE: This is a non-deterministic benchmark and should not be used as a test assertion.
It is intended as a development tool for performance analysis only.

Usage:
    python benchmarks/benchmark_move_generation.py
    # or
    python -m benchmarks.benchmark_move_generation
"""

import sys
import os
import random
import time

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.board import Board, Player
from engine.move_generator import LegalMoveGenerator
import engine.move_generator as move_gen_module
from tests.utils_game_states import generate_random_valid_state


def benchmark_naive_generator(generator, board, player, num_runs: int):
    """
    Benchmark the naive (grid-based) move generator.
    
    Args:
        generator: LegalMoveGenerator instance
        board: Board state
        player: Player to generate moves for
        num_runs: Number of times to run the generator
        
    Returns:
        Tuple of (average_time_ms, num_moves_generated)
    """
    times = []
    num_moves = None
    
    for _ in range(num_runs):
        start = time.perf_counter()
        moves = generator._get_legal_moves_naive(board, player)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)  # Convert to milliseconds
        if num_moves is None:
            num_moves = len(moves)
    
    avg_time = sum(times) / len(times)
    return avg_time, num_moves


def benchmark_frontier_grid_generator(generator, board, player, num_runs: int):
    """
    Benchmark the frontier-based move generator with grid-based legality.
    
    Args:
        generator: LegalMoveGenerator instance
        board: Board state
        player: Player to generate moves for
        num_runs: Number of times to run the generator
        
    Returns:
        Tuple of (average_time_ms, num_moves_generated, frontier_size)
    """
    # Save original flags
    original_frontier = move_gen_module.USE_FRONTIER_MOVEGEN
    original_bitboard = move_gen_module.USE_BITBOARD_LEGALITY
    
    try:
        move_gen_module.USE_FRONTIER_MOVEGEN = True
        move_gen_module.USE_BITBOARD_LEGALITY = False
        
        times = []
        num_moves = None
        frontier_size = len(board.get_frontier(player))
        
        for _ in range(num_runs):
            start = time.perf_counter()
            moves = generator._get_legal_moves_frontier(board, player)
            end = time.perf_counter()
            times.append((end - start) * 1000.0)  # Convert to milliseconds
            if num_moves is None:
                num_moves = len(moves)
        
        avg_time = sum(times) / len(times)
        return avg_time, num_moves, frontier_size
    finally:
        move_gen_module.USE_FRONTIER_MOVEGEN = original_frontier
        move_gen_module.USE_BITBOARD_LEGALITY = original_bitboard


def benchmark_frontier_bitboard_generator(generator, board, player, num_runs: int):
    """
    Benchmark the frontier-based move generator with bitboard legality.
    
    Args:
        generator: LegalMoveGenerator instance
        board: Board state
        player: Player to generate moves for
        num_runs: Number of times to run the generator
        
    Returns:
        Tuple of (average_time_ms, num_moves_generated, frontier_size)
    """
    # Save original flags
    original_frontier = move_gen_module.USE_FRONTIER_MOVEGEN
    original_bitboard = move_gen_module.USE_BITBOARD_LEGALITY
    
    try:
        move_gen_module.USE_FRONTIER_MOVEGEN = True
        move_gen_module.USE_BITBOARD_LEGALITY = True
        
        times = []
        num_moves = None
        frontier_size = len(board.get_frontier(player))
        
        for _ in range(num_runs):
            start = time.perf_counter()
            moves = generator._get_legal_moves_frontier(board, player)
            end = time.perf_counter()
            times.append((end - start) * 1000.0)  # Convert to milliseconds
            if num_moves is None:
                num_moves = len(moves)
        
        avg_time = sum(times) / len(times)
        return avg_time, num_moves, frontier_size
    finally:
        move_gen_module.USE_FRONTIER_MOVEGEN = original_frontier
        move_gen_module.USE_BITBOARD_LEGALITY = original_bitboard


def main():
    """Run the benchmark and print results."""
    print("=" * 80)
    print("Move Generation Performance Benchmark")
    print("=" * 80)
    print()
    print("NOTE: This is a non-deterministic benchmark for development use only.")
    print("Results may vary based on system load, Python version, and random state generation.")
    print()
    
    generator = LegalMoveGenerator()
    num_runs = 50
    
    # Define representative board states (early, mid, late game)
    states = [
        ("Early game", 5),
        ("Early game", 10),
        ("Mid game", 20),
        ("Mid game", 30),
        ("Late game", 40),
    ]
    
    results = []
    
    for state_label, num_moves in states:
        state_name = f"{state_label} ({num_moves} moves)"
        print(f"Generating state: {state_name}...")
        board, current_player = generate_random_valid_state(num_moves, seed=num_moves)
        
        # Benchmark naive generator (grid-based)
        print(f"  Benchmarking naive (grid) generator ({num_runs} runs)...")
        naive_avg, naive_moves = benchmark_naive_generator(generator, board, current_player, num_runs)
        
        # Benchmark frontier generator with grid-based legality
        print(f"  Benchmarking frontier+grid generator ({num_runs} runs)...")
        frontier_grid_avg, frontier_grid_moves, frontier_size = benchmark_frontier_grid_generator(
            generator, board, current_player, num_runs
        )
        
        # Benchmark frontier generator with bitboard legality
        print(f"  Benchmarking frontier+bitboard generator ({num_runs} runs)...")
        frontier_bitboard_avg, frontier_bitboard_moves, _ = benchmark_frontier_bitboard_generator(
            generator, board, current_player, num_runs
        )
        
        # Calculate speedups
        speedup_grid = naive_avg / frontier_grid_avg if frontier_grid_avg > 0 else float('inf')
        speedup_bitboard = naive_avg / frontier_bitboard_avg if frontier_bitboard_avg > 0 else float('inf')
        speedup_bitboard_vs_grid = frontier_grid_avg / frontier_bitboard_avg if frontier_bitboard_avg > 0 else float('inf')
        
        results.append({
            'name': state_name,
            'naive_avg': naive_avg,
            'frontier_grid_avg': frontier_grid_avg,
            'frontier_bitboard_avg': frontier_bitboard_avg,
            'speedup_grid': speedup_grid,
            'speedup_bitboard': speedup_bitboard,
            'speedup_bitboard_vs_grid': speedup_bitboard_vs_grid,
            'num_moves': naive_moves,
            'frontier_size': frontier_size,
            'player': current_player.name
        })
        
        print(f"  Done. Naive: {naive_avg:.2f}ms, Frontier+Grid: {frontier_grid_avg:.2f}ms, "
              f"Frontier+Bitboard: {frontier_bitboard_avg:.2f}ms")
        print()
    
    # Print summary table
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'State':<25} {'Naive':<10} {'Frontier+Grid':<15} {'Frontier+Bitboard':<18} "
          f"{'Speedup':<10} {'Moves':<8} {'Frontier':<10}")
    print("-" * 80)
    
    for result in results:
        speedup_str = f"{result['speedup_bitboard']:.2f}x" if result['speedup_bitboard'] != float('inf') else "N/A"
        print(f"{result['name']:<25} "
              f"{result['naive_avg']:>8.2f}ms  "
              f"{result['frontier_grid_avg']:>13.2f}ms  "
              f"{result['frontier_bitboard_avg']:>16.2f}ms  "
              f"{speedup_str:>8}  "
              f"{result['num_moves']:>6}  "
              f"{result['frontier_size']:>8}")
    
    print("=" * 80)
    print()
    print("Speedup is relative to naive generator.")
    print("Frontier size indicates the number of frontier cells used as starting points.")


if __name__ == "__main__":
    main()

