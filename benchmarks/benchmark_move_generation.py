"""
Micro-benchmark script to compare naive vs frontier-based move generation performance.

This script generates representative board states and times both generators
to provide performance comparison data.

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


def generate_random_board_state(num_moves: int, seed: int = None):
    """
    Generate a random but valid board state by playing random legal moves.
    
    Args:
        num_moves: Number of moves to make
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (board, current_player) representing the final state
    """
    if seed is not None:
        random.seed(seed)
    
    board = Board()
    generator = LegalMoveGenerator()
    
    moves_made = 0
    for _ in range(num_moves):
        player = board.current_player
        moves = generator._get_legal_moves_naive(board, player)
        
        if not moves:
            # No legal moves for current player, try next player
            board._update_current_player()
            continue
        
        # Choose a random move
        move = random.choice(moves)
        orientations = generator.piece_orientations_cache[move.piece_id]
        positions = move.get_positions(orientations)
        
        success = board.place_piece(positions, player, move.piece_id)
        if success:
            moves_made += 1
    
    return board, board.current_player


def benchmark_generator(generator, board, player, num_runs: int):
    """
    Benchmark a move generator by running it multiple times.
    
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


def benchmark_frontier_generator(generator, board, player, num_runs: int):
    """
    Benchmark the frontier-based move generator.
    
    Args:
        generator: LegalMoveGenerator instance
        board: Board state
        player: Player to generate moves for
        num_runs: Number of times to run the generator
        
    Returns:
        Tuple of (average_time_ms, num_moves_generated, frontier_size)
    """
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


def main():
    """Run the benchmark and print results."""
    print("=" * 70)
    print("Move Generation Performance Benchmark")
    print("=" * 70)
    print()
    
    generator = LegalMoveGenerator()
    num_runs = 50
    
    # Define representative board states
    states = [
        ("Early game (5 moves)", 5, 1),
        ("Early game (10 moves)", 10, 2),
        ("Mid game (20 moves)", 20, 3),
        ("Mid game (30 moves)", 30, 4),
        ("Late game (40 moves)", 40, 5),
    ]
    
    results = []
    
    for state_name, num_moves, seed in states:
        print(f"Generating state: {state_name}...")
        board, current_player = generate_random_board_state(num_moves, seed=seed)
        
        # Benchmark naive generator
        print(f"  Benchmarking naive generator ({num_runs} runs)...")
        naive_avg, naive_moves = benchmark_generator(generator, board, current_player, num_runs)
        
        # Benchmark frontier generator
        print(f"  Benchmarking frontier generator ({num_runs} runs)...")
        frontier_avg, frontier_moves, frontier_size = benchmark_frontier_generator(
            generator, board, current_player, num_runs
        )
        
        # Calculate speedup
        if frontier_avg > 0:
            speedup = naive_avg / frontier_avg
        else:
            speedup = float('inf')
        
        results.append({
            'name': state_name,
            'naive_avg': naive_avg,
            'frontier_avg': frontier_avg,
            'speedup': speedup,
            'num_moves': naive_moves,
            'frontier_size': frontier_size,
            'player': current_player.name
        })
        
        print(f"  Done. Naive: {naive_avg:.2f}ms, Frontier: {frontier_avg:.2f}ms")
        print()
    
    # Print summary table
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'State':<25} {'Naive (ms)':<12} {'Frontier (ms)':<15} {'Speedup':<10} {'Moves':<8} {'Frontier':<10}")
    print("-" * 70)
    
    for result in results:
        speedup_str = f"{result['speedup']:.2f}x" if result['speedup'] != float('inf') else "N/A"
        print(f"{result['name']:<25} "
              f"{result['naive_avg']:>10.2f}  "
              f"{result['frontier_avg']:>13.2f}  "
              f"{speedup_str:>8}  "
              f"{result['num_moves']:>6}  "
              f"{result['frontier_size']:>8}")
    
    print("=" * 70)
    print()
    print("Note: Results may vary based on system load and Python version.")
    print("Frontier size indicates the number of frontier cells used as starting points.")


if __name__ == "__main__":
    main()

