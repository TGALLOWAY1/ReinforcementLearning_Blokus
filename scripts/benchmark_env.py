"""
Performance benchmark script for Blokus environment and move generation.

This script measures:
1. Environment step() performance (steps per second)
2. Move generation performance (calls per second, average time per call)

Usage:
    PYTHONPATH=. python scripts/benchmark_env.py
    PYTHONPATH=. python scripts/benchmark_env.py --num-steps 50000
    PYTHONPATH=. python scripts/benchmark_env.py --num-steps 50000 --movegen-iterations 1000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.move_generator import LegalMoveGenerator
from envs.blokus_v0 import make_gymnasium_env
from training.config import TrainingConfig


def benchmark_env_step(num_steps: int = 10000, seed: int = 42):
    """
    Benchmark environment step() performance.
    
    Args:
        num_steps: Number of steps to run
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with benchmark results
    """
    print("=" * 80)
    print("Environment Step() Benchmark")
    print("=" * 80)
    print(f"Number of steps: {num_steps:,}")
    print(f"Seed: {seed}")
    print()
    
    # Create environment (same as training)
    config = TrainingConfig(
        mode="full",
        max_steps_per_episode=1000,
        num_envs=1
    )
    env = make_gymnasium_env(render_mode=None, max_episode_steps=config.max_steps_per_episode)
    
    # Reset environment
    obs, info = env.reset(seed=seed)
    
    # Track statistics
    total_reward = 0.0
    episode_count = 0
    steps_in_current_episode = 0
    terminated = False
    truncated = False
    
    # Benchmark loop
    print("Running benchmark...")
    start_time = time.perf_counter()
    
    for step in range(num_steps):
        # Check if agent is terminated/truncated before sampling action
        if terminated or truncated:
            # Agent is done - reset environment
            obs, info = env.reset()
            episode_count += 1
            steps_in_current_episode = 0
            terminated = False
            truncated = False
            continue
        
        # Get legal action mask
        legal_action_mask = info.get('legal_action_mask', np.zeros(env.action_space.n, dtype=bool))
        valid_actions = np.where(legal_action_mask)[0]
        
        # If no valid actions (shouldn't happen with our fix, but safety check)
        if len(valid_actions) == 0:
            # Agent has no legal moves - should be marked as terminated
            # Reset environment
            obs, info = env.reset()
            episode_count += 1
            steps_in_current_episode = 0
            continue
        
        # Sample random valid action
        action = np.random.choice(valid_actions)
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps_in_current_episode += 1
        
        # Reset if episode ended
        if terminated or truncated:
            obs, info = env.reset()
            episode_count += 1
            steps_in_current_episode = 0
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    # Calculate metrics
    steps_per_sec = num_steps / elapsed_time if elapsed_time > 0 else 0.0
    avg_reward_per_step = total_reward / num_steps if num_steps > 0 else 0.0
    
    results = {
        'num_steps': num_steps,
        'elapsed_time': elapsed_time,
        'steps_per_sec': steps_per_sec,
        'total_reward': total_reward,
        'avg_reward_per_step': avg_reward_per_step,
        'episodes_completed': episode_count,
        'avg_steps_per_episode': num_steps / episode_count if episode_count > 0 else 0.0
    }
    
    # Print results
    print()
    print("Results:")
    print(f"  Total steps: {num_steps:,}")
    print(f"  Total time: {elapsed_time:.2f}s")
    print(f"  Steps per second: {steps_per_sec:.2f}")
    print(f"  Episodes completed: {episode_count}")
    print(f"  Average steps per episode: {results['avg_steps_per_episode']:.1f}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Average reward per step: {avg_reward_per_step:.4f}")
    print()
    
    return results


def benchmark_move_generation(num_iterations: int = 1000, seed: int = 42):
    """
    Benchmark move generation performance.
    
    Args:
        num_iterations: Number of move generation calls
        seed: Random seed for board state generation
        
    Returns:
        Dictionary with benchmark results
    """
    print("=" * 80)
    print("Move Generation Benchmark")
    print("=" * 80)
    print(f"Number of iterations: {num_iterations:,}")
    print(f"Seed: {seed}")
    print()
    
    from tests.utils_game_states import generate_random_valid_state
    
    move_generator = LegalMoveGenerator()
    np.random.seed(seed)
    
    # Track statistics
    total_time = 0.0
    total_moves = 0
    min_time = float('inf')
    max_time = 0.0
    times = []
    
    print("Running benchmark...")
    
    for i in range(num_iterations):
        # Generate random board state (varying game progress)
        # Use different move counts to test various game stages
        move_count = np.random.randint(5, 40)  # Early to mid game
        board, player = generate_random_valid_state(move_count, seed=seed + i)
        
        # Benchmark move generation
        start_time = time.perf_counter()
        legal_moves = move_generator.get_legal_moves(board, player)
        elapsed_time = time.perf_counter() - start_time
        print(len(legal_moves))
        # Track statistics
        total_time += elapsed_time
        total_moves += len(legal_moves)
        min_time = min(min_time, elapsed_time)
        max_time = max(max_time, elapsed_time)
        times.append(elapsed_time)
    
    # Calculate metrics
    avg_time = total_time / num_iterations if num_iterations > 0 else 0.0
    avg_time_ms = avg_time * 1000.0
    calls_per_sec = num_iterations / total_time if total_time > 0 else 0.0
    avg_moves_per_call = total_moves / num_iterations if num_iterations > 0 else 0.0
    
    # Calculate percentiles
    times_sorted = sorted(times)
    p50_time_ms = times_sorted[len(times_sorted) // 2] * 1000.0
    p95_time_ms = times_sorted[int(len(times_sorted) * 0.95)] * 1000.0
    p99_time_ms = times_sorted[int(len(times_sorted) * 0.99)] * 1000.0
    
    results = {
        'num_iterations': num_iterations,
        'total_time': total_time,
        'avg_time': avg_time,
        'avg_time_ms': avg_time_ms,
        'min_time_ms': min_time * 1000.0,
        'max_time_ms': max_time * 1000.0,
        'p50_time_ms': p50_time_ms,
        'p95_time_ms': p95_time_ms,
        'p99_time_ms': p99_time_ms,
        'calls_per_sec': calls_per_sec,
        'total_moves_generated': total_moves,
        'avg_moves_per_call': avg_moves_per_call
    }
    
    # Print results
    print()
    print("Results:")
    print(f"  Total iterations: {num_iterations:,}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per call: {avg_time_ms:.2f}ms")
    print(f"  Calls per second: {calls_per_sec:.2f}")
    print(f"  Min time: {results['min_time_ms']:.2f}ms")
    print(f"  Max time: {results['max_time_ms']:.2f}ms")
    print(f"  Median (p50) time: {p50_time_ms:.2f}ms")
    print(f"  p95 time: {p95_time_ms:.2f}ms")
    print(f"  p99 time: {p99_time_ms:.2f}ms")
    print(f"  Total moves generated: {total_moves:,}")
    print(f"  Average moves per call: {avg_moves_per_call:.1f}")
    print()
    
    return results


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(
        description="Benchmark Blokus environment and move generation performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10000,
        help="Number of environment steps to run"
    )
    
    parser.add_argument(
        "--movegen-iterations",
        type=int,
        default=1000,
        help="Number of move generation calls to benchmark"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--skip-env",
        action="store_true",
        help="Skip environment step benchmark"
    )
    
    parser.add_argument(
        "--skip-movegen",
        action="store_true",
        help="Skip move generation benchmark"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Blokus Performance Benchmark")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Environment steps: {args.num_steps:,}")
    print(f"  Move generation iterations: {args.movegen_iterations:,}")
    print(f"  Seed: {args.seed}")
    print()
    
    all_results = {}
    
    # Run environment benchmark
    if not args.skip_env:
        env_results = benchmark_env_step(args.num_steps, args.seed)
        all_results['env'] = env_results
        print()
    
    # Run move generation benchmark
    if not args.skip_movegen:
        movegen_results = benchmark_move_generation(args.movegen_iterations, args.seed)
        all_results['movegen'] = movegen_results
        print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    if 'env' in all_results:
        print(f"Environment: {all_results['env']['steps_per_sec']:.2f} steps/s")
    
    if 'movegen' in all_results:
        print(f"Move Generation: {all_results['movegen']['calls_per_sec']:.2f} calls/s "
              f"({all_results['movegen']['avg_time_ms']:.2f}ms avg)")
    
    print("=" * 80)
    print()
    
    return all_results


if __name__ == "__main__":
    main()

