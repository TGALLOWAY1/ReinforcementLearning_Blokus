"""
Benchmark script to compare single-env vs multi-env training speed.

This script runs training with different numbers of environments and measures
wall-clock time to compare performance.

Usage:
    PYTHONPATH=. python scripts/benchmark_vecenv.py
"""

import os
import tempfile
import time

from training.config import TrainingConfig
from training.trainer import train


def benchmark_config(config: TrainingConfig, name: str):
    """
    Run training with a given config and measure wall-clock time.
    
    Args:
        config: Training configuration
        name: Descriptive name for this benchmark run
        
    Returns:
        Tuple of (elapsed_time, callback) where callback contains training results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    print(f"Config: num_envs={config.num_envs}, vec_env_type={config.vec_env_type}")
    print(f"        total_timesteps={config.total_timesteps}, mode={config.mode}")
    
    # Record start time
    start_time = time.time()
    
    # Run training
    callback = train(config)
    
    # Record end time
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Print summary
    if callback:
        print(f"\n✓ Training completed")
        print(f"  envs={config.num_envs}, "
              f"episodes={callback.episode_count}, "
              f"time={elapsed:.2f}s")
    else:
        print(f"\n✗ Training failed or returned None")
        print(f"  time={elapsed:.2f}s")
    
    return elapsed, callback


def main():
    """Run benchmarks comparing single-env vs multi-env training."""
    print("VecEnv Training Speed Benchmark")
    print("=" * 60)
    
    # Create temporary directories for checkpoints/logs
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        log_dir = os.path.join(tmpdir, "logs")
        
        # Single-env config
        single_env_config = TrainingConfig(
            mode="smoke",
            num_envs=1,
            vec_env_type="dummy",
            total_timesteps=20000,
            max_steps_per_episode=50,
            n_steps=128,
            batch_size=32,
            learning_rate=3e-4,
            logging_verbosity=0,  # Reduce noise
            enable_sanity_checks=True,
            log_action_details=False,
            checkpoint_interval_episodes=None,  # No checkpointing in benchmark
            checkpoint_dir=checkpoint_dir,
            tensorboard_log_dir=log_dir,
        )
        
        # Multi-env config
        multi_env_config = TrainingConfig(
            mode="smoke",
            num_envs=4,
            vec_env_type="dummy",
            total_timesteps=20000,
            max_steps_per_episode=50,
            n_steps=128,
            batch_size=32,
            learning_rate=3e-4,
            logging_verbosity=0,  # Reduce noise
            enable_sanity_checks=True,
            log_action_details=False,
            checkpoint_interval_episodes=None,  # No checkpointing in benchmark
            checkpoint_dir=checkpoint_dir,
            tensorboard_log_dir=log_dir,
        )
        
        # Run benchmarks
        single_time, single_callback = benchmark_config(single_env_config, "Single Environment (num_envs=1)")
        multi_time, multi_callback = benchmark_config(multi_env_config, "Multi Environment (num_envs=4)")
        
        # Print comparison
        print(f"\n{'='*60}")
        print("Benchmark Summary")
        print(f"{'='*60}")
        print(f"Single-env:  {single_time:.2f}s, episodes={single_callback.episode_count if single_callback else 'N/A'}")
        print(f"Multi-env:   {multi_time:.2f}s, episodes={multi_callback.episode_count if multi_callback else 'N/A'}")
        
        if single_time > 0 and multi_time > 0:
            speedup = single_time / multi_time
            print(f"\nSpeedup: {speedup:.2f}x")
            if speedup > 1:
                print(f"Multi-env is {speedup:.2f}x faster")
            elif speedup < 1:
                print(f"Single-env is {1/speedup:.2f}x faster")
            else:
                print("Both configurations have similar performance")


if __name__ == "__main__":
    main()

