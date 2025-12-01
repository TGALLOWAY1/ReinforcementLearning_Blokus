"""
Smoke tests for VecEnv support in training.

These tests verify that the training pipeline works correctly with:
- Single environment (num_envs=1) - backward compatibility
- Multiple environments with DummyVecEnv (num_envs>1)
- Multiple environments with SubprocVecEnv (num_envs>1)

See docs/vecenv-integration-plan.md for implementation details.

These are smoke tests with minimal training budgets to verify:
- Environment creation works
- Action masking works for all envs
- Episode tracking works per-environment
- Training completes without errors

NOTE: In multi-env mode, each single env is wrapped with ActionMasker.
The VecEnv itself is NOT wrapped with ActionMasker to avoid seed/reset
incompatibility between Gymnasium.Wrapper and DummyVecEnv.

Expected manual test commands:
    # Test DummyVecEnv
    PYTHONPATH=. pytest tests/test_vecenv_training.py::test_multi_env_dummy_vecenv -v
    
    # Test SubprocVecEnv (requires SKIP_SUBPROC_TESTS=false)
    PYTHONPATH=. SKIP_SUBPROC_TESTS=false pytest tests/test_vecenv_training.py::test_multi_env_subproc_vecenv -v
    
    # Run all VecEnv tests
    PYTHONPATH=. pytest tests/test_vecenv_training.py -v
    
Both DummyVecEnv and SubprocVecEnv tests should pass when the environment is properly configured.
SubprocVecEnv tests may be flaky in CI environments, hence the SKIP_SUBPROC_TESTS guard.
"""

import os
import tempfile

import pytest

from training.config import TrainingConfig
from training.trainer import train


def test_single_env_training():
    """
    Smoke test: Verify single-env training works (backward compatibility).
    
    This test ensures that num_envs=1 behaves identically to the original
    single-env implementation.
    """
    # Create minimal config for smoke test
    config = TrainingConfig(
        mode="smoke",
        num_envs=1,
        vec_env_type="dummy",
        max_episodes=3,  # Small number for smoke test
        max_steps_per_episode=50,
        total_timesteps=1000,  # Small budget
        n_steps=128,  # Reduced for smoke test
        batch_size=32,
        learning_rate=3e-4,
        logging_verbosity=0,  # Reduce noise
        enable_sanity_checks=True,
        log_action_details=False,
        checkpoint_interval_episodes=None,  # No checkpointing in smoke test
    )
    
    # Use temporary directory for checkpoints/logs
    with tempfile.TemporaryDirectory() as tmpdir:
        config.checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        config.tensorboard_log_dir = os.path.join(tmpdir, "logs")
        
        # Run training
        callback = train(config)
        
        # Assertions
        assert callback is not None, "Training should return a callback"
        assert callback.num_envs == 1, "Should have exactly 1 environment"
        assert callback.episode_count > 0, "Should complete at least one episode"
        assert callback.step_count > 0, "Should have taken at least one step"
        
        # Verify per-env tracking
        assert len(callback.env_episode_rewards[0]) > 0, "Should have episode rewards"
        assert len(callback.env_episode_lengths[0]) > 0, "Should have episode lengths"
        assert callback.env_episode_count[0] == callback.episode_count, "Episode count should match"
        
        # Verify backward-compat properties
        assert len(callback.episode_rewards) > 0, "Should have aggregated episode rewards"
        assert len(callback.episode_lengths) > 0, "Should have aggregated episode lengths"
        assert callback.episode_rewards == callback.env_episode_rewards[0], "Properties should match single-env data"


def test_multi_env_dummy_vecenv():
    """
    Smoke test: Verify multi-env training with DummyVecEnv works.
    
    This test verifies that:
    - Multiple environments are created correctly
    - Action masks are extracted for all envs
    - Episode tracking works per-environment
    - Training aggregates statistics correctly
    """
    # Create minimal config for smoke test
    config = TrainingConfig(
        mode="smoke",
        num_envs=2,
        vec_env_type="dummy",
        max_episodes=4,  # Small number, but should get more than num_envs total
        max_steps_per_episode=50,
        total_timesteps=2000,  # Small budget
        n_steps=128,
        batch_size=32,
        learning_rate=3e-4,
        logging_verbosity=0,  # Reduce noise
        enable_sanity_checks=True,
        log_action_details=False,
        checkpoint_interval_episodes=None,  # No checkpointing in smoke test
    )
    
    # Use temporary directory for checkpoints/logs
    with tempfile.TemporaryDirectory() as tmpdir:
        config.checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        config.tensorboard_log_dir = os.path.join(tmpdir, "logs")
        
        # Run training
        callback = train(config)
        
        # Assertions
        assert callback is not None, "Training should return a callback"
        assert callback.num_envs == 2, "Should have exactly 2 environments"
        assert callback.episode_count > callback.num_envs, (
            f"Should complete more episodes ({callback.episode_count}) than num_envs ({callback.num_envs})"
        )
        assert callback.step_count > 0, "Should have taken at least one step"
        
        # Verify per-env tracking
        for env_id in range(callback.num_envs):
            assert len(callback.env_episode_rewards[env_id]) > 0, (
                f"Env {env_id} should have episode rewards"
            )
            assert len(callback.env_episode_lengths[env_id]) > 0, (
                f"Env {env_id} should have episode lengths"
            )
            assert callback.env_episode_count[env_id] > 0, (
                f"Env {env_id} should have completed at least one episode"
            )
        
        # Verify aggregate statistics
        total_episodes = sum(callback.env_episode_count.values())
        assert total_episodes == callback.episode_count, (
            "Aggregate episode count should match sum of per-env counts"
        )
        assert len(callback.episode_rewards) == total_episodes, (
            "Aggregated rewards should match total episodes"
        )
        assert len(callback.episode_lengths) == total_episodes, (
            "Aggregated lengths should match total episodes"
        )
        
        # Verify both envs contributed episodes
        episodes_per_env = [callback.env_episode_count[i] for i in range(callback.num_envs)]
        assert all(count > 0 for count in episodes_per_env), (
            "All environments should have completed at least one episode"
        )


@pytest.mark.skipif(
    os.getenv("SKIP_SUBPROC_TESTS", "false").lower() == "true",
    reason="SubprocVecEnv tests can be flaky in CI, set SKIP_SUBPROC_TESTS=false to enable"
)
def test_multi_env_subproc_vecenv():
    """
    Smoke test: Verify multi-env training with SubprocVecEnv works.
    
    This test verifies parallel environment execution works correctly.
    Note: This test may be flaky in CI environments, so it's marked as optional.
    Set SKIP_SUBPROC_TESTS=false to enable it.
    """
    # Create minimal config for smoke test
    config = TrainingConfig(
        mode="smoke",
        num_envs=2,
        vec_env_type="subproc",
        max_episodes=4,  # Small number, but should get more than num_envs total
        max_steps_per_episode=50,
        total_timesteps=2000,  # Small budget
        n_steps=128,
        batch_size=32,
        learning_rate=3e-4,
        logging_verbosity=0,  # Reduce noise
        enable_sanity_checks=True,
        log_action_details=False,
        checkpoint_interval_episodes=None,  # No checkpointing in smoke test
    )
    
    # Use temporary directory for checkpoints/logs
    with tempfile.TemporaryDirectory() as tmpdir:
        config.checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        config.tensorboard_log_dir = os.path.join(tmpdir, "logs")
        
        # Run training
        callback = train(config)
        
        # Assertions (same as DummyVecEnv test)
        assert callback is not None, "Training should return a callback"
        assert callback.num_envs == 2, "Should have exactly 2 environments"
        assert callback.episode_count > 0, "Should complete at least one episode"
        assert callback.episode_count > callback.num_envs, (
            f"Should complete more episodes ({callback.episode_count}) than num_envs ({callback.num_envs})"
        )
        assert callback.step_count > 0, "Should have taken at least one step"
        
        # Verify per-env tracking
        for env_id in range(callback.num_envs):
            assert len(callback.env_episode_rewards[env_id]) > 0, (
                f"Env {env_id} should have episode rewards"
            )
            assert len(callback.env_episode_lengths[env_id]) > 0, (
                f"Env {env_id} should have episode lengths"
            )
            assert callback.env_episode_count[env_id] > 0, (
                f"Env {env_id} should have completed at least one episode"
            )
        
        # Verify aggregate statistics
        total_episodes = sum(callback.env_episode_count.values())
        assert total_episodes == callback.episode_count, (
            "Aggregate episode count should match sum of per-env counts"
        )
        assert len(callback.episode_rewards) == total_episodes, (
            "Aggregated rewards should match total episodes"
        )
        assert len(callback.episode_lengths) == total_episodes, (
            "Aggregated lengths should match total episodes"
        )
        
        # Verify both envs contributed episodes
        episodes_per_env = [callback.env_episode_count[i] for i in range(callback.num_envs)]
        assert all(count > 0 for count in episodes_per_env), (
            "All environments should have completed at least one episode"
        )


if __name__ == "__main__":
    # Allow running tests directly
    print("Running VecEnv training smoke tests...")
    print("\n1. Testing single-env training...")
    test_single_env_training()
    print("   ✓ Single-env test passed")
    
    print("\n2. Testing multi-env DummyVecEnv training...")
    test_multi_env_dummy_vecenv()
    print("   ✓ Multi-env DummyVecEnv test passed")
    
    print("\n3. Testing multi-env SubprocVecEnv training...")
    try:
        test_multi_env_subproc_vecenv()
        print("   ✓ Multi-env SubprocVecEnv test passed")
    except Exception as e:
        print(f"   ⚠️  SubprocVecEnv test skipped or failed: {e}")
        print("   (This is expected in some CI environments)")
    
    print("\n" + "=" * 60)
    print("All smoke tests completed!")
    print("=" * 60)

