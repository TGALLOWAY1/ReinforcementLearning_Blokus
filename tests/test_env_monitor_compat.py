"""
Tests for Monitor + GymnasiumBlokusWrapper compatibility.

These tests verify that SB3's Monitor wrapper works correctly with our
GymnasiumBlokusWrapper, especially in VecEnv contexts where Monitor wraps
each sub-environment individually.
"""

import pytest
import numpy as np

from envs.blokus_v0 import make_gymnasium_env, GymnasiumBlokusWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


def test_monitor_reset_compat_single_env():
    """
    Test that Monitor can successfully wrap GymnasiumBlokusWrapper and call reset().
    
    This simulates what happens in single-env mode where SB3 wraps:
    Monitor(ActionMasker(GymnasiumBlokusWrapper(...)))
    """
    # Create base environment
    base_env = make_gymnasium_env(render_mode=None, max_episode_steps=50)
    assert isinstance(base_env, GymnasiumBlokusWrapper)
    
    # Wrap with Monitor (as SB3 does automatically)
    monitored_env = Monitor(base_env)
    assert isinstance(monitored_env, Monitor)
    
    # Test reset with seed (as gymnasium.Wrapper.reset() does)
    obs, info = monitored_env.reset(seed=42)
    
    # Verify return types
    assert isinstance(obs, np.ndarray), f"Expected numpy array, got {type(obs)}"
    assert isinstance(info, dict), f"Expected dict, got {type(info)}"
    assert obs.shape == base_env.observation_space.shape, \
        f"Obs shape mismatch: {obs.shape} != {base_env.observation_space.shape}"
    
    # Test reset with seed and options
    obs2, info2 = monitored_env.reset(seed=123, options={})
    assert isinstance(obs2, np.ndarray)
    assert isinstance(info2, dict)
    
    # Test that reset with different seeds produces different results (likely)
    # Note: This might not always be true, but it's a sanity check
    assert obs.shape == obs2.shape


def test_monitor_reset_compat_dummy_vecenv():
    """
    Test that Monitor works correctly when wrapping sub-envs in a DummyVecEnv.
    
    This simulates what happens in VecEnv mode where SB3 wraps each sub-env:
    DummyVecEnv([Monitor(GymnasiumBlokusWrapper(...)), Monitor(...), ...])
    """
    # Create factory functions for VecEnv
    def make_env_fn(rank: int):
        def _make_env():
            env = make_gymnasium_env(render_mode=None, max_episode_steps=50)
            # In real usage, SB3 wraps with Monitor automatically
            # For this test, we wrap manually to verify compatibility
            return Monitor(env)
        return _make_env
    
    # Create VecEnv with 2 environments
    env_fns = [make_env_fn(rank=i) for i in range(2)]
    vec_env = DummyVecEnv(env_fns)
    
    assert vec_env.num_envs == 2
    
    # Test VecEnv reset (which calls reset(seed=...) on each sub-env)
    obs = vec_env.reset()
    
    # Verify return shape: (num_envs, *obs_shape)
    expected_shape = (2,) + env_fns[0]().observation_space.shape
    assert obs.shape == expected_shape, \
        f"VecEnv obs shape mismatch: {obs.shape} != {expected_shape}"
    
    # Test step
    actions = [0, 0]  # One action per env
    obs2, rewards, dones, infos = vec_env.step(actions)
    
    # Verify step return shapes
    assert obs2.shape == expected_shape
    assert len(rewards) == 2
    assert len(dones) == 2
    assert len(infos) == 2


def test_gymnasium_wrapper_reset_signature():
    """
    Test that GymnasiumBlokusWrapper.reset() accepts keyword arguments correctly.
    
    This verifies the signature matches what gymnasium.Wrapper.reset() expects
    when it calls self.env.reset(seed=seed, options=options).
    """
    env = make_gymnasium_env(render_mode=None, max_episode_steps=50)
    
    # Test reset with seed as keyword argument (as gymnasium.Wrapper does)
    obs1, info1 = env.reset(seed=42)
    assert isinstance(obs1, np.ndarray)
    assert isinstance(info1, dict)
    
    # Test reset with seed and options as keyword arguments
    obs2, info2 = env.reset(seed=123, options={})
    assert isinstance(obs2, np.ndarray)
    assert isinstance(info2, dict)
    
    # Test reset with no arguments (should work with defaults)
    obs3, info3 = env.reset()
    assert isinstance(obs3, np.ndarray)
    assert isinstance(info3, dict)


def test_gymnasium_wrapper_step_signature():
    """
    Test that GymnasiumBlokusWrapper.step() returns correct 5-tuple.
    
    Gymnasium requires: (obs, reward, terminated, truncated, info)
    """
    env = make_gymnasium_env(render_mode=None, max_episode_steps=50)
    env.reset(seed=42)
    
    # Test step
    result = env.step(0)
    
    # Verify it's a 5-tuple
    assert len(result) == 5, f"Expected 5-tuple, got {len(result)} elements"
    obs, reward, terminated, truncated, info = result
    
    # Verify types
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, (int, float, np.number))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_monitor_wrapper_chain():
    """
    Test the full wrapper chain: Monitor(GymnasiumBlokusWrapper).
    
    This verifies that Monitor can successfully wrap our wrapper and
    that all method calls work correctly.
    """
    base_env = make_gymnasium_env(render_mode=None, max_episode_steps=50)
    monitored_env = Monitor(base_env)
    
    # Verify wrapper chain
    assert isinstance(monitored_env, Monitor)
    assert monitored_env.env is base_env
    assert isinstance(base_env, GymnasiumBlokusWrapper)
    
    # Test full reset/step cycle
    obs, info = monitored_env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    
    obs2, reward, terminated, truncated, info2 = monitored_env.step(0)
    assert isinstance(obs2, np.ndarray)
    assert isinstance(reward, (int, float, np.number))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info2, dict)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])

