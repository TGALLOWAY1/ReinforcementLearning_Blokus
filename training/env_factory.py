"""
Environment factory for creating single or vectorized Blokus environments.

This module provides factory functions to create environments for training,
supporting both single-environment and vectorized environment setups.

IMPORTANT: For SubprocVecEnv compatibility, factory functions must be picklable.
The _make_env_init() function is defined at module top-level and used with functools.partial
to create picklable factory functions for SubprocVecEnv.
"""

from typing import Optional, Union, Callable
from functools import partial
import gymnasium as gym
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from sb3_contrib.common.wrappers import ActionMasker

from envs.blokus_v0 import make_gymnasium_env
from training.config import TrainingConfig


def make_single_env(
    config: TrainingConfig,
    seed: Optional[int] = None,
    mask_fn: Optional[Callable[[gym.Env], np.ndarray]] = None
) -> gym.Env:
    """
    Create a single Blokus environment, optionally wrapped with ActionMasker.
    
    This returns a Gymnasium-compatible env instance that SB3 can wrap with Monitor, etc.
    
    Args:
        config: Training configuration
        seed: Optional seed for environment reset
        mask_fn: Optional function to extract action masks. If provided, wraps env with ActionMasker.
        
    Returns:
        - If mask_fn is provided: ActionMasker(GymnasiumBlokusWrapper(BlokusEnv))
        - If mask_fn is None: GymnasiumBlokusWrapper(BlokusEnv)
        
    Note:
        - Monitor wrapper is NOT applied here. SB3 automatically wraps environments
          with Monitor via _patch_env() when creating MaskablePPO models.
        - ActionMasker is applied at the single-env level to avoid compatibility issues
          when wrapping VecEnv with ActionMasker (which causes reset(seed=...) errors).
    """
    # Create base environment
    env = make_gymnasium_env(
        render_mode=None,
        max_episode_steps=config.max_steps_per_episode
    )
    
    # Reset with seed if provided (using Gymnasium pattern)
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    
    # IMPORTANT: Wrap the single env with ActionMasker here, not the VecEnv.
    # Wrapping the VecEnv itself with ActionMasker leads to a Gymnasium seed/reset
    # compatibility issue: ActionMasker (a Gymnasium.Wrapper) would call
    # DummyVecEnv.reset(seed=..., options=...), but DummyVecEnv does not accept
    # keyword seed/options, causing a TypeError.
    if mask_fn is not None:
        env = ActionMasker(env, mask_fn)
    
    return env


def _make_env_init(
    rank: int,
    config: TrainingConfig,
    base_seed: Optional[int],
    mask_fn: Optional[Callable[[gym.Env], np.ndarray]]
) -> gym.Env:
    """
    Top-level function that creates a single environment instance with ActionMasker.
    
    This function is defined at module top-level to ensure SubprocVecEnv pickling
    compatibility. SubprocVecEnv requires that factory functions be picklable,
    which means they must be top-level functions that only receive simple,
    picklable arguments (not loggers, open files, or module-level singletons).
    
    This function is used with functools.partial to create picklable factory
    functions for SubprocVecEnv.
    
    Args:
        rank: Environment rank (0 to num_envs-1), used for seed diversity
        config: Training configuration (must be picklable)
        base_seed: Base seed value. Each env gets base_seed + rank as its seed.
        mask_fn: Function to extract action masks (must be picklable)
        
    Returns:
        ActionMasker(GymnasiumBlokusWrapper(BlokusEnv)) instance.
        - ActionMasker is applied at single-env level (not to VecEnv)
        - Monitor is applied automatically by SB3 via _patch_env()
        
    Note:
        This function only receives picklable arguments:
        - rank (int)
        - config (TrainingConfig dataclass, should be picklable)
        - base_seed (int or None)
        - mask_fn (function reference, should be picklable if it's a top-level function)
        
        It does NOT capture:
        - Loggers or other non-picklable objects
    """
    # Calculate seed for this environment
    # Different seeds ensure diversity across parallel envs
    env_seed = None
    if base_seed is not None:
        env_seed = base_seed + rank
    
    # Create environment with ActionMasker already applied at single-env level
    # This avoids the ActionMasker(DummyVecEnv) reset(seed=...) compatibility issue
    env = make_single_env(config, seed=env_seed, mask_fn=mask_fn)
    
    return env


def make_training_env(
    config: TrainingConfig,
    mask_fn: Callable[[gym.Env], np.ndarray]
) -> Union[ActionMasker, VecEnv]:
    """
    Main factory function that returns an environment ready for MaskablePPO training.
    
    Supports both single-environment (num_envs=1) and vectorized environments (num_envs>1).
    
    Args:
        config: Training configuration
        mask_fn: Function to extract action masks from environment.
            Always returns np.ndarray of shape (action_space.n,) since ActionMasker
            is now applied at the single-env level, even in multi-env mode.
        
    Returns:
        - If num_envs=1: ActionMasker(GymnasiumBlokusWrapper(BlokusEnv))
        - If num_envs>1: VecEnv containing ActionMasker-wrapped single envs
          (DummyVecEnv or SubprocVecEnv, NOT wrapped with ActionMasker)
    """
    if config.num_envs == 1:
        # Single environment path (backward compatible)
        # ActionMasker is applied inside make_single_env
        env = make_single_env(config, seed=config.env_seed, mask_fn=mask_fn)
        # DO NOT wrap env in ActionMasker here (already done in make_single_env)
        return env
    else:
        # VecEnv path (DummyVecEnv or SubprocVecEnv)
        # Use functools.partial with top-level function for SubprocVecEnv pickling compatibility
        # partial() creates a picklable callable that captures only the function and arguments
        base_seed = config.env_seed
        env_fns = [
            partial(_make_env_init, rank=i, config=config, base_seed=base_seed, mask_fn=mask_fn)
            for i in range(config.num_envs)
        ]
        
        # Choose VecEnv type
        # Both DummyVecEnv and SubprocVecEnv work with the same factory functions
        # SubprocVecEnv requires picklable factory functions (hence top-level _make_env_init + partial)
        if config.vec_env_type == "subproc":
            vec_env = SubprocVecEnv(env_fns)
        else:  # "dummy"
            vec_env = DummyVecEnv(env_fns)
        
        # NOTE: DO NOT wrap VecEnv with ActionMasker here.
        # We wrap each single environment with ActionMasker inside make_single_env.
        # Wrapping the VecEnv itself with ActionMasker leads to a Gymnasium seed/reset
        # compatibility issue: ActionMasker (a Gymnasium.Wrapper) would call
        # DummyVecEnv.reset(seed=..., options=...), but DummyVecEnv does not accept
        # keyword seed/options, causing a TypeError.
        # 
        # The action masking is already handled at the single-env level inside the VecEnv.
        # Monitor is NOT applied here - SB3 applies it automatically via _patch_env()
        
        return vec_env

