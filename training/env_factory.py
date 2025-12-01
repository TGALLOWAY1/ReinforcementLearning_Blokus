"""
Environment factory for creating single or vectorized Blokus environments.

This module provides factory functions to create environments for training,
supporting both single-environment and vectorized environment setups.
"""

from typing import Optional, Union, Callable
import gymnasium as gym
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from sb3_contrib.common.wrappers import ActionMasker

from envs.blokus_v0 import make_gymnasium_env
from training.config import TrainingConfig


def make_single_env(
    config: TrainingConfig,
    seed: Optional[int] = None
) -> gym.Env:
    """
    Create a single environment instance, ready for ActionMasker wrapping.
    
    Args:
        config: Training configuration
        seed: Optional seed for environment reset
        
    Returns:
        GymnasiumBlokusWrapper instance (unwrapped, no ActionMasker)
    """
    env = make_gymnasium_env(
        render_mode=None,
        max_episode_steps=config.max_steps_per_episode
    )
    
    # Reset with seed if provided
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    
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
            For single env: returns np.ndarray of shape (action_space.n,)
            For VecEnv: returns np.ndarray of shape (num_envs, action_space.n)
        
    Returns:
        - If num_envs=1: ActionMasker-wrapped GymnasiumBlokusWrapper
        - If num_envs>1: ActionMasker-wrapped VecEnv (DummyVecEnv or SubprocVecEnv)
    """
    if config.num_envs == 1:
        # Single environment path (backward compatible)
        env = make_single_env(config, seed=config.env_seed)
        env = ActionMasker(env, mask_fn)
        return env
    else:
        # VecEnv path
        def make_env_fn(rank: int):
            """
            Factory function for creating individual env instances.
            
            Args:
                rank: Environment rank (0 to num_envs-1)
                
            Returns:
                Function that creates and returns a single environment
            """
            def _make_env():
                # Use different seeds for each env if env_seed is provided
                # This ensures diversity across parallel envs
                env_seed = None
                if config.env_seed is not None:
                    env_seed = config.env_seed + rank
                
                env = make_single_env(config, seed=env_seed)
                return env
            
            return _make_env
        
        # Create list of env factory functions
        # Use closure to capture rank value correctly
        env_fns = [make_env_fn(rank=i) for i in range(config.num_envs)]
        
        # Choose VecEnv type
        if config.vec_env_type == "subproc":
            vec_env = SubprocVecEnv(env_fns)
        else:  # "dummy"
            vec_env = DummyVecEnv(env_fns)
        
        # Wrap VecEnv with ActionMasker (using unified mask_fn)
        vec_env = ActionMasker(vec_env, mask_fn)
        
        return vec_env

