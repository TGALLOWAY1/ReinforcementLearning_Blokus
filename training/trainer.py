"""
Main training script for Blokus RL using MaskablePPO.

This module provides:
- Structured training configuration (smoke-test and full modes)
- Seed control for reproducibility
- Detailed logging and sanity checks
- Episode and step limits
- Config file and CLI argument support
- Periodic checkpointing and resume functionality

Summary of training setup:
- Uses Stable-Baselines3's MaskablePPO with action masking
- Environment: BlokusEnv wrapped for Gymnasium compatibility
- Action masking ensures only legal moves are selected
- Supports both smoke-test mode (quick verification) and full training mode

Current model saving behavior:
- Models are saved using Stable-Baselines3's built-in model.save() method
- Periodic checkpoints are saved every N episodes (configurable via checkpoint_interval_episodes)
- Checkpoints include: model weights, optimizer state, training config, and metadata
- Final checkpoint is saved at end of training
- Old checkpoints are automatically cleaned up (keeps last N checkpoints)
- Resume functionality allows continuing training from any saved checkpoint
- Checkpoints are organized in: checkpoints/<agent_id>/<run_id>/ep<episode>.zip
- All checkpoints are logged to MongoDB TrainingRun records

Current hyperparameter usage:
- Hyperparameters can be defined in agent config files (YAML/JSON) in config/agents/
- Agent configs include: learning_rate, gamma, network architecture, PPO-specific params
- Configs can be loaded via --agent-config CLI argument
- If no agent config is provided, defaults are used (learning_rate=3e-4, gamma=0.99, etc.)
- Hyperparameter sweeps can be run using training/run_sweep.py
- All hyperparameters are logged to TrainingRun records for comparison
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import torch

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from training.config import TrainingConfig, create_arg_parser, parse_args_to_config
from training.env_factory import make_training_env
from training.seeds import set_seed
from training.run_logger import create_training_run_logger, TrainingRunLogger
from training.checkpoints import (
    save_checkpoint, load_checkpoint, get_checkpoint_path,
    cleanup_old_checkpoints, get_checkpoint_episode
)
from training.agent_config import load_agent_config, AgentConfig
from training.reproducibility import get_reproducibility_metadata, log_reproducibility_info
from utils.logging_setup import setup_training_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Diagnostic logging for action masking (can be disabled)
MASK_DEBUG_LOGGING = True  # Set to False to disable diagnostic logging
_mask_fn_logger = logging.getLogger(__name__ + ".mask_fn_diagnostics")


def mask_fn(env):
    """
    Unified mask function that works for both single env and VecEnv.
    
    This function is called by ActionMasker wrapper to get the action mask for MaskablePPO.
    
    Args:
        env: Either:
            - ActionMasker-wrapped GymnasiumBlokusWrapper (single env)
            - ActionMasker-wrapped VecEnv (vectorized envs)
        
    Returns:
        - For single env: np.ndarray of shape (action_space.n,)
        - For VecEnv: np.ndarray of shape (num_envs, action_space.n)
    """
    # Detect VecEnv by checking for num_envs attribute
    if hasattr(env, 'num_envs'):
        return _mask_fn_vecenv(env)
    else:
        return _mask_fn_single(env)


def _extract_mask_from_wrapped_env(wrapped_env, blokus_env, agent_name, action_space_n):
    """
    Helper function to extract mask from a wrapped environment.
    
    Handles edge cases: terminated agents, empty masks, shape validation.
    
    Args:
        wrapped_env: GymnasiumBlokusWrapper instance
        blokus_env: Underlying BlokusEnv instance
        agent_name: Agent name string (e.g., "player_0")
        action_space_n: Expected action space size
        
    Returns:
        Boolean numpy array of shape (action_space_n,)
    """
    # Check if agent is already terminated - if so, return a safe mask
    if agent_name in blokus_env.terminations and blokus_env.terminations[agent_name]:
        if MASK_DEBUG_LOGGING:
            _mask_fn_logger.warning(
                f"mask_fn({agent_name}): Agent is TERMINATED, returning fallback mask"
            )
        # Agent is terminated, return a mask with at least one action enabled
        # This prevents errors when the policy tries to sample from a terminated state
        mask = np.zeros(action_space_n, dtype=np.bool_)
        mask[0] = True  # Enable first action as a safe fallback
        if MASK_DEBUG_LOGGING:
            _mask_fn_logger.debug(
                f"mask_fn({agent_name}): Fallback mask - shape={mask.shape}, "
                f"dtype={mask.dtype}, sum={mask.sum()}"
            )
        return mask
    
    # Get the mask for our agent (player_0)
    # The info dict should always be up to date after reset/step
    if agent_name not in blokus_env.infos:
        if MASK_DEBUG_LOGGING:
            _mask_fn_logger.error(
                f"mask_fn({agent_name}): Agent not found in environment infos!"
            )
        raise ValueError(f"Agent {agent_name} not found in environment infos")
    
    mask = blokus_env.infos[agent_name]["legal_action_mask"]
    # Convert to numpy boolean array
    mask = np.asarray(mask, dtype=np.bool_)
    
    # Ensure mask has correct shape
    if mask.shape[0] != action_space_n:
        _mask_fn_logger.error(
            f"mask_fn({agent_name}): SHAPE MISMATCH - "
            f"mask.shape={mask.shape} != action_space.n={action_space_n}"
        )
        raise ValueError(f"Mask shape {mask.shape} doesn't match action space {action_space_n}")
    
    # CRITICAL: Handle case where no legal actions are available
    mask_sum = mask.sum()
    if mask_sum == 0:
        # Agent has no legal moves - this will break MaskablePPO's MaskableCategorical
        _mask_fn_logger.error(
            f"mask_fn({agent_name}): NO LEGAL ACTIONS PRESENT IN MASK - "
            f"this will break MaskablePPO! "
            f"env.infos[{agent_name}]['can_move']={blokus_env.infos[agent_name].get('can_move', 'N/A')}, "
            f"env.infos[{agent_name}]['legal_moves_count']={blokus_env.infos[agent_name].get('legal_moves_count', 'N/A')}"
        )
        # For now, we'll still return the empty mask to see the error
        # (This is diagnostic only - we'll fix behavior in next step)
    
    return mask


def _mask_fn_single(env):
    """
    Extract mask from single environment (existing logic, preserved for backward compatibility).
    
    Unwrapping chain: env (ActionMasker) -> env.env (GymnasiumBlokusWrapper) -> env.env.env (BlokusEnv)
    But ActionMasker passes the unwrapped env, so: env (GymnasiumBlokusWrapper) -> env.env (BlokusEnv)
    
    Args:
        env: GymnasiumBlokusWrapper instance (unwrapped by ActionMasker)
        
    Returns:
        Boolean numpy array of shape (action_space.n,)
    """
    # Access the underlying BlokusEnv: env (GymnasiumBlokusWrapper) -> env.env (BlokusEnv)
    blokus_env = env.env
    agent_name = env.agent_name
    action_space_n = env.action_space.n
    
    # DIAGNOSTIC: Track call count for limiting logs
    if not hasattr(mask_fn, '_call_count'):
        mask_fn._call_count = 0
    mask_fn._call_count += 1
    
    # Extract mask using helper function
    mask = _extract_mask_from_wrapped_env(env, blokus_env, agent_name, action_space_n)
    
    # DIAGNOSTIC: Log mask properties
    mask_sum = mask.sum()
    should_log = MASK_DEBUG_LOGGING and (
        mask_fn._call_count <= 20 or  # Log first 20 calls
        mask_sum == 0 or  # Always log when mask is empty
        mask.shape[0] != action_space_n  # Always log shape mismatches
    )
    
    if should_log:
        _mask_fn_logger.info(
            f"mask_fn({agent_name}) [SINGLE-ENV] call #{mask_fn._call_count}: "
            f"mask.shape={mask.shape}, "
            f"mask.dtype={mask.dtype}, "
            f"mask.sum()={mask_sum}, "
            f"action_space.n={action_space_n}, "
            f"env.infos[{agent_name}]['legal_moves_count']={blokus_env.infos[agent_name].get('legal_moves_count', 'N/A')}"
        )
        if mask_sum > 0:
            # Log sample of legal action indices
            legal_indices = np.where(mask)[0]
            sample_size = min(10, len(legal_indices))
            _mask_fn_logger.debug(
                f"mask_fn({agent_name}): Sample legal action indices: {legal_indices[:sample_size].tolist()}"
            )
    
    return mask


def _mask_fn_vecenv(env):
    """
    Extract masks from vectorized environment.
    
    When ActionMasker wraps a VecEnv, it passes the unwrapped VecEnv to mask_fn.
    Unwrapping chain for each sub-env:
    - env (VecEnv, unwrapped by ActionMasker) -> env.envs[i] (GymnasiumBlokusWrapper)
    - wrapped_env (GymnasiumBlokusWrapper) -> wrapped_env.env (BlokusEnv)
    
    Args:
        env: VecEnv instance (unwrapped by ActionMasker before calling mask_fn)
        
    Returns:
        Boolean numpy array of shape (num_envs, action_space.n)
    """
    num_envs = env.num_envs
    action_space_n = env.action_space.n
    
    # DIAGNOSTIC: Track call count for limiting logs
    if not hasattr(mask_fn, '_call_count'):
        mask_fn._call_count = 0
    mask_fn._call_count += 1
    
    if MASK_DEBUG_LOGGING and mask_fn._call_count <= 5:
        _mask_fn_logger.info(
            f"mask_fn [VECENV] call #{mask_fn._call_count}: "
            f"num_envs={num_envs}, action_space.n={action_space_n}"
        )
    
    masks = []
    for i in range(num_envs):
        # Get sub-environment from VecEnv
        # Each sub-env in env.envs is a GymnasiumBlokusWrapper (not wrapped by ActionMasker)
        wrapped_env = env.envs[i]
        
        # Extract underlying BlokusEnv and agent name
        # Unwrapping: wrapped_env (GymnasiumBlokusWrapper) -> wrapped_env.env (BlokusEnv)
        blokus_env = wrapped_env.env
        agent_name = wrapped_env.agent_name
        
        # Extract mask using helper function (reuses single-env logic)
        mask = _extract_mask_from_wrapped_env(wrapped_env, blokus_env, agent_name, action_space_n)
        masks.append(mask)
    
    # Stack masks into batch: shape (num_envs, action_space.n)
    batched_mask = np.stack(masks, axis=0)
    
    # Validate batched mask shape
    assert batched_mask.shape == (num_envs, action_space_n), (
        f"Batched mask shape mismatch: expected ({num_envs}, {action_space_n}), "
        f"got {batched_mask.shape}"
    )
    
    # DIAGNOSTIC: Log batched mask properties
    if MASK_DEBUG_LOGGING and mask_fn._call_count <= 5:
        mask_sums = [m.sum() for m in masks]
        _mask_fn_logger.info(
            f"mask_fn [VECENV] batched mask: "
            f"shape={batched_mask.shape}, "
            f"mask_sums={mask_sums}, "
            f"min={min(mask_sums)}, max={max(mask_sums)}"
        )
    
    return batched_mask


class TrainingCallback(BaseCallback):
    """
    Custom callback for training with episode limits, sanity checks, and detailed logging.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        run_logger: Optional[TrainingRunLogger] = None,
        model: Optional[Any] = None,
        verbose: int = 0,
        num_envs: Optional[int] = None
    ):
        super().__init__(verbose)
        self.config = config
        self.run_logger = run_logger
        self.model = model
        
        # Determine number of environments
        # Priority: explicit num_envs > detect from model.env > default to 1
        if num_envs is not None:
            self.num_envs = num_envs
        elif model is not None and hasattr(model.env, 'num_envs'):
            self.num_envs = model.env.num_envs
        else:
            self.num_envs = 1
        
        # Per-environment tracking (dicts keyed by env_id: 0 to num_envs-1)
        self.env_episode_rewards = {i: [] for i in range(self.num_envs)}
        self.env_episode_lengths = {i: [] for i in range(self.num_envs)}
        self.env_current_reward = {i: 0.0 for i in range(self.num_envs)}
        self.env_current_length = {i: 0 for i in range(self.num_envs)}
        self.env_episode_count = {i: 0 for i in range(self.num_envs)}
        self.env_last_info = {i: None for i in range(self.num_envs)}  # Store last info dict per env for win detection
        
        # Global step tracking (increments once per SB3 step, not per env)
        self.step_count = 0
        
        # Timing for speed metrics (using perf_counter for high-precision elapsed time)
        self.start_time = time.perf_counter()
        self.last_log_time = time.perf_counter()
        self.last_log_step_count = 0
        
        # Last step info (for debugging, uses first env's data)
        self.last_obs = None
        self.last_action = None
        self.last_reward = None
        self.last_done = None
        self.run_id = run_logger.run_id if run_logger else None
    
    @property
    def episode_count(self) -> int:
        """Backward compatibility: total episodes across all envs."""
        return sum(self.env_episode_count.values())
    
    @episode_count.setter
    def episode_count(self, value: int):
        """Backward compatibility: set episode count (distributes to first env)."""
        # When resuming, we set the episode count on the first env
        # This maintains compatibility with existing resume logic
        if self.num_envs > 0:
            self.env_episode_count[0] = value
    
    @property
    def episode_rewards(self) -> List[float]:
        """Backward compatibility: all episode rewards flattened across all envs."""
        return [r for rewards in self.env_episode_rewards.values() for r in rewards]
    
    @property
    def episode_lengths(self) -> List[int]:
        """Backward compatibility: all episode lengths flattened across all envs."""
        return [l for lengths in self.env_episode_lengths.values() for l in lengths]
    
    @property
    def current_episode_reward(self) -> float:
        """Backward compatibility: current episode reward (first env only)."""
        return self.env_current_reward[0] if self.num_envs > 0 else 0.0
    
    @property
    def current_episode_length(self) -> int:
        """Backward compatibility: current episode length (first env only)."""
        return self.env_current_length[0] if self.num_envs > 0 else 0
        
    def _on_step(self) -> bool:
        """
        Called at each step during training.
        
        Processes all environments in the batch (for VecEnv) or single environment.
        
        Returns:
            True to continue training, False to stop
        """
        # Get current info from SB3 callback locals
        # SB3 provides these as lists (for vectorized envs) or single values (for single env)
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        obs = self.locals.get("obs", None)
        actions = self.locals.get("actions", [])
        
        # Process all environments in batch
        for env_id in range(self.num_envs):
            # Extract values for this environment (handle both list and single value)
            reward = rewards[env_id] if env_id < len(rewards) else (rewards[0] if rewards else 0.0)
            done = dones[env_id] if env_id < len(dones) else (dones[0] if dones else False)
            info = infos[env_id] if env_id < len(infos) else (infos[0] if infos else {})
            action = actions[env_id] if env_id < len(actions) else (actions[0] if actions else None)
            
            # Update per-env tracking
            self.env_current_reward[env_id] += reward
            self.env_current_length[env_id] += 1
            
            # Store last step info from first env (for backward compatibility)
            if env_id == 0:
                self.last_obs = obs
                self.last_action = action
                self.last_reward = reward
                self.last_done = done
            
            # Store last info dict per env (needed for win detection on terminal steps)
            if done:
                self.env_last_info[env_id] = info
            
            # Sanity checks (per env)
            if self.config.enable_sanity_checks:
                self._sanity_check(reward, obs, action, info)
            
            # Detailed logging in smoke-test mode (only for first env to avoid spam)
            if env_id == 0 and self.config.log_action_details and self.episode_count < 3 and self.env_current_length[env_id] <= 10:
                self._log_step_details(reward, done, action, info)
            
            # Check episode termination (per env)
            if done:
                self._on_episode_end(env_id)
            
            # Check step limit per episode (per env)
            if self.env_current_length[env_id] >= self.config.max_steps_per_episode:
                logger.warning(
                    f"Episode {self.env_episode_count[env_id] + 1} (env {env_id}) reached max_steps_per_episode "
                    f"({self.config.max_steps_per_episode}), truncating"
                )
                self._on_episode_end(env_id)
        
        # Step count increments once per SB3 step (not per env)
        self.step_count += 1
        
        # Check episode limit (aggregate across all envs)
        total_episodes = self.episode_count
        if self.config.max_episodes is not None and total_episodes >= self.config.max_episodes:
            logger.info(f"Reached max_episodes limit ({self.config.max_episodes}), stopping training")
            return False
        
        return True
    
    def _on_episode_end(self, env_id: int = 0):
        """
        Handle episode end for a specific environment.
        
        Args:
            env_id: Environment ID (0 to num_envs-1)
        """
        # Update per-env episode count
        self.env_episode_count[env_id] += 1
        episode_num = self.env_episode_count[env_id]
        
        # Store episode metrics
        self.env_episode_rewards[env_id].append(self.env_current_reward[env_id])
        self.env_episode_lengths[env_id].append(self.env_current_length[env_id])
        
        # Determine win status from terminal step info
        win = self._compute_win_from_info(env_id)
        
        # Log to MongoDB if logger is available
        if self.run_logger:
            self.run_logger.log_episode(
                episode=episode_num,
                total_reward=self.env_current_reward[env_id],
                steps=self.env_current_length[env_id],
                win=win
            )
        
        # Log episode completion (only for first env or if verbose/multi-env)
        if env_id == 0 or (self.config.mode == "smoke" or self.config.logging_verbosity >= 1):
            env_suffix = f" (env {env_id})" if self.num_envs > 1 else ""
            win_str = f"win={win:.1f}" if win is not None else "win=None"
            logger.info(
                f"Episode {episode_num}{env_suffix} completed: "
                f"reward={self.env_current_reward[env_id]:.2f}, "
                f"length={self.env_current_length[env_id]}, "
                f"{win_str}"
            )
        
        # Save checkpoint if interval is reached (based on aggregate episode count)
        total_episodes = self.episode_count
        if (self.model and 
            self.config.checkpoint_interval_episodes and 
            total_episodes % self.config.checkpoint_interval_episodes == 0 and
            env_id == 0):  # Only save checkpoint once per interval (use first env as trigger)
            try:
                checkpoint_path = get_checkpoint_path(
                    checkpoint_dir=self.config.checkpoint_dir,
                    run_id=self.run_id or "unknown",
                    episode=total_episodes,
                    agent_id="ppo_agent"
                )
                
                save_checkpoint(
                    model=self.model,
                    checkpoint_path=checkpoint_path,
                    episode=total_episodes,
                    config=self.config.to_dict(),
                    run_id=self.run_id
                )
                
                # Log checkpoint to MongoDB
                if self.run_logger:
                    self.run_logger.log_checkpoint(total_episodes, checkpoint_path)
                
                # Cleanup old checkpoints
                if self.run_id:
                    cleanup_old_checkpoints(
                        checkpoint_dir=self.config.checkpoint_dir,
                        run_id=self.run_id,
                        keep_last_n=self.config.keep_last_n_checkpoints,
                        agent_id="ppo_agent"
                    )
                
                logger.info(f"Saved checkpoint at episode {total_episodes}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
        
        # Reset per-env episode tracking
        self.env_current_reward[env_id] = 0.0
        self.env_current_length[env_id] = 0
        self.env_last_info[env_id] = None  # Clear last info for next episode
        
        # Log episode statistics periodically (aggregate across all envs)
        total_episodes = self.episode_count
        if total_episodes % 10 == 0 and total_episodes > 0:
            # Get last 10 episodes across all envs
            all_rewards = self.episode_rewards
            all_lengths = self.episode_lengths
            if len(all_rewards) >= 10:
                avg_reward = np.mean(all_rewards[-10:])
                std_reward = np.std(all_rewards[-10:])
                avg_length = np.mean(all_lengths[-10:])
                std_length = np.std(all_lengths[-10:])
                
                # Calculate speed metrics (using perf_counter for high-precision timing)
                current_time = time.perf_counter()
                elapsed_since_last_log = current_time - self.last_log_time
                steps_since_last_log = self.step_count - self.last_log_step_count
                
                if elapsed_since_last_log > 0:
                    steps_per_sec = steps_since_last_log / elapsed_since_last_log
                    # Environment steps per second (accounting for vectorization)
                    env_steps_per_sec = steps_per_sec * self.num_envs
                else:
                    steps_per_sec = 0.0
                    env_steps_per_sec = 0.0
                
                # Update tracking
                self.last_log_time = current_time
                self.last_log_step_count = self.step_count
                
                logger.info(
                    f"Episodes {total_episodes - 9}-{total_episodes} (last 10): "
                    f"reward={avg_reward:.2f}±{std_reward:.2f}, "
                    f"length={avg_length:.1f}±{std_length:.1f}, "
                    f"speed={steps_per_sec:.1f} steps/s ({env_steps_per_sec:.1f} env steps/s)"
                )
    
    def _sanity_check(self, reward: float, obs: Any, action: Any, info: Dict):
        """Perform sanity checks on training data."""
        # Check for NaN/Inf in reward
        if not np.isfinite(reward):
            logger.error(f"Non-finite reward detected: {reward}")
            if self.config.mode == "smoke":
                raise ValueError(f"Non-finite reward in smoke-test mode: {reward}")
        
        # Check for NaN/Inf in observation
        if obs is not None:
            if isinstance(obs, np.ndarray):
                if not np.all(np.isfinite(obs)):
                    logger.error("Non-finite values detected in observation")
                    if self.config.mode == "smoke":
                        raise ValueError("Non-finite values in observation in smoke-test mode")
        
        # Check action validity
        if action is not None:
            if isinstance(action, (int, np.integer)):
                if action < 0:
                    logger.warning(f"Negative action detected: {action}")
            elif isinstance(action, np.ndarray):
                if not np.all(np.isfinite(action)):
                    logger.error("Non-finite values detected in action")
                    if self.config.mode == "smoke":
                        raise ValueError("Non-finite values in action in smoke-test mode")
    
    def _log_step_details(self, reward: float, done: bool, action: Any, info: Dict):
        """Log detailed step information (for smoke-test mode)."""
        logger.debug(
            f"Episode {self.episode_count + 1}, Step {self.current_episode_length}: "
            f"action={action}, reward={reward:.4f}, done={done}"
        )
        if "legal_moves_count" in info:
            logger.debug(f"  Legal moves available: {info['legal_moves_count']}")
        if "score" in info:
            logger.debug(f"  Current score: {info['score']}")
        if "pieces_remaining" in info:
            logger.debug(f"  Pieces remaining: {info['pieces_remaining']}")
    
    def _compute_win_from_info(self, env_id: int) -> Optional[float]:
        """
        Compute win value for player_0 from terminal step info dict.
        
        Win calculation:
        - 1.0 if player0_won=True and is_tie=False (player_0 wins uniquely)
        - 0.5 if is_tie=True and player_0 (id=1) in winner_ids (player_0 ties)
        - 0.0 otherwise (player_0 loses)
        - None if info dict doesn't contain game result fields (unexpected)
        
        Args:
            env_id: Environment ID to get info from
            
        Returns:
            Win value (1.0, 0.5, 0.0) or None if info is missing
        """
        info = self.env_last_info[env_id]
        
        if info is None:
            logger.warning(
                f"Episode end for env {env_id}: No info dict available. "
                f"Win detection requires terminal step info dict."
            )
            return None
        
        # Check if game result fields are present
        if "final_scores" not in info or "winner_ids" not in info or "is_tie" not in info:
            logger.warning(
                f"Episode end for env {env_id}: Missing game result fields in info dict. "
                f"Expected: final_scores, winner_ids, is_tie. "
                f"Got keys: {list(info.keys())}. "
                f"This may indicate the episode was truncated or game result wasn't computed."
            )
            return None
        
        # Extract game result information
        is_tie = info["is_tie"]
        winner_ids = info["winner_ids"]
        player0_won = info.get("player0_won", False)  # Convenience flag from env
        
        # Player_0 corresponds to Player.RED which has value=1
        player_0_id = 1  # Player.RED.value
        
        # Compute win value according to specification
        if player0_won and not is_tie:
            # Player_0 wins uniquely
            return 1.0
        elif is_tie and player_0_id in winner_ids:
            # Player_0 ties for highest score
            return 0.5
        else:
            # Player_0 loses (or tie where player_0 not in winner_ids, which shouldn't happen)
            return 0.0


def train(config: TrainingConfig, run_dir: Optional[Path] = None, experiment_name: Optional[str] = None) -> Optional[TrainingCallback]:
    """
    Train a MaskablePPO agent on the Blokus environment.
    
    Args:
        config: Training configuration object
        run_dir: Optional run directory for logs (if None, creates timestamped directory)
        experiment_name: Optional experiment name for run directory
        
    Returns:
        TrainingCallback instance (for testing/inspection), or None if training failed
    """
    # Set up logging with file output
    if run_dir is None:
        run_dir, log_file = setup_training_logging(
            base_run_dir=Path("runs"),
            experiment_name=experiment_name or config.mode,
            level=logging.INFO if config.logging_verbosity >= 1 else logging.ERROR
        )
        logger.info(f"Created run directory: {run_dir}")
        logger.info(f"Log file: {log_file}")
    else:
        # Use provided run directory, set up logging
        from utils.logging_setup import setup_logging
        log_file = setup_logging(
            run_dir,
            "training",
            level=logging.INFO if config.logging_verbosity >= 1 else logging.ERROR
        )
        logger.info(f"Using run directory: {run_dir}")
        logger.info(f"Log file: {log_file}")
    
    # Set up logging level
    if config.logging_verbosity == 0:
        logging.getLogger().setLevel(logging.ERROR)
    elif config.logging_verbosity == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif config.logging_verbosity >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load agent config if provided
    agent_config = None
    if config.agent_config_path:
        try:
            agent_config = load_agent_config(Path(config.agent_config_path))
            if agent_config:
                # Override training config with agent config values
                config.learning_rate = agent_config.learning_rate
                config.batch_size = agent_config.batch_size
                config.n_steps = agent_config.n_steps
        except Exception as e:
            logger.error(f"Failed to load agent config: {e}")
            raise
    
    # Log run start
    logger.info("=" * 80)
    logger.info("Starting Training Run")
    logger.info("=" * 80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # Log reproducibility information
    log_reproducibility_info(logger)
    logger.info("")
    
    # Log comprehensive configuration
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    config.log_config(logger)
    
    # Log additional config details (after agent_config is loaded)
    logger.info("")
    logger.info("Additional Configuration:")
    if agent_config:
        logger.info(f"  Agent Config: {config.agent_config_path}")
        logger.info(f"  Agent ID: {agent_config.agent_id}")
        logger.info(f"  Agent Name: {agent_config.name}")
        logger.info(f"  Network Architecture: {agent_config.net_arch}")
        logger.info(f"  PPO Clip Range: {agent_config.clip_range}")
        logger.info(f"  PPO Entropy Coef: {agent_config.ent_coef}")
        logger.info(f"  PPO VF Coef: {agent_config.vf_coef}")
        logger.info(f"  PPO Max Grad Norm: {agent_config.max_grad_norm}")
    else:
        logger.info("  Agent Config: Using defaults")
    logger.info("=" * 80)
    logger.info("")
    
    # Create training run logger
    # If resuming, add resume info to config metadata
    config_dict = config.to_dict()
    
    # Add reproducibility metadata
    if "metadata" not in config_dict:
        config_dict["metadata"] = {}
    config_dict["metadata"].update(get_reproducibility_metadata())
    
    # Add agent config info to metadata
    if agent_config:
        if "metadata" not in config_dict:
            config_dict["metadata"] = {}
        config_dict["metadata"]["agent_config"] = {
            "path": config.agent_config_path,
            "agent_id": agent_config.agent_id,
            "name": agent_config.name,
            "version": agent_config.version,
            "sweep_variant": agent_config.sweep_variant,
            "config_short_name": agent_config.get_config_short_name()
        }
        # Also add agent config hyperparameters to main config
        config_dict["agent_hyperparameters"] = agent_config.to_dict()
    
    if config.resume_from_checkpoint:
        if "metadata" not in config_dict:
            config_dict["metadata"] = {}
        config_dict["metadata"]["resumed_from_checkpoint"] = config.resume_from_checkpoint
        # Try to get episode from checkpoint
        try:
            resume_episode = get_checkpoint_episode(config.resume_from_checkpoint)
            if resume_episode:
                config_dict["metadata"]["resumed_from_episode"] = resume_episode
        except Exception:
            pass
    
    run_logger = None
    try:
        # Use agent config info if available
        agent_id = agent_config.agent_id if agent_config else "ppo_agent"
        algorithm = agent_config.algorithm if agent_config else "MaskablePPO"
        
        run_logger = create_training_run_logger(
            config=config_dict,
            agent_id=agent_id,
            algorithm=algorithm
        )
        if run_logger:
            logger.info(f"Training run logging enabled: run_id={run_logger.run_id}")
        else:
            logger.info("Training run logging disabled (MongoDB not available)")
    except Exception as e:
        logger.warning(f"Failed to initialize training run logger: {e}")
        run_logger = None
    
    # Set seeds for reproducibility
    set_seed(
        seed=config.random_seed,
        env_seed=config.env_seed,
        agent_seed=config.agent_seed,
        log=True
    )
    
    # Initialize the environment using factory (supports single-env and VecEnv)
    # Pass mask_fn to avoid circular import (env_factory no longer imports from trainer)
    env = make_training_env(config, mask_fn)
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.tensorboard_log_dir, exist_ok=True)
    
    # Disable PyTorch distribution validation if configured
    # NOTE: Blokus uses a huge Discrete(36400) action space.
    # The softmax over 36,400 logits can produce probabilities that sum to ~0.99998
    # instead of exactly 1 due to floating point rounding.
    # PyTorch's Distribution.validate_args=True enforces a strict Simplex constraint
    # and throws spurious errors for this case.
    if config.disable_distribution_validation:
        from torch.distributions.distribution import Distribution
        Distribution.set_default_validate_args(False)
        logger.info("Disabled PyTorch distribution validation (Simplex constraint) for large action space compatibility")
    
    # Initialize or load MaskablePPO model
    model = None
    start_episode = 0
    
    if config.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {config.resume_from_checkpoint}")
        try:
            model, checkpoint_config, extra_state = load_checkpoint(
                config.resume_from_checkpoint,
                env=env
            )
            
            # Extract episode number from checkpoint
            start_episode = get_checkpoint_episode(config.resume_from_checkpoint) or 0
            logger.info(f"Loaded checkpoint from episode {start_episode}")
            
            # Update model's environment (required for SB3)
            model.set_env(env)
            
            # Note: When resuming, a new TrainingRun will be created
            # The resume information will be stored in the config's metadata
            logger.info(f"Resuming training from episode {start_episode}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Cannot resume from checkpoint: {e}")
    else:
        # Initialize new model
        logger.info("Initializing MaskablePPO model...")
        
        # Get policy kwargs from agent config if available
        policy_kwargs = None
        if agent_config:
            policy_kwargs = agent_config.get_policy_kwargs()
            logger.info(f"Using network architecture: {agent_config.net_arch}")
        
        # Build model arguments
        model_kwargs = {
            "policy": agent_config.policy if agent_config else "MlpPolicy",
            "env": env,
            "verbose": config.logging_verbosity,
            "tensorboard_log": config.tensorboard_log_dir,
            "batch_size": config.batch_size,
            "n_steps": config.n_steps,
            "learning_rate": config.learning_rate,
        }
        
        # Add policy kwargs if available
        if policy_kwargs:
            model_kwargs["policy_kwargs"] = policy_kwargs
        
        # Add PPO-specific parameters if agent config is available
        if agent_config:
            model_kwargs["gamma"] = agent_config.gamma
            model_kwargs["n_epochs"] = agent_config.n_epochs
            model_kwargs["clip_range"] = agent_config.clip_range
            model_kwargs["ent_coef"] = agent_config.ent_coef
            model_kwargs["vf_coef"] = agent_config.vf_coef
            model_kwargs["max_grad_norm"] = agent_config.max_grad_norm
        
        model = MaskablePPO(**model_kwargs)
    
    # Create training callback
    # num_envs is determined from config or detected from model.env
    # This enables per-env tracking for VecEnv while maintaining single-env compatibility
    callback = TrainingCallback(
        config, 
        run_logger=run_logger, 
        model=model,
        verbose=config.logging_verbosity,
        num_envs=config.num_envs  # Pass num_envs explicitly for clarity
    )
    
    # Set initial episode count if resuming
    # This sets the episode count on the first env (for backward compatibility)
    if start_episode > 0:
        callback.episode_count = start_episode
    
    # DIAGNOSTIC: Inspect MaskableCategorical probabilities for numerical precision issues
    # This is only run in smoke mode to diagnose the Simplex constraint violation
    if config.mode == "smoke":
        logger.info("=" * 80)
        logger.info("DIAGNOSTIC: Inspecting MaskableCategorical probability distribution")
        logger.info("=" * 80)
        
        try:
            # Get a fresh observation and action mask from the environment
            # Reset environment to get initial state
            obs, info = env.reset()
            
            # Get action mask using the mask_fn
            action_mask = mask_fn(env)
            
            # Convert to tensors
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool).unsqueeze(0)  # Add batch dimension
            
            logger.debug(f"Diagnostic obs shape: {obs_tensor.shape}")
            logger.debug(f"Diagnostic mask shape: {mask_tensor.shape}, dtype: {mask_tensor.dtype}")
            logger.debug(f"Diagnostic mask sum: {mask_tensor.sum().item()} legal actions")
            
            # Get the distribution from the policy
            with torch.no_grad():
                # Use the policy's forward method to get the distribution
                # This mimics what happens in collect_rollouts
                features = model.policy.extract_features(obs_tensor)
                latent_pi = model.policy.mlp_extractor.forward_actor(features)
                action_logits = model.policy.action_net(latent_pi)
                
                logger.debug(f"Action logits shape: {action_logits.shape}")
                logger.debug(f"Action logits range: [{action_logits.min().item():.4f}, {action_logits.max().item():.4f}]")
                
                # Create distribution from logits (before masking)
                # This is how sb3_contrib does it internally
                from sb3_contrib.common.maskable.distributions import MaskableCategorical
                
                # Create distribution with validate_args=False to inspect probabilities even if they fail validation
                try:
                    dist = MaskableCategorical(logits=action_logits, validate_args=False)
                    # Apply masking (this is where the issue occurs)
                    dist.apply_masking(mask_tensor)
                    
                    # Get the underlying probabilities
                    probs = dist.distribution.probs
                except Exception as e:
                    logger.warning(f"Failed to create distribution with validate_args=False: {e}")
                    logger.info("Trying with validate_args=True to see the actual error...")
                    # Try with validate_args=True to see the actual error
                    try:
                        dist = MaskableCategorical(logits=action_logits, validate_args=True)
                        dist.apply_masking(mask_tensor)
                        probs = dist.distribution.probs
                    except ValueError as ve:
                        logger.error(f"Simplex constraint violation (expected): {ve}")
                        # Even if validation fails, try to get the probabilities
                        # by creating without validation
                        dist = MaskableCategorical(logits=action_logits, validate_args=False)
                        dist.apply_masking(mask_tensor)
                        probs = dist.distribution.probs
                        logger.info("Extracted probabilities despite validation error for inspection")
                
                # DIAGNOSTIC: Log probability properties
                logger.info(f"Probability tensor shape: {probs.shape}")
                logger.info(f"Probability min: {probs.min().item():.10e}")
                logger.info(f"Probability max: {probs.max().item():.10e}")
                logger.info(f"Probability mean: {probs.mean().item():.10e}")
                
                # Check for NaNs and Infs
                has_nan = torch.isnan(probs).any()
                has_inf = torch.isinf(probs).any()
                logger.info(f"Has NaN: {has_nan.item()}")
                logger.info(f"Has Inf: {has_inf.item()}")
                
                # Assert no NaNs or Infs
                assert not has_nan.item(), "Found NaN in probability distribution!"
                assert not has_inf.item(), "Found Inf in probability distribution!"
                
                # Check probability sum (this is the critical check)
                prob_sum = probs.sum(dim=-1)
                logger.info(f"Probability sum (should be ~1.0): {prob_sum.item():.10f}")
                
                # Calculate difference from 1.0
                diff_from_one = 1.0 - prob_sum.item()
                logger.info(f"Difference from 1.0: {diff_from_one:.10e}")
                
                # Log some sample probabilities
                sample_size = min(20, probs.shape[-1])
                sample_probs = probs[0, :sample_size].cpu().numpy()
                logger.debug(f"Sample probabilities (first {sample_size}): {sample_probs}")
                
                # Check if sum is close to 1 (within numerical precision)
                is_close_to_one = abs(diff_from_one) < 1e-4
                logger.info(f"Sum is close to 1.0 (within 1e-4): {is_close_to_one}")
                
                if not is_close_to_one:
                    logger.warning(
                        f"WARNING: Probability sum is {prob_sum.item():.10f}, "
                        f"diff from 1.0 is {diff_from_one:.10e}. "
                        f"This may cause Simplex constraint violation!"
                    )
                
                # Check if all probabilities are non-negative
                has_negative = (probs < 0).any()
                logger.info(f"Has negative probabilities: {has_negative.item()}")
                assert not has_negative.item(), "Found negative probabilities!"
                
                logger.info("=" * 80)
                logger.info("DIAGNOSTIC: Probability inspection complete")
                logger.info("=" * 80)
                
        except Exception as e:
            logger.error(f"DIAGNOSTIC: Error during probability inspection: {e}")
            logger.error("This diagnostic failed, but training will continue...")
            import traceback
            traceback.print_exc()
    
    # Train the model
    logger.info(f"Starting training in {config.mode.upper()} mode...")
    logger.info(f"Target: {config.total_timesteps} total timesteps")
    
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            progress_bar=True,
            callback=callback
        )
        
        # Update status to completed
        if run_logger:
            run_logger.update_status("completed")
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if run_logger:
            run_logger.update_status("stopped", error_message="Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        if run_logger:
            run_logger.update_status("failed", error_message=str(e))
        if config.mode == "smoke":
            raise  # Re-raise in smoke-test mode to catch issues early
        else:
            logger.warning("Continuing despite error (not in smoke-test mode)")
    
    # Log final statistics (aggregate across all environments)
    total_episodes = callback.episode_count
    if total_episodes > 0:
        total_time = time.perf_counter() - callback.start_time
        total_steps = callback.step_count
        total_env_steps = total_steps * callback.num_envs
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("Training Summary")
        logger.info("=" * 80)
        logger.info(f"Total episodes: {total_episodes}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Total environment steps: {total_env_steps}")
        logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        
        # Speed metrics
        if total_time > 0:
            steps_per_sec = total_steps / total_time
            env_steps_per_sec = total_env_steps / total_time
            logger.info(f"Average speed: {steps_per_sec:.1f} steps/s ({env_steps_per_sec:.1f} env steps/s)")
        
        # Aggregate episode rewards and lengths across all envs
        all_episode_rewards = callback.episode_rewards
        all_episode_lengths = callback.episode_lengths
        
        if all_episode_rewards:
            logger.info("")
            logger.info("Episode Statistics:")
            logger.info(f"  Average reward: {np.mean(all_episode_rewards):.2f} ± {np.std(all_episode_rewards):.2f}")
            logger.info(f"  Average episode length: {np.mean(all_episode_lengths):.1f} ± {np.std(all_episode_lengths):.1f}")
            logger.info(f"  Best episode reward: {np.max(all_episode_rewards):.2f}")
            logger.info(f"  Worst episode reward: {np.min(all_episode_rewards):.2f}")
            
            # Optionally show per-env stats if using multiple envs
            if callback.num_envs > 1:
                logger.info("")
                logger.info("Per-environment statistics:")
                for env_id in range(callback.num_envs):
                    env_rewards = callback.env_episode_rewards[env_id]
                    if env_rewards:
                        logger.info(
                            f"  Env {env_id}: {len(env_rewards)} episodes, "
                            f"avg reward: {np.mean(env_rewards):.2f} ± {np.std(env_rewards):.2f}, "
                            f"best: {np.max(env_rewards):.2f}, "
                            f"worst: {np.min(env_rewards):.2f}"
                        )
        
        logger.info("=" * 80)
    
    # Save final model checkpoint
    if callback.episode_count > 0:
        # Use structured checkpoint path if we have a run_id
        if run_logger and run_logger.run_id:
            final_checkpoint_path = get_checkpoint_path(
                checkpoint_dir=config.checkpoint_dir,
                run_id=run_logger.run_id,
                episode=callback.episode_count,
                agent_id="ppo_agent"
            )
        else:
            # Fallback to old format for backward compatibility
            final_checkpoint_path = os.path.join(config.checkpoint_dir, "ppo_blokus")
        
        logger.info(f"Saving final model checkpoint to {final_checkpoint_path}")
        try:
            save_checkpoint(
                model=model,
                checkpoint_path=final_checkpoint_path,
                episode=callback.episode_count,
                config=config.to_dict(),
                run_id=run_logger.run_id if run_logger else None
            )
            logger.info("Final checkpoint saved successfully")
            
            # Log checkpoint if logger is available
            if run_logger:
                run_logger.log_checkpoint(callback.episode_count, final_checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")
            # Fallback to simple save
            try:
                model.save(final_checkpoint_path)
                logger.info("Saved model using simple save method")
            except Exception as e2:
                logger.error(f"Failed to save model: {e2}")
    
    # Save config for reproducibility
    config_path = os.path.join(config.checkpoint_dir, "training_config.yaml")
    try:
        config.save_to_file(Path(config_path))
        logger.info(f"Training config saved to {config_path}")
    except Exception as e:
        logger.warning(f"Failed to save config: {e}")
    
    # Return callback for testing/inspection
    return callback


def main():
    """Main entry point for training script."""
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Parse config from args
    config = parse_args_to_config(args)
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()
