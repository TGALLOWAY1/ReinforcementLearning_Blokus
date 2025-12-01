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
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from envs.blokus_v0 import make_gymnasium_env
from training.config import TrainingConfig, create_arg_parser, parse_args_to_config
from training.seeds import set_seed
from training.run_logger import create_training_run_logger, TrainingRunLogger
from training.checkpoints import (
    save_checkpoint, load_checkpoint, get_checkpoint_path,
    cleanup_old_checkpoints, get_checkpoint_episode
)
from training.agent_config import load_agent_config, AgentConfig
from training.reproducibility import get_reproducibility_metadata, log_reproducibility_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def mask_fn(env):
    """
    Helper function to extract the legal action mask from the environment.
    
    Args:
        env: The GymnasiumBlokusWrapper environment (ActionMasker passes the unwrapped env)
        
    Returns:
        The legal action mask array
    """
    # Access the underlying BlokusEnv: env (GymnasiumBlokusWrapper) -> env.env (BlokusEnv)
    blokus_env = env.env
    agent_name = env.agent_name
    
    # Get the mask for our agent (player_0)
    # The info dict should always be up to date after reset/step
    if agent_name in blokus_env.infos:
        mask = blokus_env.infos[agent_name]["legal_action_mask"]
        # Convert to numpy boolean array
        mask = np.asarray(mask, dtype=np.bool_)
        
        # Ensure mask has correct shape
        if mask.shape[0] != env.action_space.n:
            raise ValueError(f"Mask shape {mask.shape} doesn't match action space {env.action_space.n}")
        
        # Ensure at least one action is available
        if not np.any(mask):
            # This shouldn't happen in normal gameplay, but handle it
            raise ValueError(f"No legal actions available for {agent_name}")
        
        return mask
    else:
        raise ValueError(f"Agent {agent_name} not found in environment infos")


class TrainingCallback(BaseCallback):
    """
    Custom callback for training with episode limits, sanity checks, and detailed logging.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        run_logger: Optional[TrainingRunLogger] = None,
        model: Optional[Any] = None,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.config = config
        self.run_logger = run_logger
        self.model = model
        self.episode_count = 0
        self.step_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.last_obs = None
        self.last_action = None
        self.last_reward = None
        self.last_done = None
        self.run_id = run_logger.run_id if run_logger else None
        
    def _on_step(self) -> bool:
        """
        Called at each step during training.
        
        Returns:
            True to continue training, False to stop
        """
        # Track step
        self.step_count += 1
        self.current_episode_length += 1
        
        # Get current info from SB3 callback locals
        # SB3 provides these as lists (for vectorized envs), but we use single env
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        obs = self.locals.get("obs", None)
        actions = self.locals.get("actions", [])
        
        # Extract values (handle both list and single value)
        reward = rewards[0] if rewards else 0.0
        done = dones[0] if dones else False
        info = infos[0] if infos else {}
        action = actions[0] if actions else None
        
        self.current_episode_reward += reward
        self.last_obs = obs
        self.last_action = action
        self.last_reward = reward
        self.last_done = done
        
        # Sanity checks
        if self.config.enable_sanity_checks:
            self._sanity_check(reward, obs, action, info)
        
        # Detailed logging in smoke-test mode
        if self.config.log_action_details and self.episode_count < 3 and self.current_episode_length <= 10:
            self._log_step_details(reward, done, action, info)
        
        # Check episode termination
        if done:
            self._on_episode_end()
        
        # Check episode limit
        if self.config.max_episodes is not None and self.episode_count >= self.config.max_episodes:
            logger.info(f"Reached max_episodes limit ({self.config.max_episodes}), stopping training")
            return False
        
        # Check step limit per episode
        if self.current_episode_length >= self.config.max_steps_per_episode:
            logger.warning(
                f"Episode {self.episode_count + 1} reached max_steps_per_episode "
                f"({self.config.max_steps_per_episode}), truncating"
            )
            self._on_episode_end()
            return True
        
        return True
    
    def _on_episode_end(self):
        """Handle episode end."""
        self.episode_count += 1
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)
        
        # Log to MongoDB if logger is available
        if self.run_logger:
            # Determine win status (simplified: positive reward indicates good performance)
            # TODO: Implement proper win detection based on game outcome
            win = None  # Will be None for now since we don't have game outcome in callback
            self.run_logger.log_episode(
                episode=self.episode_count,
                total_reward=self.current_episode_reward,
                steps=self.current_episode_length,
                win=win
            )
        
        if self.config.mode == "smoke" or self.config.logging_verbosity >= 1:
            logger.info(
                f"Episode {self.episode_count} completed: "
                f"reward={self.current_episode_reward:.2f}, "
                f"length={self.current_episode_length}"
            )
        
        # Save checkpoint if interval is reached
        if (self.model and 
            self.config.checkpoint_interval_episodes and 
            self.episode_count % self.config.checkpoint_interval_episodes == 0):
            try:
                checkpoint_path = get_checkpoint_path(
                    checkpoint_dir=self.config.checkpoint_dir,
                    run_id=self.run_id or "unknown",
                    episode=self.episode_count,
                    agent_id="ppo_agent"
                )
                
                save_checkpoint(
                    model=self.model,
                    checkpoint_path=checkpoint_path,
                    episode=self.episode_count,
                    config=self.config.to_dict(),
                    run_id=self.run_id
                )
                
                # Log checkpoint to MongoDB
                if self.run_logger:
                    self.run_logger.log_checkpoint(self.episode_count, checkpoint_path)
                
                # Cleanup old checkpoints
                if self.run_id:
                    cleanup_old_checkpoints(
                        checkpoint_dir=self.config.checkpoint_dir,
                        run_id=self.run_id,
                        keep_last_n=self.config.keep_last_n_checkpoints,
                        agent_id="ppo_agent"
                    )
                
                logger.info(f"Saved checkpoint at episode {self.episode_count}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
        
        # Reset episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        # Log episode statistics periodically
        if self.episode_count % 10 == 0 and self.episode_count > 0:
            avg_reward = np.mean(self.episode_rewards[-10:])
            avg_length = np.mean(self.episode_lengths[-10:])
            logger.info(
                f"Last 10 episodes - Avg reward: {avg_reward:.2f}, "
                f"Avg length: {avg_length:.1f}"
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


def train(config: TrainingConfig) -> None:
    """
    Train a MaskablePPO agent on the Blokus environment.
    
    Args:
        config: Training configuration object
    """
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
                agent_config.log_config(logger)
                # Override training config with agent config values
                config.learning_rate = agent_config.learning_rate
                config.batch_size = agent_config.batch_size
                config.n_steps = agent_config.n_steps
        except Exception as e:
            logger.error(f"Failed to load agent config: {e}")
            raise
    
    # Log reproducibility information
    log_reproducibility_info(logger)
    
    # Log configuration
    config.log_config(logger)
    
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
    
    # Initialize the environment with max_episode_steps
    env = make_gymnasium_env(
        render_mode=None,
        max_episode_steps=config.max_steps_per_episode
    )
    
    # Set environment seed if specified
    if config.env_seed is not None:
        env.reset(seed=config.env_seed)
    else:
        env.reset()
    
    # Wrap environment with ActionMasker
    env = ActionMasker(env, mask_fn)
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.tensorboard_log_dir, exist_ok=True)
    
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
    callback = TrainingCallback(
        config, 
        run_logger=run_logger, 
        model=model,
        verbose=config.logging_verbosity
    )
    
    # Set initial episode count if resuming
    if start_episode > 0:
        callback.episode_count = start_episode
    
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
    
    # Log final statistics
    if callback.episode_count > 0:
        logger.info("=" * 80)
        logger.info("Training Summary")
        logger.info("=" * 80)
        logger.info(f"Total episodes: {callback.episode_count}")
        logger.info(f"Total steps: {callback.step_count}")
        if callback.episode_rewards:
            logger.info(f"Average reward: {np.mean(callback.episode_rewards):.2f}")
            logger.info(f"Average episode length: {np.mean(callback.episode_lengths):.1f}")
            logger.info(f"Best episode reward: {np.max(callback.episode_rewards):.2f}")
            logger.info(f"Worst episode reward: {np.min(callback.episode_rewards):.2f}")
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
