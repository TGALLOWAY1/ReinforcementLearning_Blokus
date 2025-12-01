"""
Main training script for Blokus RL using MaskablePPO.
"""

import argparse
import os
import sys
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from envs.blokus_v0 import make_gymnasium_env


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


def train(args):
    """
    Train a MaskablePPO agent on the Blokus environment.
    
    Args:
        args: Command line arguments containing training hyperparameters
    """
    # Initialize the environment
    env = make_gymnasium_env()
    
    # Wrap environment with ActionMasker
    env = ActionMasker(env, mask_fn)
    
    # Initialize MaskablePPO model
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs",
        batch_size=64,
        n_steps=args.n_steps,
        learning_rate=args.lr,
    )
    
    # Train the model
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)
    
    # Save the model
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/ppo_blokus")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MaskablePPO agent on Blokus")
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100000,
        help="Total number of timesteps to train (default: 100000)"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2048,
        help="Number of steps to collect per update. Lowering this to 512 speeds up the first update (default: 2048)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    
    args = parser.parse_args()
    train(args)
