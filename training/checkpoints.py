"""
Checkpoint utilities for saving and loading training state.

This module provides functions to save and load complete training checkpoints,
including model weights, optimizer state, training configuration, and metadata.

Current model saving behavior:
- Models are saved using Stable-Baselines3's built-in `model.save()` method
- Final model is saved at end of training to `checkpoints/ppo_blokus`
- No periodic checkpointing is currently implemented
- No resume functionality exists
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from sb3_contrib import MaskablePPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("Stable-Baselines3 not available for checkpointing")


def get_checkpoint_path(
    checkpoint_dir: str,
    run_id: str,
    episode: int,
    agent_id: Optional[str] = None
) -> str:
    """
    Generate a checkpoint file path.
    
    Directory structure: checkpoints/<agent_id>/<run_id>/ep<episode>.zip
    
    Args:
        checkpoint_dir: Base checkpoint directory
        run_id: Training run ID
        episode: Episode number
        agent_id: Agent identifier (default: "ppo_agent")
        
    Returns:
        Full path to checkpoint file
    """
    agent_id = agent_id or "ppo_agent"
    run_dir = os.path.join(checkpoint_dir, agent_id, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    checkpoint_filename = f"ep{episode:06d}.zip"
    return os.path.join(run_dir, checkpoint_filename)


def save_checkpoint(
    model: Any,
    checkpoint_path: str,
    episode: int,
    config: Dict[str, Any],
    run_id: Optional[str] = None,
    extra_state: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save a complete training checkpoint.
    
    This saves:
    - Model weights and optimizer state (via SB3's model.save())
    - Training configuration
    - Episode number and metadata
    - Any extra state provided
    
    Args:
        model: Stable-Baselines3 model instance
        checkpoint_path: Path where checkpoint should be saved
        episode: Current episode number
        config: Training configuration dictionary
        run_id: Optional training run ID
        extra_state: Optional additional state to save
        
    Returns:
        Path to saved checkpoint
    """
    if not SB3_AVAILABLE:
        raise RuntimeError("Stable-Baselines3 not available for checkpointing")
    
    # Ensure directory exists
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model using SB3's built-in method
    # This saves model weights, optimizer state, and other SB3 internal state
    model.save(checkpoint_path)
    logger.info(f"Saved model checkpoint to {checkpoint_path}")
    
    # Save additional metadata in a companion JSON file
    metadata_path = checkpoint_path.replace('.zip', '_metadata.json')
    metadata = {
        "episode": episode,
        "run_id": run_id,
        "config": config,
        "timestamp": datetime.utcnow().isoformat(),
        "extra_state": extra_state or {}
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.debug(f"Saved checkpoint metadata to {metadata_path}")
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    env: Optional[Any] = None
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (.zip)
        env: Optional environment instance (required for SB3 model loading)
        
    Returns:
        Tuple of (model, config, extra_state)
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If SB3 is not available or loading fails
    """
    if not SB3_AVAILABLE:
        raise RuntimeError("Stable-Baselines3 not available for checkpointing")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model using SB3's built-in method
    # This restores model weights, optimizer state, and SB3 internal state
    if env is None:
        raise ValueError("Environment is required to load SB3 model")
    
    try:
        model = MaskablePPO.load(checkpoint_path, env=env)
        logger.info(f"Loaded model checkpoint from {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")
    
    # Load metadata if available
    metadata_path = checkpoint_path.replace('.zip', '_metadata.json')
    config = {}
    extra_state = {}
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                config = metadata.get("config", {})
                extra_state = metadata.get("extra_state", {})
                logger.debug(f"Loaded checkpoint metadata from {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint metadata: {e}")
    
    return model, config, extra_state


def get_checkpoint_episode(checkpoint_path: str) -> Optional[int]:
    """
    Extract episode number from checkpoint path or metadata.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Episode number if found, None otherwise
    """
    # Try to extract from filename (epXXXXXX.zip)
    filename = os.path.basename(checkpoint_path)
    if filename.startswith("ep") and filename.endswith(".zip"):
        try:
            episode_str = filename[2:-4]  # Remove "ep" prefix and ".zip" suffix
            return int(episode_str)
        except ValueError:
            pass
    
    # Try to load from metadata
    metadata_path = checkpoint_path.replace('.zip', '_metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get("episode")
        except Exception:
            pass
    
    return None


def list_checkpoints(
    checkpoint_dir: str,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None
) -> list:
    """
    List all checkpoints in a directory.
    
    Args:
        checkpoint_dir: Base checkpoint directory
        run_id: Optional run ID to filter by
        agent_id: Optional agent ID to filter by
        
    Returns:
        List of checkpoint paths, sorted by episode number
    """
    checkpoints = []
    
    if run_id and agent_id:
        # Specific run directory
        run_dir = os.path.join(checkpoint_dir, agent_id, run_id)
        if os.path.exists(run_dir):
            for filename in os.listdir(run_dir):
                if filename.endswith('.zip') and filename.startswith('ep'):
                    checkpoint_path = os.path.join(run_dir, filename)
                    episode = get_checkpoint_episode(checkpoint_path)
                    if episode is not None:
                        checkpoints.append((episode, checkpoint_path))
    else:
        # Search all runs
        agent_id = agent_id or "ppo_agent"
        agent_dir = os.path.join(checkpoint_dir, agent_id)
        if os.path.exists(agent_dir):
            for run_dir_name in os.listdir(agent_dir):
                run_dir = os.path.join(agent_dir, run_dir_name)
                if os.path.isdir(run_dir):
                    for filename in os.listdir(run_dir):
                        if filename.endswith('.zip') and filename.startswith('ep'):
                            checkpoint_path = os.path.join(run_dir, filename)
                            episode = get_checkpoint_episode(checkpoint_path)
                            if episode is not None:
                                checkpoints.append((episode, checkpoint_path))
    
    # Sort by episode number
    checkpoints.sort(key=lambda x: x[0])
    return [path for _, path in checkpoints]


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    run_id: str,
    keep_last_n: int,
    agent_id: Optional[str] = None
) -> int:
    """
    Remove old checkpoints, keeping only the most recent N.
    
    Args:
        checkpoint_dir: Base checkpoint directory
        run_id: Training run ID
        keep_last_n: Number of checkpoints to keep
        agent_id: Optional agent identifier
        
    Returns:
        Number of checkpoints removed
    """
    agent_id = agent_id or "ppo_agent"
    run_dir = os.path.join(checkpoint_dir, agent_id, run_id)
    
    if not os.path.exists(run_dir):
        return 0
    
    # Get all checkpoints with their episodes
    checkpoints = []
    for filename in os.listdir(run_dir):
        if filename.endswith('.zip') and filename.startswith('ep'):
            checkpoint_path = os.path.join(run_dir, filename)
            episode = get_checkpoint_episode(checkpoint_path)
            if episode is not None:
                checkpoints.append((episode, checkpoint_path))
    
    # Sort by episode (descending)
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    
    # Remove old checkpoints
    removed_count = 0
    for episode, checkpoint_path in checkpoints[keep_last_n:]:
        try:
            os.remove(checkpoint_path)
            # Also remove metadata file if it exists
            metadata_path = checkpoint_path.replace('.zip', '_metadata.json')
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            removed_count += 1
            logger.debug(f"Removed old checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")
    
    if removed_count > 0:
        logger.info(f"Cleaned up {removed_count} old checkpoint(s), kept {min(keep_last_n, len(checkpoints))}")
    
    return removed_count

