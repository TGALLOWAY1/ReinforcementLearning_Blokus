"""
Seed initialization utilities for reproducibility in RL training.
"""

import logging
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, skipping torch seed initialization")


def set_seed(
    seed: Optional[int],
    env_seed: Optional[int] = None,
    agent_seed: Optional[int] = None,
    log: bool = True
) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Base random seed (sets Python, NumPy, PyTorch if available)
        env_seed: Optional separate seed for environment (defaults to seed)
        agent_seed: Optional separate seed for agent (defaults to seed)
        log: Whether to log seed values
    """
    if seed is None:
        if log:
            logger.info("No seed provided, using random initialization")
        return
    
    # Use base seed for env/agent if not separately specified
    if env_seed is None:
        env_seed = seed
    if agent_seed is None:
        agent_seed = seed
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (if available)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    if log:
        logger.info(f"Seeds initialized:")
        logger.info(f"  Base seed: {seed}")
        logger.info(f"  Environment seed: {env_seed}")
        logger.info(f"  Agent seed: {agent_seed}")


def set_env_seed(seed: Optional[int]) -> None:
    """Set seed specifically for environment."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


def set_agent_seed(seed: Optional[int]) -> None:
    """Set seed specifically for agent/RL algorithm."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

