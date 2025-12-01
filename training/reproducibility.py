"""
Reproducibility utilities for capturing code version, git hash, and environment info.

This module provides functions to capture metadata that helps ensure
reproducibility of training runs.
"""

import logging
import os
import subprocess
import sys
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_git_hash() -> Optional[str]:
    """
    Get the current git commit hash.
    
    Returns:
        Git commit hash or None if not available
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_branch() -> Optional[str]:
    """
    Get the current git branch name.
    
    Returns:
        Git branch name or None if not available
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_python_version() -> str:
    """Get Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_environment_info() -> Dict[str, Any]:
    """
    Get environment and system information.
    
    Returns:
        Dictionary with environment info
    """
    info = {
        "python_version": get_python_version(),
        "platform": sys.platform,
    }
    
    # Try to get package versions
    try:
        import torch
        info["torch_version"] = torch.__version__
    except ImportError:
        pass
    
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except ImportError:
        pass
    
    try:
        import gymnasium
        info["gymnasium_version"] = gymnasium.__version__
    except ImportError:
        pass
    
    try:
        import stable_baselines3
        info["sb3_version"] = stable_baselines3.__version__
    except ImportError:
        pass
    
    return info


def get_reproducibility_metadata() -> Dict[str, Any]:
    """
    Get complete reproducibility metadata.
    
    Returns:
        Dictionary with git hash, branch, code version, and environment info
    """
    metadata = {
        "code_version": "1.0.0",  # TODO: Extract from pyproject.toml or __version__
        "git_hash": get_git_hash(),
        "git_branch": get_git_branch(),
        "environment": get_environment_info()
    }
    
    return metadata


def log_reproducibility_info(logger: logging.Logger):
    """Log reproducibility information."""
    metadata = get_reproducibility_metadata()
    
    logger.info("Reproducibility Information:")
    logger.info(f"  Code Version: {metadata.get('code_version', 'Unknown')}")
    if metadata.get("git_hash"):
        logger.info(f"  Git Hash: {metadata['git_hash']}")
    if metadata.get("git_branch"):
        logger.info(f"  Git Branch: {metadata['git_branch']}")
    logger.info(f"  Python Version: {metadata['environment'].get('python_version', 'Unknown')}")
    
    if "torch_version" in metadata["environment"]:
        logger.info(f"  PyTorch Version: {metadata['environment']['torch_version']}")
    if "sb3_version" in metadata["environment"]:
        logger.info(f"  Stable-Baselines3 Version: {metadata['environment']['sb3_version']}")

