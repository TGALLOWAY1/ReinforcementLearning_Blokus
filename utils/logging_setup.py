"""
Logging setup utilities for training runs.

This module provides functions to configure logging with both console and file output,
creating timestamped run directories for organized log storage.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Path,
    run_name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> Path:
    """
    Set up logging with both console and file handlers.
    
    Creates a log file in the specified directory with the run name.
    Configures both file and console handlers with consistent formatting.
    
    Args:
        log_dir: Directory where log file should be created
        run_name: Name for the log file (without extension)
        level: Logging level (default: logging.INFO)
        format_string: Optional custom format string. If None, uses default format.
        
    Returns:
        Path to the created log file
        
    Example:
        >>> log_file = setup_logging(Path("runs/exp1"), "training_run", logging.INFO)
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("This will be logged to both console and file")
    """
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file path
    log_file = log_dir / f"{run_name}.log"
    
    # Default format: timestamp, level, logger name, message
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create formatter
    formatter = logging.Formatter(
        format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    # (useful if setup_logging is called multiple times)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return log_file


def create_run_directory(
    base_dir: Path = Path("runs"),
    experiment_name: Optional[str] = None
) -> Path:
    """
    Create a timestamped run directory for organizing training outputs.
    
    Directory format: <base_dir>/<YYYYMMDD>_<HHMMSS>_<experiment_name>/
    If experiment_name is None, uses "run" as default.
    
    Args:
        base_dir: Base directory for runs (default: "runs")
        experiment_name: Optional experiment name (default: "run")
        
    Returns:
        Path to the created run directory
        
    Example:
        >>> run_dir = create_run_directory(experiment_name="ppo_baseline")
        >>> # Creates: runs/20250115_143022_ppo_baseline/
    """
    # Create base directory if it doesn't exist
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use experiment name or default
    exp_name = experiment_name or "run"
    
    # Create run directory name
    run_dir_name = f"{timestamp}_{exp_name}"
    run_dir = base_dir / run_dir_name
    
    # Create directory
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def setup_training_logging(
    base_run_dir: Path = Path("runs"),
    experiment_name: Optional[str] = None,
    level: int = logging.INFO
) -> tuple:
    """
    Set up complete logging infrastructure for a training run.
    
    Creates a timestamped run directory and sets up both file and console logging.
    This is the main entry point for training logging setup.
    
    Args:
        base_run_dir: Base directory for runs (default: "runs")
        experiment_name: Optional experiment name (default: "run")
        level: Logging level (default: logging.INFO)
        
    Returns:
        Tuple of (run_directory, log_file_path)
        
    Example:
        >>> run_dir, log_file = setup_training_logging(experiment_name="ppo_v1")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Training started")
    """
    # Create run directory
    run_dir = create_run_directory(base_run_dir, experiment_name)
    
    # Set up logging
    log_file = setup_logging(run_dir, "training", level)
    
    return run_dir, log_file

