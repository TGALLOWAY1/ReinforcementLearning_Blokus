"""
Hyperparameter sweep runner for quick sanity checks.

This script runs short training runs with multiple agent configurations
to quickly compare hyperparameter settings before committing to long training runs.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from typing import Optional

from training.agent_config import AgentConfig, find_agent_configs
from training.config import TrainingConfig
from training.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_sweep(
    config_paths: List[Path],
    base_config: Optional[TrainingConfig] = None,
    episodes_per_config: int = 100,
    mode: str = "smoke"
) -> None:
    """
    Run hyperparameter sweep across multiple agent configs.
    
    Args:
        config_paths: List of agent config file paths
        base_config: Base training config (optional)
        episodes_per_config: Number of episodes to run per config
        mode: Training mode ("smoke" or "full")
    """
    if not config_paths:
        logger.error("No agent config files found")
        return
    
    logger.info("=" * 80)
    logger.info("Hyperparameter Sweep")
    logger.info("=" * 80)
    logger.info(f"Configs to test: {len(config_paths)}")
    logger.info(f"Episodes per config: {episodes_per_config}")
    logger.info(f"Mode: {mode}")
    logger.info("=" * 80)
    
    results = []
    
    for i, config_path in enumerate(config_paths, 1):
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Running config {i}/{len(config_paths)}: {config_path.name}")
        logger.info("=" * 80)
        
        try:
            # Load agent config
            agent_config = AgentConfig.from_file(config_path)
            logger.info(f"Agent: {agent_config.get_config_name()}")
            
            # Create training config
            if base_config is None:
                sweep_config = TrainingConfig()
            else:
                # Create a copy
                sweep_config = TrainingConfig(**base_config.to_dict())
            
            # Override for sweep mode
            sweep_config.mode = mode
            sweep_config.max_episodes = episodes_per_config
            sweep_config.agent_config_path = str(config_path)
            
            # Adjust for short runs
            if mode == "smoke":
                sweep_config.total_timesteps = min(sweep_config.total_timesteps, 10000)
                sweep_config.max_steps_per_episode = 100
                sweep_config.checkpoint_interval_episodes = None  # No checkpoints in sweeps
            
            # Run training
            try:
                train(sweep_config)
                results.append({
                    "config": config_path.name,
                    "agent_config": agent_config,
                    "status": "completed"
                })
                logger.info(f"✓ Completed: {config_path.name}")
            except Exception as e:
                logger.error(f"✗ Failed: {config_path.name} - {e}")
                results.append({
                    "config": config_path.name,
                    "agent_config": agent_config,
                    "status": "failed",
                    "error": str(e)
                })
        
        except Exception as e:
            logger.error(f"✗ Failed to load config {config_path.name}: {e}")
            results.append({
                "config": config_path.name,
                "status": "error",
                "error": str(e)
            })
    
    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Sweep Summary")
    logger.info("=" * 80)
    
    completed = [r for r in results if r.get("status") == "completed"]
    failed = [r for r in results if r.get("status") != "completed"]
    
    logger.info(f"Completed: {len(completed)}/{len(results)}")
    logger.info(f"Failed: {len(failed)}/{len(results)}")
    
    if completed:
        logger.info("")
        logger.info("Completed configs:")
        for result in completed:
            agent_config = result.get("agent_config")
            if agent_config:
                logger.info(f"  - {agent_config.get_config_name()}")
    
    if failed:
        logger.info("")
        logger.info("Failed configs:")
        for result in failed:
            logger.info(f"  - {result['config']}: {result.get('error', 'Unknown error')}")
    
    logger.info("")
    logger.info("View results in Training History: http://localhost:5173/training")
    logger.info("=" * 80)


def main():
    """Main entry point for sweep script."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweep across multiple agent configs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "config_pattern",
        type=str,
        help="Glob pattern for agent config files (e.g., 'config/agents/ppo_agent_sweep_*.yaml')"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run per config (default: 100)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["smoke", "full"],
        default="smoke",
        help="Training mode (default: smoke)"
    )
    
    parser.add_argument(
        "--base-config",
        type=str,
        default=None,
        help="Path to base training config file (optional)"
    )
    
    args = parser.parse_args()
    
    # Find config files
    config_paths = find_agent_configs(args.config_pattern)
    
    if not config_paths:
        logger.error(f"No agent config files found matching pattern: {args.config_pattern}")
        logger.info("Example patterns:")
        logger.info("  config/agents/ppo_agent_sweep_*.yaml")
        logger.info("  config/agents/*.yaml")
        sys.exit(1)
    
    # Load base config if provided
    base_config = None
    if args.base_config:
        try:
            base_config = TrainingConfig.from_file(Path(args.base_config))
        except Exception as e:
            logger.error(f"Failed to load base config: {e}")
            sys.exit(1)
    
    # Run sweep
    run_sweep(
        config_paths=config_paths,
        base_config=base_config,
        episodes_per_config=args.episodes,
        mode=args.mode
    )


if __name__ == "__main__":
    main()

