"""
TrainingRun logger for MongoDB integration.

This module provides utilities to log training runs to MongoDB,
including episode metrics, status updates, and checkpoint tracking.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import MongoDB dependencies
try:
    import os
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    from webapi.db.models import (
        CheckpointPath,
        EpisodeMetric,
        RollingWinRate,
        TrainingRun,
    )
    from webapi.db.mongo import get_database
    MONGODB_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    logger.warning(f"MongoDB not available for training run logging: {e}")
    MONGODB_AVAILABLE = False
    TrainingRun = None
    EpisodeMetric = None


class TrainingRunLogger:
    """
    Logger for training runs that writes to MongoDB.
    
    This class handles creating, updating, and querying TrainingRun records
    during RL training sessions.
    """
    
    def __init__(self, run_id: Optional[str] = None, agent_id: str = "ppo_agent", algorithm: str = "MaskablePPO"):
        """
        Initialize the training run logger.
        
        Args:
            run_id: Optional run ID (will generate UUID if not provided)
            agent_id: Agent identifier for logging
            algorithm: Algorithm name for logging
        """
        self.run_id = run_id or str(uuid.uuid4())
        self.agent_id = agent_id
        self.algorithm = algorithm
        self.episodes: List[Dict[str, Any]] = []
        self.rolling_win_rates: List[Dict[str, Any]] = []
        self.window_size = 100  # For rolling win rate calculation
        self._run_created = False
        
    def create_run(
        self,
        config: Dict[str, Any]
    ) -> bool:
        """
        Create a new TrainingRun record in MongoDB.
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not MONGODB_AVAILABLE:
            logger.warning("MongoDB not available, skipping run creation")
            return False
        
        try:
            # Run async operation in sync context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, schedule the coroutine
                asyncio.create_task(self._create_run_async(config))
            else:
                loop.run_until_complete(self._create_run_async(config))
            
            self._run_created = True
            logger.info(f"Created TrainingRun: {self.run_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create training run: {e}")
            return False
    
    async def _create_run_async(self, config: Dict[str, Any]):
        """Async helper to create run in MongoDB."""
        try:
            db = get_database()
            training_run = TrainingRun(
                run_id=self.run_id,
                agent_id=self.agent_id,
                algorithm=self.algorithm,
                config=config,
                status="running",
                start_time=datetime.utcnow(),
                metrics={
                    "episodes": [],
                    "rolling_win_rate": []
                }
            )
            
            await db.training_runs.insert_one(training_run.dict(by_alias=True))
            logger.info(f"TrainingRun {self.run_id} created in MongoDB")
        except Exception as e:
            logger.error(f"Error creating training run in MongoDB: {e}")
            raise
    
    def log_episode(
        self,
        episode: int,
        total_reward: float,
        steps: int,
        win: Optional[float] = None,
        epsilon: Optional[float] = None
    ) -> bool:
        """
        Log metrics for a single episode.
        
        Args:
            episode: Episode number
            total_reward: Total reward for the episode
            steps: Number of steps in the episode
            win: Win value (1.0=win, 0.5=tie, 0.0=loss) or None if unavailable
            epsilon: Exploration rate (if applicable)
            
        Returns:
            True if successful, False otherwise
        """
        if not MONGODB_AVAILABLE or not self._run_created:
            return False
        
        episode_metric = {
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "win": win,
            "epsilon": epsilon
        }
        
        self.episodes.append(episode_metric)
        
        # Update rolling win rate if win information is available
        if win is not None:
            self._update_rolling_win_rate(episode, win)
        
        try:
            # Update MongoDB asynchronously
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._update_episodes_async())
            else:
                loop.run_until_complete(self._update_episodes_async())
            
            return True
        except Exception as e:
            logger.error(f"Failed to log episode {episode}: {e}")
            return False
    
    def _update_rolling_win_rate(self, episode: int, win: float):
        """
        Update rolling win rate calculation.
        
        Win values: 1.0 = win, 0.5 = tie, 0.0 = loss
        Win rate is calculated as: (wins + 0.5 * ties) / total_episodes
        This gives partial credit for ties.
        """
        # Calculate win rate over last N episodes
        recent_episodes = [e for e in self.episodes if e.get("win") is not None]
        if len(recent_episodes) >= self.window_size:
            recent_wins = recent_episodes[-self.window_size:]
            # Win rate = (wins + 0.5 * ties) / total
            # win > 0.5 counts as full win, win == 0.5 counts as half, win < 0.5 counts as loss
            win_rate = sum(e["win"] for e in recent_wins) / len(recent_wins)
        else:
            recent_wins = recent_episodes
            win_rate = sum(e["win"] for e in recent_wins) / len(recent_wins) if recent_wins else 0.0
        
        self.rolling_win_rates.append({
            "episode": episode,
            "win_rate": win_rate
        })
    
    async def _update_episodes_async(self):
        """Async helper to update episodes in MongoDB."""
        try:
            db = get_database()
            
            # Get current run
            run = await db.training_runs.find_one({"run_id": self.run_id})
            if not run:
                logger.warning(f"TrainingRun {self.run_id} not found in MongoDB")
                return
            
            # Update episodes and rolling win rate
            update_data = {
                "$set": {
                    "metrics.episodes": [e for e in self.episodes],
                    "metrics.rolling_win_rate": [r for r in self.rolling_win_rates]
                }
            }
            
            await db.training_runs.update_one(
                {"run_id": self.run_id},
                update_data
            )
        except Exception as e:
            logger.error(f"Error updating episodes in MongoDB: {e}")
    
    def update_status(
        self,
        status: str,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update the status of the training run.
        
        Args:
            status: New status ("running", "completed", "stopped", "failed")
            error_message: Optional error message if status is "failed"
            
        Returns:
            True if successful, False otherwise
        """
        if not MONGODB_AVAILABLE or not self._run_created:
            return False
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._update_status_async(status, error_message))
            else:
                loop.run_until_complete(self._update_status_async(status, error_message))
            
            logger.info(f"Updated TrainingRun {self.run_id} status to {status}")
            return True
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            return False
    
    async def _update_status_async(self, status: str, error_message: Optional[str] = None):
        """Async helper to update status in MongoDB."""
        try:
            db = get_database()
            
            update_data = {
                "$set": {
                    "status": status,
                    "end_time": datetime.utcnow() if status in ["completed", "stopped", "failed"] else None
                }
            }
            
            if error_message:
                update_data["$set"]["metadata.error_message"] = error_message
            
            await db.training_runs.update_one(
                {"run_id": self.run_id},
                update_data
            )
        except Exception as e:
            logger.error(f"Error updating status in MongoDB: {e}")
    
    def log_checkpoint(self, episode: int, checkpoint_path: str) -> bool:
        """
        Log a checkpoint path.
        
        Args:
            episode: Episode number when checkpoint was saved
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            True if successful, False otherwise
        """
        if not MONGODB_AVAILABLE or not self._run_created:
            return False
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._log_checkpoint_async(episode, checkpoint_path))
            else:
                loop.run_until_complete(self._log_checkpoint_async(episode, checkpoint_path))
            
            return True
        except Exception as e:
            logger.error(f"Failed to log checkpoint: {e}")
            return False
    
    async def _log_checkpoint_async(self, episode: int, checkpoint_path: str):
        """Async helper to log checkpoint in MongoDB."""
        try:
            db = get_database()
            
            checkpoint = {
                "episode": episode,
                "path": checkpoint_path
            }
            
            await db.training_runs.update_one(
                {"run_id": self.run_id},
                {"$push": {"checkpoint_paths": checkpoint}}
            )
        except Exception as e:
            logger.error(f"Error logging checkpoint in MongoDB: {e}")


def create_training_run_logger(
    config: Dict[str, Any],
    agent_id: str = "ppo_agent",
    algorithm: str = "MaskablePPO"
) -> Optional[TrainingRunLogger]:
    """
    Create and initialize a TrainingRunLogger.
    
    Args:
        config: Training configuration dictionary
        agent_id: Agent identifier for logging
        algorithm: Algorithm name for logging
        
    Returns:
        TrainingRunLogger instance or None if MongoDB is unavailable
    """
    logger_instance = TrainingRunLogger(agent_id=agent_id, algorithm=algorithm)
    
    if logger_instance.create_run(config):
        return logger_instance
    else:
        return None

