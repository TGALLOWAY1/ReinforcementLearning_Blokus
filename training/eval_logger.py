"""
Evaluation logger for MongoDB integration.

This module provides utilities to log evaluation results to MongoDB,
linking them to TrainingRun records.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try to import MongoDB dependencies
try:
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    from webapi.db.mongo import get_database
    from webapi.db.models import EvaluationRun
    MONGODB_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    logger.warning(f"MongoDB not available for evaluation logging: {e}")
    MONGODB_AVAILABLE = False
    EvaluationRun = None


def log_evaluation_result(
    training_run_id: str,
    checkpoint_path: str,
    opponent_type: str,
    games_played: int,
    win_rate: float,
    avg_reward: float,
    avg_game_length: float
) -> bool:
    """
    Log an evaluation result to MongoDB.
    
    Args:
        training_run_id: Training run ID
        checkpoint_path: Path to checkpoint used
        opponent_type: Type of opponent ("random", "heuristic", "self_play")
        games_played: Number of games played
        win_rate: Win rate (0.0 to 1.0)
        avg_reward: Average reward per game
        avg_game_length: Average game length in steps
        
    Returns:
        True if successful, False otherwise
    """
    if not MONGODB_AVAILABLE:
        logger.warning("MongoDB not available, skipping evaluation logging")
        return False
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_log_evaluation_async(
                training_run_id, checkpoint_path, opponent_type,
                games_played, win_rate, avg_reward, avg_game_length
            ))
        else:
            loop.run_until_complete(_log_evaluation_async(
                training_run_id, checkpoint_path, opponent_type,
                games_played, win_rate, avg_reward, avg_game_length
            ))
        
        logger.info(f"Logged evaluation result: {opponent_type} - {win_rate:.2%} win rate")
        return True
    except Exception as e:
        logger.error(f"Failed to log evaluation result: {e}")
        return False


async def _log_evaluation_async(
    training_run_id: str,
    checkpoint_path: str,
    opponent_type: str,
    games_played: int,
    win_rate: float,
    avg_reward: float,
    avg_game_length: float
):
    """Async helper to log evaluation in MongoDB."""
    try:
        db = get_database()
        evaluation_run = EvaluationRun(
            training_run_id=training_run_id,
            checkpoint_path=checkpoint_path,
            opponent_type=opponent_type,
            games_played=games_played,
            win_rate=win_rate,
            avg_reward=avg_reward,
            avg_game_length=avg_game_length,
            created_at=datetime.utcnow()
        )
        
        await db.evaluation_runs.insert_one(evaluation_run.dict(by_alias=True))
        logger.info(f"EvaluationRun logged to MongoDB for {opponent_type}")
    except Exception as e:
        logger.error(f"Error logging evaluation in MongoDB: {e}")
        raise

