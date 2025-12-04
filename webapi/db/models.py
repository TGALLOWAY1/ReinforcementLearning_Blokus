"""
MongoDB models/schemas for Blokus RL Web API.

This module defines Pydantic models for MongoDB documents representing:
- TrainingRun: Records of RL training sessions
- EvaluationRun: Records of model evaluation sessions

These models are designed to be stored in MongoDB collections and can be extended
as the RL training and evaluation features are implemented.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from pydantic import BaseModel, Field
from bson import ObjectId


# Simplified ObjectId handling for Pydantic V2 compatibility
# We'll use Optional[str] for id fields and convert to ObjectId when needed


class EpisodeMetric(BaseModel):
    """Metrics for a single training episode."""
    episode: int = Field(..., description="Episode number")
    total_reward: float = Field(..., description="Total reward for the episode")
    steps: int = Field(..., description="Number of steps in the episode")
    win: Optional[bool] = Field(None, description="Whether the agent won (if applicable)")
    epsilon: Optional[float] = Field(None, description="Exploration rate (if applicable)")
    
    class Config:
        json_encoders = {ObjectId: str}


class RollingWinRate(BaseModel):
    """Rolling win rate metric at a specific episode."""
    episode: int = Field(..., description="Episode number")
    win_rate: float = Field(..., description="Win rate (0.0 to 1.0)")
    
    class Config:
        json_encoders = {ObjectId: str}


class CheckpointPath(BaseModel):
    """Checkpoint path information."""
    episode: int = Field(..., description="Episode number when checkpoint was saved")
    path: str = Field(..., description="File path to the checkpoint")
    
    class Config:
        json_encoders = {ObjectId: str}


class TrainingRun(BaseModel):
    """
    Model representing a training run in MongoDB.
    
    This schema stores information about RL training sessions, including
    configuration, metrics, and checkpoints.
    """
    id: Optional[str] = Field(default=None, alias="_id")
    run_id: str = Field(..., description="Unique identifier for the training run (UUID)")
    agent_id: str = Field(..., description="Identifier for the agent (e.g., 'dqn_agent', 'ppo_agent')")
    algorithm: str = Field(..., description="RL algorithm used (e.g., 'DQN', 'PPO', 'A2C')")
    config: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters and training configuration")
    status: str = Field(
        default="running",
        description="Current status of the training run (running, completed, stopped, or failed)"
    )
    start_time: datetime = Field(default_factory=datetime.utcnow, description="Training start time")
    end_time: Optional[datetime] = Field(None, description="Training end time (null if still running)")
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Training metrics including episodes, win rates, etc."
    )
    checkpoint_paths: List[CheckpointPath] = Field(
        default_factory=list,
        description="List of saved checkpoint paths"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (git hash, env version, etc.)"
    )
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str, datetime: lambda v: v.isoformat()}
        json_schema_extra = {
            "example": {
                "run_id": "550e8400-e29b-41d4-a716-446655440000",
                "agent_id": "ppo_agent",
                "algorithm": "PPO",
                "config": {
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "n_steps": 2048
                },
                "status": "running",
                "start_time": "2024-01-01T00:00:00Z",
                "metrics": {
                    "episodes": [
                        {
                            "episode": 1,
                            "total_reward": 10.5,
                            "steps": 50,
                            "win": True,
                            "epsilon": 0.1
                        }
                    ],
                    "rolling_win_rate": [
                        {
                            "episode": 1,
                            "win_rate": 0.5
                        }
                    ]
                },
                "checkpoint_paths": [
                    {
                        "episode": 100,
                        "path": "checkpoints/ppo_agent_ep100.zip"
                    }
                ],
                "metadata": {
                    "git_hash": "abc123",
                    "env_version": "1.0.0"
                }
            }
        }


class EvaluationRun(BaseModel):
    """
    Model representing an evaluation run in MongoDB.
    
    This schema stores information about model evaluation sessions,
    including performance metrics against different opponents.
    """
    id: Optional[str] = Field(default=None, alias="_id")
    training_run_id: str = Field(..., description="Reference to the TrainingRun that was evaluated")
    checkpoint_path: str = Field(..., description="Path to the model checkpoint used for evaluation")
    opponent_type: str = Field(..., description="Type of opponent (e.g., 'random', 'heuristic', 'self_play')")
    games_played: int = Field(..., description="Number of games played during evaluation")
    win_rate: float = Field(..., description="Win rate (0.0 to 1.0)")
    avg_reward: float = Field(..., description="Average reward per game")
    avg_game_length: float = Field(..., description="Average game length in steps")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Evaluation run creation time")
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str, datetime: lambda v: v.isoformat()}
        json_schema_extra = {
            "example": {
                "training_run_id": "550e8400-e29b-41d4-a716-446655440000",
                "checkpoint_path": "checkpoints/ppo_agent_ep100.zip",
                "opponent_type": "random",
                "games_played": 100,
                "win_rate": 0.75,
                "avg_reward": 15.3,
                "avg_game_length": 45.2,
                "created_at": "2024-01-01T12:00:00Z"
            }
        }

