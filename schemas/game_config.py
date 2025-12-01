"""
Pydantic schemas for game configuration.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class PlayerType(str, Enum):
    """Types of players in a game."""
    HUMAN = "human"
    RANDOM = "random"
    HEURISTIC = "heuristic"
    MCTS = "mcts"


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    type: PlayerType
    name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    seed: Optional[int] = None


class GameConfig(BaseModel):
    """Configuration for a Blokus game."""
    players: List[AgentConfig] = Field(..., min_items=2, max_items=4)
    max_moves: int = Field(default=1000, ge=1, le=10000)
    time_limit_per_move: Optional[float] = Field(default=None, ge=0.1, le=300.0)
    auto_play: bool = Field(default=True, description="Whether agents play automatically")
    
    class Config:
        json_schema_extra = {
            "example": {
                "players": [
                    {"type": "human", "name": "Player 1"},
                    {"type": "random", "name": "Random Bot", "seed": 42},
                    {"type": "heuristic", "name": "Smart Bot", "parameters": {"piece_size": 1.0}},
                    {"type": "mcts", "name": "MCTS Bot", "parameters": {"iterations": 1000}}
                ],
                "max_moves": 1000,
                "time_limit_per_move": 30.0,
                "auto_play": True
            }
        }
