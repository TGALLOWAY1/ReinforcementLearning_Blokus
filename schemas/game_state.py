"""
Game state schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class Player(str, Enum):
    """Player enumeration."""
    RED = "RED"
    BLUE = "BLUE"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class AgentType(str, Enum):
    """Available agent types."""
    RANDOM = "random"
    HEURISTIC = "heuristic"
    MCTS = "mcts"
    HUMAN = "human"


class GameStatus(str, Enum):
    """Game status enumeration."""
    WAITING = "waiting"
    IN_PROGRESS = "in_progress"
    FINISHED = "finished"
    ERROR = "error"


class Position(BaseModel):
    """Represents a position on the board."""
    row: int = Field(ge=0, le=19)
    col: int = Field(ge=0, le=19)


class Move(BaseModel):
    """Represents a move in the game."""
    piece_id: int = Field(ge=1, le=21, description="ID of the piece to place")
    orientation: int = Field(ge=0, description="Orientation index of the piece")
    anchor_row: int = Field(ge=0, le=19, description="Row position of the anchor")
    anchor_col: int = Field(ge=0, le=19, description="Column position of the anchor")


class PlayerConfig(BaseModel):
    """Configuration for a player."""
    player: Player
    agent_type: AgentType
    agent_config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class GameConfig(BaseModel):
    """Configuration for creating a new game."""
    players: List[PlayerConfig] = Field(min_items=2, max_items=4)
    game_id: Optional[str] = None
    auto_start: bool = Field(default=True, description="Whether to start the game automatically")


class GameState(BaseModel):
    """Current state of the game."""
    game_id: str
    status: GameStatus
    current_player: Player
    board: List[List[int]] = Field(description="20x20 board state")
    scores: Dict[str, int] = Field(description="Scores for each player")
    pieces_used: Dict[str, List[int]] = Field(description="Pieces used by each player")
    move_count: int
    game_over: bool
    winner: Optional[Player] = None
    legal_moves: List[Move] = Field(description="Available legal moves for current player")
    created_at: datetime
    updated_at: datetime
    players: Optional[List[PlayerConfig]] = None
    heatmap: Optional[List[List[float]]] = Field(
        default=None,
        description="20x20 grid where 1.0 = legal move position, 0.0 = illegal"
    )


class StateUpdate(BaseModel):
    """WebSocket state update message."""
    type: str = Field(description="Type of update: 'game_state', 'move_made', 'game_over', 'error'")
    game_id: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class MoveRequest(BaseModel):
    """Request to make a move."""
    move: Move
    player: Optional[Player] = None  # If None, uses current player


class MoveResponse(BaseModel):
    """Response to a move request."""
    success: bool
    message: str
    game_state: Optional[GameState] = None


class GameCreateResponse(BaseModel):
    """Response when creating a new game."""
    game_id: str
    game_state: GameState
    message: str


class AgentInfo(BaseModel):
    """Information about an available agent."""
    type: AgentType
    name: str
    description: str
    config_schema: Optional[Dict[str, Any]] = None


class GameListResponse(BaseModel):
    """Response for listing games."""
    games: List[GameState]
    total: int


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
