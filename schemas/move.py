"""
Pydantic schemas for game moves.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Player(str, Enum):
    """Player enumeration."""
    RED = "RED"
    BLUE = "BLUE"
    GREEN = "GREEN"
    YELLOW = "YELLOW"


class MoveRequest(BaseModel):
    """Request to make a move."""
    player: Player
    piece_id: int = Field(..., ge=1, le=21, description="ID of the piece to place")
    orientation: int = Field(..., ge=0, description="Orientation index of the piece")
    anchor_row: int = Field(..., ge=0, le=19, description="Row position of the anchor")
    anchor_col: int = Field(..., ge=0, le=19, description="Column position of the anchor")
    
    class Config:
        json_schema_extra = {
            "example": {
                "player": "red",
                "piece_id": 1,
                "orientation": 0,
                "anchor_row": 0,
                "anchor_col": 0
            }
        }


class MoveResponse(BaseModel):
    """Response after making a move."""
    success: bool
    message: str
    new_score: Optional[int] = None
    game_over: bool = False
    winner: Optional[Player] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Move successful",
                "new_score": 5,
                "game_over": False,
                "winner": None
            }
        }


class Position(BaseModel):
    """Position on the board."""
    row: int = Field(..., ge=0, le=19)
    col: int = Field(..., ge=0, le=19)


class Move(BaseModel):
    """A move that was made."""
    player: Player
    piece_id: int
    orientation: int
    anchor_row: int
    anchor_col: int
    positions: List[Position] = Field(description="All positions occupied by this move")
    score_delta: int = Field(description="Score change from this move")
    move_number: int = Field(description="Move number in the game")
    
    class Config:
        json_schema_extra = {
            "example": {
                "player": "red",
                "piece_id": 1,
                "orientation": 0,
                "anchor_row": 0,
                "anchor_col": 0,
                "positions": [{"row": 0, "col": 0}],
                "score_delta": 5,
                "move_number": 1
            }
        }
