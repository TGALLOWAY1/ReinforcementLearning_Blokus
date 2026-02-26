"""
Pydantic schemas for game state updates.
"""

from typing import Any, List, Optional

from pydantic import BaseModel, Field

from .move import Move, Player, Position


class BoardState(BaseModel):
    """Current state of the game board."""
    cells: List[List[Optional[Player]]] = Field(description="20x20 board state")
    move_count: int = Field(description="Total number of moves made")
    
    class Config:
        json_schema_extra = {
            "example": {
                "cells": [
                    ["red", None, None, ...],
                    [None, None, None, ...],
                    ...
                ],
                "move_count": 5
            }
        }


class PlayerState(BaseModel):
    """State of a player."""
    player: Player
    score: int = Field(ge=0)
    pieces_used: List[int] = Field(description="IDs of pieces already used")
    pieces_remaining: List[int] = Field(description="IDs of pieces still available")
    is_active: bool = Field(description="Whether it's this player's turn")
    
    class Config:
        json_schema_extra = {
            "example": {
                "player": "red",
                "score": 15,
                "pieces_used": [1, 2, 3],
                "pieces_remaining": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                "is_active": True
            }
        }


class LegalMove(BaseModel):
    """A legal move available to the current player."""
    piece_id: int
    orientation: int
    anchor_row: int
    anchor_col: int
    positions: List[Position] = Field(description="Positions this move would occupy")
    
    class Config:
        json_schema_extra = {
            "example": {
                "piece_id": 1,
                "orientation": 0,
                "anchor_row": 0,
                "anchor_col": 0,
                "positions": [{"row": 0, "col": 0}]
            }
        }


class GameState(BaseModel):
    """Complete game state."""
    game_id: str
    current_player: Player
    board: BoardState
    players: List[PlayerState]
    legal_moves: List[LegalMove]
    game_over: bool = False
    winner: Optional[Player] = None
    last_move: Optional[Move] = None
    heatmap: Optional[List[List[float]]] = Field(
        default=None,
        description="20x20 grid where 1.0 = legal move position, 0.0 = illegal"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "game_id": "game_123",
                "current_player": "red",
                "board": {
                    "cells": [...],
                    "move_count": 5
                },
                "players": [...],
                "legal_moves": [...],
                "game_over": False,
                "winner": None,
                "last_move": None
            }
        }


class StateUpdate(BaseModel):
    """WebSocket message for state updates."""
    type: str = Field(description="Type of update: 'state', 'move', 'error'")
    game_id: str
    data: Any = Field(description="Update data (GameState, Move, or error message)")
    timestamp: float = Field(description="Unix timestamp of the update")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "state",
                "game_id": "game_123",
                "data": {
                    "current_player": "red",
                    "board": {...},
                    "players": [...],
                    "legal_moves": [...]
                },
                "timestamp": 1640995200.0
            }
        }


class GameSummary(BaseModel):
    """Summary of a game."""
    game_id: str
    status: str = Field(description="'active', 'completed', 'error'")
    players: List[PlayerState]
    winner: Optional[Player] = None
    total_moves: int
    duration_seconds: Optional[float] = None
    created_at: float
    last_updated: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "game_id": "game_123",
                "status": "active",
                "players": [...],
                "winner": None,
                "total_moves": 5,
                "duration_seconds": None,
                "created_at": 1640995200.0,
                "last_updated": 1640995200.0
            }
        }
