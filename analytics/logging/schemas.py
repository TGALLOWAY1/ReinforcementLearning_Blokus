
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class StepLog(BaseModel):
    """Schema for per-step move log."""
    game_id: str
    timestamp: float
    seed: Optional[int]
    turn_index: int
    player_id: int
    
    # Action details
    action: Dict[str, Any]
    
    # Board state (optional/minimal)
    board_hash: Optional[str] = None
    
    # Basic counts
    pieces_remaining: Optional[List[int]] = None # List of pieces remaining for this player
    legal_moves_before: int
    legal_moves_after: int
    
    # Derived metrics
    metrics: Dict[str, Any]

class GameResultLog(BaseModel):
    """Schema for game end result."""
    game_id: str
    timestamp: float
    final_scores: Dict[str, int] # Keyed by player_id (str to be safe for JSON)
    winner_id: Optional[int]
    num_turns: int
    agent_ids: Dict[str, str] # map player_id -> agent_name/id
    seat_order: List[int]
