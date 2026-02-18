
from typing import List, Tuple, Set, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

# Re-export key types from engine/board.py if feasible, or redefine if we want decoupling.
# Given this is internal analytics, reusing engine types is fine.
# We assume the engine is available in the python path.

BOARD_SIZE = 20
CENTER_COORD = ((BOARD_SIZE - 1) / 2, (BOARD_SIZE - 1) / 2)

# Manhattan distance
def mdist(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Center mask radius
CENTER_MASK_RADIUS_K = 4

def is_in_center_mask(pos: Tuple[int, int], k: int = CENTER_MASK_RADIUS_K) -> bool:
    return mdist(pos, CENTER_COORD) <= k

@dataclass
class MetricInput:
    """Standard input for metric functions."""
    state: Any # Board object
    move: Any # Move object (with get_positions logic accessible or precomputed)
    next_state: Any # Board object
    player_id: int # ID of the player making the move
    opponents: List[int] # IDs of opponents
    
    # Precomputed/cached values to avoid re-deriving
    placed_squares: Optional[List[Tuple[int, int]]] = None
    precomputed_values: Optional[Dict[str, Any]] = None # Generic cache for things like mobility counts
    
    def get_placed_squares(self) -> List[Tuple[int, int]]:
        if self.placed_squares is not None:
            return self.placed_squares
        # Logic to derive from self.move if not provided
        # This assumes self.move matches engine.Move or similar
        # For now, we expect the caller to populate this if easier
        return []

from .center import compute_center_metrics
from .territory import compute_territory_metrics
from .mobility import compute_mobility_metrics, get_mobility_counts
from .blocking import compute_blocking_metrics
from .corners import compute_corner_metrics
from .proximity import compute_proximity_metrics
from .pieces import compute_piece_metrics


