
from typing import Any, Dict

import numpy as np

from . import BOARD_SIZE, MetricInput


def compute_proximity_metrics(inp: MetricInput) -> Dict[str, Any]:
    """
    Computes proximity to opponents.
    
    Outputs:
    - dist_to_opp_min: Minimum Manhattan distance from placed piece to any opponent piece.
    - is_close_3: 1 if dist_to_opp_min <= 3, else 0.
    """
    
    placed = inp.get_placed_squares()
    if not placed:
        # No piece placed?
        return {
            "dist_to_opp_min": float(BOARD_SIZE * 2),
            "is_close_3": 0
        }
    
    # Identify opponent squares
    # Use state.grid (before move) to be safe/standard, though opponents don't change.
    grid = inp.state.grid
    opp_mask = np.isin(grid, inp.opponents)
    
    # If no opponents on board yet
    if not np.any(opp_mask):
         return {
            "dist_to_opp_min": float(BOARD_SIZE * 2),
            "is_close_3": 0
        }

    # Get opponent coordinates: (N, 2) array
    opp_coords = np.argwhere(opp_mask)
    
    # Placed coordinates: (M, 2) array
    placed_coords = np.array(placed)
    
    # Compute distances efficiently with broadcasting
    # shape: (M, N, 2) - difference in each coordinate
    diffs = placed_coords[:, np.newaxis, :] - opp_coords[np.newaxis, :, :]
    
    # Manhattan distance: sum of abs differences
    # shape: (M, N)
    dists = np.sum(np.abs(diffs), axis=2)
    
    min_dist = np.min(dists)
    
    return {
        "dist_to_opp_min": float(min_dist),
        "is_close_3": 1 if min_dist <= 3 else 0
    }
