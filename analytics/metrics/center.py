
from typing import Any, Dict

from . import CENTER_COORD, MetricInput, is_in_center_mask, mdist


def compute_center_metrics(inp: MetricInput) -> Dict[str, Any]:
    """
    Computes center control metrics.
    
    Outputs:
    - center_distance: Mean Manhattan distance of placed squares to center.
    - center_gain: Number of placed squares within the center mask (radius k).
    """
    placed = inp.get_placed_squares()
    if not placed:
        return {
            "center_distance": 0.0, # Or None? 0.0 for empty move seems safer for stats
            "center_gain": 0
        }
        
    # d_center: mean distance
    dists = [mdist(sq, CENTER_COORD) for sq in placed]
    avg_dist = sum(dists) / len(dists)
    
    # center_gain: count in mask
    gain = sum(1 for sq in placed if is_in_center_mask(sq))
    
    return {
        "center_distance": avg_dist,
        "center_gain": gain
    }
