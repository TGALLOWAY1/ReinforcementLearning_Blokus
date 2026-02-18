
from typing import Dict, Any, Tuple, Set
import numpy as np
from . import MetricInput, BOARD_SIZE

def compute_territory_metrics(inp: MetricInput) -> Dict[str, Any]:
    """
    Computes territory shape and expansion metrics.
    
    Outputs:
    - area_gain
    - perimeter_after
    - delta_perimeter
    - compactness_after
    - bbox_fill_after
    """
    
    # We need access to the grid from state and next_state
    # Assuming inp.state and inp.next_state are Board objects with a .grid attribute
    
    def get_player_mask(board) -> np.ndarray:
        return board.grid == inp.player_id

    mask_before = get_player_mask(inp.state)
    mask_after = get_player_mask(inp.next_state)
    
    area_before = np.sum(mask_before)
    area_after = np.sum(mask_after)
    
    area_gain = int(area_after - area_before)
    
    def calculate_perimeter(mask: np.ndarray) -> int:
        if not np.any(mask):
            return 0
        
        # Simple convolution-like approach or just iterating occupied cells
        # Since we have numpy, let's use it.
        # Perimeter = sum of edges adjacent to 0 in the mask (where mask is 1)
        # Pad mask to handle boundaries
        padded = np.pad(mask, 1, mode='constant', constant_values=0)
        
        # Edges in each direction
        # diff != 0 means transition 0->1 or 1->0
        # We only care about edges of the "1" region.
        # But for shape perimeter, any transition is a shared edge.
        # Sum of transitions where one side is 1 and other is 0.
        
        perimeter_count = 0
        # Vertical edges
        perimeter_count += np.sum(np.abs(np.diff(padded.astype(int), axis=0)))
        # Horizontal edges
        perimeter_count += np.sum(np.abs(np.diff(padded.astype(int), axis=1)))
        
        return int(perimeter_count)

    def calculate_bbox_area(mask: np.ndarray) -> int:
        if not np.any(mask):
            return 0
        rows, cols = np.where(mask)
        h = np.max(rows) - np.min(rows) + 1
        w = np.max(cols) - np.min(cols) + 1
        return int(h * w)

    perim_before = calculate_perimeter(mask_before)
    perim_after = calculate_perimeter(mask_after)
    delta_perimeter = perim_after - perim_before
    
    bbox_a = calculate_bbox_area(mask_after)
    
    # Compactness: area / perimeter^2 (isoperimetric quotient variant)
    # C = 4*pi*A / P^2 for circles. Here just A/P^2 is fine as requested.
    compactness = 0.0
    if perim_after > 0:
        compactness = area_after / (perim_after ** 2)
    
    # BBox fill
    bbox_fill = 0.0
    if bbox_a > 0:
        bbox_fill = area_after / bbox_a

    return {
        "area_gain": area_gain,
        "perimeter_after": perim_after,
        "delta_perimeter": delta_perimeter,
        "compactness_after": compactness,
        "bbox_fill_after": bbox_fill
    }
