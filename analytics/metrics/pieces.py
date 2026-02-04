
from typing import Dict, Any
import numpy as np
from . import MetricInput
from engine.pieces import PieceGenerator

def compute_piece_metrics(inp: MetricInput) -> Dict[str, Any]:
    """
    Computes piece-related metrics.
    
    Outputs:
    - piece_size: Size of the placed piece.
    - piece_complexity: Perimeter of the piece shape (as complexity proxy).
    """
    
    # Extract piece_id from move object
    # Assuming move has piece_id attribute
    piece_id = getattr(inp.move, 'piece_id', None)
    if piece_id is None:
        return {
            "piece_size": 0,
            "piece_complexity": 0
        }
        
    piece = PieceGenerator.get_piece_by_id(piece_id)
    if not piece:
        return {
            "piece_size": 0,
            "piece_complexity": 0
        }
        
    size = piece.size
    
    # Calculate complexity (perimeter of the base shape)
    # Reuse the logic from territory.py? 
    # Or just simple count of edges 
    
    def calculate_shape_perimeter(mask: np.ndarray) -> int:
        padded = np.pad(mask, 1, mode='constant', constant_values=0)
        perimeter_count = 0
        perimeter_count += np.sum(np.abs(np.diff(padded.astype(int), axis=0)))
        perimeter_count += np.sum(np.abs(np.diff(padded.astype(int), axis=1)))
        return int(perimeter_count)

    complexity = calculate_shape_perimeter(piece.shape)
    
    return {
        "piece_size": size,
        "piece_complexity": complexity
    }
