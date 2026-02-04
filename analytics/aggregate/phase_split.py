
from typing import List, Tuple

def get_phase_label(move_index: int, total_moves: int) -> str:
    """
    Returns 'Opening', 'Midgame', or 'Endgame' based on move index (0-based) and total moves.
    
    Opening: first 25%
    Midgame: middle 50%
    Endgame: last 25%
    """
    if total_moves == 0:
        return "Unknown"
    
    # Simple thresholds
    t1 = int(total_moves * 0.25)
    t2 = int(total_moves * 0.75)
    
    # Ensure at least one move in opening if total_moves > 0?
    # With strict math:
    # 4 moves: t1=1, t2=3. 
    # idx 0 (<1) -> Opening
    # idx 1, 2 (<3) -> Mid
    # idx 3 (>=3) -> End
    # Seems reasonable.
    
    if move_index < t1:
        return "Opening"
    elif move_index < t2:
        return "Midgame"
    else:
        return "Endgame"
