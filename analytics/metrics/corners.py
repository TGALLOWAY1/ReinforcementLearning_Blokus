
from typing import Any, Dict

from engine.board import Player

from . import MetricInput


def compute_corner_metrics(inp: MetricInput) -> Dict[str, Any]:
    """
    Computes corner economy metrics (frontier size).
    
    Relies on Board.get_frontier() which matches the definition:
    - Empty
    - Diagonally adjacent to self
    - Not orthogonally adjacent to self
    """
    
    def get_corners(board, pid: int) -> int:
        # Check if get_frontier exists, else fail/implement manual fallback?
        # Assuming board matches engine.board.Board
        return len(board.get_frontier(Player(pid)))

    me = inp.player_id
    opponents = inp.opponents
    
    c_me_before = get_corners(inp.state, me)
    c_me_after = get_corners(inp.next_state, me)
    
    c_opp_before_sum = sum(get_corners(inp.state, o) for o in opponents)
    c_opp_after_sum = sum(get_corners(inp.next_state, o) for o in opponents)
    
    # Corner block: sum of corners lost by opponents
    # This assumes loss is purely due to me filling them (or potentially placing next to them? No, def doesn't care about opp adjacency)
    # Actually, is it just net change? 
    # Spec: "corner_block = sum(max(0, C_o_before - C_o_after) for o in opponents)"
    
    corner_block = 0
    for o in opponents:
        before = get_corners(inp.state, o)
        after = get_corners(inp.next_state, o)
        corner_block += max(0, before - after)
        
    return {
        "corners_me_before": c_me_before,
        "corners_me_after": c_me_after,
        "corners_me_delta": c_me_after - c_me_before,
        "corners_opp_before_sum": c_opp_before_sum,
        "corners_opp_after_sum": c_opp_after_sum,
        "corner_block": corner_block
    }
