
from typing import Dict, Any, List, Tuple
from . import MetricInput
from engine.move_generator import LegalMoveGenerator
from engine.board import Player, Board

# Reusable generator instance if needed, though it might be stateful?
# LegalMoveGenerator.__init__ takes no args and loads pieces. It caches stuff.
# Better to have one instance if possible.
_shared_move_generator = None

def get_move_generator():
    global _shared_move_generator
    if _shared_move_generator is None:
        _shared_move_generator = LegalMoveGenerator()
    return _shared_move_generator

def get_mobility_counts(inp: MetricInput) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Helper to get or compute mobility counts for before/after states."""
    counts_before = inp.precomputed_values.get('mobility_counts_before') if inp.precomputed_values else None
    counts_after = inp.precomputed_values.get('mobility_counts_after') if inp.precomputed_values else None
    
    # helper to compute if missing
    def compute_counts(board: Board, players: List[int]) -> Dict[int, int]:
        gen = get_move_generator()
        out = {}
        for pid in players:
            # We assume pid is valid for Player(pid)
            p_enum = Player(pid)
            moves = gen.get_legal_moves(board, p_enum)
            out[pid] = len(moves)
        return out

    all_players = [inp.player_id] + inp.opponents
    
    if counts_before is None:
        counts_before = compute_counts(inp.state, all_players)
        
    if counts_after is None:
        counts_after = compute_counts(inp.next_state, all_players)
        
    return counts_before, counts_after

def compute_mobility_metrics(inp: MetricInput) -> Dict[str, Any]:
    """
    Computes mobility metrics (legal move counts).
    
    Inputs expected in inp.precomputed_values (optional):
    - mobility_counts_before: Dict[int, int] (player_id -> count)
    - mobility_counts_after: Dict[int, int]
    
    If not provided, they will be computed (expensive).
    """
    counts_before, counts_after = get_mobility_counts(inp)
        
    # Analyze
    me = inp.player_id
    m_me_before = counts_before.get(me, 0)
    m_me_after = counts_after.get(me, 0)
    
    diff = m_me_after - m_me_before
    ratio = m_me_after / m_me_before if m_me_before > 0 else 0.0
    
    opp_before_sum = sum(counts_before.get(o, 0) for o in inp.opponents)
    opp_after_sum = sum(counts_after.get(o, 0) for o in inp.opponents)
    opp_delta_sum = opp_after_sum - opp_before_sum
    
    return {
        "mobility_me_before": m_me_before,
        "mobility_me_after": m_me_after,
        "mobility_me_delta": diff,
        "mobility_me_ratio": ratio,
        "mobility_opp_before_sum": opp_before_sum,
        "mobility_opp_after_sum": opp_after_sum,
        "mobility_opp_delta_sum": opp_delta_sum,
        
        # Pass these through for use by other metrics (like blocking)
        # The caller (logger) might merge these dicts, so we can return rich objects if needed.
        # But for now we stick to the requested scalar outputs.
        # However, blocking needs per-opponent deltas.
        # We'll return a special key for detailed counts if needed, but blocking.py probably re-accesses precomputed values.
    }
