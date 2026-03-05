from typing import Dict, Any, List
from .board import Board, Player
from .advanced_metrics import compute_dead_zones
import time
import logging

logger = logging.getLogger(__name__)

def compute_player_metrics(board: Board, player: Player, move_generator=None, fast_mode: bool = True) -> Dict[str, float]:
    """
    Compute metrics for a single player.
    
    Args:
        board: The current board state.
        player: The player to compute metrics for.
        move_generator: Optional move generator to compute exact mobility.
        fast_mode: If True, skip expensive metrics (like exact mobility and full deadzone parsing per player).
        
    Returns:
        Dictionary of metrics (frontierSize, mobility, deadSpace).
    """
    frontier = board.get_frontier(player)
    frontier_size = len(frontier)
    
    mobility = 0.0
    if not fast_mode and move_generator is not None:
        # Exact legal moves
        moves = move_generator.get_legal_moves(board, player)
        mobility = float(len(moves))
    else:
        # Fast proxy: frontier size correlates with mobility
        mobility = float(frontier_size * 2)
        
    # Dead space is currently a global board calculation `compute_dead_zones`.
    # For MVP, we'll assign the count of dead zones touching the player's pieces, or just total dead space.
    # To keep it fast in fast_mode, we might skip it or do a simplified approximation.
    # Actually, the user asked for "deadSpace (your existing deadzone metric)".
    # `compute_dead_zones` returns a 20x20 boolean matrix.
    # In fast mode, we skip computing it if it's too slow.
    dead_space = 0.0
    if not fast_mode:
        dz_matrix = compute_dead_zones(board)
        # Count dead zones adjacent to this player
        # For simplicity MVP, we just count how many dead zone cells are in the player's frontier/influence.
        # As an even simpler MVP: count total dead zone cells on board (though this is identical for all players).
        # We will count dead zone cells that are orthogonally adjacent to the player's pieces.
        adj_count = 0
        for r in range(board.SIZE):
            for c in range(board.SIZE):
                if dz_matrix[r][c]:
                    # Check if adjacent to player
                    is_adj = False
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < board.SIZE and 0 <= nc < board.SIZE:
                            if board.grid[nr][nc] == player.value:
                                is_adj = True
                                break
                    if is_adj:
                        adj_count += 1
        dead_space = float(adj_count)
        
    return {
        "frontierSize": float(frontier_size),
        "mobility": mobility,
        "deadSpace": dead_space,
    }

def collect_all_player_metrics(board: Board, move_generator=None, fast_mode: bool = True) -> Dict[str, Dict[str, float]]:
    """Collect metrics for all players."""
    metrics = {}
    
    # If not in fast mode, we only need to compute dead zones once per state, not per player.
    # But for now, we just let `compute_player_metrics` handle it.
    # Optimization: compute dead zones once here explicitly if not fast_mode.
    dz_matrix = None
    if not fast_mode:
        dz_matrix = compute_dead_zones(board)
        
    for p in Player:
        p_metrics = compute_player_metrics(board, p, move_generator, fast_mode=True) # Always fast mobility inside
        
        # Override deadSpace if we precomputed
        if dz_matrix:
            adj_count = 0
            for r in range(board.SIZE):
                for c in range(board.SIZE):
                    if dz_matrix[r][c]:
                        is_adj = False
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < board.SIZE and 0 <= nc < board.SIZE:
                                if board.grid[nr][nc] == p.value:
                                    is_adj = True
                                    break
                        if is_adj:
                            adj_count += 1
            p_metrics["deadSpace"] = float(adj_count)
            
            # If not fast mode, also do exact mobility
            if move_generator:
                moves = move_generator.get_legal_moves(board, p)
                p_metrics["mobility"] = float(len(moves))
                
        metrics[p.name] = p_metrics
        
    return metrics

def compute_move_telemetry_delta(
    ply: int,
    mover_id: str,
    move_id: str,
    before: Dict[str, Dict[str, float]],
    after: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    """
    Compute delta vectors from before/after snapshots.
    """
    if mover_id not in before or mover_id not in after:
        return {}
        
    # Delta Self
    delta_self = {}
    for k in after[mover_id]:
        delta_self[k] = after[mover_id][k] - before[mover_id].get(k, 0.0)
        
    # Delta Opponents
    delta_opp_total = {}
    delta_opp_by_player = {}
    
    for opp_name, opp_after_metrics in after.items():
        if opp_name == mover_id:
            continue
            
        opp_before_metrics = before.get(opp_name, {})
        delta_opp_by_player[opp_name] = {}
        
        for k in opp_after_metrics:
            diff = opp_after_metrics[k] - opp_before_metrics.get(k, 0.0)
            delta_opp_by_player[opp_name][k] = diff
            delta_opp_total[k] = delta_opp_total.get(k, 0.0) + diff
            
    return {
        "ply": ply,
        "moverId": mover_id,
        "moveId": move_id,
        "deltaSelf": delta_self,
        "deltaOppTotal": delta_opp_total,
        "deltaOppByPlayer": delta_opp_by_player,
    }
