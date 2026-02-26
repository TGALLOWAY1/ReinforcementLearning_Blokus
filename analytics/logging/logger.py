
import os
import time
from typing import Any, Dict, Optional

from analytics.metrics import (
    MetricInput,
    compute_blocking_metrics,
    compute_center_metrics,
    compute_corner_metrics,
    compute_mobility_metrics,
    compute_piece_metrics,
    compute_proximity_metrics,
    compute_territory_metrics,
)
from engine.board import Board, Player
from engine.move_generator import LegalMoveGenerator, Move
from engine.pieces import PieceGenerator

from .schemas import GameResultLog, StepLog


class StrategyLogger:
    def __init__(self, log_dir: str = "logs/analytics"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.steps_path = os.path.join(log_dir, "steps.jsonl")
        self.results_path = os.path.join(log_dir, "results.jsonl")
        
        # Cache for move expansion
        self.move_generator = LegalMoveGenerator()
        
        # Buffer? Or direct write? Direct write for now.
        
    def log_step(self, item: StepLog):
        with open(self.steps_path, "a") as f:
            f.write(item.model_dump_json() + "\n")
            
    def log_result(self, item: GameResultLog):
        with open(self.results_path, "a") as f:
            f.write(item.model_dump_json() + "\n")
            
    def on_reset(self, game_id: str, seed: int, agent_ids: Dict[int, str], config: Dict[str, Any]):
        # Store context if needed, or just log a "START" event?
        # User spec schema doesn't have START event, just step and end.
        self.current_game_id = game_id
        self.current_seed = seed
        self.agent_map = agent_ids
        # could log metadata
        pass
        
    def on_step(self, game_id: str, turn_index: int, player_id: int, 
                state: Board, move: Move, next_state: Board):
        
        # 1. Expand move to placed squares
        # We need the piece orientations. Move object stores index (orientation_id).
        # But Move object expects us to pass the list of orientations to get_positions().
        orientations = self.move_generator.piece_orientations_cache.get(move.piece_id)
        if not orientations:
            # Fallback (should not happen if engine works)
            logging_orientations = PieceGenerator.generate_orientations_for_piece(move.piece_id, PieceGenerator.get_piece_by_id(move.piece_id).shape)
            placed_squares = move.get_positions(logging_orientations)
        else:
            placed_squares = move.get_positions(orientations)
            
        placed_tuples = [(p.row, p.col) for p in placed_squares]
        
        # 2. Identify opponents
        opponents = [p.value for p in Player if p.value != player_id]
        
        # 3. Construct MetricInput
        inp = MetricInput(
            state=state,
            move=move,
            next_state=next_state,
            player_id=player_id,
            opponents=opponents,
            placed_squares=placed_tuples,
            precomputed_values={} 
        )
        
        # 4. Compute metrics
        # Order matters mainly for precomputation (mobility -> blocking)
        
        # Mobility first (computes counts and populates precomputed_values indirectly? No, need to merge returns)
        mob_metrics = compute_mobility_metrics(inp)
        
        # Optimization: We can inject the counts back into inp for blocking to use
        # mobility.py returns raw counts in the metrics dict (e.g. "mobility_me_before")
        # constructing the counts dict expected by blocking.py is explicit work.
        # mobility.py DOES logic to compute ALL player counts.
        # But it only returns summarized metrics.
        # Wait, I updated mobility.py to expose `get_mobility_counts(inp)` which looks at `precomputed_values`
        # BUT `compute_mobility_metrics` CALLS `get_mobility_counts`.
        # So if I want to AVOID recomputing in blocking, I must populate `precomputed_values` explicitly before calling blocking.
        
        # Optimization: Explicitly compute mobility counts once here.
        from analytics.metrics.mobility import get_mobility_counts
        counts_before, counts_after = get_mobility_counts(inp)
        inp.precomputed_values['mobility_counts_before'] = counts_before
        inp.precomputed_values['mobility_counts_after'] = counts_after
        # Now call metrics. mobility metrics will reuse these.
        
        metrics = {}
        metrics.update(compute_center_metrics(inp))
        metrics.update(compute_territory_metrics(inp))
        metrics.update(compute_mobility_metrics(inp)) # Re-uses precomputed
        metrics.update(compute_blocking_metrics(inp)) # Re-uses precomputed
        metrics.update(compute_corner_metrics(inp))
        metrics.update(compute_proximity_metrics(inp))
        metrics.update(compute_piece_metrics(inp))
        
        # 5. Log
        # Action details
        action_dict = {
            "piece_id": move.piece_id,
            "orientation": move.orientation,
            "anchor_row": move.anchor_row,
            "anchor_col": move.anchor_col
        }
        
        # Basic counts needed for schema
        legal_moves_before = counts_before.get(player_id, 0)
        legal_moves_after = counts_after.get(player_id, 0)
        
        # pieces remaining
        # next_state has updated usage
        pieces_used = next_state.player_pieces_used.get(Player(player_id), set())
        # All piece IDs are 1..21.
        remaining = [i for i in range(1, 22) if i not in pieces_used]
        
        log_entry = StepLog(
            game_id=game_id,
            timestamp=time.time(),
            seed=getattr(self, 'current_seed', None),
            turn_index=turn_index,
            player_id=player_id,
            action=action_dict,
            legal_moves_before=legal_moves_before,
            legal_moves_after=legal_moves_after,
            pieces_remaining=remaining,
            metrics=metrics
        )
        
        self.log_step(log_entry)

    def on_game_end(self, game_id: str, final_scores: Dict[int, int], winner_id: Optional[int], num_turns: int):
        # Convert keys to str for JSON
        scores_str = {str(k): v for k, v in final_scores.items()}
        
        # Seat order: simply [1, 2, 3, 4] for standard Blokus?
        seat_order = [p.value for p in Player]
        
        # agent_ids mapping 
        # (reverse mapping logic if self.agent_map is needed, but we used simple dict before)
        agent_map_str = {str(k): v for k, v in getattr(self, 'agent_map', {}).items()}
        
        log_entry = GameResultLog(
            game_id=game_id,
            timestamp=time.time(),
            final_scores=scores_str,
            winner_id=winner_id,
            num_turns=num_turns,
            agent_ids=agent_map_str,
            seat_order=seat_order
        )
        
        self.log_result(log_entry)
