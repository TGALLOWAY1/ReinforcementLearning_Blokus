"""Feature extraction utilities for win-probability modeling."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from engine.advanced_metrics import (
    compute_center_proximity,
    compute_corner_differential,
    compute_dead_zones,
    compute_opponent_adjacency,
)
from engine.board import Board, Player
from engine.mobility_metrics import compute_player_mobility_metrics
from engine.move_generator import LegalMoveGenerator
from engine.pieces import PieceGenerator


SNAPSHOT_FEATURE_COLUMNS: List[str] = [
    "frontier_size",
    "corner_differential",
    "center_proximity",
    "opponent_adjacency",
    "deadzone_count",
    "deadzone_fraction",
    "mobility_total_placements",
    "mobility_total_orientation_normalized",
    "mobility_total_cell_weighted",
    "mobility_bucket_1",
    "mobility_bucket_2",
    "mobility_bucket_3",
    "mobility_bucket_4",
    "mobility_bucket_5",
    "blocking_exposure_legal_moves_opp_sum",
    "utility_frontier_plus_mobility",
    "pieces_used_count",
    "pieces_remaining_count",
    "remaining_squares",
    "remaining_size_1_count",
    "remaining_size_2_count",
    "remaining_size_3_count",
    "remaining_size_4_count",
    "remaining_size_5_count",
    "remaining_key_piece_17",
    "remaining_key_piece_19",
    "remaining_key_piece_20",
    "phase_ply",
    "phase_turn_index",
    "phase_board_occupancy",
    "phase_piece_usage_ratio",
    "phase_progress_turn_ratio",
    "phase_progress_placement_ratio",
    "player_board_occupancy",
]


@dataclass(frozen=True)
class SnapshotRuntimeContext:
    """Context values reused across all players at one snapshot."""

    ply: int
    turn_index: int
    board_occupancy: float
    deadzone_count: int
    deadzone_fraction: float
    progress_turn_ratio: float
    progress_placement_ratio: float


@lru_cache(maxsize=1)
def _piece_size_lookup() -> Dict[int, int]:
    return {piece.id: piece.size for piece in PieceGenerator.get_all_pieces()}


@lru_cache(maxsize=1)
def _total_piece_squares() -> int:
    return int(sum(_piece_size_lookup().values()))


def _remaining_piece_ids(pieces_used: Iterable[int]) -> List[int]:
    used = set(int(piece_id) for piece_id in pieces_used)
    return [piece_id for piece_id in range(1, 22) if piece_id not in used]


def _remaining_squares(pieces_used: Iterable[int]) -> int:
    sizes = _piece_size_lookup()
    used = set(int(piece_id) for piece_id in pieces_used)
    used_squares = sum(sizes.get(piece_id, 0) for piece_id in used)
    return int(_total_piece_squares() - used_squares)


def _remaining_sizes_by_bucket(remaining_ids: Iterable[int]) -> Dict[int, int]:
    sizes = _piece_size_lookup()
    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for piece_id in remaining_ids:
        bucket = int(sizes.get(piece_id, 0))
        if bucket in counts:
            counts[bucket] += 1
    return counts


def build_snapshot_runtime_context(
    board: Board,
    *,
    turn_index: int,
    max_turns: int,
) -> SnapshotRuntimeContext:
    """Compute global snapshot context fields from current board state."""
    deadzones = compute_dead_zones(board)
    deadzone_count = int(sum(1 for row in deadzones for cell in row if cell))
    board_cells = board.SIZE * board.SIZE
    occupied_cells = int(np.count_nonzero(board.grid))
    board_occupancy = occupied_cells / board_cells if board_cells else 0.0
    progress_turn_ratio = float(turn_index / max(1, max_turns))
    progress_placement_ratio = float(board.move_count / max(1, max_turns))
    return SnapshotRuntimeContext(
        ply=int(board.move_count),
        turn_index=int(turn_index),
        board_occupancy=float(board_occupancy),
        deadzone_count=deadzone_count,
        deadzone_fraction=float(deadzone_count / board_cells if board_cells else 0.0),
        progress_turn_ratio=progress_turn_ratio,
        progress_placement_ratio=progress_placement_ratio,
    )


def extract_player_snapshot_features(
    board: Board,
    *,
    player: Player,
    context: SnapshotRuntimeContext,
    move_generator: Optional[LegalMoveGenerator] = None,
) -> Dict[str, float]:
    """Extract one player's feature vector for win-probability modeling."""
    move_gen = move_generator or LegalMoveGenerator()
    legal_moves = move_gen.get_legal_moves(board, player)
    pieces_used = sorted(board.player_pieces_used[player])
    mobility = compute_player_mobility_metrics(legal_moves, pieces_used)
    frontier_size = float(len(board.get_frontier(player)))
    opponent_legal_moves_sum = 0.0
    for opponent in Player:
        if opponent == player:
            continue
        opponent_legal_moves_sum += float(len(move_gen.get_legal_moves(board, opponent)))
    remaining_ids = _remaining_piece_ids(pieces_used)
    remaining_by_size = _remaining_sizes_by_bucket(remaining_ids)
    remaining_count = len(remaining_ids)
    used_count = len(pieces_used)
    piece_usage_ratio = float(used_count / 21.0)
    player_board_occupancy = float(np.count_nonzero(board.grid == player.value) / (board.SIZE * board.SIZE))
    utility = frontier_size + float(mobility.totalCellWeighted)

    return {
        "frontier_size": frontier_size,
        "corner_differential": float(compute_corner_differential(board, player)),
        "center_proximity": float(compute_center_proximity(board, player)),
        "opponent_adjacency": float(compute_opponent_adjacency(board, player)),
        "deadzone_count": float(context.deadzone_count),
        "deadzone_fraction": float(context.deadzone_fraction),
        "mobility_total_placements": float(mobility.totalPlacements),
        "mobility_total_orientation_normalized": float(mobility.totalOrientationNormalized),
        "mobility_total_cell_weighted": float(mobility.totalCellWeighted),
        "mobility_bucket_1": float(mobility.buckets.get(1, 0.0)),
        "mobility_bucket_2": float(mobility.buckets.get(2, 0.0)),
        "mobility_bucket_3": float(mobility.buckets.get(3, 0.0)),
        "mobility_bucket_4": float(mobility.buckets.get(4, 0.0)),
        "mobility_bucket_5": float(mobility.buckets.get(5, 0.0)),
        "blocking_exposure_legal_moves_opp_sum": float(opponent_legal_moves_sum),
        "utility_frontier_plus_mobility": float(utility),
        "pieces_used_count": float(used_count),
        "pieces_remaining_count": float(remaining_count),
        "remaining_squares": float(_remaining_squares(pieces_used)),
        "remaining_size_1_count": float(remaining_by_size[1]),
        "remaining_size_2_count": float(remaining_by_size[2]),
        "remaining_size_3_count": float(remaining_by_size[3]),
        "remaining_size_4_count": float(remaining_by_size[4]),
        "remaining_size_5_count": float(remaining_by_size[5]),
        "remaining_key_piece_17": float(1 if 17 in remaining_ids else 0),
        "remaining_key_piece_19": float(1 if 19 in remaining_ids else 0),
        "remaining_key_piece_20": float(1 if 20 in remaining_ids else 0),
        "phase_ply": float(context.ply),
        "phase_turn_index": float(context.turn_index),
        "phase_board_occupancy": float(context.board_occupancy),
        "phase_piece_usage_ratio": piece_usage_ratio,
        "phase_progress_turn_ratio": float(context.progress_turn_ratio),
        "phase_progress_placement_ratio": float(context.progress_placement_ratio),
        "player_board_occupancy": player_board_occupancy,
    }


def coerce_feature_dict(features: Mapping[str, Any]) -> Dict[str, float]:
    """Convert feature mapping into the canonical float-only feature dict."""
    output: Dict[str, float] = {}
    for column in SNAPSHOT_FEATURE_COLUMNS:
        value = features.get(column, 0.0)
        output[column] = float(value)
    return output

