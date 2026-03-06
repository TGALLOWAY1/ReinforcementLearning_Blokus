"""
Mobility metrics per spec: docs/metrics/mobility.md

P_i = placements per piece, O_i = unique orientations, S_i = piece size.
PN_i = P_i / O_i, MW_i = PN_i * S_i.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from .move_generator import LegalMoveGenerator, Move
from .pieces import PieceGenerator


@dataclass
class PlayerMobilityMetrics:
    totalPlacements: int
    totalOrientationNormalized: float
    totalCellWeighted: float
    buckets: Dict[int, float]
    mobilityEntropy: float
    pieceTop1Share: float
    anchorTop1Share: float


def _get_orientation_counts() -> Dict[int, int]:
    """O_i: unique orientation count per piece. Matches LegalMoveGenerator cache."""
    gen = LegalMoveGenerator()
    return {piece_id: len(gen.piece_orientations_cache[piece_id]) for piece_id in range(1, 22)}


def _get_piece_sizes() -> Dict[int, int]:
    """S_i: piece size (1-5)."""
    pieces = PieceGenerator.get_all_pieces()
    return {p.id: p.size for p in pieces}


def compute_player_mobility_metrics(
    legal_moves: List[Move],
    pieces_used: List[int],
) -> PlayerMobilityMetrics:
    """
    Compute mobility metrics from legal moves.

    Args:
        legal_moves: List of Move objects (piece_id, orientation, anchor_row, anchor_col)
        pieces_used: Piece IDs already placed by the player (excluded)

    Returns:
        PlayerMobilityMetrics matching docs/metrics/mobility.md
    """
    pieces_used_set = set(pieces_used)

    # P_i: count placements per piece
    P: Dict[int, int] = {}
    for move in legal_moves:
        pid = move.piece_id
        if 1 <= pid <= 21:
            P[pid] = P.get(pid, 0) + 1

    O = _get_orientation_counts()
    S = _get_piece_sizes()

    total_placements = 0
    total_orientation_normalized = 0.0
    total_cell_weighted = 0.0
    buckets: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

    # For Piece Concentration
    max_piece_placements = 0

    # For Anchor Concentration
    # A move's "anchor" is its (anchor_row, anchor_col) coordinate
    anchor_counts: Dict[tuple, int] = {}
    for move in legal_moves:
        anchor = (move.anchor_row, move.anchor_col)
        anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1

    for piece_id in range(1, 22):
        if piece_id in pieces_used_set:
            continue
        Pi = P.get(piece_id, 0)
        Oi = O.get(piece_id, 1)
        Si = S.get(piece_id, 0)
        if Si < 1 or Si > 5:
            continue

        if Pi > max_piece_placements:
            max_piece_placements = Pi

        PNi = Pi / Oi if Oi > 0 else 0.0
        MWi = PNi * Si

        total_placements += Pi
        total_orientation_normalized += PNi
        total_cell_weighted += MWi
        buckets[Si] = buckets.get(Si, 0) + MWi

    # Compute Entropy
    import math
    mobility_entropy = 0.0
    if total_placements > 0:
        entropy_sum = 0.0
        for piece_id, Pi in P.items():
            if piece_id not in pieces_used_set and Pi > 0:
                q_p = Pi / total_placements
                entropy_sum -= q_p * math.log(q_p)

        # Normalize by log(num_available_pieces) to get [0, 1]
        num_avail = 21 - len(pieces_used_set)
        if num_avail > 1:
            mobility_entropy = entropy_sum / math.log(num_avail)
        else:
            mobility_entropy = 0.0

    piece_top1_share = 0.0
    if total_placements > 0:
        piece_top1_share = max_piece_placements / total_placements

    anchor_top1_share = 0.0
    if total_placements > 0 and anchor_counts:
        max_anchor_placements = max(anchor_counts.values())
        anchor_top1_share = max_anchor_placements / total_placements

    return PlayerMobilityMetrics(
        totalPlacements=total_placements,
        totalOrientationNormalized=total_orientation_normalized,
        totalCellWeighted=total_cell_weighted,
        buckets=buckets,
        mobilityEntropy=mobility_entropy,
        pieceTop1Share=piece_top1_share,
        anchorTop1Share=anchor_top1_share,
    )


def compute_player_mobility_metrics_from_dicts(
    legal_moves: List[Dict[str, Any]],
    pieces_used: List[int],
) -> PlayerMobilityMetrics:
    """
    Same as compute_player_mobility_metrics but accepts dicts with piece_id or pieceId.
    Used when receiving API payloads.
    """
    # Convert to Move-like objects for counting
    class MoveLike:
        def __init__(self, piece_id: int):
            self.piece_id = piece_id

    moves = []
    for m in legal_moves:
        pid = m.get("piece_id") or m.get("pieceId")
        if pid is not None:
            moves.append(MoveLike(pid))
    return compute_player_mobility_metrics(moves, pieces_used)
