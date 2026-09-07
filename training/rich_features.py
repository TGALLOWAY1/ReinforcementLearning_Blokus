"""Rich Blokus-specific state features for temporal-difference learning.

This module extends the eight Layer-6 state-evaluator features
(:data:`mcts.state_evaluator.FEATURE_NAMES`) with a much wider set of
Blokus-strategy features used to train the TD value model
(:mod:`training.td_learning`).

Design goals
------------
* **Versioned & stable.** :data:`FEATURE_SET_VERSION` names this exact feature
  layout. :data:`RICH_FEATURE_NAMES` is an ordered, append-only list — once a
  feature ships it must keep its name and position so persisted training
  artifacts stay readable.
* **Superset of the agent's evaluator features.** Every name in
  ``mcts.state_evaluator.FEATURE_NAMES`` is present here with identical
  semantics, so a TD-learned weight vector can be *projected* back onto the
  eight features the live agent's :class:`BlokusStateEvaluator` understands
  (see :func:`training.td_learning.project_to_agent_weights`).
* **Reuse, not reinvent.** Frontier, mobility, and advanced-metric helpers from
  the engine are reused rather than re-implemented.
* **Cache the expensive bits.** Legal-move enumeration is the dominant cost; a
  per-extraction :class:`FeatureCache` memoises legal moves per player so a
  single snapshot (which touches every player's mobility) enumerates each
  player's moves at most once.

All returned values are floats, finite (never NaN/inf), and normalised to keep
the linear TD value in a sane range.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.board import Board, Player
from engine.move_generator import LegalMoveGenerator
from engine.mobility_metrics import compute_player_mobility_metrics
from mcts.state_evaluator import (
    BlokusStateEvaluator,
    FEATURE_NAMES as SE_FEATURE_NAMES,
    PHASE_EARLY_THRESHOLD,
    PHASE_LATE_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

FEATURE_SET_VERSION = "rich_blokus_v2"  # v2: + contested-territory block (6 features)

_PLAYERS: List[Player] = list(Player)

# ---------------------------------------------------------------------------
# Piece metadata (ids 1..21). Reuse the same source the engine uses.
# ---------------------------------------------------------------------------

_PIECE_SIZES: Optional[Dict[int, int]] = None
_TOTAL_PIECE_AREA: int = 0
_PIECE_COUNT_BY_SIZE: Dict[int, int] = {}
_TOTAL_PIECE_COUNT: int = 0
# "Key" / high-value pentominoes that are awkward to place if left late.
_AWKWARD_PIECE_IDS = (17, 19, 20)


def _piece_sizes() -> Dict[int, int]:
    global _PIECE_SIZES, _TOTAL_PIECE_AREA, _PIECE_COUNT_BY_SIZE, _TOTAL_PIECE_COUNT
    if _PIECE_SIZES is None:
        from engine.pieces import PieceGenerator

        sizes: Dict[int, int] = {}
        counts: Dict[int, int] = {}
        for piece in PieceGenerator.get_all_pieces():
            sizes[piece.id] = piece.size
            counts[piece.size] = counts.get(piece.size, 0) + 1
        _PIECE_SIZES = sizes
        _TOTAL_PIECE_AREA = int(sum(sizes.values()))
        _PIECE_COUNT_BY_SIZE = counts
        _TOTAL_PIECE_COUNT = len(sizes)
    return _PIECE_SIZES


def total_piece_area() -> int:
    _piece_sizes()
    return _TOTAL_PIECE_AREA


def _count_of_size(size: int) -> int:
    """How many distinct pieces of a given cell-count exist in the piece set.

    Derived from the engine's actual piece generator (standard Blokus set:
    1/1/2/5/12 pieces of sizes 1–5 since 2026-09; earlier builds shipped a
    non-standard 1/1/2/6/11 set), so feature normalisation divisors stay
    correct if the piece set ever changes.
    """
    _piece_sizes()
    return max(_PIECE_COUNT_BY_SIZE.get(size, 0), 1)


def total_piece_count() -> int:
    _piece_sizes()
    return max(_TOTAL_PIECE_COUNT, 1)


# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------

_MAX_FRONTIER = 40.0
_MAX_REACHABLE = 60.0
_MAX_LEGAL_MOVES = 400.0  # rough upper bound on a single player's legal placements
_BOARD_CELLS = float(Board.SIZE * Board.SIZE)  # 400
_MAX_PIECE_DIM = float(Board.SIZE)  # for distance normalisation
_MAX_REGION = _BOARD_CELLS

# ---------------------------------------------------------------------------
# Ordered, append-only feature name list
# ---------------------------------------------------------------------------

# Extra rich features (everything beyond the eight SE features). Keep this list
# append-only; new features go at the end.
_RICH_EXTRA_NAMES: List[str] = [
    # Mobility / move-space
    "legal_move_count",
    "legal_move_count_small_pieces",
    "legal_move_count_medium_pieces",
    "legal_move_count_large_pieces",
    "playable_piece_count",
    "total_playable_piece_area",
    "avg_legal_move_area",
    "max_legal_move_area",
    # Corner / frontier
    "corner_count",
    "corner_quality_score",
    "frontier_size",
    "frontier_density",
    "frontier_to_opponent_distance",
    "new_corner_generation_potential",
    # Piece inventory
    "remaining_piece_count",
    "remaining_singletons",
    "remaining_dominoes",
    "remaining_trominoes",
    "remaining_tetrominoes",
    "remaining_pentominoes",
    "awkward_piece_penalty",
    "piece_diversity_score",
    # Territory and blocking
    "largest_reachable_region",
    "trapped_region_count",
    "trapped_region_area",
    "opponent_mobility_avg",
    "opponent_mobility_max",
    "opponent_mobility_min",
    "opponent_corner_pressure",
    "leader_mobility_pressure",
    # Score / race
    "score_margin_vs_leader",
    "score_margin_vs_next_player",
    "rank_so_far",
    "completion_ratio",
    "remaining_area",
    # Center / board-position
    "edge_pressure",
    "quadrant_balance",
    # Contested territory (v2 block, append-only): who can actually realize
    # their remaining piece area. Exclusive = reachable by me and no opponent.
    "exclusive_reachable_area",
    "contested_reachable_area",
    "reachable_piece_bound",
    "opponent_exclusive_max",
    "exclusive_area_margin",
    "reachable_bound_margin",
]

# Full ordered list: the eight SE features first (so the projection slice is the
# leading block), then the rich extras. Names are unique.
RICH_FEATURE_NAMES: List[str] = list(SE_FEATURE_NAMES) + [
    n for n in _RICH_EXTRA_NAMES if n not in SE_FEATURE_NAMES
]


# ---------------------------------------------------------------------------
# Per-extraction cache (legal moves are the expensive part)
# ---------------------------------------------------------------------------


@dataclass
class FeatureCache:
    """Memoises per-player legal moves within a single snapshot extraction.

    A snapshot computes mobility for every player (the focal player plus each
    opponent for the opponent-mobility features), so without caching a 4-player
    snapshot would enumerate legal moves up to 16 times. The cache keys on
    ``player.value`` and assumes the board is not mutated while it is alive —
    callers create a fresh cache per board state.
    """

    move_generator: LegalMoveGenerator = field(default_factory=LegalMoveGenerator)
    _legal_moves: Dict[int, List] = field(default_factory=dict)
    _reachable: Dict[int, frozenset] = field(default_factory=dict)

    def legal_moves(self, board: Board, player: Player) -> List:
        key = int(player.value)
        cached = self._legal_moves.get(key)
        if cached is None:
            cached = self.move_generator.get_legal_moves(board, player)
            self._legal_moves[key] = cached
        return cached

    def reachable_cells(self, board: Board, player: Player) -> frozenset:
        """Memoised frontier-flood reachable set (incl. frontier cells)."""
        key = int(player.value)
        cached = self._reachable.get(key)
        if cached is None:
            cached = frozenset(_reachable_cells(board, board.get_frontier(player)))
            self._reachable[key] = cached
        return cached


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_phase(board: Board) -> str:
    """Game phase from board occupancy (matches the evaluator's thresholds)."""
    occ = float(np.count_nonzero(board.grid)) / _BOARD_CELLS
    if occ < PHASE_EARLY_THRESHOLD:
        return "early"
    if occ < PHASE_LATE_THRESHOLD:
        return "mid"
    return "late"


def board_occupancy(board: Board) -> float:
    return float(np.count_nonzero(board.grid)) / _BOARD_CELLS


def _remaining_piece_ids(pieces_used) -> List[int]:
    used = {int(p) for p in pieces_used}
    return [pid for pid in range(1, 22) if pid not in used]


def _player_scores(board: Board) -> Dict[int, int]:
    """Squares-covered score proxy per player (cheap; no bonuses)."""
    grid = board.grid
    return {p.value: int(np.count_nonzero(grid == p.value)) for p in _PLAYERS}


def _mobility_count(cache: FeatureCache, board: Board, player: Player) -> int:
    return len(cache.legal_moves(board, player))


def _reachable_cells(board: Board, frontier) -> set:
    """Empty cells orthogonally connected to the player's frontier (incl. the
    frontier cells themselves) — the space this player could still expand into."""
    grid = board.grid
    size = board.SIZE
    frontier_set = set(frontier)
    reachable = set()
    queue = deque(frontier_set)
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                if grid[nr, nc] == 0 and (nr, nc) not in reachable and (nr, nc) not in frontier_set:
                    reachable.add((nr, nc))
                    queue.append((nr, nc))
    return reachable | frontier_set


def _largest_reachable_region_and_trapped(
    board: Board, frontier
) -> Tuple[int, int, int]:
    """BFS over empty cells from the player's frontier.

    Returns ``(largest_reachable_region, trapped_region_count, trapped_region_area)``.

    * *reachable* = empty cells connected (orthogonally) to a frontier cell.
    * *trapped* = connected components of empty cells that touch **no** frontier
      cell (regions this player can never expand into).
    """
    grid = board.grid
    size = board.SIZE
    frontier_set = set(frontier)
    largest_reachable = len(_reachable_cells(board, frontier))

    # Connected components of all empties; those disjoint from the frontier are
    # "trapped" for this player.
    visited = np.zeros((size, size), dtype=bool)
    trapped_count = 0
    trapped_area = 0
    for r in range(size):
        for c in range(size):
            if grid[r, c] != 0 or visited[r, c]:
                continue
            # BFS this component.
            comp = []
            touches_frontier = False
            q = deque([(r, c)])
            visited[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                if (cr, cc) in frontier_set:
                    touches_frontier = True
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < size and 0 <= nc < size and not visited[nr, nc] and grid[nr, nc] == 0:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            if not touches_frontier:
                trapped_count += 1
                trapped_area += len(comp)
    return largest_reachable, trapped_count, trapped_area


def _frontier_to_opponent_distance(board: Board, player: Player, frontier) -> float:
    """Min Manhattan distance from any frontier cell to any opponent cell."""
    if not frontier:
        return 1.0  # normalised max distance (no frontier ⇒ no pressure)
    grid = board.grid
    opp_coords = np.argwhere((grid != 0) & (grid != player.value))
    if opp_coords.size == 0:
        return 1.0
    best = None
    for (fr, fc) in frontier:
        d = np.min(np.abs(opp_coords[:, 0] - fr) + np.abs(opp_coords[:, 1] - fc))
        if best is None or d < best:
            best = int(d)
            if best == 0:
                break
    return min(float(best) / (2.0 * board.SIZE), 1.0)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Leaf-evaluator feature subsets (cost tiers)
# ---------------------------------------------------------------------------
#
# extract_rich_features enumerates legal moves for ALL FOUR players (the
# opponent-mobility features) which costs ~15-27 ms/leaf — far too slow to call
# even once per MCTS simulation. The groups below classify features by the
# expensive computation they require so the leaf evaluator can request only an
# affordable subset (:func:`extract_leaf_features`).

# Features that require enumerating EVERY player's legal moves (the dominant
# cost, ~11-22 ms): the three opponent-mobility aggregates plus the score
# leader's mobility.
OPPONENT_MOBILITY_FEATURES: frozenset = frozenset({
    "opponent_mobility_avg",
    "opponent_mobility_max",
    "opponent_mobility_min",
    "leader_mobility_pressure",
})

# Features that require enumerating the FOCAL player's legal moves (~3-5 ms).
FOCAL_MOBILITY_FEATURES: frozenset = frozenset({
    "legal_move_count",
    "legal_move_count_small_pieces",
    "legal_move_count_medium_pieces",
    "legal_move_count_large_pieces",
    "playable_piece_count",
    "total_playable_piece_area",
    "avg_legal_move_area",
    "max_legal_move_area",
    "corner_quality_score",
    "new_corner_generation_potential",
})

# Features that require the whole-board reachable-region / trapped-region BFS
# (~0.5-1 ms).
TERRITORY_FEATURES: frozenset = frozenset({
    "largest_reachable_region",
    "trapped_region_count",
    "trapped_region_area",
    "frontier_density",
})

# v2 contested-territory block: needs the reachable flood for ALL FOUR players
# (memoised per board on the FeatureCache, so a 4-perspective snapshot pays for
# each player's flood exactly once).
CONTESTED_TERRITORY_FEATURES: frozenset = frozenset({
    "exclusive_reachable_area",
    "contested_reachable_area",
    "reachable_piece_bound",
    "opponent_exclusive_max",
    "exclusive_area_margin",
    "reachable_bound_margin",
})

# The cheap remainder (SE-8, score/rank/margin, piece inventory, frontier
# counts, board-position): no legal-move enumeration and no territory BFS,
# ~0.3 ms on top of the SE-8 features.
_ALL_FEATURE_SET: frozenset = frozenset(RICH_FEATURE_NAMES)
SCORE_LEAF_FEATURES: frozenset = (
    _ALL_FEATURE_SET
    - OPPONENT_MOBILITY_FEATURES
    - FOCAL_MOBILITY_FEATURES
    - TERRITORY_FEATURES
    - CONTESTED_TERRITORY_FEATURES
)

# Named cost tiers exposed to the agent via ``rich_leaf_feature_subset``.
#   "full"            — all 45 features (~15-27 ms/leaf); usually too slow.
#   "no_opp_mobility" — drops only the all-player opponent enumeration
#                       (~4-6 ms/leaf); keeps focal mobility + territory.
#   "score"           — cheap, high-signal subset (~0.3 ms/leaf); the default.
LEAF_FEATURE_SUBSETS: Dict[str, frozenset] = {
    "full": _ALL_FEATURE_SET,
    "no_opp_mobility": _ALL_FEATURE_SET - OPPONENT_MOBILITY_FEATURES,
    "score": SCORE_LEAF_FEATURES,
}


def extract_rich_features(
    board: Board,
    player: Player,
    *,
    cache: Optional[FeatureCache] = None,
    evaluator: Optional[BlokusStateEvaluator] = None,
) -> Dict[str, float]:
    """Return the full ``rich_blokus_v1`` feature dict for *player*.

    The result always contains exactly :data:`RICH_FEATURE_NAMES`, every value a
    finite float. ``cache`` (optional) shares legal-move enumeration across the
    players evaluated for one board state.
    """
    return _extract_features_impl(
        board, player, include=_ALL_FEATURE_SET, cache=cache, evaluator=evaluator
    )


def extract_leaf_features(
    board: Board,
    player: Player,
    *,
    subset: str = "score",
    cache: Optional[FeatureCache] = None,
    evaluator: Optional[BlokusStateEvaluator] = None,
) -> Dict[str, float]:
    """Return a *subset* of the rich features for cheap MCTS-leaf evaluation.

    Computes only the features in ``LEAF_FEATURE_SUBSETS[subset]`` (skipping the
    expensive legal-move enumeration / territory BFS for everything else) and
    zeroes the rest. The included features carry **identical values** to
    :func:`extract_rich_features` (same gated implementation), so a TD weight
    vector trained on the full features can be applied directly — the zeroed
    terms simply drop out of the dot product.

    Args:
        subset: One of ``"full"``, ``"no_opp_mobility"``, or ``"score"``.
    """
    include = LEAF_FEATURE_SUBSETS.get(subset)
    if include is None:
        raise ValueError(
            f"unknown leaf feature subset '{subset}'; "
            f"expected one of {sorted(LEAF_FEATURE_SUBSETS)}"
        )
    return _extract_features_impl(
        board, player, include=include, cache=cache, evaluator=evaluator
    )


def _extract_features_impl(
    board: Board,
    player: Player,
    *,
    include: frozenset,
    cache: Optional[FeatureCache] = None,
    evaluator: Optional[BlokusStateEvaluator] = None,
) -> Dict[str, float]:
    """Shared extraction core for the full and leaf-subset feature paths.

    Only the blocks whose features intersect ``include`` are computed; the
    expensive focal-mobility, territory, and opponent-mobility blocks are gated
    so a cheap subset never pays for them. Features outside ``include`` are
    emitted as ``0.0``. When ``include`` is the full feature set this is
    behaviourally identical to the original single-pass extraction.
    """
    cache = cache or FeatureCache()
    evaluator = evaluator or _shared_evaluator()
    sizes = _piece_sizes()
    total_area = total_piece_area()

    need_focal = bool(include & FOCAL_MOBILITY_FEATURES)
    need_territory = bool(include & TERRITORY_FEATURES)
    need_opp_mobility = bool(include & OPPONENT_MOBILITY_FEATURES)
    need_contested = bool(include & CONTESTED_TERRITORY_FEATURES)

    # --- Eight SE features (identical semantics to the live agent) -----------
    feats: Dict[str, float] = dict(evaluator.extract_features(board, player))

    pieces_used = board.player_pieces_used[player]
    remaining_ids = _remaining_piece_ids(pieces_used)
    frontier = board.get_frontier(player)

    # --- Mobility / move-space (gated: focal legal-move enumeration) ---------
    n_moves = 0
    playable_ids: set = set()
    if need_focal:
        legal_moves = cache.legal_moves(board, player)
        n_moves = len(legal_moves)
        small = medium = large = 0
        areas: List[int] = []
        for mv in legal_moves:
            pid = int(getattr(mv, "piece_id"))
            sz = sizes.get(pid, 0)
            areas.append(sz)
            playable_ids.add(pid)
            if sz <= 2:
                small += 1
            elif sz == 3:
                medium += 1
            else:
                large += 1
        playable_area = sum(sizes.get(pid, 0) for pid in playable_ids)
        avg_area = (sum(areas) / len(areas)) if areas else 0.0
        max_area = float(max(areas)) if areas else 0.0

        feats["legal_move_count"] = min(n_moves / _MAX_LEGAL_MOVES, 1.0)
        feats["legal_move_count_small_pieces"] = min(small / _MAX_LEGAL_MOVES, 1.0)
        feats["legal_move_count_medium_pieces"] = min(medium / _MAX_LEGAL_MOVES, 1.0)
        feats["legal_move_count_large_pieces"] = min(large / _MAX_LEGAL_MOVES, 1.0)
        feats["playable_piece_count"] = len(playable_ids) / total_piece_count()
        feats["total_playable_piece_area"] = playable_area / max(total_area, 1)
        feats["avg_legal_move_area"] = avg_area / 5.0
        feats["max_legal_move_area"] = max_area / 5.0

    # --- Corner / frontier (cheap counts always; quality gated on focal) -----
    feats["frontier_size"] = min(len(frontier) / _MAX_FRONTIER, 1.0)
    feats["corner_count"] = min(len(frontier) / _MAX_FRONTIER, 1.0)
    if need_focal:
        # corner quality: legal placements available per anchor (anchors that
        # lead to many moves are higher quality).
        feats["corner_quality_score"] = (
            min((n_moves / max(len(frontier), 1)) / 20.0, 1.0) if frontier else 0.0
        )
        # new-corner potential: distinct anchor cells reachable by current legal
        # moves, proxied by playable-piece diversity × frontier breadth.
        feats["new_corner_generation_potential"] = min(
            (len(playable_ids) * len(frontier)) / 200.0, 1.0
        )
    if need_territory:
        largest_reach, trapped_count, trapped_area = (
            _largest_reachable_region_and_trapped(board, frontier)
        )
        feats["frontier_density"] = (
            min(len(frontier) / max(largest_reach, 1), 1.0) if largest_reach else 0.0
        )
        # --- Territory and blocking -----------------------------------------
        feats["largest_reachable_region"] = min(largest_reach / _MAX_REACHABLE, 1.0)
        feats["trapped_region_count"] = min(trapped_count / 10.0, 1.0)
        feats["trapped_region_area"] = min(trapped_area / _BOARD_CELLS, 1.0)
    feats["frontier_to_opponent_distance"] = _frontier_to_opponent_distance(
        board, player, frontier
    )

    # --- Contested territory (v2 block; gated: 4-player reachable floods) ----
    if need_contested:
        mine = cache.reachable_cells(board, player)
        opp_sets = {
            opp: cache.reachable_cells(board, opp)
            for opp in _PLAYERS if opp != player
        }
        union_opp: frozenset = frozenset().union(*opp_sets.values()) if opp_sets else frozenset()
        exclusive = mine - union_opp
        contested = mine & union_opp

        def _remaining_area(p: Player) -> float:
            return float(sum(
                sizes.get(pid, 0)
                for pid in _remaining_piece_ids(board.player_pieces_used[p])
            ))

        all_sets = {player: mine, **opp_sets}

        def _bound(p: Player) -> float:
            reach = all_sets[p]
            others = frozenset().union(
                *(s for q, s in all_sets.items() if q != p)
            )
            excl = reach - others
            cont = reach & others
            return min(_remaining_area(p), len(excl) + 0.5 * len(cont))

        my_bound = min(_remaining_area(player), len(exclusive) + 0.5 * len(contested))
        opp_excls = {
            opp: len(reach - frozenset().union(
                *(s for q, s in all_sets.items() if q != opp)
            ))
            for opp, reach in opp_sets.items()
        }
        opp_bounds = [_bound(opp) for opp in opp_sets]
        max_opp_excl = max(opp_excls.values()) if opp_excls else 0
        max_opp_bound = max(opp_bounds) if opp_bounds else 0.0

        feats["exclusive_reachable_area"] = min(len(exclusive) / _MAX_REGION, 1.0)
        feats["contested_reachable_area"] = min(len(contested) / _MAX_REGION, 1.0)
        feats["reachable_piece_bound"] = min(my_bound / max(total_area, 1), 1.0)
        feats["opponent_exclusive_max"] = min(max_opp_excl / _MAX_REGION, 1.0)
        feats["exclusive_area_margin"] = max(
            -1.0, min((len(exclusive) - max_opp_excl) / _MAX_REGION, 1.0)
        )
        feats["reachable_bound_margin"] = max(
            -1.0, min((my_bound - max_opp_bound) / max(total_area, 1), 1.0)
        )

    # --- Piece inventory (cheap) --------------------------------------------
    rem_by_size = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for pid in remaining_ids:
        sz = sizes.get(pid, 0)
        if sz in rem_by_size:
            rem_by_size[sz] += 1
    feats["remaining_piece_count"] = len(remaining_ids) / total_piece_count()
    feats["remaining_singletons"] = rem_by_size[1] / _count_of_size(1)
    feats["remaining_dominoes"] = rem_by_size[2] / _count_of_size(2)
    feats["remaining_trominoes"] = rem_by_size[3] / _count_of_size(3)
    feats["remaining_tetrominoes"] = rem_by_size[4] / _count_of_size(4)
    feats["remaining_pentominoes"] = rem_by_size[5] / _count_of_size(5)
    awkward = sum(5 if pid in _AWKWARD_PIECE_IDS else (sizes.get(pid, 0))
                  for pid in remaining_ids)
    feats["awkward_piece_penalty"] = min(awkward / max(total_area, 1), 1.0)
    distinct_sizes = sum(1 for v in rem_by_size.values() if v > 0)
    feats["piece_diversity_score"] = distinct_sizes / 5.0

    # --- Opponent mobility (gated: all-player legal-move enumeration) --------
    if need_opp_mobility:
        opp_mobilities: List[int] = []
        for opp in _PLAYERS:
            if opp == player:
                continue
            opp_mobilities.append(_mobility_count(cache, board, opp))
        if opp_mobilities:
            feats["opponent_mobility_avg"] = min(
                (sum(opp_mobilities) / len(opp_mobilities)) / _MAX_LEGAL_MOVES, 1.0
            )
            feats["opponent_mobility_max"] = min(max(opp_mobilities) / _MAX_LEGAL_MOVES, 1.0)
            feats["opponent_mobility_min"] = min(min(opp_mobilities) / _MAX_LEGAL_MOVES, 1.0)
        else:
            feats["opponent_mobility_avg"] = 0.0
            feats["opponent_mobility_max"] = 0.0
            feats["opponent_mobility_min"] = 0.0

    # opponent corner pressure: total opponent frontier near this player's
    # pieces (cheap — frontier sets only, no enumeration).
    opp_frontier_total = 0
    for opp in _PLAYERS:
        if opp == player:
            continue
        opp_frontier_total += len(board.get_frontier(opp))
    feats["opponent_corner_pressure"] = min(opp_frontier_total / (3.0 * _MAX_FRONTIER), 1.0)

    # --- Score / race (cheap — square counts only) --------------------------
    scores = _player_scores(board)
    if need_opp_mobility:
        # leader mobility pressure: mobility of the current score leader.
        leader = max((p for p in _PLAYERS if p != player), key=lambda p: scores[p.value])
        feats["leader_mobility_pressure"] = min(
            _mobility_count(cache, board, leader) / _MAX_LEGAL_MOVES, 1.0
        )
    my_score = scores[player.value]
    others = sorted((scores[p.value] for p in _PLAYERS if p != player), reverse=True)
    leader_score = others[0] if others else 0
    # "Next player" = the opponent adjacent to me in the score ranking (the player
    # just above me, or the runner-up if I am leading) — distinct from the leader.
    higher = [s for s in others if s > my_score]
    if higher:
        next_score = min(higher)  # closest opponent ranked above me
    elif len(others) >= 2:
        next_score = others[1]  # I'm leading ⇒ gap to the second-best opponent
    elif others:
        next_score = others[0]
    else:
        next_score = 0
    feats["score_margin_vs_leader"] = float(np.tanh((my_score - leader_score) / 20.0))
    feats["score_margin_vs_next_player"] = float(np.tanh((my_score - next_score) / 20.0))
    rank = 1 + sum(1 for s in others if s > my_score)
    feats["rank_so_far"] = (4 - rank) / 3.0  # 1st⇒1.0, 4th⇒0.0
    feats["completion_ratio"] = (total_piece_count() - len(remaining_ids)) / total_piece_count()
    rem_area = sum(sizes.get(pid, 0) for pid in remaining_ids)
    feats["remaining_area"] = rem_area / max(total_area, 1)

    # --- Center / board-position --------------------------------------------
    player_mask = board.grid == player.value
    coords = np.argwhere(player_mask)
    if coords.size:
        # edge pressure: fraction of own cells on the outer ring.
        on_edge = np.sum(
            (coords[:, 0] == 0)
            | (coords[:, 0] == board.SIZE - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == board.SIZE - 1)
        )
        feats["edge_pressure"] = float(on_edge) / float(len(coords))
        # quadrant balance: 1 - normalised spread of cell counts across quadrants.
        mid = board.SIZE / 2.0
        q = [0, 0, 0, 0]
        for (r, c) in coords:
            idx = (0 if r < mid else 2) + (0 if c < mid else 1)
            q[idx] += 1
        q_arr = np.array(q, dtype=float)
        mean = q_arr.mean()
        spread = (q_arr.std() / mean) if mean > 0 else 0.0
        feats["quadrant_balance"] = float(max(0.0, 1.0 - spread))
    else:
        feats["edge_pressure"] = 0.0
        feats["quadrant_balance"] = 0.0

    # --- Sanitise: ensure exactly the declared names, all finite -------------
    out: Dict[str, float] = {}
    for name in RICH_FEATURE_NAMES:
        v = feats.get(name, 0.0)
        fv = float(v)
        if not np.isfinite(fv):
            fv = 0.0
        out[name] = fv
    return out


# ---------------------------------------------------------------------------
# Shared evaluator (default weights; only used for raw feature extraction)
# ---------------------------------------------------------------------------

_SHARED_EVALUATOR: Optional[BlokusStateEvaluator] = None


def _shared_evaluator() -> BlokusStateEvaluator:
    global _SHARED_EVALUATOR
    if _SHARED_EVALUATOR is None:
        _SHARED_EVALUATOR = BlokusStateEvaluator()
    return _SHARED_EVALUATOR


def feature_vector(features: Dict[str, float]) -> np.ndarray:
    """Project a feature dict onto the canonical ordered vector."""
    return np.array([float(features.get(n, 0.0)) for n in RICH_FEATURE_NAMES], dtype=float)
