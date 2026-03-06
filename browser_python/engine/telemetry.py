import logging
import random
from typing import Any, Dict

from .advanced_metrics import (
    compute_center_proximity,
    compute_dead_space_split,
    compute_effective_frontier,
    compute_frontier_spread,
)
from .board import Board, Player
from .metrics_config import TelemetryConfig
from .mobility_metrics import compute_player_mobility_metrics

logger = logging.getLogger(__name__)

def simulate_mobility_stability(
    board: Board,
    player: Player,
    move_generator,
    fast_mode: bool
) -> Dict[str, float]:
    rng = random.Random(TelemetryConfig.DETERMINISM_SEED)
    k_samples = TelemetryConfig.MOBILITY_SAMPLES_K if fast_mode else 8

    current_moves = move_generator.get_legal_moves(board, player)
    current_mobility = len(current_moves)
    if current_mobility == 0:
        return {
            "mobilityNextMean": 0.0,
            "mobilityNextP10": 0.0,
            "mobilityNextMin": 0.0,
            "mobilityDropRisk": 0.0,
        }

    sampled_mobilities = []
    opponents = [p for p in Player if p != player]

    for opp in opponents:
        opp_moves = move_generator.get_legal_moves(board, opp)
        if not opp_moves:
            continue

        # Shuffle with seeded RNG so it's deterministic but tests typical branches
        rng.shuffle(opp_moves)
        samples_to_evaluate = opp_moves[:k_samples]

        for move in samples_to_evaluate:
            test_board = board.copy()
            # Fast placement using cached orientations
            piece_positions = move.get_positions(move_generator.piece_orientations_cache[move.piece_id])
            test_board.place_piece(piece_positions, opp, move.piece_id)

            next_moves = move_generator.get_legal_moves(test_board, player)
            sampled_mobilities.append(float(len(next_moves)))

    if not sampled_mobilities:
        return {
            "mobilityNextMean": float(current_mobility),
            "mobilityNextP10": float(current_mobility),
            "mobilityNextMin": float(current_mobility),
            "mobilityDropRisk": 0.0,
        }

    sampled_mobilities.sort()
    n = len(sampled_mobilities)
    mean_mob = sum(sampled_mobilities) / n
    min_mob = sampled_mobilities[0]
    p10_idx = max(0, int(0.10 * n) - 1)
    p10_mob = sampled_mobilities[p10_idx]

    return {
        "mobilityNextMean": mean_mob,
        "mobilityNextP10": p10_mob,
        "mobilityNextMin": min_mob,
        "mobilityDropRisk": max(0.0, current_mobility - p10_mob),
    }

def compute_player_metrics(board: Board, player: Player, move_generator=None, fast_mode: bool = True) -> Dict[str, float]:
    frontier = board.get_frontier(player)
    frontier_size = len(frontier)

    # Base expansion
    comp_count, quad_cov = compute_frontier_spread(board, player)
    eff_frontier = compute_effective_frontier(board, player)

    # Phase calculation via material remaining
    # Initial area = 89 per player
    from .pieces import PieceGenerator
    gen = PieceGenerator()
    all_pieces = {p.id: p for p in gen.get_all_pieces()}

    p_pieces_used = list(board.player_pieces_used[player])
    p_has_move = {pid: False for pid in range(1, 22) if pid not in p_pieces_used}

    remaining_area = sum(all_pieces[pid].size for pid in p_has_move.keys())
    phase_ratio = max(0.0, min(1.0, 1.0 - (remaining_area / 89.0)))  # 0.0 early, 1.0 late

    center_dist = compute_center_proximity(board, player)
    center_raw = 9.0 - center_dist
    center_weighted = center_raw * phase_ratio

    metrics: Dict[str, float] = {
        "frontierSize": float(frontier_size),
        "effectiveFrontier": float(eff_frontier),
        "frontierComponentCount": float(comp_count),
        "frontierQuadrantCoverage": float(quad_cov),
        "centerControl": float(center_raw),
        "centerControlWeighted": float(center_weighted),
        "pieceLockRisk": 0.0,
        "remainingArea": float(remaining_area),
    }

    if move_generator is not None:
        moves = move_generator.get_legal_moves(board, player)
        metrics["mobility"] = float(len(moves))

        # True Mobility via mobility_metrics
        p_pieces_used = list(board.player_pieces_used[player])
        mob_metrics = compute_player_mobility_metrics(moves, p_pieces_used)

        metrics["mobilityWeighted"] = mob_metrics.totalCellWeighted
        metrics["mobilityEntropy"] = mob_metrics.mobilityEntropy
        metrics["pieceTop1Share"] = mob_metrics.pieceTop1Share
        metrics["anchorTop1Share"] = mob_metrics.anchorTop1Share

        # Stability
        stab = simulate_mobility_stability(board, player, move_generator, fast_mode)
        metrics.update(stab)

        # Piece lock logic
        p_has_move = {pid: False for pid in range(1, 22) if pid not in p_pieces_used}
        for m in moves:
            if m.piece_id in p_has_move:
                p_has_move[m.piece_id] = True
        metrics["pieceLockRisk"] = float(sum(1 for can_place in p_has_move.values() if not can_place))

        # New pieces metrics like lockedArea
        from .pieces import PieceGenerator
        gen = PieceGenerator()
        all_pieces = {p.id: p for p in gen.get_all_pieces()}
        locked_area = sum(all_pieces[pid].size for pid, can_place in p_has_move.items() if not can_place)
        metrics["lockedArea"] = float(locked_area)

        # Calculate critical (placements <= 3)
        placement_counts = {pid: 0 for pid in range(1, 22) if pid not in p_pieces_used}
        for m in moves:
            if m.piece_id in placement_counts:
                placement_counts[m.piece_id] += 1

        metrics["criticalPiecesCount"] = float(sum(1 for cnt in placement_counts.values() if cnt > 0 and cnt <= 3))

        if placement_counts:
            placed_vals = [c for c in placement_counts.values() if c > 0]
            metrics["bottleneckScore"] = float(min(placed_vals)) if placed_vals else 0.0
        else:
            metrics["bottleneckScore"] = 0.0

        largest_remaining = max((all_pieces[pid].size for pid in p_has_move.keys()), default=0)
        metrics["largestRemainingPiece"] = float(largest_remaining)

        unload_pot = sum(all_pieces[pid].size for pid in p_has_move.keys() if placement_counts.get(pid, 0) > 0)
        metrics["unloadPotential"] = float(unload_pot)

    else:
        # Fallbacks when no generator available
        metrics["mobility"] = float(frontier_size * 2)
        metrics["mobilityWeighted"] = metrics["mobility"] * 3
        metrics["mobilityEntropy"] = 0.0
        metrics["pieceTop1Share"] = 0.0
        metrics["anchorTop1Share"] = 0.0
        metrics["mobilityNextMean"] = metrics["mobility"]
        metrics["mobilityNextP10"] = metrics["mobility"]
        metrics["mobilityNextMin"] = metrics["mobility"]
        metrics["mobilityDropRisk"] = 0.0
        metrics["lockedArea"] = 0.0
        metrics["criticalPiecesCount"] = 0.0
        metrics["bottleneckScore"] = 0.0
        metrics["largestRemainingPiece"] = 0.0
        metrics["unloadPotential"] = 0.0

    return metrics

def collect_all_player_metrics(board: Board, move_generator=None, fast_mode: bool = True) -> Dict[str, Dict[str, float]]:
    metrics = {}

    for p in Player:
        p_metrics = compute_player_metrics(board, p, move_generator, fast_mode=fast_mode)

        # Dead space causality split
        ds_self, ds_opp = compute_dead_space_split(board, p)
        p_metrics["deadSpaceNearSelf"] = ds_self
        p_metrics["deadSpaceNearOpponents"] = ds_opp
        p_metrics["deadSpace"] = ds_self + ds_opp  # Fallback for UI backwards compatibility

        metrics[p.name] = p_metrics

    return metrics

def compute_move_telemetry_delta(
    ply: int,
    mover_id: str,
    move_id: str,
    before: Dict[str, Dict[str, float]],
    after: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
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

    # Advantage Deltas (using polarity map)
    def compute_adv(state: Dict[str, Dict[str, float]], player_id: str, k: str) -> float:
        if player_id not in state or k not in state[player_id]:
            return 0.0
        my_val = state[player_id][k]
        opp_vals = [state[p][k] for p in state if p != player_id and k in state[p]]
        mean_opp = sum(opp_vals) / len(opp_vals) if opp_vals else 0.0

        polarity = TelemetryConfig.METRIC_POLARITY.get(k, 1)
        return float(polarity * (my_val - mean_opp))

    delta_adv = {}
    for k in TelemetryConfig.METRIC_POLARITY.keys():
        if mover_id in after and k in after[mover_id]:
            before_adv = compute_adv(before, mover_id, k)
            after_adv = compute_adv(after, mover_id, k)
            delta_adv[k + "Adv"] = after_adv - before_adv

    # Win Proxy Score
    def sum_win_proxy(state: Dict[str, Dict[str, float]], player_id: str) -> float:
        score = 0.0
        for base_key, weight in TelemetryConfig.WIN_PROXY_WEIGHTS.items():
            k_base = base_key.replace("Adv", "")
            if player_id in state and k_base in state[player_id]:
               adv = compute_adv(state, player_id, k_base)
               score += adv * weight
        return score

    win_proxy_before = sum_win_proxy(before, mover_id)
    win_proxy_after = sum_win_proxy(after, mover_id)
    delta_adv["winProxy"] = win_proxy_after - win_proxy_before

    return {
        "ply": ply,
        "moverId": mover_id,
        "moveId": move_id,
        "deltaSelf": delta_self,
        "deltaOppTotal": delta_opp_total,
        "deltaOppByPlayer": delta_opp_by_player,
        "deltaAdvantage": delta_adv,
        "winProxyDelta": delta_adv["winProxy"],
    }
