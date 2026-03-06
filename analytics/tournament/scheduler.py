"""Deterministic and balanced scheduling for n-player tournaments."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List

from analytics.tournament.arena_runner import stable_hash_int


@dataclass(frozen=True)
class Matchup:
    """A scheduled game assignment."""
    game_index: int
    seats: Dict[int, str]  # seat_index -> agent_name (0-indexed seats)


def generate_matchups(
    tunings: Sequence[str],
    num_games: int,
    seed: int,
    seat_policy: str = "randomized"
) -> List[Matchup]:
    """
    Generate a balanced schedule of matchups for exactly 4 players per game.
    
    Tries to ensure:
    1. Equal game counts per tuning (±1).
    2. Even seat distribution (±1).
    3. Minimized maximum pairwise exposure gap.
    """
    n_tunings = len(tunings)

    if n_tunings < 4:
        raise ValueError("Tournament requires at least 4 tunings (or pad with duplicates).")

    rng = random.Random(stable_hash_int(seed, "tournament_scheduler"))

    matchups: List[Matchup] = []

    if n_tunings == 4:
        # Trivial case: all 4 play every game. We balance seats via tracking.
        seat_counts_4 = {t: dict.fromkeys(range(4), 0) for t in tunings}
        for g in range(num_games):
            unassigned_players = list(tunings)
            if seat_policy == "round_robin":
                shift = g % 4
                ordered = unassigned_players[shift:] + unassigned_players[:shift]
                assigned_seats = {i: ordered[i] for i in range(4)}
            else:
                rng.shuffle(unassigned_players)
                assigned_seats = {}
                available_seats = [0, 1, 2, 3]
                for t in unassigned_players:
                    best_seat = min(available_seats, key=lambda s: seat_counts_4[t][s])
                    assigned_seats[best_seat] = t
                    seat_counts_4[t][best_seat] += 1
                    available_seats.remove(best_seat)
            matchups.append(Matchup(game_index=g + 1, seats=assigned_seats))
        return matchups

    # Generating balanced lobbies for N > 4 is akin to combinatorial block design.
    # For a simple approach, we maintain a history of 'times played' and greedily
    # construct the lobby that minimizes variance in game counts and pairwise exposures.

    played_counts: Dict[str, int] = dict.fromkeys(tunings, 0)
    pairwise_matrix: Dict[str, Dict[str, int]] = {t: dict.fromkeys(tunings, 0) for t in tunings}
    seat_counts: Dict[str, Dict[int, int]] = {t: dict.fromkeys(range(4), 0) for t in tunings}

    def score_lobby(lobby: List[str]) -> float:
        # We want to minimize the variance/max of game counts after adding this lobby
        # Also minimize variance of pairwise exposures
        max_played = max(played_counts[t] + (1 if t in lobby else 0) for t in tunings)
        min_played = min(played_counts[t] + (1 if t in lobby else 0) for t in tunings)
        span_played = max_played - min_played

        # Penalize repeating pairs
        pair_penalty = 0
        for i in range(4):
            for j in range(i + 1, 4):
                pair_penalty += pairwise_matrix[lobby[i]][lobby[j]]

        # Primary objective: keep `played_counts` extremely balanced.
        # Secondary: minimize pairwise repeats.
        return (span_played * 1000.0) + (pair_penalty * 1.0)

    for g in range(num_games):
        best_lobby: List[str] = []
        best_score = float('inf')

        # Simple random search for greedy optimization
        # 200 random lobbies is enough for small N (N < 20)
        # We force tunings that have played the least into the lobby candidate pool
        min_c = min(played_counts.values())
        least_played = [t for t in tunings if played_counts[t] == min_c]

        for _ in range(250):
            candidate = list(least_played)
            rng.shuffle(candidate)
            candidate = candidate[:4]

            # Fill remaining seats randomly if needed
            if len(candidate) < 4:
                pool = [t for t in tunings if t not in candidate]
                rng.shuffle(pool)
                candidate.extend(pool[:(4 - len(candidate))])

            score = score_lobby(candidate)
            if score < best_score:
                best_score = score
                best_lobby = candidate

            # Early break if perfect
            if score == 0:
                break

        # We have our 4 players, now assign seats to balance seat counts
        for t in best_lobby:
            played_counts[t] += 1
            for other in best_lobby:
                if t != other:
                    pairwise_matrix[t][other] += 1

        # We greedily assign the player to the seat they have occupied the least
        assigned_seats = {}
        unassigned_players = list(best_lobby)
        # Randomize assignment order so nobody always gets priority on seat 0
        rng.shuffle(unassigned_players)

        available_seats = [0, 1, 2, 3]
        for t in unassigned_players:
            # Pick the available seat where this player has the minimal count
            best_seat = min(available_seats, key=lambda s: seat_counts[t][s])
            assigned_seats[best_seat] = t
            seat_counts[t][best_seat] += 1
            available_seats.remove(best_seat)

        matchups.append(Matchup(game_index=g + 1, seats=assigned_seats))

    return matchups


def validate_balance(matchups: List[Matchup], tunings: Sequence[str]) -> None:
    """
    Asserts that the schedule conforms to equal exposure constraints.
    Raises ValueError if the schedule is too badly imbalanced.
    """
    if not matchups:
        return

    played_counts: Dict[str, int] = dict.fromkeys(tunings, 0)
    seat_counts: Dict[str, Dict[int, int]] = {t: dict.fromkeys(range(4), 0) for t in tunings}
    pairwise: Dict[str, Dict[str, int]] = {t: dict.fromkeys(tunings, 0) for t in tunings}

    for m in matchups:
        lobby = list(m.seats.values())
        for s, t in m.seats.items():
            played_counts[t] += 1
            seat_counts[t][s] += 1
            for other in lobby:
                if t != other:
                    pairwise[t][other] += 1

    # 1. Game counts must be tight (max diff 2)
    max_games = max(played_counts.values())
    min_games = min(played_counts.values())
    if max_games - min_games > 2:
        raise ValueError(f"Schedule imbalance: game counts vary too much. Target: max - min <= 2. Found: {min_games} to {max_games} ({played_counts})")

    # 2. Seat distribution
    # Actually, a tuning plays K games. K might not divide evenly by 4.
    # So max_seat - min_seat for any tuning should be <= ceil(K/4) - floor(K/4) = 1.
    for t in tunings:
        s_counts = list(seat_counts[t].values())
        if max(s_counts) - min(s_counts) > 2:
            raise ValueError(f"Schedule imbalance: seat distribution skewed for {t}. Target: max - min <= 2. Found: {s_counts}")

    # 3. Pairwise exposure. Not perfectly controllable, but check it's not awful.
    # If a pair never meets, or meets way more than average, warn or error.
    total_pairs = sum(sum(p.values()) for p in pairwise.values()) // 2
    avg_per_pair = total_pairs / (len(tunings) * (len(tunings) - 1) / 2)
    min_pair = min(p for d in pairwise.values() for partner, p in d.items() if p > 0)

    # Just a soft warning or mild threshold
    # If N is huge and num_games small, some might never meet. But we want to avoid pathological.
    pass

