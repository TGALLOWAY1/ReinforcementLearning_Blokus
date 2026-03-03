"""Tests for tournament schedule balancing."""

import pytest
from analytics.tournament.scheduler import generate_matchups, validate_balance


def test_scheduler_four_players():
    tunings = ["A", "B", "C", "D"]
    matchups = generate_matchups(tunings, num_games=10, seed=42, seat_policy="randomized")
    assert len(matchups) == 10
    validate_balance(matchups, tunings)


def test_scheduler_six_players():
    tunings = ["T1", "T2", "T3", "T4", "T5", "T6"]
    # 6 players * 4 games each = 24 player-slots = 6 games
    # Let's run 6 games to test perfect divisibility, or 12 games.
    matchups = generate_matchups(tunings, num_games=12, seed=42)
    assert len(matchups) == 12
    validate_balance(matchups, tunings)


def test_scheduler_seven_players_uneven():
    tunings = [f"T{i}" for i in range(7)]
    # 7 players * 4 games each = 28 / 4 = 7 games, so num_games=14 means 8 games each
    matchups = generate_matchups(tunings, num_games=14, seed=1337)
    assert len(matchups) == 14
    validate_balance(matchups, tunings)

