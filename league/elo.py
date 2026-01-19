"""
Elo rating helpers for Blokus agent league.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EloConfig:
    k_factor: float = 32.0
    draw_score: float = 0.5


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_ratings(rating_a: float, rating_b: float, result_a: float, config: EloConfig) -> tuple[float, float]:
    expected_a = expected_score(rating_a, rating_b)
    expected_b = expected_score(rating_b, rating_a)
    new_a = rating_a + config.k_factor * (result_a - expected_a)
    new_b = rating_b + config.k_factor * ((1.0 - result_a) - expected_b)
    return new_a, new_b
