"""Protocol v3: fresh run seeds, round-robin seats, parallel iteration-pinned
games, paired statistics, and the two M1 gates (clone calibration and
discrimination)."""

import json

import pytest

from training.evaluation import protocol_v3 as p3

CHEAP_AGENTS = [
    {"name": "greedy", "type": "greedy", "thinking_time_ms": None, "params": {}},
    {"name": "heuristic", "type": "heuristic", "thinking_time_ms": None, "params": {}},
    {"name": "random_a", "type": "random", "thinking_time_ms": None, "params": {}},
    {"name": "random_b", "type": "random", "thinking_time_ms": None, "params": {}},
]


def test_run_seed_is_fresh_per_run_but_reproducible():
    a = p3.derive_run_seed("clone", "2026-09-07T10:00:00")
    b = p3.derive_run_seed("clone", "2026-09-07T10:00:01")
    c = p3.derive_run_seed("clone", "2026-09-07T10:00:00")
    assert a != b and a == c
    assert 0 < a < 2**31


def test_parallel_games_are_reproducible_and_seat_balanced(tmp_path):
    first = p3.play_games(CHEAP_AGENTS, num_games=4, seed=123, workers=2, out_dir=tmp_path / "a",
                          label="t")
    second = p3.play_games(CHEAP_AGENTS, num_games=4, seed=123, workers=2, out_dir=tmp_path / "b",
                           label="t")
    assert [g["agent_scores"] for g in first] == [g["agent_scores"] for g in second]
    assert (tmp_path / "a" / "games.jsonl").exists()
    seats = {a["name"]: [0, 0, 0, 0] for a in CHEAP_AGENTS}
    for g in first:
        for seat, name in g["seat_assignment"].items():
            seats[name][int(seat) - 1] += 1
    assert all(counts == [1, 1, 1, 1] for counts in seats.values()), seats


def test_report_has_pairwise_stats_and_seat_counts(tmp_path):
    games = p3.play_games(CHEAP_AGENTS, num_games=4, seed=5, workers=1, out_dir=tmp_path, label="t")
    report = p3.analyze(games, CHEAP_AGENTS, label="t", seed=5, stat_seed=1)
    assert report["protocol"] == p3.PROTOCOL_VERSION
    assert report["completed_games"] == 4
    assert "greedy vs random_a" in report["pairwise"]
    assert set(report["pairwise"]["greedy vs random_a"]) >= {"observed_diff", "p_value", "n_games"}
    assert report["seat_counts"]["greedy"] == [1, 1, 1, 1]


def _fake_report(diff, p):
    return {"pairwise": {"champion vs clone": {"observed_diff": diff, "p_value": p, "n_games": 100}},
            "completed_games": 100}


def test_clone_gate_passes_only_within_noise():
    assert p3.clone_gate(_fake_report(1.2, 0.55), "champion", "clone", min_games=100)["passed"]
    assert not p3.clone_gate(_fake_report(4.0, 0.55), "champion", "clone", min_games=100)["passed"]
    assert not p3.clone_gate(_fake_report(1.0, 0.10), "champion", "clone", min_games=100)["passed"]
    assert not p3.clone_gate(_fake_report(1.0, 0.55), "champion", "clone", min_games=101)["passed"]


def test_discrimination_gate_requires_every_pair_to_separate():
    rep = {"pairwise": {"a vs b": {"observed_diff": 12.0, "p_value": 0.001, "n_games": 100},
                        "b vs c": {"observed_diff": 5.0, "p_value": 0.20, "n_games": 100}},
           "completed_games": 100}
    assert p3.discrimination_gate(rep, [("a", "b")], min_games=100)["passed"]
    assert not p3.discrimination_gate(rep, [("a", "b"), ("b", "c")], min_games=100)["passed"]
