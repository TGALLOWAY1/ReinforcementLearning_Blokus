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
    first = p3.play_games(CHEAP_AGENTS, num_games=24, seed=123, workers=2, out_dir=tmp_path / "a",
                          label="t")
    second = p3.play_games(CHEAP_AGENTS, num_games=24, seed=123, workers=1, out_dir=tmp_path / "b",
                           label="t")
    assert [g["agent_scores"] for g in first] == [g["agent_scores"] for g in second]
    assert (tmp_path / "a" / "games.jsonl").exists()
    seats = {a["name"]: [0, 0, 0, 0] for a in CHEAP_AGENTS}
    for g in first:
        for seat, name in g["seat_assignment"].items():
            seats[name][int(seat) - 1] += 1
    assert all(counts == [6, 6, 6, 6] for counts in seats.values()), seats


def test_report_has_pairwise_stats_and_seat_counts(tmp_path):
    games = p3.play_games(CHEAP_AGENTS, num_games=24, seed=5, workers=1, out_dir=tmp_path, label="t")
    report = p3.analyze(games, CHEAP_AGENTS, label="t", seed=5, stat_seed=1)
    assert report["protocol"] == p3.PROTOCOL_VERSION
    assert report["completed_games"] == 24
    assert "greedy vs random_a" in report["pairwise"]
    assert set(report["pairwise"]["greedy vs random_a"]) >= {"observed_diff", "p_value", "n_games"}
    assert report["seat_counts"]["greedy"] == [6, 6, 6, 6]


def _fake_report(diff, p, n=240):
    return {"pairwise": {"champion vs clone": {"observed_diff": diff, "p_value": p, "n_games": n,
                                               "ci90_low": diff - 1.8, "ci90_high": diff + 1.8}},
            "completed_games": n, "excluded_games": 0}


def test_clone_gate_passes_only_within_noise():
    assert p3.clone_gate(_fake_report(1.2, 0.55), "champion", "clone")["passed"]
    assert not p3.clone_gate(_fake_report(4.0, 0.55), "champion", "clone")["passed"]
    assert not p3.clone_gate(_fake_report(1.0, 0.03), "champion", "clone")["passed"]
    assert not p3.clone_gate(_fake_report(1.0, 0.55, n=200), "champion", "clone")["passed"]


def test_discrimination_gate_requires_every_pair_to_separate():
    rep = {"pairwise": {"a vs b": {"observed_diff": 12.0, "p_value": 0.001, "n_games": 120},
                        "b vs c": {"observed_diff": 5.0, "p_value": 0.20, "n_games": 120}},
           "completed_games": 120, "excluded_games": 0}
    assert p3.discrimination_gate(rep, [("a", "b")])["passed"]
    assert not p3.discrimination_gate(rep, [("a", "b"), ("b", "c")])["passed"]
    rep["excluded_games"] = 1
    assert not p3.discrimination_gate(rep, [("a", "b")])["passed"]
    rep["excluded_games"] = 0
    rep["pairwise"]["a vs b"]["n_games"] = 100
    assert not p3.discrimination_gate(rep, [("a", "b")], min_games=100)["passed"]  # floor is 120


# --- review round 1 (harness-statistics) ------------------------------------
def test_seat_schedule_cycles_all_24_permutations_and_balances_successors():
    names = ["a", "b", "c", "d"]
    seen = set()
    seats = {n: [0, 0, 0, 0] for n in names}
    successors = {}
    for g in range(24):
        assignment = p3.seat_assignment(names, g)
        order = [assignment[str(s)] for s in (1, 2, 3, 4)]
        seen.add(tuple(order))
        for s, n in enumerate(order):
            seats[n][s] += 1
        for x, y in zip(order, order[1:]):
            successors[(x, y)] = successors.get((x, y), 0) + 1
    assert len(seen) == 24
    assert all(v == [6, 6, 6, 6] for v in seats.values())
    assert len(successors) == 12 and set(successors.values()) == {6}
    assert p3.seat_assignment(names, 24) == p3.seat_assignment(names, 0)
    with pytest.raises(ValueError):
        p3.play_games(CHEAP_AGENTS, num_games=20, seed=1, workers=1, out_dir="/tmp/x", label="t")


def _pair(mean, lo, hi, p, n=240):
    return {"observed_diff": mean, "p_value": p, "n_games": n, "ci90_low": lo, "ci90_high": hi}


def test_clone_gate_is_an_equivalence_test_with_a_hard_game_floor():
    ok = {"pairwise": {"champion vs clone": _pair(1.0, -0.8, 2.8, 0.40)}, "excluded_games": 0}
    assert p3.clone_gate(ok, "champion", "clone")["passed"]
    wide = {"pairwise": {"champion vs clone": _pair(2.5, 0.6, 4.4, 0.02)}, "excluded_games": 0}
    assert not p3.clone_gate(wide, "champion", "clone")["passed"]          # CI leaves +/-4 band
    sig = {"pairwise": {"champion vs clone": _pair(2.0, 0.5, 3.5, 0.03)}, "excluded_games": 0}
    assert not p3.clone_gate(sig, "champion", "clone")["passed"]           # bias significant
    few = {"pairwise": {"champion vs clone": _pair(0.5, -1.0, 2.0, 0.6, n=100)}, "excluded_games": 0}
    assert not p3.clone_gate(few, "champion", "clone")["passed"]           # below 240 games
    assert not p3.clone_gate(few, "champion", "clone", min_games=50)["passed"]  # floor cannot be lowered
    errs = {"pairwise": {"champion vs clone": _pair(1.0, -0.8, 2.8, 0.40)}, "excluded_games": 1}
    assert not p3.clone_gate(errs, "champion", "clone")["passed"]          # any errored game fails


def test_worker_failures_become_error_records_and_are_excluded(tmp_path):
    bad = [dict(a) for a in CHEAP_AGENTS]
    bad[1] = {"name": "broken", "type": "no_such_agent_type", "thinking_time_ms": None, "params": {}}
    games = p3.play_games(bad, num_games=24, seed=9, workers=2, out_dir=tmp_path, label="t")
    assert len(games) == 24 and all(g["is_valid_result"] is False for g in games)
    assert all("no_such_agent_type" in (g.get("error") or "") for g in games)
    assert sum(1 for _ in (tmp_path / "games.jsonl").open()) == 24
    report = p3.analyze(games, bad, label="t", seed=9)
    assert report["completed_games"] == 0 and report["excluded_games"] == 24


def test_pairwise_entries_carry_confidence_intervals(tmp_path):
    games = p3.play_games(CHEAP_AGENTS, num_games=24, seed=5, workers=1, out_dir=tmp_path, label="t")
    report = p3.analyze(games, CHEAP_AGENTS, label="t", seed=5)
    t = report["pairwise"]["greedy vs random_a"]
    assert t["ci90_low"] <= t["observed_diff"] <= t["ci90_high"]
    assert "sd_diff" in t and t["n_games"] == 24
    assert report["excluded_games"] == 0 and report["seat_policy"] == "all_permutations"
