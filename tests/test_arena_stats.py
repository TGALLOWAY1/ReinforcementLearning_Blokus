from __future__ import annotations

import math

from analytics.tournament.arena_stats import compute_summary


def _sample_games():
    return [
        {
            "game_id": "g1",
            "error": None,
            "duration_sec": 10.0,
            "seat_assignment": {"1": "a", "2": "b", "3": "c", "4": "d"},
            "winner_agents": ["a"],
            "agent_scores": {"a": 10, "b": 8, "c": 6, "d": 4},
            "agent_move_stats": {
                "a": {
                    "moves": 10,
                    "total_time_ms": 1000,
                    "total_simulations": 2000,
                    "moves_with_simulations": 10,
                },
                "b": {
                    "moves": 10,
                    "total_time_ms": 1200,
                    "total_simulations": 2200,
                    "moves_with_simulations": 10,
                },
                "c": {
                    "moves": 10,
                    "total_time_ms": 1300,
                    "total_simulations": 2100,
                    "moves_with_simulations": 10,
                },
                "d": {
                    "moves": 10,
                    "total_time_ms": 1400,
                    "total_simulations": 2000,
                    "moves_with_simulations": 10,
                },
            },
        },
        {
            "game_id": "g2",
            "error": None,
            "duration_sec": 12.0,
            "seat_assignment": {"1": "b", "2": "a", "3": "d", "4": "c"},
            "winner_agents": ["b"],
            "agent_scores": {"a": 7, "b": 11, "c": 6, "d": 6},
            "agent_move_stats": {
                "a": {
                    "moves": 8,
                    "total_time_ms": 800,
                    "total_simulations": 1600,
                    "moves_with_simulations": 8,
                },
                "b": {
                    "moves": 8,
                    "total_time_ms": 900,
                    "total_simulations": 1700,
                    "moves_with_simulations": 8,
                },
                "c": {
                    "moves": 8,
                    "total_time_ms": 950,
                    "total_simulations": 1500,
                    "moves_with_simulations": 8,
                },
                "d": {
                    "moves": 8,
                    "total_time_ms": 980,
                    "total_simulations": 1550,
                    "moves_with_simulations": 8,
                },
            },
        },
        {
            "game_id": "g3",
            "error": None,
            "duration_sec": 11.0,
            "seat_assignment": {"1": "c", "2": "d", "3": "a", "4": "b"},
            "winner_agents": ["c", "d"],
            "agent_scores": {"a": 5, "b": 6, "c": 9, "d": 9},
            "agent_move_stats": {
                "a": {
                    "moves": 6,
                    "total_time_ms": 600,
                    "total_simulations": 1200,
                    "moves_with_simulations": 6,
                },
                "b": {
                    "moves": 6,
                    "total_time_ms": 720,
                    "total_simulations": 1300,
                    "moves_with_simulations": 6,
                },
                "c": {
                    "moves": 6,
                    "total_time_ms": 750,
                    "total_simulations": 1250,
                    "moves_with_simulations": 6,
                },
                "d": {
                    "moves": 6,
                    "total_time_ms": 780,
                    "total_simulations": 1280,
                    "moves_with_simulations": 6,
                },
            },
        },
    ]


def test_compute_summary_core_counts_and_pairwise():
    games = _sample_games()
    summary = compute_summary(
        games,
        run_id="run_x",
        run_seed=42,
        seat_policy="round_robin",
        agent_names=["a", "b", "c", "d"],
        thinking_time_ms_by_agent={"a": 100, "b": 100, "c": 100, "d": 100},
        run_config={"num_games": 3},
    )

    assert summary["num_games"] == 3
    assert summary["completed_games"] == 3
    assert summary["error_games"] == 0

    total_win_points = sum(v["win_points"] for v in summary["win_stats"].values())
    assert math.isclose(total_win_points, 3.0)

    assert summary["pairwise_total_comparisons"] == 18
    ab = summary["pairwise_matchups"]["a__vs__b"]
    assert ab["a_beats_b"] == 1
    assert ab["b_beats_a"] == 2
    assert ab["tie"] == 0
    assert ab["total"] == 3

    score_mean_a = summary["score_stats"]["a"]["mean"]
    assert score_mean_a is not None
    assert math.isclose(score_mean_a, (10 + 7 + 5) / 3.0)

    eff_a = summary["time_sim_efficiency"]["a"]
    assert math.isclose(eff_a["avg_time_ms_per_move"], 100.0)
    assert math.isclose(eff_a["avg_simulations_per_move"], 200.0)
    assert math.isclose(eff_a["simulations_per_second"], 2000.0)


def test_compute_summary_ignores_error_games_in_wins():
    games = _sample_games()
    games.append(
        {
            "game_id": "g_error",
            "error": "boom",
            "seat_assignment": {"1": "a", "2": "b", "3": "c", "4": "d"},
            "winner_agents": ["a"],
            "agent_scores": {"a": 99, "b": 0, "c": 0, "d": 0},
            "agent_move_stats": {},
        }
    )
    summary = compute_summary(
        games,
        run_id="run_y",
        run_seed=1,
        seat_policy="randomized",
        agent_names=["a", "b", "c", "d"],
        thinking_time_ms_by_agent={"a": 50, "b": 50, "c": 50, "d": 50},
        run_config={"num_games": 4},
    )
    assert summary["num_games"] == 4
    assert summary["completed_games"] == 3
    assert summary["error_games"] == 1
    total_win_points = sum(v["win_points"] for v in summary["win_stats"].values())
    assert math.isclose(total_win_points, 3.0)

