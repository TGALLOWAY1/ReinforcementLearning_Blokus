"""Tests for arena summary statistics with missing agents."""

from analytics.tournament.arena_stats import compute_summary


def test_compute_summary_missing_agents():
    # Simulate a scenario where 6 tunings exist, but each game only has 4.
    agent_names = ["T1", "T2", "T3", "T4", "T5", "T6"]

    # Game 1: T1, T2, T3, T4. T1 wins.
    # Game 2: T3, T4, T5, T6. T5 wins.
    # T1 played 1, T2 played 1, T3 played 2, T4 played 2, T5 played 1, T6 played 1

    games = [
        {
            "game_id": "g1",
            "seat_assignment": {"1": "T1", "2": "T2", "3": "T3", "4": "T4"},
            "winner_agents": ["T1"],
            "agent_scores": {"T1": 100, "T2": 80, "T3": 60, "T4": 40},
            "agent_move_stats": {
                "T1": {"moves": 20, "total_time_ms": 1000},
                "T2": {"moves": 20, "total_time_ms": 1000},
                "T3": {"moves": 20, "total_time_ms": 1000},
                "T4": {"moves": 20, "total_time_ms": 1000},
            },
            "duration_sec": 4.0
        },
        {
            "game_id": "g2",
            "seat_assignment": {"1": "T3", "2": "T4", "3": "T5", "4": "T6"},
            "winner_agents": ["T5"],
            "agent_scores": {"T3": 50, "T4": 50, "T5": 110, "T6": 20},
            "agent_move_stats": {
                "T3": {"moves": 25, "total_time_ms": 1250},
                "T4": {"moves": 25, "total_time_ms": 1250},
                "T5": {"moves": 25, "total_time_ms": 1250},
                "T6": {"moves": 25, "total_time_ms": 1250},
            },
            "duration_sec": 5.0
        }
    ]

    thinking_times = dict.fromkeys(agent_names, 50)

    summary = compute_summary(
        games=games,
        run_id="test_missing_agents",
        run_seed=0,
        seat_policy="round_robin",
        agent_names=agent_names,
        thinking_time_ms_by_agent=thinking_times,
        run_config={}
    )

    win_stats = summary["win_stats"]

    # Assert games played counts are perfectly captured despite missingness
    assert win_stats["T1"]["games_played"] == 1.0
    assert win_stats["T3"]["games_played"] == 2.0
    assert win_stats["T5"]["games_played"] == 1.0

    # Assert win rates
    assert win_stats["T1"]["win_rate"] == 1.0
    assert win_stats["T3"]["win_rate"] == 0.0
    assert win_stats["T5"]["win_rate"] == 1.0

    # Assert efficiency stats avoid division by zero for T1 in game 2
    eff = summary["time_sim_efficiency"]
    assert eff["T1"]["moves"] == 20
    assert eff["T3"]["moves"] == 45  # 20 + 25


