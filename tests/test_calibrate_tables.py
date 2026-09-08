"""The M1 gate tables must be iteration-pinned, single-worker, and built from
the recorded champion / serving / D-016 configurations (regression pin)."""

from mcts_lab import calibrate as cal


def _pinned(agent):
    p = agent["params"]
    return (agent["thinking_time_ms"] is None and p["deterministic_time_budget"] is False
            and p["iterations"] == cal.SERVING_ITERATIONS and p["num_workers"] == 1)


def test_clone_table_is_champion_twice_plus_fixed_anchors():
    table = cal.clone_table()
    assert [a["name"] for a in table] == ["champion", "champion_clone", "greedy", "random"]
    assert table[0]["params"] == table[1]["params"]
    assert _pinned(table[0]) and _pinned(table[1])
    assert table[0]["params"]["rollout_policy"] == "greedy_sample"


def test_discrimination_table_pins_all_three_search_configs_at_serving_budget():
    table = cal.discrimination_table()
    assert [a["name"] for a in table] == ["gen140", "serving_v2", "d016_250", "greedy"]
    assert all(_pinned(a) for a in table if a["type"] == "mcts")
    assert table[1]["params"]["rollout_policy"] == "random"          # served registry v2
    assert table[2]["params"]["progressive_widening_enabled"] is True  # D-016
    assert table[2]["params"]["value_model_path"].endswith("value_v2_v2_mixed.joblib")


def test_pentobi_table_pins_search_configs_and_levels():
    table = cal.pentobi_table()
    assert [a["name"] for a in table] == ["gen140", "d016_250", "pentobi_l3", "pentobi_l7"]
    assert all(_pinned(a) for a in table if a["type"] == "mcts")
    assert [a["params"]["level"] for a in table if a["type"] == "pentobi"] == [3, 7]
    assert all(a["params"]["use_book"] is False for a in table if a["type"] == "pentobi")
