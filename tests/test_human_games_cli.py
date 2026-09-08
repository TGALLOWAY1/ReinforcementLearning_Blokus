"""M5 measurement: summarise the logged human games (webapi/game_log.py) per
Pentobi level with the pre-registered gate, scored from the AGENT's side:
Pentobi takes first place in >= 70% of 20 seat-rotated games (a human tie for
first counts against the agent) and no agent loss is attributable to a
fallback, timeout or error."""

import json

from mcts_lab.human_games import GATE_GAMES, GATE_FIRST_PLACE, summarize_dir, wilson_interval
from webapi.game_log import GameLogWriter

AI = {"level": 7, "seed": 1, "time_budget_ms": 10000}


def _players(human_seat):
    out = []
    for colour in ("RED", "BLUE", "GREEN", "YELLOW"):
        if colour == human_seat:
            out.append({"player": colour, "agent_type": "human", "agent_config": {}})
        else:
            out.append({"player": colour, "agent_type": "pentobi", "agent_config": dict(AI)})
    return out


def _game(writer, game_id, human_seat, scores, *, fallback=False, finish=True, level=7, ai_pass_reason=None):
    ai = dict(AI, level=level)
    players = [dict(p, agent_config=dict(ai) if p["agent_type"] == "pentobi" else {}) for p in _players(human_seat)]
    writer.start(game_id, players=players, scoring_mode="standard", meta={"app_profile": "research"})
    writer.record_move(game_id, player=human_seat, is_human=True,
                       move={"piece_id": 1, "orientation": 0, "anchor_row": 0, "anchor_col": 0},
                       stats={"timeSpentMs": 5000})
    ai_seat = next(p["player"] for p in players if p["agent_type"] == "pentobi")
    writer.record_move(game_id, player=ai_seat, is_human=False,
                       move={"piece_id": 1, "orientation": 0, "anchor_row": 5, "anchor_col": 5},
                       stats={"timeSpentMs": 1200, "level": level, "fallback": "timeout" if fallback else None})
    if ai_pass_reason:
        writer.record_pass(game_id, player=ai_seat, is_human=False, reason=ai_pass_reason)
    if finish:
        winner = max(scores, key=scores.get)
        writer.finish(game_id, scores=scores, winner=winner, status="finished")


def test_wilson_interval_is_sane():
    lo, hi = wilson_interval(14, 20)  # 90% Wilson for 14/20: (0.516, 0.836)
    assert 0.50 < lo < 0.53 and 0.82 < hi < 0.85
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_summary_counts_completed_human_games_per_level(tmp_path):
    writer = GameLogWriter(tmp_path)
    _game(writer, "a", "RED", {"RED": 80, "BLUE": 70, "GREEN": 60, "YELLOW": 50})
    _game(writer, "b", "BLUE", {"RED": 80, "BLUE": 70, "GREEN": 60, "YELLOW": 50})
    _game(writer, "c", "GREEN", {"RED": 60, "BLUE": 60, "GREEN": 60, "YELLOW": 50})   # tie for first
    _game(writer, "d", "YELLOW", {"RED": 80, "BLUE": 70, "GREEN": 60, "YELLOW": 50}, finish=False)  # abandoned
    _game(writer, "e", "RED", {"RED": 90, "BLUE": 70, "GREEN": 60, "YELLOW": 50}, level=3)
    summary = summarize_dir(tmp_path)
    assert summary["games_found"] == 5
    assert summary["completed_human_games"] == 4
    assert summary["unfinished_human_games"] == ["d"]
    by_level = {row["level"]: row for row in summary["levels"]}
    assert by_level[7]["games"] == 3
    assert by_level[7]["human_first_place"] == 2          # a (outright) and c (tied first)
    assert by_level[7]["agent_first_place"] == 1          # only b: the human was not first
    assert by_level[7]["agent_first_place_rate"] == round(1 / 3, 4)
    assert by_level[7]["human_outright_wins"] == 1
    assert by_level[7]["seats"] == {"RED": 1, "BLUE": 1, "GREEN": 1, "YELLOW": 0}
    assert by_level[7]["mean_human_score"] == round((80 + 70 + 60) / 3, 1)
    assert by_level[7]["mean_human_rank"] == round((1 + 2 + 1) / 3, 2)
    assert by_level[7]["fallback_moves"] == 0
    assert by_level[3]["games"] == 1 and by_level[3]["agent_first_place"] == 0


def test_human_winning_most_games_fails_the_gate(tmp_path):
    writer = GameLogWriter(tmp_path)
    for i in range(20):
        human = ["RED", "BLUE", "GREEN", "YELLOW"][i % 4]
        scores = {"RED": 60, "BLUE": 60, "GREEN": 60, "YELLOW": 60}
        scores[human] = 90 if i < 14 else 30   # human first in 14 of 20
        _game(writer, f"g{i}", human, scores)
    row = summarize_dir(tmp_path)["levels"][0]
    assert row["games"] == GATE_GAMES and row["human_first_place"] == 14
    assert row["agent_first_place"] == 6 and row["agent_first_place_rate"] == 0.3
    assert row["gate"]["first_place_ok"] is False and row["gate"]["pass"] is False
    assert GATE_FIRST_PLACE == 0.7


def test_agent_loss_with_a_fallback_or_timeout_fails_the_gate(tmp_path):
    writer = GameLogWriter(tmp_path)
    for i in range(20):
        human = ["RED", "BLUE", "GREEN", "YELLOW"][i % 4]
        scores = {"RED": 60, "BLUE": 60, "GREEN": 60, "YELLOW": 60}
        scores[human] = 30 if i < 15 else 90   # agent first in 15 of 20, human first in the last 5
        _game(writer, f"g{i}", human, scores, fallback=(i == 0),                     # agent won: harmless
              ai_pass_reason="timeout" if i == 19 else None)                       # agent lost after a timeout
    row = summarize_dir(tmp_path)["levels"][0]
    assert row["agent_first_place_rate"] == 0.75 and row["gate"]["first_place_ok"] is True
    assert row["fallback_moves"] == 2
    assert row["attributable_losses"] == 1
    assert row["gate"]["fallback_ok"] is False and row["gate"]["pass"] is False


def test_gate_passes_with_clean_record(tmp_path):
    writer = GameLogWriter(tmp_path)
    for i in range(20):
        human = ["RED", "BLUE", "GREEN", "YELLOW"][i % 4]
        scores = {"RED": 60, "BLUE": 60, "GREEN": 60, "YELLOW": 60}
        scores[human] = 30 if i < 14 else 90   # agent first in 14 of 20
        _game(writer, f"g{i}", human, scores, fallback=(i == 0))
    row = summarize_dir(tmp_path)["levels"][0]
    assert row["agent_first_place_rate"] == 0.7 and row["attributable_losses"] == 0
    assert row["gate"]["pass"] is True
    assert row["agent_first_place_ci90"][0] < 0.7 < row["agent_first_place_ci90"][1]


def test_cli_prints_a_table_and_writes_json(tmp_path, capsys):
    from mcts_lab.human_games import main

    writer = GameLogWriter(tmp_path / "logs")
    _game(writer, "a", "RED", {"RED": 80, "BLUE": 70, "GREEN": 60, "YELLOW": 50})
    out = tmp_path / "summary.json"
    assert main(["--dir", str(tmp_path / "logs"), "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "level 7" in printed and "Pentobi first place 0/1" in printed
    assert json.loads(out.read_text())["completed_human_games"] == 1
