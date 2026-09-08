"""Game log for the human protocol (M5): one JSON file per game, rewritten
atomically after every event so an abandoned or crashed game still leaves a
readable record. Player types, seats, agent settings, per-move timing and
fallbacks, final scores and the code/engine versions are all recorded."""

import json
import os

import pytest

from webapi.game_log import GAME_LOG_VERSION, GameLogWriter, load_game_log


@pytest.fixture
def writer(tmp_path):
    return GameLogWriter(tmp_path)


PLAYERS = [
    {"player": "RED", "agent_type": "human", "agent_config": {}},
    {"player": "BLUE", "agent_type": "pentobi", "agent_config": {"level": 7, "seed": 1}},
    {"player": "GREEN", "agent_type": "pentobi", "agent_config": {"level": 7, "seed": 2}},
    {"player": "YELLOW", "agent_type": "pentobi", "agent_config": {"level": 7, "seed": 3}},
]


def test_start_writes_a_readable_record_with_provenance(writer, tmp_path):
    path = writer.start("g1", players=PLAYERS, scoring_mode="standard", meta={"profile": "research"})
    assert path == tmp_path / "g1.json"
    rec = load_game_log(path)
    assert rec["log_version"] == GAME_LOG_VERSION
    assert rec["game_id"] == "g1" and rec["status"] == "in_progress"
    assert rec["players"] == PLAYERS
    assert rec["seats"] == {"RED": "human", "BLUE": "pentobi", "GREEN": "pentobi", "YELLOW": "pentobi"}
    assert rec["has_human"] is True and rec["human_seats"] == ["RED"]
    assert rec["scoring_mode"] == "standard"
    assert rec["meta"]["profile"] == "research"
    assert "state_schema_version" in rec["versions"] and "action_schema_version" in rec["versions"]
    assert "repo_commit" in rec["versions"]
    assert rec["moves"] == [] and rec["result"] is None


def test_moves_and_passes_are_appended_in_order(writer):
    writer.start("g2", players=PLAYERS, scoring_mode="standard", meta={})
    writer.record_move("g2", player="RED", is_human=True, move={"piece_id": 21, "orientation": 0,
                       "anchor_row": 0, "anchor_col": 0}, stats={"timeSpentMs": 1234})
    writer.record_move("g2", player="BLUE", is_human=False, move={"piece_id": 1, "orientation": 0,
                       "anchor_row": 0, "anchor_col": 19}, stats={"timeSpentMs": 40, "fallback": "timeout"})
    writer.record_pass("g2", player="GREEN", is_human=False, reason="no_moves")
    rec = load_game_log(writer.path_for("g2"))
    assert [m["player"] for m in rec["moves"]] == ["RED", "BLUE", "GREEN"]
    assert rec["moves"][0]["is_human"] is True and rec["moves"][0]["move"]["piece_id"] == 21
    assert rec["moves"][1]["stats"]["fallback"] == "timeout"
    assert rec["moves"][2] == {"index": 2, "player": "GREEN", "is_human": False, "is_pass": True,
                               "reason": "no_moves", "move": None, "stats": {},
                               "at": rec["moves"][2]["at"]}
    assert rec["fallbacks"] == [{"index": 1, "player": "BLUE", "fallback": "timeout"}]


def test_finish_records_result_and_status(writer):
    writer.start("g3", players=PLAYERS, scoring_mode="standard", meta={})
    writer.finish("g3", scores={"RED": 80, "BLUE": 75, "GREEN": 60, "YELLOW": 60}, winner="RED", status="finished")
    rec = load_game_log(writer.path_for("g3"))
    assert rec["status"] == "finished"
    assert rec["result"]["scores"]["RED"] == 80 and rec["result"]["winner"] == "RED"
    assert rec["result"]["ranks"] == {"RED": 1, "BLUE": 2, "GREEN": 3, "YELLOW": 3}
    assert rec["result"]["human_first_place"] is True
    assert rec["finished_at"] is not None


def test_unknown_game_is_ignored_not_fatal(writer):
    writer.record_pass("nope", player="RED", is_human=True, reason="no_moves")
    writer.finish("nope", scores={}, winner=None, status="finished")
    assert not os.path.exists(writer.path_for("nope"))


def test_file_is_valid_json_after_every_write(writer):
    writer.start("g4", players=PLAYERS, scoring_mode="standard", meta={})
    for i in range(5):
        writer.record_pass("g4", player="RED", is_human=True, reason="no_moves")
        json.loads(writer.path_for("g4").read_text())
    assert not list(writer.path_for("g4").parent.glob("*.tmp"))


def test_ai_timeout_or_error_passes_count_as_fallbacks(writer):
    """The GameManager records an AI turn lost to its outer timeout or an
    exception as a pass; the human protocol must see those as attributable."""
    writer.start("g5", players=PLAYERS, scoring_mode="standard", meta={})
    writer.record_pass("g5", player="BLUE", is_human=False, reason="timeout")
    writer.record_pass("g5", player="GREEN", is_human=False, reason="error")
    writer.record_pass("g5", player="YELLOW", is_human=False, reason="no_moves")
    writer.record_pass("g5", player="RED", is_human=True, reason="no_moves")
    rec = load_game_log(writer.path_for("g5"))
    assert rec["fallbacks"] == [{"index": 0, "player": "BLUE", "fallback": "pass_timeout"},
                                {"index": 1, "player": "GREEN", "fallback": "pass_error"}]


def test_log_file_is_readable_by_other_users(writer):
    path = writer.start("g6", players=PLAYERS, scoring_mode="standard", meta={})
    assert (path.stat().st_mode & 0o777) == 0o644
