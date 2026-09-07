"""The teacher-dataset validator must refuse records produced under an older
state/action schema (e.g. the pre-2026-09 piece catalogue) with a clear error,
not a traceback and not a pass.
"""

import json

from engine.board import STATE_SCHEMA_VERSION, Board, Player
from engine.move_generator import ACTION_SCHEMA_VERSION, get_shared_generator
from training.experiments import teacher_selfplay as ts


def _one_record(board: Board, player: Player) -> dict:
    gen = get_shared_generator()
    legal = [m.to_dict() for m in gen.get_legal_moves(board, player)]
    return {
        "game_id": "g0000", "player_id": player.value, "state": board.to_dict(),
        "legal_actions": legal, "selected_action": legal[0],
        "search": [{"action": legal[0], "visits": 1, "total_reward": 0.0}],
        "policy_target": [1.0],
        "final_scores": {"1": 10, "2": 9, "3": 8, "4": 7},
        "final_ranks": {"1": 1, "2": 2, "3": 3, "4": 4},
        "record_schema_version": ts.RECORD_SCHEMA_VERSION,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
    }


def _write_dataset(tmp_path, record):
    (tmp_path / "game_0000.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"games_completed": 1, "records_written": 1}))
    return tmp_path


def test_validator_accepts_a_current_schema_record(tmp_path, capsys):
    rec = _one_record(Board(), Player.RED)
    assert ts.validate(_write_dataset(tmp_path, rec)) == 0


def test_validator_rejects_old_schema_records_with_a_clear_error(tmp_path, capsys):
    rec = _one_record(Board(), Player.RED)
    rec["state_schema_version"] = "board_state_v1"
    rec["state"]["schema_version"] = "board_state_v1"
    rec["action_schema_version"] = "move_v1"
    assert ts.validate(_write_dataset(tmp_path, rec)) == 1
    out = capsys.readouterr().out
    assert "schema" in out.lower()
