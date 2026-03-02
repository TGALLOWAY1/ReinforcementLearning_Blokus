from __future__ import annotations

import csv
from pathlib import Path

from analytics.tournament.arena_runner import RunConfig, run_experiment
from analytics.winprob.features import SNAPSHOT_FEATURE_COLUMNS


def _build_random_run_config(output_root: Path, checkpoints: list[int]) -> RunConfig:
    return RunConfig.from_dict(
        {
            "agents": [
                {"name": "r1", "type": "random"},
                {"name": "r2", "type": "random"},
                {"name": "r3", "type": "random"},
                {"name": "r4", "type": "random"},
            ],
            "num_games": 1,
            "seed": 123,
            "seat_policy": "round_robin",
            "output_root": str(output_root),
            "max_turns": 16,
            "snapshots": {
                "enabled": True,
                "strategy": "fixed_ply",
                "checkpoints": checkpoints,
            },
        }
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def test_snapshot_row_count_matches_games_x_checkpoints_x_players(tmp_path: Path):
    run_config = _build_random_run_config(tmp_path, checkpoints=[1, 2])
    result = run_experiment(run_config)
    run_dir = Path(result["run_dir"])
    snapshots_csv = run_dir / "snapshots.csv"
    assert snapshots_csv.exists()
    rows = _read_csv_rows(snapshots_csv)
    # 1 game * 2 checkpoints * 4 players
    assert len(rows) == 8


def test_snapshot_rows_have_feature_columns_and_end_labels(tmp_path: Path):
    run_config = _build_random_run_config(tmp_path, checkpoints=[1])
    result = run_experiment(run_config)
    run_dir = Path(result["run_dir"])
    rows = _read_csv_rows(run_dir / "snapshots.csv")
    assert len(rows) == 4
    sample = rows[0]
    for column in SNAPSHOT_FEATURE_COLUMNS:
        assert column in sample
    assert sample["final_score"] != ""
    assert sample["final_rank"] != ""
    assert sample["label_is_winner"] in {"0", "1"}
    # Leakage guard: labels are stored in dedicated fields, not in feature columns.
    assert "final_score" not in SNAPSHOT_FEATURE_COLUMNS
    assert "label_is_winner" not in SNAPSHOT_FEATURE_COLUMNS

