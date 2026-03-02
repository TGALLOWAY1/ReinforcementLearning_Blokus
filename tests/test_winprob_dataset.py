from __future__ import annotations

import pandas as pd

from analytics.winprob.dataset import build_pairwise_dataset, split_pairwise_by_game
from analytics.winprob.features import SNAPSHOT_FEATURE_COLUMNS


def _snapshot_rows_for_game(game_id: str, checkpoint_index: int = 0):
    rows = []
    scores = [10, 8, 6, 4]
    for idx, player_id in enumerate([1, 2, 3, 4]):
        row = {
            "run_id": "run_x",
            "game_id": game_id,
            "checkpoint_index": checkpoint_index,
            "player_id": player_id,
            "agent_name": f"a{player_id}",
            "final_score": scores[idx],
            "phase_board_occupancy": 0.2,
        }
        for feat_idx, feature in enumerate(SNAPSHOT_FEATURE_COLUMNS):
            row[feature] = float((player_id * 10) + feat_idx)
        rows.append(row)
    return rows


def test_pairwise_dataset_has_balanced_directional_rows():
    df = pd.DataFrame(_snapshot_rows_for_game("g1"))
    pairwise_df, metadata = build_pairwise_dataset(df)
    # 4 players -> 6 unordered pairs -> 12 directional rows.
    assert len(pairwise_df) == 12
    assert metadata["rows"] == 12
    assert set(pairwise_df["label"].unique().tolist()) == {0, 1}


def test_split_by_game_preserves_game_separation():
    rows = _snapshot_rows_for_game("g1") + _snapshot_rows_for_game("g2")
    df = pd.DataFrame(rows)
    pairwise_df, _ = build_pairwise_dataset(df)
    train_df, test_df, meta = split_pairwise_by_game(pairwise_df, test_size=0.5, seed=42)
    assert not train_df.empty
    assert not test_df.empty
    assert set(train_df["game_id"].unique()).isdisjoint(set(test_df["game_id"].unique()))
    assert meta["train_games"] == 1
    assert meta["test_games"] == 1

