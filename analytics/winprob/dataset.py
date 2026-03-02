"""Dataset loading and pairwise transformation for win-probability modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from analytics.winprob.features import SNAPSHOT_FEATURE_COLUMNS


def resolve_snapshot_path(path: str) -> Path:
    """Resolve snapshots dataset path from file or run directory."""
    input_path = Path(path)
    if input_path.is_file():
        return input_path
    if input_path.is_dir():
        parquet_path = input_path / "snapshots.parquet"
        if parquet_path.exists():
            return parquet_path
        csv_path = input_path / "snapshots.csv"
        if csv_path.exists():
            return csv_path
    raise FileNotFoundError(
        f"Could not resolve snapshots dataset from '{path}'. "
        "Expected snapshots.parquet or snapshots.csv."
    )


def load_snapshots_dataframe(path: str) -> pd.DataFrame:
    """Load snapshot dataset from parquet or csv."""
    resolved = resolve_snapshot_path(path)
    if resolved.suffix.lower() == ".parquet":
        return pd.read_parquet(resolved)
    if resolved.suffix.lower() == ".csv":
        return pd.read_csv(resolved)
    raise ValueError(f"Unsupported snapshot dataset extension: {resolved.suffix}")


def phase_bucket_from_occupancy(
    board_occupancy: float,
    *,
    early_max: float = 0.33,
    mid_max: float = 0.66,
) -> str:
    if board_occupancy < early_max:
        return "early"
    if board_occupancy < mid_max:
        return "mid"
    return "late"


def build_pairwise_dataset(
    snapshots_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = SNAPSHOT_FEATURE_COLUMNS,
    tie_policy: str = "drop",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build pairwise rows from per-player snapshots.

    For each snapshot checkpoint in each game, creates pairwise rows:
    - x = features_i - features_j
    - y = 1 if final_score_i > final_score_j else 0
    Includes reciprocal rows (j-i, 1-y) to balance the dataset.
    """
    missing = [column for column in feature_columns if column not in snapshots_df.columns]
    if missing:
        raise ValueError(f"Snapshot dataset missing feature columns: {missing}")

    required = [
        "run_id",
        "game_id",
        "checkpoint_index",
        "player_id",
        "agent_name",
        "final_score",
        "phase_board_occupancy",
    ]
    missing_required = [column for column in required if column not in snapshots_df.columns]
    if missing_required:
        raise ValueError(f"Snapshot dataset missing required columns: {missing_required}")

    rows: List[Dict[str, Any]] = []
    ties_dropped = 0
    grouped = snapshots_df.groupby(["run_id", "game_id", "checkpoint_index"], sort=True)
    for (run_id, game_id, checkpoint_index), group in grouped:
        players = group.sort_values("player_id")
        if len(players) < 2:
            continue
        phase_occupancy = float(players["phase_board_occupancy"].mean())
        phase_bucket = phase_bucket_from_occupancy(phase_occupancy)
        values = players[list(feature_columns)].astype(float).to_numpy(dtype=float)
        player_ids = players["player_id"].astype(int).to_numpy()
        agent_names = players["agent_name"].astype(str).to_numpy()
        final_scores = players["final_score"].astype(float).to_numpy()

        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                score_i = final_scores[i]
                score_j = final_scores[j]
                if score_i == score_j:
                    if tie_policy == "drop":
                        ties_dropped += 1
                        continue
                    raise ValueError(
                        f"Unsupported tie_policy '{tie_policy}'. Expected 'drop'."
                    )
                diff_ij = values[i] - values[j]
                y_ij = 1 if score_i > score_j else 0
                base = {
                    "run_id": run_id,
                    "game_id": game_id,
                    "checkpoint_index": int(checkpoint_index),
                    "phase_board_occupancy": phase_occupancy,
                    "phase_bucket": phase_bucket,
                    "player_i_id": int(player_ids[i]),
                    "player_j_id": int(player_ids[j]),
                    "agent_i": agent_names[i],
                    "agent_j": agent_names[j],
                }
                row_ij = dict(base)
                for feature_idx, feature_name in enumerate(feature_columns):
                    row_ij[feature_name] = float(diff_ij[feature_idx])
                row_ij["label"] = int(y_ij)
                rows.append(row_ij)

                row_ji = dict(base)
                row_ji["player_i_id"] = int(player_ids[j])
                row_ji["player_j_id"] = int(player_ids[i])
                row_ji["agent_i"] = agent_names[j]
                row_ji["agent_j"] = agent_names[i]
                for feature_idx, feature_name in enumerate(feature_columns):
                    row_ji[feature_name] = float(-diff_ij[feature_idx])
                row_ji["label"] = int(1 - y_ij)
                rows.append(row_ji)

    pairwise_df = pd.DataFrame(rows)
    metadata = {
        "rows": int(len(pairwise_df)),
        "games": int(pairwise_df["game_id"].nunique()) if not pairwise_df.empty else 0,
        "checkpoints": int(pairwise_df["checkpoint_index"].nunique())
        if not pairwise_df.empty
        else 0,
        "ties_dropped": int(ties_dropped),
        "tie_policy": tie_policy,
    }
    return pairwise_df, metadata


def split_pairwise_by_game(
    pairwise_df: pd.DataFrame,
    *,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Split pairwise dataset by game_id to prevent leakage."""
    if pairwise_df.empty:
        raise ValueError("Pairwise dataset is empty.")
    game_ids = sorted(pairwise_df["game_id"].unique().tolist())
    rng = np.random.RandomState(seed)
    shuffled = list(game_ids)
    rng.shuffle(shuffled)
    split_index = int(round(len(shuffled) * (1.0 - test_size)))
    split_index = max(1, min(split_index, len(shuffled) - 1))
    train_games = set(shuffled[:split_index])
    test_games = set(shuffled[split_index:])
    train_df = pairwise_df[pairwise_df["game_id"].isin(train_games)].copy()
    test_df = pairwise_df[pairwise_df["game_id"].isin(test_games)].copy()
    metadata = {
        "train_games": len(train_games),
        "test_games": len(test_games),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }
    return train_df, test_df, metadata

