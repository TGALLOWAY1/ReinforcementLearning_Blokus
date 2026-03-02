# Arena Datasets

This document describes datasets produced by `scripts/arena.py` for analysis and win-probability modeling.

## Files

Per run directory (`arena_runs/<run_id>/`):

- `games.jsonl`: canonical per-game records.
- `snapshots.parquet`: preferred ML dataset format (written when parquet engine is available).
- `snapshots.csv`: always-written snapshot fallback.

## `games.jsonl` Schema (per row)

Core fields:

- `run_id`, `game_id`, `game_index`, `game_seed`
- `seat_assignment`: map `{player_id -> agent_name}`
- `winner_ids`, `winner_agents`, `winner_id`, `is_tie`
- `final_scores`, `final_ranks`
- `agent_scores`, `agent_ranks`
- `moves_made`, `turn_count`, `passes`, `invalid_actions`
- `duration_sec`, `truncated`, `error`
- `agent_move_stats`

## Snapshot Schema (per row, one row per player per checkpoint)

Metadata:

- `run_id`, `game_id`, `game_index`, `game_seed`
- `checkpoint_index`, `checkpoint_ply`
- `ply`, `turn_index`, `current_player_id`
- `player_id`, `player_name`, `agent_name`, `seat_index`

Labels (filled at game end):

- `winner_id`, `winner_ids_json`, `label_is_winner`
- `final_score`, `final_rank`, `final_scores_json`
- `is_tie`

Feature columns:

- `frontier_size`
- `corner_differential`
- `center_proximity`
- `opponent_adjacency`
- `deadzone_count`
- `deadzone_fraction`
- `mobility_total_placements`
- `mobility_total_orientation_normalized`
- `mobility_total_cell_weighted`
- `mobility_bucket_1`
- `mobility_bucket_2`
- `mobility_bucket_3`
- `mobility_bucket_4`
- `mobility_bucket_5`
- `blocking_exposure_legal_moves_opp_sum`
- `utility_frontier_plus_mobility`
- `pieces_used_count`
- `pieces_remaining_count`
- `remaining_squares`
- `remaining_size_1_count`
- `remaining_size_2_count`
- `remaining_size_3_count`
- `remaining_size_4_count`
- `remaining_size_5_count`
- `remaining_key_piece_17`
- `remaining_key_piece_19`
- `remaining_key_piece_20`
- `phase_ply`
- `phase_turn_index`
- `phase_board_occupancy`
- `phase_piece_usage_ratio`
- `phase_progress_turn_ratio`
- `phase_progress_placement_ratio`
- `player_board_occupancy`

## Loading in Python

```python
from pathlib import Path
import pandas as pd

run_dir = Path("arena_runs/<run_id>")
snapshots_path = run_dir / "snapshots.parquet"
if snapshots_path.exists():
    df = pd.read_parquet(snapshots_path)
else:
    df = pd.read_csv(run_dir / "snapshots.csv")

games = pd.read_json(run_dir / "games.jsonl", lines=True)
```

## Pairwise Training Dataset

For modeling, snapshot rows are transformed into pairwise examples:

- group by `(run_id, game_id, checkpoint_index)`
- for each player pair `(i, j)`, compute features as `x = features_i - features_j`
- label `y = 1` if `final_score_i > final_score_j`, else `0`
- reciprocal rows `(j, i)` are added for symmetry
- ties are currently dropped (`tie_policy=drop`)

Reference implementation:

- `analytics/winprob/dataset.py`
- `scripts/train_winprob_v1.py`
- `scripts/train_winprob_v2.py`
