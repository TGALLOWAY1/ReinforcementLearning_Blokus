# Plan: Build the MCTS Evaluation Function

## Current State

The codebase has strong **infrastructure** already built but is missing the **training pipeline** and **trained models**:

| Component | Status | Location |
|-----------|--------|----------|
| Feature extraction (38 features) | Done | `analytics/winprob/features.py` |
| Pairwise dataset builder | Done | `analytics/winprob/dataset.py` |
| Learned evaluator (inference) | Done | `mcts/learned_evaluator.py` |
| MCTS integration (leaf eval, progressive bias, reward shaping) | Done | `mcts/mcts_agent.py` |
| Arena benchmarking | Done | `scripts/arena_tuning.py` |
| **Data generation at scale** | **Missing** | — |
| **Model training script** | **Missing** | — |
| **Trained model artifact** | **Missing** | — |
| **End-to-end validation** | **Missing** | — |

## Implementation Steps

### Step 1: Self-Play Data Generation Script
**File:** `scripts/generate_training_data.py`

Generate game snapshots with feature vectors by running MCTS-vs-MCTS (or MCTS-vs-Heuristic) games at scale. For each game:
- Play full games using `FastMCTSAgent` or `MCTSAgent`
- At configurable checkpoint intervals (e.g., every 4 plies), extract all 38 features for all 4 players via `extract_player_snapshot_features()`
- Record `final_score` per player after the game ends (needed as labels)
- Save as parquet with columns matching `dataset.py`'s expected schema: `run_id, game_id, checkpoint_index, player_id, agent_name, final_score, phase_board_occupancy, + 38 feature columns`

**CLI interface:**
```
python scripts/generate_training_data.py \
  --num-games 500 \
  --agent-type fast_mcts \
  --thinking-time-ms 100 \
  --checkpoint-interval 4 \
  --output data/snapshots.parquet \
  --seed 42 \
  --workers 4
```

**Key decisions:**
- Use `FastMCTSAgent` with moderate time budgets (50-200ms) for practical generation speed
- Checkpoint every 4 plies to get ~15-20 snapshots per game across early/mid/late phases
- Target 500-1000 games initially (yields ~30K-60K snapshots → ~180K-360K pairwise rows)
- Parallelize with `ProcessPoolExecutor` across games

### Step 2: Model Training Script
**File:** `scripts/train_eval_model.py`

Train a pairwise win-probability model using the existing `dataset.py` pipeline. Two model types, matching `learned_evaluator.py`'s supported formats:

**A) Logistic Regression baseline (`pairwise_logreg`):**
- Load snapshots → `build_pairwise_dataset()` → `split_pairwise_by_game()`
- `sklearn.pipeline.Pipeline` with `StandardScaler` + `LogisticRegression`
- Save artifact via `joblib.dump()` with keys: `model_type, feature_columns, pipeline`

**B) Gradient Boosted Trees with phase models (`pairwise_gbt_phase`):**
- Same data pipeline, but train separate `GradientBoostingClassifier` (or `HistGradientBoostingClassifier` for speed) for each phase bucket (early/mid/late)
- Also train a single fallback model on all data
- Save artifact with keys: `model_type, feature_columns, phase_models, fallback_model`

**CLI interface:**
```
python scripts/train_eval_model.py \
  --data data/snapshots.parquet \
  --model-type pairwise_gbt_phase \
  --output models/eval_v1.pkl \
  --test-size 0.2 \
  --seed 42
```

**Output:** Print train/test accuracy, log-loss, and calibration metrics. Save the `.pkl` artifact.

### Step 3: Validate Model in MCTS
**File:** `scripts/validate_eval_model.py`

Verify the trained model actually improves play:

1. **Inference speed check** — Time `predict_player_win_probability()` per call. Must be <5ms to not dominate the MCTS time budget.
2. **Head-to-head arena** — Run the arena tuning system comparing:
   - `base` (no learned eval, pure rollout MCTS)
   - `leaf_eval` (replace rollout with model prediction)
   - `leaf_eval + progressive_bias` (bias tree search toward model-preferred moves)
3. Report win rates across configurations.

**CLI interface:**
```
python scripts/validate_eval_model.py \
  --model models/eval_v1.pkl \
  --num-games 100 \
  --thinking-time-ms 100
```

This is essentially a thin wrapper around `arena_tuning.py` that registers a new `TuningSet` dynamically with the model path.

### Step 4: Register Model-Based Tuning Sets
**File:** Edit `analytics/tournament/tuning.py`

Update the existing `_LEAF_EVAL_BASE` params and add a new tuning set that points to the real trained model instead of the dummy:

```python
register_tuning_set(TuningSet(
    name="eval_model_v1",
    tunings=[
        MctsTuning("base_rollout", {**_BASE_PARAMS}),
        MctsTuning("leaf_eval_only", {
            **_BASE_PARAMS,
            "leaf_evaluation_enabled": True,
            "learned_model_path": "models/eval_v1.pkl",
        }),
        MctsTuning("leaf_eval_bias_0.25", {
            **_BASE_PARAMS,
            "leaf_evaluation_enabled": True,
            "progressive_bias_enabled": True,
            "progressive_bias_weight": 0.25,
            "learned_model_path": "models/eval_v1.pkl",
        }),
        MctsTuning("leaf_eval_bias_shaping", {
            **_BASE_PARAMS,
            "leaf_evaluation_enabled": True,
            "progressive_bias_enabled": True,
            "progressive_bias_weight": 0.25,
            "potential_shaping_enabled": True,
            "learned_model_path": "models/eval_v1.pkl",
        }),
    ]
))
```

## Execution Order

```
Step 1 → Step 2 → Step 3 → Step 4
  │         │         │
  │         │         └─ validates model improves play
  │         └─ trains .pkl artifact
  └─ generates training data (parquet)
```

Steps 1 and 2 are the core work. Step 3 validates. Step 4 wires it into the existing tournament infrastructure.

## Dependencies

All sklearn-based — no new heavy dependencies needed:
- `scikit-learn` (LogisticRegression, GradientBoostingClassifier, Pipeline, StandardScaler)
- `joblib` (already imported in `learned_evaluator.py`)
- `pandas`, `numpy` (already used)

## Risks / Considerations

- **Data quality:** Random/heuristic agent games may not produce diverse enough board states. Consider mixing agent types (random, heuristic, MCTS at different budgets) for training data.
- **Feature cost:** `extract_player_snapshot_features()` computes legal moves for ALL 4 players (expensive). During MCTS leaf evaluation this runs at every leaf — monitor that inference time stays <5ms.
- **Phase model coverage:** If training data skews toward early/mid game (games where players get stuck), late-game model may underfit. Monitor phase distribution in generated data.
