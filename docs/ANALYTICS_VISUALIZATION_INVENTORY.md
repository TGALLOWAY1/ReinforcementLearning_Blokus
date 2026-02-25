# Blokus RL: Analytics & Visualization Codebase Inventory

**Purpose:** Detailed inventory of legal move logic, analytics history, charts/visualizations, what-if tooling, training export, and MCTS explainability. No new features—analysis only.

---

## A) Architecture Map (High Level)

### Major Subsystems

| Subsystem | File Paths | Key Exports | Responsibility | Data Flow |
|-----------|------------|-------------|----------------|-----------|
| **Game Engine** | `engine/game.py`, `engine/board.py`, `engine/move_generator.py`, `engine/pieces.py`, `engine/bitboard.py` | `BlokusGame`, `Board`, `LegalMoveGenerator`, `Move`, `PiecePlacement`, `PieceGenerator` | Core Blokus rules, move generation, placement validation | Engine is source of truth for legality; webapi calls `game.get_legal_moves()` |
| **State Store (Frontend)** | `frontend/src/store/gameStore.ts` | `useGameStore`, `setGameState`, `legalMovesHistory`, `useLegalMoves`, `useLegalMovesByPiece` | Client-side game state, WebSocket sync, mobility history | WebSocket receives `game_state` → `setGameState` → computes mobility metrics → appends to `legalMovesHistory` |
| **UI Components** | `frontend/src/components/*.tsx`, `frontend/src/pages/*.tsx` | `Board`, `RightPanel`, `LegalMovesBarChart`, `LegalMovesPerTurnPlot`, `MobilityBucketsChart`, `PolicyView`, `ValueView`, `PieceTray` | Rendering, user interaction | Components read from `gameStore`; charts use `legalMovesHistory` |
| **Analytics (Backend)** | `analytics/logging/logger.py`, `analytics/metrics/*.py`, `analytics/aggregate/*.py` | `StrategyLogger`, `StepLog`, `GameResultLog`, `compute_*_metrics` | Per-step JSONL logging, metric computation | **Not wired to webapi**—only used by `scripts/generate_analytics_data.py` |
| **Agents** | `agents/fast_mcts_agent.py`, `agents/registry.py`, `mcts/mcts_agent.py` | `FastMCTSAgent`, `MCTSAgent` | Move selection, MCTS search | `webapi` calls `agent.choose_move()` or `agent.select_action()` with `legal_moves` from engine |
| **WebAPI** | `webapi/app.py`, `webapi/game_manager.py` | Routes, `GameManager`, `_get_game_state`, `_broadcast_game_state` | HTTP/WS, game state serialization, heatmap | `_get_game_state` builds `GameState` with `legal_moves`, `heatmap`; broadcasts to frontend |

### Data Flow Summary

```
Engine (get_legal_moves) → webapi _get_game_state → GameState JSON + heatmap
                                                    ↓
WebSocket broadcast → frontend gameStore.setGameState → legalMovesHistory append
                                                    ↓
Charts (LegalMovesPerTurnPlot, MobilityBucketsChart) read legalMovesHistory
```

---

## B) Source of Truth for Legality

### Canonical Implementation

| Responsibility | File | Function/Class | Notes |
|----------------|------|----------------|-------|
| **Generate legal moves** | `engine/move_generator.py` | `LegalMoveGenerator.get_legal_moves(board, player)` | Single public API; delegates to `_get_legal_moves_frontier` (default) or `_get_legal_moves_naive` |
| **Validate proposed move** | `engine/move_generator.py` | `LegalMoveGenerator.is_move_legal(board, player, move)` | Used by `BlokusGame.make_move()` |
| **Placement validation** | `engine/board.py` | `Board.can_place_piece(piece_positions, player)` | Checks overlap, adjacency, corner connection |
| **Bounds + orientation** | `engine/pieces.py` | `PiecePlacement.can_place_piece_at(board_shape, piece_shape, row, col)` | Per-cell bounds check |

### Orientation Enumeration & Symmetry Dedupe

- **File:** `engine/pieces.py`
- **Function:** `PieceGenerator.get_piece_rotations_and_reflections(piece)` → used by `generate_orientations_for_piece()`
- **Logic:** 8 variants (4 rotations + 4 reflected rotations); deduplicated by `normalize_offsets` + `seen_offsets` key (tuple of normalized offsets)
- **Cache:** `LegalMoveGenerator.piece_orientations_cache` and `piece_position_cache` precompute all orientations at init

### Legal Move Return Shape

- **Engine:** `List[Move]` where `Move` = `(piece_id, orientation, anchor_row, anchor_col)`
- **API:** Converted to `List[dict]` with keys `piece_id`, `orientation`, `anchor_row`, `anchor_col` (and `positions` in some payloads)
- **Frontend:** `LegalMove[]` from `gameState.legal_moves` with `piece_id`, `orientation`, `anchor_row`, `anchor_col`, `positions`

---

## C) Analytics Storage + Lifecycle

### 1. Frontend: `legalMovesHistory` (gameStore)

| Field | Description |
|-------|-------------|
| **File** | `frontend/src/store/gameStore.ts` |
| **Storage** | `legalMovesHistory: LegalMovesHistoryEntry[]` |
| **Schema** | `{ turn: number; byPlayer: Record<string, PlayerMobilityMetrics> }` |
| **PlayerMobilityMetrics** | `totalPlacements`, `totalOrientationNormalized`, `totalCellWeighted`, `buckets: Record<1|2|3|4|5, number>` |

**Hook points:**
- Updated in `setGameState` when `gameState` changes and turn advances (`move_count` or `current_player` changes)
- Trigger: WebSocket `game_state` message → `setGameState(gameState)`

**Reset:**
- New game: `prev?.game_id !== gameState.game_id` → `legalMovesHistory: [entry]`
- Game cleared: `gameState: null` → `legalMovesHistory: []`

**Compute:** `computeMobilityMetrics(legal_moves, pieces_used)` in `frontend/src/utils/mobilityMetrics.ts`

**Data per entry:** Only current player’s metrics for that turn (not all players). `byPlayer` is `{ [current_player]: metrics }`.

---

### 2. Backend: `StrategyLogger` (analytics/logging)

| Field | Description |
|-------|-------------|
| **File** | `analytics/logging/logger.py` |
| **Storage** | `logs/analytics/steps.jsonl`, `results.jsonl` (or custom `log_dir`) |
| **Schema** | `StepLog` (game_id, timestamp, seed, turn_index, player_id, action, legal_moves_before, legal_moves_after, pieces_remaining, metrics) |

**Hook points:**
- `on_reset(game_id, seed, agent_ids, config)` — called at game start
- `on_step(game_id, turn_index, player_id, state_before, move, next_state)` — called after each move
- `on_game_end(game_id, final_scores, winner_id, num_turns)` — called at game end

**Wiring:** **Not wired to webapi or game loop.** Only used by `scripts/generate_analytics_data.py`, which runs standalone games with `StrategyLogger`.

**Reset:** New `StrategyLogger` instance per run; `log_dir` can be cleared or overwritten.

**Metrics computed:** `compute_center_metrics`, `compute_territory_metrics`, `compute_mobility_metrics`, `compute_blocking_metrics`, `compute_corner_metrics`, `compute_proximity_metrics`, `compute_piece_metrics` (see `analytics/metrics/`).

---

### 3. Engine: `game_history`

| Field | Description |
|-------|-------------|
| **File** | `engine/game.py` |
| **Storage** | `BlokusGame.game_history: List[dict]` — in-memory only |
| **Schema** | `{ move, player, board_state: None }` (board_state not copied for performance) |

**Hook:** Appended in `make_move()` on success. **Reset:** `reset()` clears `game_history`.

**Not persisted** — used only for replay/debug within a single process.

---

### 4. Training: `train_metrics.jsonl`

| Field | Description |
|-------|-------------|
| **File** | `rl/train.py` |
| **Path** | `{log_dir}/train_metrics.jsonl` |
| **Schema** | `{ step, elapsed_sec, steps_per_sec, games_per_sec_est, games_est, eval: {wins, losses, draws, elo}, league_size? }` |

**Hook:** Written at each evaluation interval.

---

## D) Inventory of Existing Visualizations

| Component | File | UI Location | What It Plots | Axes / Units | Update Trigger | Limitations |
|-----------|------|-------------|---------------|--------------|----------------|-------------|
| **LegalMovesBarChart** | `frontend/src/components/LegalMovesBarChart.tsx` | RightPanel (deploy), Research (RightPanel) | Legal moves per piece (1–21) | X: piece IDs; Y: count | `gameState.legal_moves` via `useLegalMovesByPiece` | Only current player; only available pieces |
| **LegalMovesPerTurnPlot** | `frontend/src/components/LegalMovesPerTurnPlot.tsx` | RightPanel (deploy + research) | Cell-weighted mobility per player over turns | X: turn; Y: totalCellWeighted (MW) | `legalMovesHistory` | One entry per turn; only current player per turn; carry-forward for missing players |
| **MobilityBucketsChart** | `frontend/src/components/MobilityBucketsChart.tsx` | RightPanel (deploy + research) | 5 line charts (mono→pento) with normalized Y | X: turn; Y: 0–100% of max for that size | `legalMovesHistory` | One entry per turn; carry-forward when not player’s turn |
| **PolicyView** | `frontend/src/components/AgentVisualizations.tsx` | RightPanel (deploy + research) | 20×20 heatmap | Binary: 1.0 = legal cell, 0.0 = illegal | `gameState.heatmap` | Falls back to random if no heatmap; binary only |
| **ValueView** | `frontend/src/components/AgentVisualizations.tsx` | RightPanel (research) | Win probability bar chart | Placeholder/demo | Static/dummy | Not real value function |
| **TrainingRunDetail** | `frontend/src/pages/TrainingRunDetail.tsx` | `/runs/:runId` | Episode reward line chart | X: episode; Y: reward | `run.metrics` from API | Training runs only |
| **ResearchSidebar** | `frontend/src/components/ResearchSidebar.tsx` | Research layout | Sparkline (reward trend) | Dummy data | Static | Not wired to real metrics |

**Normalization:**
- LegalMovesPerTurnPlot: raw `totalCellWeighted` (MW)
- MobilityBucketsChart: Y normalized per size bucket (0–100% of max for that size)
- PolicyView: binary (1.0/0.0)

---

## E) What-If / Move Impact Tooling

**No explicit what-if UI.** The following are related but not user-facing:

1. **Analytics `StepLog` metrics:** `mobility_me_before`, `mobility_me_after`, `mobility_me_delta`, etc. in `steps.jsonl` — computed by `StrategyLogger.on_step()` after state transition. Not exposed in UI.

2. **MCTS simulation:** `FastMCTSAgent` and `MCTSAgent` simulate moves internally (no board copy in fast path; uses move simulation). No per-move comparison or Δ display.

3. **Fast MCTS heuristic:** `_quick_move_evaluation` sorts by piece size and center distance; no explicit Δ mobility or comparison UI.

**Conclusion:** No “what-if” or counterfactual move evaluation in the UI. Analytics `steps.jsonl` has before/after metrics but is not wired to the game or UI.

---

## F) Training / Export / Replay

| System | Location | Format | Purpose |
|--------|----------|--------|---------|
| **StrategyLogger steps** | `analytics/logging/logger.py` → `logs/analytics/steps.jsonl` | JSONL | Per-step analytics (not wired to webapi) |
| **StrategyLogger results** | `analytics/logging/logger.py` → `logs/analytics/results.jsonl` | JSONL | Per-game results |
| **Aggregate pipeline** | `analytics/aggregate/aggregate_games.py`, `aggregate_agents.py` | Parquet | `game_summary.parquet`, `agent_summary.parquet` |
| **Training metrics** | `rl/train.py` → `{log_dir}/train_metrics.jsonl` | JSONL | Per-eval step, elapsed, wins/losses/draws, Elo |
| **League registry** | `league/pdl.py` → `league_registry.jsonl` | JSONL | Checkpoint registry for Stage 3 |
| **Replay API** | `webapi/app.py` → `_replay_to_index`, `get_game_replay` | HTTP | `GET /api/analysis/{game_id}/replay?move_index=N` |
| **State hashing** | `board_hash` in `StepLog` | Optional | Schema supports it; currently `null` in logs |

**Replay:** `_replay_to_index` re-runs `move_records` from MongoDB to reconstruct board at `move_index`. Returns `board`, `move_count`, `pieces_used`, etc.

**League/ELO:** `league/league.py` — `LeagueDB`, `update_elo`, `update_elo_after_4p_match`. `league/elo.py` — `EloConfig`, `update_ratings`.

**Batch scripts:** `scripts/generate_analytics_data.py` (runs games with StrategyLogger), `scripts/run_sample_games.py`, `benchmarks/bench_selfplay_league.py`.

---

## G) Gaps / Duplication Risk List

### Partially Implemented or Duplicated

1. **Two analytics histories:**
   - **Frontend:** `legalMovesHistory` — mobility metrics, computed from `legal_moves` on each turn broadcast.
   - **Backend:** `StrategyLogger` → `steps.jsonl` — richer metrics (mobility, corners, blocking, etc.) but **not wired** to webapi or live games.

2. **Two legal-move sources:**
   - Engine `get_legal_moves()` is canonical.
   - Frontend receives `gameState.legal_moves` from API; no separate generator.

3. **PolicyView heatmap:** Uses `gameState.heatmap` (binary legal moves). ValueView shows placeholder; no real policy or value from RL/MCTS.

### Confusing Overlaps

- **Mobility:** Frontend `computeMobilityMetrics` (PN_i, MW_i, buckets) vs `analytics/metrics/mobility.py` (`compute_mobility_metrics`, `get_mobility_counts`). Different definitions and storage; not shared.
- **Heatmap:** Computed in `webapi/app.py` `_get_game_state` and in `webapi/game_manager.py` — same logic, two code paths.

### Features Not Wired to UI

- `StrategyLogger` (analytics) — only used by standalone script.
- `steps.jsonl` / `results.jsonl` — no UI to view or replay.
- MCTS `visits` / `total_reward` — stored in `FastMCTSNode` but not returned by `think()`; only `nodesEvaluated`, `timeSpentMs`, `maxDepthReached` in stats.
- `board_hash` in StepLog — schema supports it; not populated.

### Recommended Next Steps (Before Implementation)

1. **Clarify analytics scope:** Decide whether to wire `StrategyLogger` into webapi or keep frontend-only mobility.
2. **Unify mobility metrics:** Align frontend `mobilityMetrics.ts` with backend `analytics/metrics/mobility.py` if both are needed.
3. **MCTS explainability:** Extend `FastMCTSAgent.think()` to return per-move visits/Q if explainability is desired.
4. **Consolidate heatmap:** Single implementation in `_get_game_state` (or shared helper).
5. **ValueView:** Either wire to real value function or remove/relabel as placeholder.

---

*Generated from codebase analysis. No implementation changes made.*
