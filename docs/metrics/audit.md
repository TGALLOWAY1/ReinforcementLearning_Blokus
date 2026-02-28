# Metrics Codebase Audit

## 1. Current State Diagram

```mermaid
flowchart TD
    %% Base Engine State
    Game[BlokusGame] --> Board[Board State: grid, pieces_used]
    Game --> MoveGen[LegalMoveGenerator]
    
    %% Incremental States
    Board --> Frontier[Frontier Cells]
    MoveGen --> Cache[piece_orientations_cache\npiece_frontier_fail_cache]
    
    %% Compute Triggers
    Game -->|get_state() called by| Bridge[WebWorkerGameBridge\n(browser_python)]
    
    %% On-demand Computing (Per State Fetch / Move)
    Bridge -->|Compute| AdvMetrics[advanced_metrics.py]
    Bridge -->|Compute| MobMetrics[mobility_metrics.py]
    
    %% Specific Metrics
    AdvMetrics -->|compute_dead_zones| DeadZones
    AdvMetrics -->|compute_territory_control| TerritoryInfluence
    AdvMetrics -->|compute_corner_differential| CornerDiff
    AdvMetrics -->|compute_center_proximity| CenterProx
    AdvMetrics -->|compute_piece_penalty| PiecePen
    AdvMetrics -->|compute_opponent_adjacency| OppAdj
    
    MobMetrics -->|compute_player_mobility_metrics| Mob
    Mob --> Placements[Total Placements]
    Mob --> OrientNorm[Normalized Orientation]
    Mob --> CellWeighted[Cell-Weighted]
```

## 2. Existing Metrics Inventory

| Metric Name | Location (File / Function) | Computation Trigger | Caching Strategy |
| --- | --- | --- | --- |
| **Frontier Size** | `engine/game.py` (`get_state`) | Per UI State Update | Incrementally updated on `make_move` |
| **Dead Zones** | `engine/advanced_metrics.py` (`compute_dead_zones`) | Per UI State Update | Recomputed (flood-fill BFS) |
| **Territory Control** | `engine/advanced_metrics.py` (`compute_territory_control`) | Per UI State Update | Recomputed (Manhattan distance spread) |
| **Corner Differential** | `engine/advanced_metrics.py` (`compute_corner_differential`) | Per UI State Update | Recomputed (derived from cached frontiers) |
| **Center Proximity** | `engine/advanced_metrics.py` (`compute_center_proximity`) | Per UI State Update | Recomputed (grid scan) |
| **Piece Penalty** | `engine/advanced_metrics.py` (`compute_piece_penalty`) | Per UI State Update | Recomputed (constant time sum) |
| **Opponent Adjacency** | `engine/advanced_metrics.py` (`compute_opponent_adjacency`) | Per UI State Update | Recomputed (grid scan for touching pieces) |
| **Mobility Metrics** | `engine/mobility_metrics.py` | Per UI State Update | Recomputed based on generated legal moves |

**Note on "When Computed":** Everything through the WebWorker `get_state()` method is computed synchronously when requested by the frontend. This means recomputing heavy metrics like flood-fill block the browser worker.

## 3. Computational Hotspots Profile

Based on architectural review and baseline `performance_test.py`:
- **Legal Move Generation**: The heaviest computation. Iterates through frontiers and checks anchor legality using bitboards. Cached per turn state via `piece_frontier_fail`.
- **Applying / Undoing a move**: Applying a move updates grid and incrementally maintains `board.player_frontiers[player]`.
- **Frontier Recompute**: Full recompute `_compute_full_frontier` exists but is skipped in favor of `update_frontier_after_move` (critical incremental path).
- **Flood-Fill / Connected Components**: `compute_dead_zones` inside `advanced_metrics.py`. It runs a BFS for all unoccupied cells. Currently, no incremental update path exists for this (recalculated from scratch).

## 4. Testing Harness Audit

- **Unit tests for board invariants**: Extensive coverage (`test_board.py`, `test_engine.py`, `test_legality_bitboard_equivalence.py`).
- **Regression tests for metrics**: `test_mobility_metrics.py` covers mobility, and `test_frontier_basic.py` covers incremental frontiers. However, no "golden snapshot" tests exist for the full suite in `advanced_metrics.py`.
- **Replay loader**: Implemented via `WebWorkerGameBridge.load_game` in `worker_bridge.py` which accepts an action history array and replays the engine states sequentially.

## 5. Documentation Audit

- Details on standard mobility metrics live in `docs/metrics/mobility.md`.
- No canonical documentation exists for `advanced_metrics.py` (Territory, Dead zones, etc.) other than code comments.
- **Decision:** The `docs/metrics/` directory is the canonical location. This `audit.md` should serve as the index, while detailed metric specs reside alongside `mobility.md`.

## 6. Gaps vs New Plan

- **Incremental Dead Zones**: We must implement an incremental solution for dead zones / territories as recomputing via BFS every move is too slow for deeper MCTS rollouts or fast UI scrubbers.
- **Missing Snapshot Tests**: Need golden regression tests for advanced metrics before refactoring.
- **Frontend Offloading**: Heavy metrics block the main execution flow; might consider making them optional or asynchronous during training.

## 7. Performance Constraints + Approach Recommendation

**Constraint**: Blokus move generation needs to stay sub-millisecond per move for Monte Carlo simulations. The current bottleneck shifts slightly to the `get_state` telemetry parsing overhead when sending UI updates.
**Recommendation**:
1. Implement incremental tracking for Dead Zones and Territory Control. Instead of mapping distance per empty cell, maintain disjoint sets or local delta updates upon piece placement.
2. Ensure metrics used strictly for training vs UI visualization are decoupled (disable dead-zone BFS during headless simulation unless explicitly required by the reward model).
