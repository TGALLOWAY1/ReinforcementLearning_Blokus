# Blokus RL Metrics Audit & Upgrade Plan

## 1. Codebase Audit Summary

### Where Metrics Are Computed
Metrics computation is split across several files in the `engine/` directory:
- **`engine/telemetry.py`**: The core conductor for per-move telemetry. `collect_all_player_metrics()` loops through all players to compute the base payload (frontier size, mobility, dead space, center control, piece lock risk). It computes the "Before" and "After" state vectors in `compute_move_telemetry_delta()`.
- **`engine/advanced_metrics.py`**: Contains deeper spatial metrics used by the AI and telemetry: `compute_corner_differential`, `compute_territory_control`, `compute_piece_penalty`, `compute_center_proximity`, and `compute_dead_zones`.
- **`engine/mobility_metrics.py`**: An existing module that contains `compute_player_mobility_metrics()`, which analyzes a list of legal moves to extract total placements, orientation counts, and board buckets, although it's currently primarily used by `worker_bridge.py` directly rather than the unified telemetry pipeline.

### Board State Representation & Copying
- **Representation**: `engine/board.py` represents the state both as a 20x20 `numpy.ndarray` (`self.grid` where 0=empty, 1-4=players) and as bitmasks (`self.occupied_bits`, `self.player_bits`).
- **Frontiers**: Iteratively maintained in `self.player_frontiers` as sets of `(row, col)` tuples.
- **Copying**: `Board.copy()` provides a deep copy mechanism, cloning the numpy grid, piece sets, and bitboards. This makes branching simulations (e.g. for "Mobility Stability" sampling) safe and efficient, as we do not leak mutations.

### Legal Move Generation
- **Source**: `engine/move_generator.py` -> `LegalMoveGenerator`.
- **Methodology**: It employs a highly-optimized frontier-based generator (`USE_FRONTIER_MOVEGEN=True`) that uses the player's `player_frontiers` to limit anchor queries, tested against bitboard collision logic (`is_placement_legal_bitboard_coords`). 
- **Complexity**: $O(|Frontier| \cdot |Unused Pieces| \cdot |Orientations| \cdot |Offsets|)$. This allows it to run very quickly compared to a naive full-board scan, making it feasible to perform full generation in fast-mode and sampling steps.

### Telemetry / Logging Pipeline
- **Storage**: In `engine/game.py`, `make_move()` intercepts state. If `_TELEMETRY_AVAILABLE` is true, it calls `collect_all_player_metrics` *before* applying the move, runs the move, and calls it *after*. The delta is then attached to `self.game_history`.
- **Frontend Fetching**: The Pyodide WebWorker evaluates `worker_bridge.py`, which accesses `game.game_history` and serialized the raw+delta vectors to JSON (`get_state()`), feeding the React UI components (`MoveDeltaPanel`, `MoveImpactWaterfall`, etc.).
- **Fast-Mode**: Controlled by `telemetry_fast_mode` flag. Currently, in `telemetry.py`, if fast-mode is True and no `move_generator` is passed, mobility is loosely approximated as `frontier_size * 2`. (Note: Recent changes often pass `move_generator` to force exact computation, but replacing the crude baseline with piece-aware logic is the goal).

---

## 2. Upgrade Implementation Plan (Checklist)

### A) Effective Frontier & Spread
- [ ] Add `compute_effective_frontier(board, player)` taking into account local empty-space rules without full enumeration (fast-mode compliant).
- [ ] Add `frontier_component_count` using spatial clustering/connected components (existing logic in `worker_bridge.py` can be refactored into the engine).

### B) True Mobility & Stability
- [ ] Ensure "Piece-aware Mobility" arrays tracking placements-per-piece are part of the core metric payload (extending `mobility_metrics.py`).
- [ ] Implement `mobility_entropy` over the piece allocation distribution.
- [ ] Implement `simulate_mobility_stability(board, player, K_samples)`: clone board, make plausible random opposing moves, record mean/min/p10 drops. Ensure seeded RNG tracking.
- [ ] Add anchor concentration `anchor_top1_share`.

### C) Center Control & Phase 
- [ ] Calculate `game_phase` progression factor `[0..1]` based on `board.move_count`.
- [ ] Combine continuous `center_proximity` with piece count and phase weight to yield `center_control_phase_weighted`.

### D) Dead Space Refinement
- [ ] Upgrade `compute_dead_zones()` to isolate `denial` vs `wasted_cavity`.
- [ ] Expose `dead_space_efficiency`.

### E) Piece Viability / Outcomes
- [ ] Calculate `locked_area` (sum of piece sizes for trapped pieces).
- [ ] Implement `prob_lock_next_turn` inside the stability sampler.
- [ ] Provide "Win Proxy Score" & comparative advantage formulas (`adv = self - mean(opps)`).

### F) Move Delta Generation
- [ ] Modify `compute_move_telemetry_delta` in `telemetry.py` to calculate Advantage Deltas (`Δadv`) and Future Deltas based on the new core metrics.

### G) UI, Tests & Polish
- [ ] Refactor React views (`MoveDeltaPanel`, Recharts bindings) to render the new fields nicely, grouping by "Expansion", "Flexibility", etc.
- [ ] Add robust unit tests under `tests/` asserting determinism, bounds, and correctness.
- [ ] Provide a `scripts/verify_metrics.py` simulation runner.
