````markdown
# Development Plan: Move Generation Optimization (M6)

Focus areas:

1. Frontier-based move generation (avoid scanning full board)
2. Bitboard/bitmask representation for fast legality checks
3. Precomputation + lightweight caching

The goal is to **reduce `calculate_legal_moves` time by 5–10×** without changing game correctness.

---

## 0. Scope & Files

Expected primary touch points (adjust to your actual paths):

- `engine/board.py`
- `engine/move_generator.py`
- `engine/pieces.py`
- `engine/game.py` (integration)
- `tests/test_move_generation.py` (new or extended)
- `tests/test_bitboards.py` (new)
- `benchmarks/benchmark_move_gen.py` (new optional)

We’ll keep the current/naive move generator under a feature flag at first so we can compare behavior and performance.

---

## 1. Frontier-Based Move Generation

**Goal:** Only generate candidate moves from **frontier cells** (empty cells diagonally adjacent to a player’s existing pieces, and not orthogonally adjacent to them).

### 1.1 Design

- For each player, maintain a **frontier set**:
  - Frontier cell = empty (unoccupied) cell such that:
    - It is diagonally adjacent to at least one of that player’s placed cells.
    - It is *not* orthogonally adjacent to any of that player’s cells.
- On each successful placement for player `p`:
  - Update `p`’s frontier **incrementally** based on the cells in the newly placed piece.
- `move_generator.generate_legal_moves(player_id)`:
  - Ask `board` for frontier cells for `player_id`.
  - For each frontier cell, for each piece orientation, for each “anchor square” of the piece:
    - Anchor the piece at that frontier cell and test legality.
  - Return all legal placements.

### 1.2 Implementation Steps

1. **Audit current move generation**
   - Locate the existing `generate_legal_moves` (or equivalent) in `engine/move_generator.py`.
   - Document current complexity/behavior:
     - How many board cells are scanned per call?
     - Does it handle first move rules specially?
   - Add a simple logging/timing wrapper (to be removed later) so you can compare before/after.

2. **Add frontier tracking to `Board`**
   - In `engine/board.py`:
     - Decide representation:
       - Start simple: `Set[Tuple[int, int]]` per player.
       - Optionally add a bitmask version later (see Section 2).
     - Add internal state:
       - `self.frontiers: Dict[player_id, Set[Coord]]`
     - Implement:
       - `compute_initial_frontier(player_id)` for the starting configuration.
         - For Blokus, typically each player starts from a specific corner.
         - Initial frontier is that corner cell.
       - `recompute_frontier(player_id)` (slow, full recompute – for debugging/testing).
       - `update_frontier_after_move(player_id, placed_cells: List[Coord])`:
         - For each newly placed cell:
           - Remove it from frontier if present.
           - For each of its 4 diagonal neighbors:
             - If neighbor is on-board and empty:
               - Add to frontier **if** not orth-adjacent to any of player’s cells.
         - For each of its 4 orthogonal neighbors:
           - If an orth neighbor is in frontier, remove it (it’s now illegal as a frontier).
     - Ensure `Board.place_piece(...)` calls `update_frontier_after_move`.

3. **Provide frontier accessors**
   - In `engine/board.py`, add:
     - `get_frontier(player_id) -> Iterable[Coord]`
     - `debug_rebuild_frontier(player_id)` that runs `recompute_frontier` and optionally asserts equality with the incremental version (in debug or tests).

4. **Refactor move generation to use frontier**
   - In `engine/move_generator.py`:
     - Add a new function:
       - `generate_legal_moves_frontier(board, player_id, remaining_pieces)`.
     - Logic:
       1. Fetch `frontier = board.get_frontier(player_id)`.
       2. For each `piece` in `remaining_pieces`:
          - For each `orientation` of that piece:
            - For each selected **anchor index** in that orientation (we’ll define anchors in Section 3):
              - For each `frontier_cell`:
                - Compute candidate placement by aligning anchor cell to `frontier_cell`.
                - Call a legality check (initially your existing one; in Section 2 we swap in bitmasks).
                - If legal, append to candidate list.
     - Add a config / feature flag to choose between current generator and new frontier-based generator, e.g.:
       - `USE_FRONTIER_GENERATOR = True` controlled by env var or settings.

5. **Handle first-move special rules**
   - Some Blokus variants require the **first move** to cover a specific corner.
   - Ensure that, for turn 1 of a player:
     - `frontier` is set such that it only permits moves that include the starting corner, or
     - The first move uses a special-case generator that enforces the corner rule and then normal frontier afterward.

### 1.3 Testing & Verification

1. **Unit tests for frontier logic**
   - Add cases in `tests/test_move_generation.py`:
     - Scenario: empty board, first placement for player 0 at corner (0,0):
       - Validate expected frontier cells.
     - Scenario: place a 3–4-cell piece and verify:
       - Frontier includes new diagonals.
       - Frontier excludes new orth neighbors and occupied cells.
     - Scenario: call `debug_rebuild_frontier` and assert it equals the incremental frontier.

2. **Behavioral equivalence tests**
   - Keep the old generator as `generate_legal_moves_naive`.
   - Write property-like tests:
     - For a set of random valid board states (can be generated with self-play or fixtures):
       - Compare sets:
         - `set(naive_moves)` vs `set(frontier_moves)`.
       - Assert equality for:
         - Early game (few pieces placed).
         - Mid game.
         - Late game (few or no legal moves).
   - If there’s a discrepancy, dump the board state and failing moves for debugging.

3. **Performance benchmarks**
   - In `benchmarks/benchmark_move_gen.py`:
     - Generate representative board states from stored game logs or random self-play.
     - Measure:
       - Avg time for `naive` vs `frontier` over N runs.
     - Ensure no regression for small boards; expect large improvement for mid/late game.

---

## 2. Bitboard / Bitmask Representation

**Goal:** Replace cell-by-cell legality checks with a handful of bitwise operations.

### 2.1 Design

- Represent board and pieces as bitmasks over a **linearized index**:
  - `index = row * BOARD_WIDTH + col` (e.g., 20×20 → 0..399).
- Board bitmasks:
  - `occupied_bits: int` — all occupied cells.
  - `player_bits[player_id]: int` — cells occupied by specific player.
- Piece/orientation masks (anchored at origin (0,0)):
  - `shape_mask: int`
  - `diagonal_border_mask: int` — potential diagonal contact cells around the shape.
  - `orth_border_mask: int` — potential orthogonal contact cells around the shape.
- Shift piece masks to candidate anchor positions and evaluate:
  - `overlaps = (shape_mask_shifted & occupied_bits) != 0`
  - `touches_orth = (orth_mask_shifted & player_bits[player_id]) != 0`
  - `touches_diag = (diag_mask_shifted & player_bits[player_id]) != 0`
- Also ensure the entire shifted shape is within board bounds (no wrap-around).

### 2.2 Implementation Steps

1. **Define bitboard utilities**
   - Create `engine/bitboard.py` (or similar) with:
     - `BOARD_WIDTH`, `BOARD_HEIGHT`, `NUM_CELLS = BOARD_WIDTH * BOARD_HEIGHT`.
     - Helpers:
       - `coord_to_bit(row, col) -> int` (returns a mask with a single bit set).
       - `coords_to_mask(coords: List[Coord]) -> int`.
       - `shift_mask(mask: int, d_row: int, d_col: int) -> Optional[int]` that:
         - Returns `None` if shifting would push any bit off-board or wrap across rows.
         - Otherwise returns the shifted mask.
     - Note: shifting by `(d_row, d_col)` is equivalent to `(d_row * BOARD_WIDTH + d_col)` in linear index, but you need to prevent wrapping.

2. **Add bitboard state to `Board`**
   - In `engine/board.py`:
     - Add:
       - `self.occupied_bits: int = 0`
       - `self.player_bits: Dict[player_id, int] = defaultdict(int)`
     - In `place_piece(player_id, cells: List[Coord])`:
       - Compute `mask = coords_to_mask(cells)`.
       - Update:
         - `self.occupied_bits |= mask`
         - `self.player_bits[player_id] |= mask`
       - Keep the existing grid representation for now, so you have both.

3. **Precompute piece masks (shape + borders)**
   - In `engine/pieces.py`:
     - For each piece and orientation (see Section 3 for precomputation):
       - Precompute:
         - `shape_coords` anchored at (0,0).
         - `shape_mask = coords_to_mask(shape_coords)`.
       - Compute diagonal neighbors:
         - For each `cell` in `shape_coords`, add diagonals `(±1, ±1)` into a set, excluding coords that are inside the shape itself.
         - `diag_mask = coords_to_mask(diag_coords)`.
       - Compute orth neighbors:
         - For each `cell` in `shape_coords`, add `(±1, 0)`, `(0, ±1)` that are not in the shape.
         - `orth_mask = coords_to_mask(orth_coords)`.
       - Store these in the piece orientation metadata, e.g.:
         ```python
         @dataclass
         class PieceOrientation:
             id: int
             offsets: List[Coord]
             anchor_indices: List[int]  # from Section 3
             shape_mask: int
             diag_mask: int
             orth_mask: int
         ```

4. **Bitboard legality check**
   - Add a function, e.g. in `engine/move_generator.py` or `engine/bitboard.py`:
     ```python
     def is_placement_legal_bitboard(
         board: Board,
         player_id: int,
         orientation: PieceOrientation,
         anchor_board_coord: Coord,
         anchor_piece_index: int,
     ) -> bool:
         ...
     ```
   - Steps:
     1. Derive `(anchor_piece_row, anchor_piece_col)` from `orientation.offsets[anchor_piece_index]`.
     2. Compute `(d_row, d_col)` = `anchor_board_coord - anchor_piece_coord`.
     3. Call `shift_mask` on `shape_mask`, `diag_mask`, `orth_mask` with `(d_row, d_col)`:
        - If any returns `None` → placement is off-board → illegal.
     4. Check overlap:
        - If `shape_shifted & board.occupied_bits != 0`: illegal.
     5. Check orth adjacency:
        - If `orth_shifted & board.player_bits[player_id] != 0`: illegal.
     6. Check diagonal adjacency:
        - If `diag_shifted & board.player_bits[player_id] == 0`: illegal (must touch diagonally at least once).
     7. Any extra piece-specific rules (e.g., first move must cover corner) can be handled outside or via an additional mask check.

5. **Wire bitboard checks into frontier generator**
   - In `generate_legal_moves_frontier`:
     - Replace the existing cell-by-cell legality function with `is_placement_legal_bitboard`.
   - Keep the old cell-based legality function around temporarily for:
     - Testing equivalence.
     - Debugging.

### 2.3 Testing & Verification

1. **Bitboard <-> grid consistency**
   - In `tests/test_bitboards.py`:
     - Start with an empty board.
     - Place random legal pieces on grid using existing logic.
     - After each placement, compare:
       - `board.occupied_bits` vs grid:
         - For every cell `(r,c)`:
           - `grid[r][c] != 0` ↔ `occupied_bits & coord_to_bit(r,c) != 0`.
       - `player_bits[p]` vs player-owned cells in grid.
     - Add a helper assertion:
       - `assert_board_consistent(board)` used in multiple tests.

2. **Bitboard legality vs old legality**
   - Use the same random state generator as in Section 1.3.
   - For each candidate move (piece, orientation, anchor):
     - Evaluate with original legality checker and bitboard checker.
     - Assert they agree (legal vs illegal).
   - Run tests for:
     - First moves.
     - Mid-game with many pieces.
     - Edge/corner-heavy positions.

3. **Performance checks**
   - Extend `benchmarks/benchmark_move_gen.py` to compare:
     - Old legality vs bitboard legality (with frontier turned on or off).
   - Record speedup factors for documentation.

---

## 3. Precomputation + Lightweight Caching

**Goal:** Avoid per-call work that can be done once at startup or once per turn. Keep caching simple and safe.

### 3.1 Design

- Precompute **everything static** at module import time:
  - All unique piece orientations (rotations/reflections).
  - For each orientation:
    - Cell offsets relative to an anchor.
    - A small set of **anchor indices** (1–4 cells that make good attachment points).
    - Bitboard masks: `shape_mask`, `diag_mask`, `orth_mask`.
- Introduce **per-turn, in-memory caches** where it’s cheap & safe:
  - Example: If no orientation of piece `P` can be placed anchored at frontier cell `F` (off-board, overlap, etc.), record that fact and skip re-testing every anchor of `P` at `F` during that same `generate_legal_moves` call.

### 3.2 Implementation Steps

1. **Precompute piece orientations (shapes)**
   - In `engine/pieces.py`, centralize orientation generation:
     1. For each base piece definition:
        - Generate all rotations (0°, 90°, 180°, 270°).
        - For each rotation, optionally generate reflections (depending on Blokus rules).
     2. Normalize orientations:
        - Translate each shape so its min row and col are `0`.
     3. Deduplicate:
        - Sort the cell list and use it as a key in a set/dict to remove duplicates.
     4. Store the final list:
        ```python
        ALL_PIECES: Dict[PieceId, List[PieceOrientation]]
        ```
   - Make sure this logic runs once on module import or through an explicit `init_pieces()` called during engine setup.

2. **Choose anchor indices for each orientation**
   - For each `orientation.offsets` list:
     - Compute candidate anchor indices using simple heuristics, e.g.:
       - The cell with minimum `(row + col)` (top-leftmost).
       - The cell with maximum `(row + col)` (bottom-rightmost).
       - Optionally add one or two “extreme” cells (like furthest from centroid).
     - Store them as `anchor_indices: List[int]` in the `PieceOrientation` class.
   - The goal: 2–4 anchors per orientation instead of every cell.

3. **Attach precomputed bitmasks**
   - Reuse the code from Section 2.2 Step 3:
     - For each `PieceOrientation`, compute and store:
       - `shape_mask`
       - `diag_mask`
       - `orth_mask`

4. **Introduce a per-call cache in move generation (optional but helpful)**
   - In `generate_legal_moves_frontier`, at the **start of the function**:
     - Initialize e.g.:
       ```python
       per_call_cache = {
           "piece_frontier_fail": set(),  # (piece_id, frontier_idx)
       }
       ```
   - Before testing any anchors for a `(piece_id, frontier_coord)` pair:
     - If `(piece_id, frontier_coord)` is in `piece_frontier_fail`, skip.
   - After testing all anchors of `piece_id` at `frontier_coord`:
     - If none were legal, add `(piece_id, frontier_coord)` to the fail set.
   - This avoids redundant work when multiple anchors of the same piece/orientation are clearly impossible at that frontier cell.
   - Keep this cache **local to a single call** of `generate_legal_moves_frontier` to avoid statefulness bugs.

5. **Remove redundant runtime computations**
   - Grep for any repeated rotation/reflection/normalization operations inside hot paths (like per-move loops).
   - Replace them with lookups into the precomputed `ALL_PIECES` structure.
   - Ensure `generate_legal_moves` only iterates over ready-made `PieceOrientation` objects.

### 3.3 Testing & Verification

1. **Orientation correctness tests**
   - In `tests/test_pieces.py`:
     - For each piece:
       - Assert the number of precomputed orientations matches manually verified counts (Blokus standards).
       - For each orientation:
         - Assert no duplicated coordinates.
         - Assert shapes are contiguous (optionally).
         - Assert that no two orientations have identical coordinate sets.

2. **Anchor sanity tests**
   - Ensure that `anchor_indices` are valid indices into `offsets`.
   - Ensure that different anchor indices correspond to different coordinates (no duplicates).
   - Optionally add a simple property:
     - For at least one anchor in each orientation, there exists at least one legal placement on an empty board starting from a corner.

3. **Precomputation vs old on-the-fly logic**
   - If you still have the old inline orientation / rotation logic:
     - For random base pieces and random transforms:
       - Generate orientation both with old logic and new precomputed logic.
       - Assert equality of shape coords after normalization.

4. **Cache correctness**
   - For the per-call `(piece_id, frontier_coord)` cache:
     - Add a debug mode test where:
       - The cache is disabled.
       - The cache is enabled.
       - Both executions produce identical move sets.

5. **Overall regression tests**
   - Use the tests from Sections 1 and 2 to ensure:
     - Precomputation + frontier + bitboards together produce exactly the same set of legal moves as the original naive implementation (within numerical tolerance of ordering, but equivalent as sets).
   - Run a set of small full games (self-play) with:
     - Original generator.
     - Optimized generator.
     - Verify:
       - Game ends after same number of moves or with compatible outcomes.
       - No illegal moves are ever generated (assert at runtime in debug builds).

---

## 4. Rollout Plan

1. **Phase 1 – Frontier only**
   - Implement frontier tracking & frontier-based generator.
   - Keep existing legality checks.
   - Ensure full correctness via equivalence tests & unit tests.
   - Benchmark and record improvements.

2. **Phase 2 – Bitboards + precomputation**
   - Introduce bitboard representation under a feature flag.
   - Precompute orientations + masks + anchors.
   - Swap legality checks to bitboard version.
   - Ensure equivalence with old logic and frontier-based generator.
   - Benchmark again; record total speedup.

3. **Phase 3 – Tidy & default**
   - Remove or demote old naive generator (or keep only for debug).
   - Enable `frontier + bitboards + precompute` as default in training.
   - Add docs:
     - `docs/move-generation-optimization.md` summarizing the approach, data structures, and benchmark results.

---

## 5. Success Criteria

- **Correctness**
  - For a large random sample of board states, `optimized_moves` and `naive_moves` are identical as sets for each player.
  - No runtime assertion of illegal moves is ever triggered in debug self-play.

- **Performance**
  - Measured average time per `calculate_legal_moves` call reduced by:
    - ≥ 3× in early game.
    - ≥ 5–10× in mid/late game.
  - Overall training loop (episodes/sec) improves significantly without other bottlenecks taking over.

- **Maintainability**
  - Code clearly separates:
    - Board representation (`Board`, bitboards, frontiers).
    - Piece definitions & precomputation (`pieces.py`).
    - Move generation strategy (`move_generator.py`).
  - Tests cover all three layers with good failure messages.

---
````
