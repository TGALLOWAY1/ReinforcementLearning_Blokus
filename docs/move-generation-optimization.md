# Move Generation Optimization (M6)

## Overview

Blokus move generation is a critical performance bottleneck. With a 20×20 board, 21 pieces per player, and up to 8 orientations per piece, the naive approach of scanning every board cell for every piece orientation results in millions of candidate placements to evaluate. In mid-game states, this can take several seconds per move.

**M6 Goals:**
- Reduce time spent in "calculate legal moves" by 5–10× without changing correctness
- Maintain 100% equivalence with the naive reference implementation
- Provide optional performance optimizations (heuristic anchors) with safety fallbacks

**Results:**
- Frontier-based generation reduces search space by focusing on relevant cells
- Bitboard legality checks provide O(1) overlap/adjacency tests vs O(N) grid scans
- Combined optimizations achieve significant speedup while maintaining correctness

---

## High-Level Design

### Frontier-Based Generation

**Key Insight:** In Blokus, pieces must be placed diagonally adjacent to your own pieces (except the first move). This means legal placements can only occur at cells that are:
- Diagonally adjacent to at least one of your pieces
- NOT orthogonally adjacent to any of your pieces

These cells form the "frontier" - a much smaller set than the full 400-cell board.

**Algorithm:**
1. For each player, maintain a set of frontier cells (updated when pieces are placed)
2. When generating moves, only consider frontier cells as potential anchor points
3. This dramatically reduces the search space, especially in later game stages

**Example:** In a mid-game state, the frontier might contain 20-30 cells vs 400 total cells, reducing candidate placements by 10-20×.

### Bitboard Representation

The board state is represented as bitmasks:
- **20×20 board = 400 cells = 400 bits** (Python ints are arbitrary precision)
- `occupied_bits`: Bitmask of all occupied cells
- `player_bits[player]`: Bitmask of cells occupied by each player

**Benefits:**
- O(1) overlap checks: `shape_mask & occupied_bits != 0`
- O(1) adjacency checks: `neighbor_mask & player_bits[player] != 0`
- No grid iteration needed for legality checks

### Coords-Based Legality

The bitboard legality check works directly from placement coordinates, avoiding anchor/shift complexity:

**Legality Rules:**
1. **No Overlap**: `shape_mask & occupied_bits == 0`
2. **No Orth Adjacency**: `orth_mask & player_bits[player] == 0`
3. **Diag Adjacency Required** (if not first move): `diag_mask & player_bits[player] != 0`
4. **First Move Corner**: `shape_mask & start_corner_bit != 0`

**Why Coords-Based:**
- Simpler logic: no anchor translation or mask shifting
- More robust: computes neighbors dynamically from final coordinates
- Easier to verify: direct correspondence with grid-based checks

---

## Data Structures

### Board

```python
class Board:
    grid: np.ndarray  # 20×20 grid (0=empty, 1-4=players)
    occupied_bits: int  # Bitmask of all occupied cells
    player_bits: Dict[Player, int]  # Bitmask per player
    player_frontiers: Dict[Player, Set[Tuple[int, int]]]  # Frontier cells per player
```

**Bitboard State Maintenance:**
- Updated on `place_piece()`: sets bits in `occupied_bits` and `player_bits[player]`
- Updated on `remove_piece()`: clears bits
- Maintained in sync with grid via `assert_bitboard_consistent()`

### PieceOrientation

```python
@dataclass
class PieceOrientation:
    piece_id: int
    orientation_id: int
    offsets: List[Tuple[int, int]]  # Normalized to (0,0) anchor
    shape_mask: int  # Bitmask for piece cells
    diag_mask: int   # Bitmask for diagonal neighbors
    orth_mask: int   # Bitmask for orthogonal neighbors
    anchor_indices: List[int]  # Heuristic anchor points
```

**Precomputation:**
- All 21 pieces × unique orientations (typically 4-8 per piece)
- Masks computed once at module load time
- Anchor indices selected via heuristics (top-left, bottom-right, furthest-from-centroid)

### Move Representation

```python
class Move:
    piece_id: int
    orientation: int  # Index into orientations list
    anchor_row: int
    anchor_col: int
```

**Coordinate-Based Comparison:**
- Moves are compared by `(piece_id, sorted_coords)` for robust equivalence testing
- Abstracts away orientation indices and anchor points

---

## Algorithms

### Public API: `get_legal_moves()`

```python
def get_legal_moves(board: Board, player: Player) -> List[Move]:
    if USE_FRONTIER_MOVEGEN:
        return self._get_legal_moves_frontier(board, player)
    else:
        return self._get_legal_moves_naive(board, player)
```

**Default Path:** Frontier + bitboard (fast, correct)
**Debug Path:** Naive generator (full-board scan, grid-based)

### Frontier Move Generation

```python
def _get_legal_moves_frontier(board, player):
    frontier_cells = board.get_frontier(player)
    
    for piece in available_pieces:
        for orientation in piece_orientations:
            for frontier_cell in frontier_cells:
                for anchor_index in anchor_indices:
                    # Calculate anchor position
                    anchor_row = frontier_cell.row - offset[anchor_index].row
                    anchor_col = frontier_cell.col - offset[anchor_index].col
                    
                    # Build placement coordinates
                    placement_coords = [
                        (anchor_row + offset.row, anchor_col + offset.col)
                        for offset in orientation.offsets
                    ]
                    
                    # Check legality (bitboard or grid-based)
                    if is_placement_legal(placement_coords):
                        legal_moves.append(Move(...))
```

**Anchor Selection:**
- **Exact Mode** (default): Try all offsets as anchors
- **Heuristic Mode**: Try only `anchor_indices`, with per-orientation fallback to all offsets if no moves found

### Bitboard Legality (Coords-Based)

```python
def is_placement_legal_bitboard_coords(board, player, placement_coords, is_first_move):
    # 1. Build shape mask from coords
    shape_mask = coords_to_mask(placement_coords)
    
    # 2. Check overlap
    if shape_mask & board.occupied_bits != 0:
        return False
    
    # 3. Compute neighbors dynamically
    diag_neighbors = compute_diag_neighbors(placement_coords)
    orth_neighbors = compute_orth_neighbors(placement_coords)
    
    diag_mask = coords_to_mask(diag_neighbors)
    orth_mask = coords_to_mask(orth_neighbors)
    
    # 4. Check orth adjacency (not allowed)
    if orth_mask & board.player_bits[player] != 0:
        return False
    
    # 5. Check diag adjacency (required if not first move)
    if not is_first_move:
        if diag_mask & board.player_bits[player] == 0:
            return False
    
    # 6. Check first move corner
    if is_first_move:
        if shape_mask & start_corner_bit == 0:
            return False
    
    return True
```

**Key Design:** Neighbors computed dynamically from placement coordinates, ensuring all neighbors are captured (including those that were negative in normalized space).

---

## Performance Optimizations

### Anchor Heuristics

**Problem:** Trying all offsets as anchors for every piece/orientation/frontier combination is redundant. Many anchors produce the same placements or no valid placements.

**Solution:** Precompute strategic anchor points per orientation:
- Top-left cell (min row+col)
- Bottom-right cell (max row+col)
- Furthest-from-centroid cell (max distance from average position)
- Small pieces (≤4 cells): Use all offsets (heuristic overhead not worth it)

**Safety:** Per-orientation fallback mechanism:
- If heuristic anchors find no legal moves for an orientation, try all offsets
- Ensures heuristics never miss legal moves
- Only activates when needed (maintains performance in common case)

### Per-Call Caching

**Problem:** Same (piece, orientation, frontier_cell) combinations are tested multiple times within a single move generation call.

**Solution:** Cache `(piece_id, orientation_idx, frontier_coord)` combinations that yield no legal moves, skipping redundant checks within the same call.

**Granularity:** Cache key includes `orientation_idx` to avoid false negatives (different orientations may succeed at same frontier cell).

### Code Hoisting

**Optimizations:**
- Hoist `ALL_PIECE_ORIENTATIONS.get(piece.id, [])` outside loops
- Hoist `piece_orientations_cache[piece.id]` outside loops
- Reduce redundant dictionary lookups

### Benchmark Results

From `benchmarks/benchmark_move_generation.py`:

**Empty Board (First Move):**
- Naive: ~50-100ms
- Frontier+Grid: ~20-40ms (2-3× speedup)
- Frontier+Bitboard: ~10-20ms (5-10× speedup)

**Mid-Game State (10-15 moves):**
- Naive: ~500-2000ms
- Frontier+Grid: ~100-300ms (5-7× speedup)
- Frontier+Bitboard: ~50-150ms (10-20× speedup)

**Late-Game State (30+ moves):**
- Naive: ~2000-5000ms
- Frontier+Grid: ~200-500ms (10× speedup)
- Frontier+Bitboard: ~100-250ms (20× speedup)

**Key Insight:** Speedup increases with game progress as frontier size shrinks relative to board size.

---

## Configuration & Flags

### Feature Flags

#### `BLOKUS_USE_FRONTIER_MOVEGEN` (default: `True`)

Controls whether to use frontier-based generation or naive full-board scan.

- `True` (default): Use frontier-based generation (fast)
- `False`: Use naive generator (slow, for debugging)

**When to disable:**
- Debugging move generation correctness issues
- Comparing performance between generators
- Testing equivalence

#### `BLOKUS_USE_BITBOARD_LEGALITY` (default: `True`)

Controls whether to use bitboard-based legality checks or grid-based checks.

- `True` (default): Use bitboard legality (fast, O(1) checks)
- `False`: Use grid-based legality (slower, O(N) scans)

**When to disable:**
- Debugging bitboard correctness issues
- Comparing bitboard vs grid legality
- Testing equivalence

#### `BLOKUS_USE_HEURISTIC_ANCHORS` (default: `False`)

Controls whether to use heuristic anchor selection or try all offsets.

- `False` (default): Try all offsets as anchors (exact mode, slower but guaranteed complete)
- `True`: Use heuristic anchors with fallback (faster, should be equivalent due to fallback)

**When to enable:**
- Performance-critical scenarios where slight overhead is acceptable
- Testing heuristic fallback mechanism
- Production deployments after thorough testing

**Safety:** Per-orientation fallback ensures no moves are missed, but fallback only triggers when heuristic anchors find zero moves for an orientation.

### Debug Flags

#### `BLOKUS_MOVEGEN_DEBUG` (default: `False`)

Enable timing logs for move generation.

**Output:** Logs move generation time, frontier size, slowest piece, etc.

**Use case:** Performance profiling

#### `BLOKUS_MOVEGEN_DEBUG_EQUIVALENCE` (default: `False`)

Enable random sampling of bitboard vs grid legality checks (5% of calls).

**Output:** Debug logs when bitboard and grid legality differ

**Use case:** Catching correctness regressions during development

#### `BLOKUS_DEBUG_BITBOARD` (default: `False`)

Enable deep bitboard vs grid comparison for specific moves.

**Output:** Detailed comparison of:
- Shape masks from coords vs shifted orientation masks
- Diag/orth neighbors computed from coords vs shifted masks
- Adjacency intersections with player bits
- Grid vs bitboard legality results

**Use case:** Debugging specific moves that fail bitboard legality but pass grid legality (or vice versa)

**Usage:**
```bash
BLOKUS_DEBUG_BITBOARD=1 pytest tests/test_move_generation_equivalence.py::test_frontier_bitboard_vs_naive_random_states -s
```

---

## Debugging & Testing

### Test Suites

#### `tests/test_move_generation_equivalence.py`

Comprehensive equivalence tests comparing naive vs frontier+bitboard:
- Empty board scenarios
- After single/multiple pieces
- Random state comparisons (15+ states)
- Coordinate-based move comparison for robustness
- Safety tests for default path

**Key Tests:**
- `test_default_move_generation_matches_naive_on_small_random_sample`: Safety test using actual defaults
- `test_frontier_bitboard_vs_naive_random_states`: Comprehensive random state comparison
- `test_heuristic_anchors_fallback_correctness`: Verifies fallback mechanism

#### `tests/test_legality_bitboard_equivalence.py`

Tests comparing grid-based vs bitboard-based legality checks:
- First move scenarios
- After piece placements
- Actual legal moves from generators
- Random state comparisons

#### `tests/test_bitboard_basic.py`

Basic bitboard functionality and consistency tests:
- Utility function tests (coord_to_bit, mask_to_coords, etc.)
- Bitboard state consistency after moves
- `test_bitboard_invariants_through_random_self_play`: Safety test for bitboard consistency

### Debug Helpers

#### `debug_compare_bitboard_vs_grid()`

Deep comparison helper for debugging specific moves:

```python
from engine.move_generator import debug_compare_bitboard_vs_grid

# Enable debug output
import os
os.environ["BLOKUS_DEBUG_BITBOARD"] = "1"

# Call with move details
debug_compare_bitboard_vs_grid(
    board, player, orientation,
    anchor_board_coord, anchor_piece_index,
    placement_coords
)
```

**Output:**
- Shape mask comparison (coords vs shifted)
- Diag/orth neighbor comparison
- Adjacency intersection details
- Grid vs bitboard legality results

#### `board.assert_bitboard_consistent()`

Verifies bitboard state matches grid state:

```python
board.assert_bitboard_consistent()  # Raises AssertionError if inconsistent
```

**Checks:**
- Every occupied cell in grid has bit set in `occupied_bits`
- Every player-owned cell has bit set in `player_bits[player]`
- No bits set for empty cells
- No cross-player bit conflicts

### Debugging Recipe

**If you suspect a legality bug:**

1. **Reproduce with naive generator:**
   ```python
   import os
   os.environ["BLOKUS_USE_FRONTIER_MOVEGEN"] = "0"
   os.environ["BLOKUS_USE_BITBOARD_LEGALITY"] = "0"
   # Test your scenario
   ```

2. **Compare with default path:**
   ```python
   # Reset to defaults
   del os.environ["BLOKUS_USE_FRONTIER_MOVEGEN"]
   del os.environ["BLOKUS_USE_BITBOARD_LEGALITY"]
   # Test same scenario
   ```

3. **Enable debug output:**
   ```bash
   BLOKUS_DEBUG_BITBOARD=1 pytest tests/test_move_generation_equivalence.py::test_frontier_bitboard_vs_naive_random_states -s
   ```

4. **Check bitboard consistency:**
   ```python
   board.assert_bitboard_consistent()  # Should not raise
   ```

5. **Use coordinate-based comparison:**
   - Compare moves by `(piece_id, sorted_coords)` not by anchor/orientation
   - This abstracts away implementation differences

**If moves are missing:**
- Check if `USE_HEURISTIC_ANCHORS` is enabled (fallback should catch, but verify)
- Verify frontier is correctly maintained (check `board.get_frontier(player)`)
- Compare frontier+bitboard vs naive on same state

**If bitboard legality fails but grid passes:**
- Use `debug_compare_bitboard_vs_grid()` for the specific move
- Check if neighbor masks match (diag/orth from coords vs shifted masks)
- Verify `board.player_bits[player]` matches grid state

---

## Implementation History

**M6-P1:** Frontier-based move generation
- Added per-player frontier tracking
- Implemented frontier-based generator
- Added equivalence tests

**M6-P2:** Bitboard-based legality
- Added bitboard utilities (`engine/bitboard.py`)
- Integrated bitboard state into Board
- Precomputed piece orientations with masks
- Implemented coords-based bitboard legality
- Added comprehensive equivalence tests
- Implemented heuristic anchors with fallback
- Made frontier+bitboard the default

**M6-P3:** Documentation and safety tests
- Made frontier+bitboard the default configuration
- Added safety tests for default path
- Added bitboard invariant tests
- Created this documentation

---

## Future Optimizations

Potential areas for further optimization:

1. **Transposition Tables:** Cache legal moves for board states
2. **Incremental Frontier Updates:** Optimize frontier maintenance
3. **SIMD Operations:** Vectorize bitboard operations (if using NumPy)
4. **Parallel Move Generation:** Generate moves for multiple pieces in parallel
5. **Move Ordering:** Heuristics to try promising moves first (for AI agents)

---

## References

- **Code:** `engine/move_generator.py`, `engine/bitboard.py`, `engine/pieces.py`
- **Tests:** `tests/test_move_generation_equivalence.py`, `tests/test_legality_bitboard_equivalence.py`
- **Benchmarks:** `benchmarks/benchmark_move_generation.py`
- **Performance Results:** `PERFORMANCE_OPTIMIZATION_RESULTS.md`

