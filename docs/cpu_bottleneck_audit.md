# Blokus CPU Bottleneck Audit

## 1. Executive Summary

**The single largest CPU bottleneck is `is_placement_legal_bitboard_coords` in `move_generator.py:676`, which consumes 82% of move generation time.** It recomputes neighbor bitmasks from scratch on every call despite `PieceOrientation` already storing precomputed `shape_mask`, `diag_mask`, and `orth_mask`. This results in 2.66M function calls to `coord_to_bit`/`coord_to_index` per 100 move-gen calls.

The irony: the "bitboard" legality path is **2-3x slower** than the grid-based path because it converts coordinates to bitmasks in Python loops rather than using precomputed masks with bitwise shifts. The benchmark confirms:

| Game Stage | Frontier+Grid | Frontier+Bitboard | Ratio |
|---|---|---|---|
| Early (5 moves) | 2.67ms | 6.32ms | 2.4x slower |
| Early (10 moves) | 7.10ms | 15.22ms | 2.1x slower |
| Mid (20 moves) | 4.80ms | 15.01ms | 3.1x slower |
| Mid (30 moves) | 2.83ms | 9.45ms | 3.3x slower |
| Late (40 moves) | 3.44ms | 11.29ms | 3.3x slower |

The fix is straightforward: use the precomputed masks from `PieceOrientation` with a single `shift_mask()` call instead of rebuilding masks from coordinates. This alone should deliver a **3-5x speedup** to the default move generation path.

Secondary bottlenecks (board copy cost in MCTS, `LegalMoveGenerator` reinstantiation, `list(Player)` allocations) are real but lower priority until the primary legality check is fixed.

---

## 2. Current Benchmark Coverage

### Existing Benchmarks

| File | Type | What It Measures | Trustworthy? |
|---|---|---|---|
| `benchmarks/benchmark_move_generation.py` | Micro | Move gen across 3 strategies × 5 game stages | **Yes** - good methodology, 50 runs each, reveals the bitboard regression |
| `scripts/benchmark_env.py` | Integration | Env step() + move gen isolation | Broken - depends on missing `envs.blokus_v0` module |
| `scripts/benchmark_vecenv.py` | Integration | Single vs multi-env training speed | Broken - same missing dependency |
| `benchmarks/bench_selfplay_league.py` | Integration | Stage 2 vs 3 rollout throughput | Good but requires training infra |
| `benchmarks/profile_stage3.py` | Profiling | cProfile of rollout steps | Good |
| `benchmarks/scan_stage3_envs.py` | Sweep | VecEnv config optimization | Good |
| `tests/performance_test.py` | End-to-end | Full game move timing | **Yes** - real game context |

### What's Missing

1. **Legality check micro-benchmark**: No isolated benchmark for `is_placement_legal_bitboard_coords` vs `is_placement_legal_bitboard` vs grid-based. This is the #1 bottleneck but has no dedicated benchmark.
2. **Board.copy() benchmark**: No measurement of copy cost in isolation, critical for MCTS.
3. **MCTS iteration throughput**: No benchmark for nodes/second in MCTS search.
4. **Action encoding/decoding**: No benchmark for move↔action_index conversion (relevant for RL env).
5. **Frontier update benchmark**: No measurement of `update_frontier_after_move` cost.

### Key Observation

The existing `benchmark_move_generation.py` already reveals the critical finding (bitboard path is slower) but the codebase defaults to `USE_BITBOARD_LEGALITY=True`. **The default configuration is using the slower path.**

---

## 3. Architecture Map of CPU-Critical Paths

### Arena/Self-Play Hot Path
```
run_single_game() [arena_runner.py:578]
  └─ while not game_over:
      ├─ game.get_legal_moves(player)           ← DOMINANT COST
      │   └─ move_generator.get_legal_moves()
      │       └─ _get_legal_moves_frontier()    ← 82% of time in bitboard legality
      ├─ agent.choose_move(board, player, moves)
      │   └─ FastMCTSAgent.think()
      │       └─ expand() → sim_board.place_piece() → get_legal_moves() again
      └─ game.make_move(move)
          └─ board.place_piece()
              ├─ can_place_piece() → _check_adjacency_rules_fast()
              ├─ coords_to_mask() → bitboard update
              └─ update_frontier_after_move()
```

### MCTS Hot Path (per `think()` call)
```
think(board, player, legal_moves, time_budget)
  └─ for iteration in range(max_iterations):
      ├─ select() → UCB1 child selection
      ├─ expand() → FastMCTSNode creation
      └─ _fast_rollout()
          ├─ board.copy()                ← COPIES PER ITERATION
          ├─ get_legal_moves()           ← CALLED PER ROLLOUT STEP
          ├─ _quick_move_evaluation()
          │   └─ sorted(legal_moves)     ← FULL SORT PER STEP
          └─ board.place_piece()
```

### Call Volume (from cProfile, 100 calls to `get_legal_moves`):
- `is_placement_legal_bitboard_coords`: 136,300 calls
- `coords_to_mask`: 327,300 calls (3× per legality check)
- `coord_to_bit`: 2,664,400 calls
- `coord_to_index`: 2,664,400 calls

---

## 4. Bottleneck Findings

### Finding 1: `is_placement_legal_bitboard_coords` Rebuilds Masks Every Call (CRITICAL)

**File**: `engine/move_generator.py:676-752`
**Evidence**: 82% of move generation time, 2.66M function calls per 100 movegen calls

The function receives `placement_coords` and rebuilds `shape_mask`, `diag_mask`, and `orth_mask` from scratch:

```python
# Line 705: Rebuilds shape mask from coords
shape_mask = coords_to_mask(placement_coords)

# Lines 712-729: Rebuilds diag and orth neighbor sets from scratch
placement_set = set(placement_coords)
diag_neighbors = set()
orth_neighbors = set()
for (r, c) in placement_coords:
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < board.SIZE and 0 <= cc < board.SIZE:
            if (rr, cc) not in placement_set:
                diag_neighbors.add((rr, cc))
    # ... same for orthogonal

diag_mask = coords_to_mask(diag_neighbors)  # Line 731
orth_mask = coords_to_mask(orth_neighbors)  # Line 732
```

But `PieceOrientation` already has precomputed `shape_mask`, `diag_mask`, and `orth_mask` (computed in `pieces.py:192-227`). The caller in `_get_legal_moves_frontier` (line 390) has the `piece_orientation` object available but doesn't pass its masks.

**The fix**: Pass `PieceOrientation` into the legality check, compute the shift offset once, and use `shift_mask(precomputed_mask, d_row, d_col)` instead of rebuilding from coordinates.

However, `shift_mask()` itself is also slow (see Finding 2).

---

### Finding 2: `shift_mask()` Does Round-Trip Coordinate Conversion (HIGH)

**File**: `engine/bitboard.py:100-145`

```python
def shift_mask(mask, d_row, d_col, strict=True):
    coords = mask_to_coords(mask)     # mask → coords (slow iteration)
    shifted_coords = []
    for row, col in coords:
        new_row = row + d_row
        new_col = col + d_col
        # ... bounds checks ...
        shifted_coords.append((new_row, new_col))
    return coords_to_mask(shifted_coords)  # coords → mask (slow)
```

This converts mask→coords→shifted_coords→mask. For a 20×20 board, `mask_to_coords` iterates up to 400 bits even for sparse masks.

**The fix**: Implement shift as pure bit operations:
```python
def shift_mask_fast(mask, d_row, d_col):
    bit_shift = d_row * BOARD_WIDTH + d_col
    if bit_shift >= 0:
        shifted = mask << bit_shift
    else:
        shifted = mask >> (-bit_shift)
    # Apply row-wrapping guard mask
    return shifted & VALID_BOARD_MASK
```

The row-wrapping guard requires a precomputed mask that prevents bits from wrapping across row boundaries. This is the key complexity, but it's a one-time precomputation.

---

### Finding 3: `coords_to_mask` / `coord_to_bit` Function Call Overhead (HIGH)

**File**: `engine/bitboard.py:62-75, 47-59`

Each call to `coords_to_mask` calls `coord_to_bit` per coordinate, which calls `coord_to_index`. For a 5-cell piece with ~20 neighbors, that's ~25 function calls. With 136K legality checks, that's 3.4M function calls.

**Evidence from profile**: `coord_to_bit` accounts for 0.849s and `coord_to_index` for 0.268s out of 3.844s total.

**The fix**: Inline the computation or use a lookup table:
```python
# Precomputed: BIT_TABLE[row][col] = 1 << (row * 20 + col)
BIT_TABLE = [[1 << (r * 20 + c) for c in range(20)] for r in range(20)]
```

---

### Finding 4: Default Config Uses Slower Bitboard Path (HIGH)

**File**: `engine/move_generator.py:69`

`USE_BITBOARD_LEGALITY` defaults to `True`, but benchmark data shows frontier+grid is 2-3x faster. Simply setting `USE_BITBOARD_LEGALITY=False` would cut move generation time by 50-70% immediately with zero code changes.

**Root cause**: The bitboard legality was intended to be faster via bitwise ops, but the current implementation rebuilds masks from coordinates in Python loops, negating the benefit.

---

### Finding 5: `placement_coords` List Comprehension in Inner Loop (MEDIUM)

**File**: `engine/move_generator.py:382-385`

```python
placement_coords = [
    (anchor_row + rel_r, anchor_col + rel_c)
    for rel_r, rel_c in relative_positions
]
```

This creates a new list of tuples for every (frontier_cell × anchor_index) combination. With 136K legality checks, that's 136K temporary lists.

**The fix**: If using precomputed shifted masks, this list becomes unnecessary entirely.

---

### Finding 6: Board.copy() Cost in MCTS (MEDIUM)

**File**: `engine/board.py:634-648`

```python
def copy(self):
    new_board = Board()                  # Creates new Board with __init__
    new_board.grid = self.grid.copy()    # 400-element numpy copy
    new_board.player_pieces_used = {k: v.copy() for k, v in ...}  # 4 set copies
    new_board.player_frontiers = {k: v.copy() for k, v in ...}    # 4 set copies (large)
    ...
```

Each copy allocates a new Board (which runs `__init__`, creating fresh dicts, numpy array, and calling `init_frontiers()`), then overwrites everything. The `Board.__init__` work is wasted.

**The fix**: Use `__new__` to skip `__init__`, or implement copy-on-write for frontiers.

---

### Finding 7: `LegalMoveGenerator` Reinstantiation (MEDIUM)

**File**: `mcts/mcts_agent.py:57, 148`; `analytics/tournament/arena_runner.py:611`

`LegalMoveGenerator()` is created fresh in many places. Each instantiation runs `_cache_piece_orientations()` which iterates all 21 pieces, computes rotations/reflections, and builds position caches. This is O(21 pieces × ~8 orientations × shape computation).

**The fix**: Use a module-level singleton or pass the generator through the call chain.

---

### Finding 8: `list(Player)` Created on Every Player Rotation (LOW)

**File**: `engine/board.py:549-551`

```python
def _update_current_player(self):
    players = list(Player)
    current_index = players.index(self.current_player)
    self.current_player = players[(current_index + 1) % len(players)]
```

Called on every move. `list(Player)` creates a new list from the enum each time.

Same pattern in `mcts/mcts_agent.py:165-168, 519-521`.

**The fix**: `_PLAYERS = list(Player)` at module level, or just use modular arithmetic on player values.

---

### Finding 9: `get_frontier()` Returns a Copy (LOW)

**File**: `engine/board.py:254`

```python
def get_frontier(self, player):
    return self.player_frontiers[player].copy()
```

Called once per `_get_legal_moves_frontier` call. The copy is O(frontier_size) and creates a new set. The caller only iterates over it and never modifies it.

**The fix**: Return the set directly (callers don't mutate it) or document that it's read-only.

---

### Finding 10: `time.perf_counter()` Calls in Move Generation (LOW)

**File**: `engine/move_generator.py:272, 297, 531-532`

Every call to `_get_legal_moves_frontier` makes 2 + (2 × num_pieces) calls to `time.perf_counter()` for per-piece timing. This is ~40 syscalls per move generation call even when debug logging is disabled.

**The fix**: Guard the timing with `if MOVEGEN_DEBUG:` checks, not just the logging.

---

## 5. Prioritized Recommendations

| Priority | Area | Bottleneck | Evidence | Recommendation | Expected Impact | Complexity | How to Benchmark |
|---|---|---|---|---|---|---|---|
| **P0** | Move Gen | `is_placement_legal_bitboard_coords` rebuilds masks from coords | 82% of movegen time, 2.66M function calls per 100 calls | Use precomputed `PieceOrientation` masks + fast bit-shift instead of coord→mask conversion | **3-5x movegen speedup** | Medium - need correct shift with row-wrap guards | `benchmark_move_generation.py` - compare frontier+bitboard before/after |
| **P0-quick** | Move Gen | Default uses slower bitboard path | Benchmark shows grid is 2-3x faster | Set `USE_BITBOARD_LEGALITY=False` as immediate fix | **2-3x movegen speedup** (instant, zero risk) | Trivial - one line change | `benchmark_move_generation.py` |
| **P1** | Bitboard | `shift_mask()` round-trips through coords | Called in `is_placement_legal_bitboard` path | Implement pure bitwise shift with row-boundary guard mask | **Enables P0 fix to work at full speed** | Medium - need row-wrapping correctness | New: `benchmark_shift_mask.py` |
| **P1** | Bitboard | `coords_to_mask`/`coord_to_bit` function call overhead | 0.849s + 0.268s = 1.1s out of 3.8s | Precompute `BIT_TABLE[r][c]` lookup, inline mask building | **30% reduction in legality check time** | Low | Profile: count of coord_to_bit calls should → 0 |
| **P2** | Board | `Board.copy()` does wasted `__init__` work | Every MCTS copy creates/discards fresh dicts | Use `__new__` + direct attribute assignment | **15-25% MCTS speedup** | Low | New: `benchmark_board_copy.py` |
| **P2** | MCTS | `LegalMoveGenerator()` reinstantiation | Each creates piece caches from scratch | Module-level singleton or dependency injection | **Removes redundant startup cost** | Low | Profile: check _cache_piece_orientations calls |
| **P3** | Move Gen | `placement_coords` list comprehension per anchor | 136K temporary list allocations | Eliminated by P0 fix (use precomputed masks) | Subsumed by P0 | N/A | N/A |
| **P3** | Board | `get_frontier()` returns defensive copy | Unnecessary copy per movegen call | Return set directly, document read-only contract | **1-2% movegen speedup** | Trivial | `benchmark_move_generation.py` |
| **P3** | Move Gen | `time.perf_counter()` in non-debug path | ~40 syscalls per movegen call | Guard timing with `if MOVEGEN_DEBUG:` | **1-3% movegen speedup** | Trivial | `benchmark_move_generation.py` |
| **P3** | Board | `list(Player)` allocation per move | New list creation per player rotation | Module-level constant | **<1% overall** | Trivial | Not worth benchmarking alone |

---

## 6. Benchmark Validation Plan

### For P0-quick (Set `USE_BITBOARD_LEGALITY=False`)

- **Benchmark**: `benchmarks/benchmark_move_generation.py`
- **Metric**: Frontier+Bitboard column should match Frontier+Grid column (same path)
- **Validation**: Run before/after, expect 2-3x improvement in default path
- **Risk**: None - grid path is already tested and was the original implementation

### For P0 (Precomputed Mask Shift)

- **Benchmark**: `benchmarks/benchmark_move_generation.py`
- **Metric**: Frontier+Bitboard time should drop below Frontier+Grid
- **New benchmark needed**: Isolated `is_placement_legal_bitboard_coords` microbenchmark
  - Setup: Generate 1000 random (board, placement_coords) pairs
  - Measure: Time per call before/after
  - Target: <1µs per call (currently ~23µs)
- **Correctness**: Run existing test suite to verify no regressions. Also run equivalence test comparing grid vs bitboard results.

### For P1 (Fast `shift_mask`)

- **New benchmark**: `benchmarks/benchmark_bitboard_ops.py`
  - Measure `shift_mask` vs `shift_mask_fast` for various piece sizes and shifts
  - Target: 10x speedup (eliminate coord round-trip)

### For P2 (Board.copy)

- **New benchmark**: `benchmarks/benchmark_board_copy.py`
  - Setup: Board at various game stages (10, 20, 30 moves)
  - Measure: Time per `board.copy()` call, 10K iterations
  - Target: 2x improvement

### New Benchmarks to Add

1. **`benchmarks/benchmark_legality_check.py`**: Isolated legality check throughput
2. **`benchmarks/benchmark_board_copy.py`**: Board copy cost at various game stages
3. **`benchmarks/benchmark_mcts_throughput.py`**: MCTS nodes/second with fixed time budget

---

## 7. Recommended Next Steps

### Top 3 Next Actions

1. **Immediate (5 min)**: Change `USE_BITBOARD_LEGALITY` default to `False` in `move_generator.py:69`. This is a one-line change that delivers 2-3x movegen speedup with zero risk. The grid path is the original, well-tested implementation. Run `benchmark_move_generation.py` to confirm.

2. **Short-term (1-2 days)**: Rewrite `is_placement_legal_bitboard_coords` to use precomputed masks from `PieceOrientation`. Change the caller in `_get_legal_moves_frontier` to pass the `PieceOrientation` object and compute the shift once: `d_row = anchor_row, d_col = anchor_col` (since offsets are normalized from (0,0)). Implement `shift_mask_fast` using pure bit operations with row-boundary guards. Target: bitboard path faster than grid path.

3. **Medium-term (1 day)**: Optimize `Board.copy()` by using `__new__` to skip `__init__`, and make `LegalMoveGenerator` a singleton. These compound with the movegen fix for MCTS workloads.

### Top 3 Benchmarks to Trust

1. `benchmarks/benchmark_move_generation.py` — directly measures the #1 bottleneck
2. `tests/performance_test.py` — end-to-end game-level timing
3. `benchmarks/bench_selfplay_league.py` — realistic training throughput (when infra available)

### Top 3 New Benchmarks to Add

1. **Legality check microbenchmark** — isolate `is_placement_legal_*` functions
2. **Board copy benchmark** — measure copy cost at various game stages
3. **MCTS node throughput** — measure iterations/second in realistic search

### Code Areas NOT to Touch Yet

- **Frontier computation** (`update_frontier_after_move`): Already incremental, not the bottleneck
- **Piece orientation generation** (`pieces.py`): One-time cost, not in hot path
- **Arena runner** (`arena_runner.py`): Orchestration overhead is negligible vs movegen
- **Agent decision logic** (`fast_mcts_agent.py`): Optimize after movegen is fixed
- **Training pipeline** (`envs/`, `training/`): Archived to separate branch (see Appendix B)

---

## Appendix A: Profile Data

### cProfile Output (100 calls to `get_legal_moves`, mid-game state)

```
9,661,501 function calls in 3.844 seconds

ncalls   tottime  cumtime  function
136300   1.423    3.163    is_placement_legal_bitboard_coords
327300   0.679    1.529    coords_to_mask
2664400  0.581    0.849    coord_to_bit
2664400  0.268    0.268    coord_to_index
138000   0.087    0.238    all (bounds check)
2370900  0.179    0.179    set.add
779600   0.152    0.152    genexpr (bounds check)
138000   0.086    0.086    listcomp (placement_coords)
```

### Benchmark Output (benchmark_move_generation.py)

```
State                  Naive      Frontier+Grid   Frontier+Bitboard  Speedup
Early game (5 moves)   175.09ms   2.67ms          6.32ms             27.72x
Early game (10 moves)  126.58ms   7.10ms          15.22ms            8.32x
Mid game (20 moves)    63.39ms    4.80ms          15.01ms            4.22x
Mid game (30 moves)    33.45ms    2.83ms          9.45ms             3.54x
Late game (40 moves)   12.19ms    3.44ms          11.29ms            1.08x
```

The "Speedup" column is frontier+grid vs naive. Note that frontier+bitboard is consistently slower than frontier+grid.

---

## Appendix B: Training Pipeline Bottlenecks (Archived Code)

The RL training code (`envs/blokus_v0.py`, `training/`) was archived to a separate git branch (commit `cc106db`). While not in the current working tree, these bottlenecks will matter when training is re-enabled.

### Observation Generation (CRITICAL when training)

`BlokusEnv._get_observation()` allocates a `(30, 20, 20)` float32 array (48KB) **every step** and populates it with cell-by-cell Python loops:

```python
# Current: O(400) Position object allocations + grid lookups
for row in range(20):
    for col in range(20):
        cell_value = board.get_cell(Position(row, col))
        obs[cell_value if cell_value != 0 else 0, row, col] = 1
```

**Fix**: Vectorize with NumPy: `obs[0] = (grid == 0).astype(np.float32)` etc. Eliminates 400 Position allocations and method calls per step.

Additionally, unused piece channels fill entire `(20,20)` planes:
```python
for i, piece_id in enumerate(range(1, 22)):
    if piece_id not in used_pieces:
        obs[5+i, :, :] = 1  # 400 float writes per unused piece
```

### Action Masking (SIGNIFICANT when training)

- Action space: `Discrete(36400)` — maps `(piece_id × orientations × row × col)`
- Mask: `np.zeros(36400, dtype=bool)` allocated every step (36KB)
- Legal moves mapped to action IDs via dict lookup per move
- When no legal moves exist, a fallback sets `mask[0] = True` which is semantically wrong

### VecEnv Overhead

- `ActionMasker` applied at single-env level, not vectorized across environments
- Each of N environments generates observations and masks independently
- No batched observation construction across environments
