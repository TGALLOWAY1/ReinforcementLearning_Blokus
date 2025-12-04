# Move Generation Notes

## Current Move Generation (Pre-M6)

### Overview

The legal move generation for Blokus is implemented in `engine/move_generator.py` in the `LegalMoveGenerator` class. The main entry point is the `get_legal_moves()` method.

### Main Function

**`LegalMoveGenerator.get_legal_moves(board: Board, player: Player) -> List[Move]`**

This is the single top-level function used by the rest of the codebase to generate all legal moves for a given player on the current board state.

### Algorithm Description

The move generation algorithm follows this high-level flow:

1. **Filter available pieces**: Get all pieces that haven't been used yet by the player
   - Iterates through all 21 pieces and filters out those in `board.player_pieces_used[player]`

2. **For each available piece**:
   - Get all orientations (rotations and reflections) from cache
   - Get cached relative positions for each orientation
   
3. **For each orientation**:
   - Get all valid anchor positions (positions where the piece can be placed without going out of bounds)
   - Uses `PiecePlacement.get_valid_anchor_positions()` which scans the board dimensions

4. **For each anchor position**:
   - **Fast rejection checks** (in order):
     - Bounds check: Verify all piece positions are within board bounds
     - Overlap check: Verify all piece positions are empty (direct grid access: `grid[r, c] != 0`)
     - First move check: If first move, verify piece covers player's start corner
   
   - **Adjacency validation** (if passes fast checks):
     - Calls `_check_adjacency_fast_inline()` which:
       - Checks edge adjacency: No edge-to-edge contact with same color pieces
       - Checks corner adjacency: Must have at least one corner connection with same color (if not first move)
       - Uses direct grid access for all checks

5. **If all checks pass**: Create a `Move` object and add to legal moves list

### Key Implementation Details

- **Caching**: Piece orientations and relative positions are pre-computed and cached in `_cache_piece_orientations()`
- **Direct grid access**: Uses `board.grid` directly (numpy array) instead of Position objects for performance
- **Early exit**: Multiple early exit points to avoid unnecessary computation
- **Nested loops**: The algorithm has nested loops:
  - Outer: Available pieces (up to 21)
  - Middle: Orientations per piece (varies, typically 1-8)
  - Inner: Anchor positions (can be hundreds per orientation on empty board)
  - Deepest: Piece positions (1-5 squares per piece)

### Special Cases Handled

1. **First move rule**: 
   - First move must cover the player's start corner
   - Start corners: RED=(0,0), BLUE=(0,19), YELLOW=(19,19), GREEN=(19,0)
   - Checked via `board.player_first_move[player]` flag

2. **End of game**:
   - When no legal moves are available, `get_legal_moves()` returns an empty list
   - Game over detection is handled separately in `game.py` by checking all players

3. **Piece exhaustion**:
   - Pieces already used are filtered out at the start
   - Tracked in `board.player_pieces_used[player]` (set of piece IDs)

### Performance Hotspots

Based on code analysis, the main performance hotspots are:

1. **Anchor position enumeration**: `PiecePlacement.get_valid_anchor_positions()` generates all possible anchor positions for each orientation, which can be O(board_size²) per orientation

2. **Nested iteration**: The triple-nested loop (pieces × orientations × anchors) creates a large search space
   - Early game: ~21 pieces × ~4 orientations × ~400 anchors = ~33,600 iterations
   - Late game: Fewer pieces, but more complex adjacency checks

3. **Adjacency checking**: `_check_adjacency_fast_inline()` is called for every candidate position that passes fast checks
   - Checks 4 edge neighbors and 4 corner neighbors for each square in the piece
   - For a 5-square piece, this is 5 × 8 = 40 grid lookups per candidate

4. **Grid scanning**: While not a full board scan, the algorithm does check many positions:
   - All anchor positions are enumerated
   - All positions within each piece are checked
   - All adjacent positions (edges and corners) are checked

### Current Timing Infrastructure

The code already includes some timing instrumentation:
- Uses `time.perf_counter()` for high-resolution timing
- Logs total time and per-piece timings at DEBUG level
- Logs slowest piece identification

### Dependencies

- `engine.board.Board`: Provides board state and grid access
- `engine.pieces.PieceGenerator`: Provides piece definitions and orientations
- `engine.pieces.PiecePlacement`: Provides anchor position calculation and position generation

### Usage in Codebase

The `get_legal_moves()` method is called from:
- `engine.game.BlokusGame.get_legal_moves()`: Main game interface
- `envs.blokus_v0.BlokusEnv._get_info()`: For action masking in RL environment
- `webapi.game_manager.GameManager.get_legal_moves()`: For web API
- Various agent implementations for move selection

### Debug Timing Hook

A debug timing hook has been added to measure move generation performance. It can be enabled by setting the environment variable `BLOKUS_MOVEGEN_DEBUG=1`.

When enabled, the hook logs timing information at INFO level:
- Format: `MoveGen: player=<name>, legal_moves=<n>, elapsed_ms=<x.xx>`
- Also logs slowest piece: `MoveGen: slowest piece=<id>, piece_time_ms=<x.xx>`

**Test Results (from quick self-play run):**
- First move (empty board): ~70-75ms, generates ~58 legal moves
- Early game moves: ~300-450ms, generates ~100-200 legal moves
- Mid-game moves: ~300-350ms, generates ~200-350 legal moves
- Piece 15 consistently appears as the slowest piece to process

The timing hook is designed to be minimally invasive - it only adds overhead when the debug flag is enabled, and uses the existing logging infrastructure.

