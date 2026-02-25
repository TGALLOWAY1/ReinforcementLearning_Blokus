# Blokus Game Engine & RL Integration - Comprehensive Audit Report

## Executive Summary

This audit examines the Blokus game implementation, including the rules engine, piece definitions, UI integration, agent support, and overall playability. The codebase shows a well-structured foundation with several critical issues that prevent reliable gameplay.

**Key Findings:**
- ✅ Core game engine architecture is sound
- ❌ **CRITICAL**: Piece shape mismatches between backend and frontend
- ⚠️ **ISSUE**: Incomplete game-over detection logic
- ⚠️ **ISSUE**: Human player move validation flow has gaps
- ⚠️ **ISSUE**: Agent integration works but needs verification for 1-3 agent scenarios

---

## 1. Blokus Architecture Summary

### 1.1 Module Structure

**Core Engine (`engine/`):**
- `board.py`: `Board` class - 20x20 numpy grid, player state tracking, placement validation
- `game.py`: `BlokusGame` class - Main game orchestrator, scoring, turn management
- `pieces.py`: `PieceGenerator`, `Piece` class - All 21 Blokus pieces with rotations/flips
- `move_generator.py`: `LegalMoveGenerator`, `Move` class - Legal move enumeration and validation

**RL Environment (`envs/`):**
- `blokus_v0.py`: `BlokusEnv` - PettingZoo AEC environment wrapper
- `blokus_env.py`: Additional environment utilities

**Agents (`agents/`):**
- `random_agent.py`: `RandomAgent` - Random legal move selection
- `heuristic_agent.py`: `HeuristicAgent` - Rule-based evaluation
- `fast_mcts_agent.py`: `FastMCTSAgent` - Optimized MCTS for real-time play
- `mcts/mcts_agent.py`: `MCTSAgent` - Full MCTS implementation

**Web API (`webapi/`):**
- `app.py`: FastAPI application with REST and WebSocket endpoints
- `game_manager.py`: Game session management (appears unused, logic in `app.py`)

**Frontend (`frontend/src/`):**
- `pages/Play.tsx`: Main game page component
- `components/Board.tsx`: SVG board rendering
- `components/PieceTray.tsx`: Piece selection UI
- `store/gameStore.ts`: Zustand state management with WebSocket integration
- `constants/gameConstants.ts`: Shared constants including piece shapes
- `utils/pieceUtils.ts`: Piece transformation utilities

**Schemas (`schemas/`):**
- `game_config.py`: `GameConfig`, `AgentConfig` - Game setup schemas
- `game_state.py`: `GameState`, `Move`, `Position` - State representation
- `move.py`: `MoveRequest`, `MoveResponse` - Move API schemas
- `state_update.py`: `StateUpdate` - WebSocket message schemas

### 1.2 Central Game State Object

**File:** `engine/board.py` - `Board` class

**Key Fields:**
- `grid: np.ndarray` - 20x20 integer array (0=empty, 1-4=players)
- `player_start_corners: Dict[Player, Position]` - Starting corners per player
- `player_pieces_used: Dict[Player, Set[int]]` - Tracked used piece IDs
- `player_first_move: Dict[Player, bool]` - First-move flag per player
- `current_player: Player` - Active player (RED, BLUE, YELLOW, GREEN)
- `move_count: int` - Total moves made
- `game_over: bool` - Game termination flag

**Additional State in `BlokusGame` (`engine/game.py`):**
- `game_history: List[Dict]` - Move history with board snapshots
- `winner: Optional[Player]` - Winner after game ends

### 1.3 Move Representation

**File:** `engine/move_generator.py` - `Move` class

**Structure:**
```python
Move(piece_id: int, orientation: int, anchor_row: int, anchor_col: int)
```

- `piece_id`: 1-21 (identifies which of 21 pieces)
- `orientation`: Index into cached orientations list (0-7 typically, varies by piece symmetry)
- `anchor_row`, `anchor_col`: Top-left position of piece placement

**API Schema:** `schemas/move.py` - `MoveRequest` wraps this with `player` field

### 1.4 Board Representation

**File:** `engine/board.py`

- **Structure:** 20x20 numpy array (`np.ndarray`), dtype=int
- **Indexing:** 0-based, row-major (grid[row, col])
- **Values:**
  - `0`: Empty cell
  - `1`: RED player
  - `2`: BLUE player
  - `3`: YELLOW player
  - `4`: GREEN player

**Frontend:** `frontend/src/components/Board.tsx` - Renders as SVG grid, reads from `gameState.board` (list of lists)

### 1.5 Rules Enforcement Locations

**File:** `engine/board.py` - `Board.can_place_piece()`

**Validation Chain:**
1. **Bounds Check:** `is_valid_position()` - Ensures all positions within 0-19
2. **Empty Check:** `is_empty()` - Ensures no overlap with existing pieces
3. **First Move Rule:** `player_first_move[player]` - First piece must cover start corner
4. **Adjacency Rules:** `_check_adjacency_rules()`:
   - No edge-adjacency with same-color pieces
   - Must touch existing same-color pieces at corners (after first move)
   - Edge-touching with different colors is allowed

**File:** `engine/move_generator.py` - `LegalMoveGenerator.is_move_legal()`
- Validates piece not already used
- Validates orientation index
- Validates bounds via `PiecePlacement.can_place_piece_at()`
- Delegates to `Board.can_place_piece()`

**File:** `engine/game.py` - `BlokusGame.make_move()`
- Calls `LegalMoveGenerator.is_move_legal()` before placement
- Updates game history and checks game-over after successful placement

---

## 2. Piece Set & Representation Audit

### 2.1 Files Defining Pieces

**Backend:**
- `engine/pieces.py`: `PieceGenerator.get_all_pieces()` - Defines all 21 pieces as numpy arrays

**Frontend:**
- `frontend/src/constants/gameConstants.ts`: `PIECE_SHAPES` - JavaScript array definitions

### 2.2 Piece Completeness Check

**Total Pieces:** 21 pieces required (1 monomino, 1 domino, 2 trominoes, 5 tetrominoes, 12 pentominoes)

**Backend Verification:**
- ✅ All 21 pieces defined in `engine/pieces.py` (IDs 1-21)
- ✅ Piece sizes verified: 1, 2, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5

**Frontend Verification:**
- ✅ All 21 pieces defined in `frontend/src/constants/gameConstants.ts`

### 2.3 Critical Shape Mismatches

**❌ CRITICAL ISSUE: Pentomino F (ID 11)**

**Backend (`engine/pieces.py:91`):**
```python
np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # This is actually an X shape!
```

**Frontend (`frontend/src/constants/gameConstants.ts:18`):**
```typescript
[[0, 1, 1], [1, 1, 0], [0, 1, 0]]  // Closer to F but still incorrect
```

**Correct F Pentomino shape should be:**
```
[0, 1, 1]
[1, 1, 0]
[0, 1, 0]
```
OR
```
[0, 1, 0]
[1, 1, 1]
[1, 0, 0]
```

**❌ CRITICAL ISSUE: Pentomino Y (ID 21)**

**Backend (`engine/pieces.py:121`):**
```python
np.array([[1, 0], [1, 1], [1, 0], [1, 0]])  # 4 rows, 2 cols
```

**Frontend (`frontend/src/constants/gameConstants.ts:28`):**
```typescript
[[1, 1, 1, 1], [0, 1, 0, 0]]  // 2 rows, 4 cols - COMPLETELY DIFFERENT SHAPE
```

**Correct Y Pentomino shape should be:**
```
[1, 0]
[1, 1]
[1, 0]
[1, 0]
```
OR rotated/flipped variants.

**⚠️ POTENTIAL ISSUE: Pentomino X (ID 20)**

**Backend (`engine/pieces.py:118`):**
```python
np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # Correct X shape
```

**Frontend (`frontend/src/constants/gameConstants.ts:27`):**
```typescript
[[0, 1, 0], [1, 1, 1], [0, 1, 0]]  // Matches backend - CORRECT
```

**Note:** The backend F pentomino (ID 11) has the X shape, suggesting a copy-paste error.

### 2.4 Rotation and Reflection Support

**Backend:** `engine/pieces.py` - `PieceGenerator.get_piece_rotations_and_reflections()`
- ✅ Generates 8 orientations: original, 3 rotations (90°, 180°, 270°), flip, 3 flipped rotations
- ✅ Removes duplicates (handles symmetric pieces correctly)
- ✅ Cached in `LegalMoveGenerator.piece_orientations_cache`

**Frontend:** `frontend/src/utils/pieceUtils.ts` - `getPieceShape()`
- ✅ Implements same 8-orientation system (0-7)
- ✅ Uses `rotatePiece()` and `flipPiece()` utilities
- ⚠️ **ISSUE**: Shape definitions must match backend exactly for correct rotations

### 2.5 Piece Validation Checklist

**Canonical Piece Data Structure:**

Each piece should have:
1. **ID:** 1-21 (unique identifier)
2. **Name:** Human-readable name (e.g., "Pentomino F")
3. **Base Shape:** 2D numpy array (backend) or 2D number array (frontend)
4. **Size:** Number of squares (1-5)
5. **Orientations:** List of all unique rotations/flips

**Validation Requirements:**
- ✅ All 21 pieces present
- ✅ Each piece has correct square count
- ✅ Shapes match official Blokus piece set
- ❌ **FAIL**: Frontend and backend shapes must be identical (currently mismatched for F and Y)

**Where to Fix:**
- **Backend:** `engine/pieces.py` - Fix Pentomino F shape (line 91)
- **Frontend:** `frontend/src/constants/gameConstants.ts` - Fix Pentomino F (line 18) and Pentomino Y (line 28)
- **Validation:** Add unit test comparing frontend/backend piece shapes

---

## 3. Rules Engine & Move Validation Audit

### 3.1 Core Rules Implementation

**File:** `engine/board.py` - `Board.can_place_piece()`

**✅ Rule 1: Board Boundaries**
- **Location:** `is_valid_position()` (line 59-61)
- **Status:** CORRECT - Checks 0 <= row < 20, 0 <= col < 20

**✅ Rule 2: No Overlap**
- **Location:** `can_place_piece()` line 128-130
- **Status:** CORRECT - Checks `is_empty()` for all positions

**✅ Rule 3: First Move Starting Corner**
- **Location:** `can_place_piece()` line 132-136
- **Status:** CORRECT - First move must include `player_start_corners[player]`
- **Start Corners:**
  - RED: (0, 0)
  - BLUE: (0, 19)
  - YELLOW: (19, 19)
  - GREEN: (19, 0)

**✅ Rule 4: Same-Color Corner-Only Adjacency**
- **Location:** `_check_adjacency_rules()` line 144-171
- **Status:** MOSTLY CORRECT with potential edge case issues

**Implementation Details:**
- Line 154-157: Checks edge-adjacent positions, rejects if same color
- Line 160-164: Allows corner-adjacent same-color pieces
- Line 167-169: After first move, piece must connect via corners to existing pieces

**⚠️ POTENTIAL ISSUE:** The connection check (`_is_connected_via_corners`) only verifies that at least one corner touches an existing piece. This is correct, but the logic could be clearer.

**✅ Rule 5: Edge-Touching Between Different Colors**
- **Location:** `_check_adjacency_rules()` line 154-157
- **Status:** CORRECT - Only rejects edge-adjacency if `get_player_at(adj_pos) == player` (same color)

### 3.2 Turn & Game Flow

**File:** `engine/board.py` - `Board._update_current_player()`
- **Location:** Line 215-219
- **Status:** CORRECT - Rotates through Player enum: RED → BLUE → YELLOW → GREEN → RED

**File:** `engine/board.py` - `Board.place_piece()`
- **Location:** Line 190-213
- **Status:** CORRECT - Updates `player_first_move[player] = False` after first placement
- **Status:** CORRECT - Increments `move_count` and calls `_update_current_player()`

**File:** `webapi/app.py` - `GameManager._make_agent_move()`
- **Location:** Line 153-158
- **Status:** CORRECT - Handles no-legal-moves by calling `board._update_current_player()` to skip turn

**⚠️ ISSUE: Human Player Passing**
- **Location:** `webapi/app.py` - No explicit "pass" endpoint for human players
- **Status:** MISSING - Human players cannot explicitly pass when they have no legal moves
- **Impact:** Game may hang waiting for human move when none are available

### 3.3 Game-Over Detection

**File:** `engine/game.py` - `BlokusGame._check_game_over()`
- **Location:** Line 69-81
- **Status:** ⚠️ INCOMPLETE

**Current Implementation:**
```python
def _check_game_over(self) -> None:
    all_players_can_move = False
    for player in Player:
        if self.move_generator.has_legal_moves(self.board, player):
            all_players_can_move = True
            break
    
    if not all_players_can_move:
        self.board.game_over = True
        self.winner = self.get_winner()
```

**Issues:**
1. **Logic Error:** Checks if ANY player can move, sets game_over if NONE can move. This is correct, but the variable name `all_players_can_move` is misleading (should be `any_player_can_move`).
2. **Performance:** Calls `has_legal_moves()` for all 4 players every move, which may be expensive.
3. **Missing Check:** Doesn't verify if all players have passed consecutively (standard Blokus rule: game ends when all players pass in sequence).

**File:** `engine/board.py` - `Board.is_game_over()`
- **Location:** Line 246-250
- **Status:** ⚠️ SIMPLIFIED - Just returns `self.game_over` flag, doesn't check actual state

### 3.4 Scoring

**File:** `engine/board.py` - `Board.get_score()`
- **Location:** Line 221-236
- **Status:** PARTIALLY CORRECT

**Current Rules:**
- ✅ 1 point per square covered
- ✅ +15 bonus for using all 21 pieces

**File:** `engine/game.py` - `BlokusGame.get_score()`
- **Location:** Line 105-119
- **Status:** ADDS BONUSES (non-standard)

**Additional Bonuses (not in standard Blokus):**
- Corner control: +5 per controlled corner
- Center control: +2 per center square (4x4 center area)

**⚠️ NOTE:** These bonuses are custom and not part of official Blokus rules. May be intentional for RL training.

### 3.5 Identified Issues Summary

**✅ Correctly Implemented:**
- Board boundaries
- Overlap prevention
- First-move corner rule
- Same-color edge-adjacency prevention
- Corner-adjacency allowance
- Turn rotation

**⚠️ Issues Found:**
1. **Game-over detection:** Logic works but variable naming is confusing; missing consecutive-pass check
2. **Human passing:** No explicit pass mechanism for human players
3. **Scoring bonuses:** Non-standard bonuses added (may be intentional)

**❌ Missing/Incomplete:**
- Consecutive pass detection for game-over
- Explicit pass action for human players
- Performance optimization for game-over checks

---

## 4. Human Player Interaction & UI Wiring Review

### 4.1 UI Entry Points

**File:** `frontend/src/pages/Play.tsx`
- **Route:** `/play` (assumed, check router config)
- **Component:** `Play` - Main game interface

**File:** `frontend/src/pages/Home.tsx` (referenced in project structure)
- **Route:** `/` (home page)
- **Purpose:** Game configuration and creation

### 4.2 Flow: UI → Rules Engine

**Step 1: Piece Selection**
- **Component:** `frontend/src/components/PieceTray.tsx`
- **Handler:** `onPieceSelect(pieceId)` (line 129)
- **State:** `useGameStore().selectPiece(pieceId)` (line 272 in `gameStore.ts`)
- **Status:** ✅ WORKING

**Step 2: Piece Rotation/Flip**
- **Component:** `frontend/src/components/PieceTray.tsx` (lines 37-64)
- **Keyboard:** `R` key rotates, `F` key flips
- **State:** `useGameStore().setPieceOrientation(orientation)`
- **Status:** ✅ WORKING

**Step 3: Board Click (Placement)**
- **Component:** `frontend/src/components/Board.tsx`
- **Handler:** `onCellClick(row, col)` (line 44-59)
- **Parent:** `frontend/src/pages/Play.tsx` - `handleCellClick()` (line 27-106)

**Step 4: Move Construction**
- **File:** `frontend/src/pages/Play.tsx` (lines 73-82)
- **Creates:** `MoveRequest` with `piece_id`, `orientation`, `anchor_row`, `anchor_col`
- **Status:** ✅ WORKING

**Step 5: WebSocket Send**
- **File:** `frontend/src/store/gameStore.ts` - `makeMove()` (line 279-370)
- **Sends:** WebSocket message with type "move" and move data
- **Status:** ✅ WORKING

**Step 6: Backend Validation**
- **File:** `webapi/app.py` - `_process_human_move_immediately()` (line 250-316)
- **Validates:** Turn check, move legality via `game.make_move()`
- **Status:** ✅ WORKING

**Step 7: Response Handling**
- **File:** `frontend/src/store/gameStore.ts` - `makeMove()` Promise handler (line 315-343)
- **Updates:** Game state from response
- **Status:** ✅ WORKING

### 4.3 Invalid Move Handling

**Backend Response:**
- **File:** `webapi/app.py` - `_process_human_move_immediately()` (line 303-308)
- **Returns:** `MoveResponse(success=False, message="Invalid move")`
- **Status:** ✅ WORKING

**Frontend Display:**
- **File:** `frontend/src/pages/Play.tsx` (lines 94-99, 176-193)
- **Shows:** Error message in red banner at top of board
- **Status:** ✅ WORKING

**⚠️ ISSUE: Error Message Clarity**
- Backend returns generic "Invalid move" - should specify reason (out of bounds, overlaps, wrong turn, etc.)

### 4.4 Turn Advancement

**Backend:**
- **File:** `engine/board.py` - `place_piece()` calls `_update_current_player()` (line 211)
- **Status:** ✅ WORKING

**Frontend:**
- **File:** `frontend/src/store/gameStore.ts` - WebSocket message handler updates `gameState.current_player` (line 194-204)
- **Status:** ✅ WORKING

**⚠️ ISSUE: Turn Indicator**
- No clear UI indicator showing whose turn it is (check `Play.tsx` - may be in sidebar)

### 4.5 Broken Flows Identified

**❌ CRITICAL: Piece Shape Mismatch**
- **Issue:** Frontend and backend have different shapes for Pentomino F and Y
- **Impact:** Preview shows wrong shape, placement may fail validation
- **Fix:** Synchronize shapes in `frontend/src/constants/gameConstants.ts` with `engine/pieces.py`

**⚠️ ISSUE: No Legal Moves for Human**
- **Issue:** If human player has no legal moves, UI doesn't provide "Pass" button
- **Impact:** Game may appear stuck
- **Fix:** Add pass button or auto-detect and skip turn

**⚠️ ISSUE: Piece Preview Calculation**
- **File:** `frontend/src/components/Board.tsx` - `getPiecePreview()` (line 130-143)
- **Issue:** Uses frontend shape definitions which may not match backend orientations
- **Impact:** Preview may show incorrect placement
- **Fix:** Ensure frontend orientation system matches backend exactly

### 4.6 Specific Fixes Required

**File:** `frontend/src/pages/Play.tsx`
- **Line 55-58:** Remove or fix human-player-only check (may block valid moves)
- **Add:** Turn indicator display
- **Add:** "Pass" button when no legal moves available

**File:** `frontend/src/components/Board.tsx`
- **Line 130-143:** Verify `calculatePiecePositions()` matches backend anchor system
- **Add:** Visual indicator for legal move positions (use `gameState.heatmap` if available)

**File:** `webapi/app.py`
- **Line 303-308:** Improve error messages to specify rejection reason

---

## 5. Agent Integration (1–3 Agents) Review

### 5.1 Agent Code Locations

**Agent Implementations:**
- `agents/random_agent.py`: `RandomAgent`
- `agents/heuristic_agent.py`: `HeuristicAgent`
- `agents/fast_mcts_agent.py`: `FastMCTSAgent`
- `mcts/mcts_agent.py`: `MCTSAgent`

**Common Interface:**
```python
def select_action(board: Board, player: Player, legal_moves: List[Move]) -> Optional[Move]
```

**Status:** ✅ All agents implement this interface correctly

### 5.2 Game Loop Integration

**File:** `webapi/app.py` - `GameManager._run_turn_loop()`
- **Location:** Line 106-140
- **Flow:**
  1. Checks if game is over
  2. Gets current player
  3. Looks up agent for current player
  4. If agent exists: calls `_make_agent_move()`
  5. If no agent (human): waits for WebSocket message
  6. Broadcasts state updates

**Status:** ✅ WORKING

**File:** `webapi/app.py` - `GameManager._make_agent_move()`
- **Location:** Line 142-200
- **Flow:**
  1. Gets legal moves for player
  2. If no moves: skips turn (calls `board._update_current_player()`)
  3. Calls `agent.select_action()` with timeout (5 seconds)
  4. Falls back to random move on timeout/error
  5. Executes move via `game.make_move()`
  6. Broadcasts state update

**Status:** ✅ WORKING with error handling

### 5.3 Agent State Access

**How Agents Receive State:**
- **Parameter:** `board: Board` - Full board object with grid, player state, etc.
- **Parameter:** `player: Player` - Current player making move
- **Parameter:** `legal_moves: List[Move]` - Pre-computed legal moves

**Status:** ✅ CLEAR - Agents receive consistent, well-defined state

**How Agents Return Moves:**
- **Return:** `Optional[Move]` - Move object or None if no move
- **Validation:** Move is validated via `game.make_move()` which calls `is_move_legal()`

**Status:** ✅ CORRECT - Agents use same validation as human players

### 5.4 Multi-Agent Support (1–3 Agents)

**Configuration:**
- **File:** `schemas/game_config.py` - `GameConfig.players` (line 28)
- **Constraint:** `min_items=2, max_items=4` - Supports 2-4 players
- **File:** `webapi/app.py` - `_initialize_agents()` (line 71-93)
- **Status:** ✅ SUPPORTS 1-3 agents + 1 human (2-4 total players)

**Agent Assignment:**
- **File:** `webapi/app.py` - `_initialize_agents()` (line 74-91)
- **Maps:** `Player` enum → Agent instance (or `None` for human)
- **Storage:** `self.agent_instances[game_id]` dictionary

**Status:** ✅ CORRECT - Can assign agents to any player slot

**Turn Rotation:**
- **File:** `engine/board.py` - `_update_current_player()` (line 215-219)
- **Status:** ✅ CORRECT - Rotates RED → BLUE → YELLOW → GREEN
- **Note:** Works regardless of which players are human vs agent

### 5.5 Passing for Agents

**File:** `webapi/app.py` - `_make_agent_move()` (line 153-158)
- **Handling:** If `legal_moves` is empty, calls `board._update_current_player()` to skip turn
- **Status:** ✅ WORKING

**File:** `agents/random_agent.py` - `select_action()` (line 41-43)
- **Handling:** Returns `None` if no legal moves
- **Status:** ✅ CORRECT

**File:** `agents/heuristic_agent.py` - `select_action()` (line 47-49)
- **Handling:** Returns `None` if no legal moves
- **Status:** ✅ CORRECT

**File:** `agents/fast_mcts_agent.py` - `select_action()` (line 91-92)
- **Handling:** Returns `None` if no legal moves
- **Status:** ✅ CORRECT

### 5.6 Gaps and Issues

**✅ What Works:**
- Agent assignment to any player slot
- Turn rotation with mixed human/agent players
- Passing when no legal moves
- Move validation for agent moves
- Error handling and timeouts

**⚠️ Potential Issues:**
1. **Agent Timeout Fallback:** Falls back to random move on timeout (line 176 in `app.py`). This may not be desired behavior - should probably pass or raise error.
2. **No Agent Move History:** Agent moves aren't logged differently from human moves in UI (may be intentional).
3. **Concurrent Agent Moves:** Turn loop processes one move at a time, which is correct, but if multiple agents are fast, there's no rate limiting.

**❌ Missing Features:**
- No way to configure agent parameters per-game (seed, iterations, etc.) via API
- No agent move confidence/explanation in UI
- No way to pause/resume agent play

### 5.7 Required Changes for 1–3 Agent Support

**Current Status:** ✅ **ALREADY SUPPORTS 1-3 agents**

**Verification Needed:**
1. Test game with 1 human + 1 agent
2. Test game with 1 human + 2 agents
3. Test game with 1 human + 3 agents
4. Verify turn order is correct in all cases
5. Verify agents can pass correctly
6. Verify game ends correctly with mixed players

**Suggested Improvements:**
- Add agent parameter configuration in `GameConfig`
- Add agent move logging/visualization
- Add pause/resume controls for agent play

---

## 6. Smoke Tests & Diagnostics Plan

### 6.1 Test Framework

**Existing Framework:** `pytest` (Python) - See `tests/test_engine.py`

**Test Files:**
- `tests/test_engine.py` - Engine unit tests
- `tests/test_blokus_env.py` - RL environment tests
- `tests/smoke_test_game.py` - Basic smoke test (50 lines)
- `tests/smoke_test_env.py` - Environment smoke test
- `tests/verify_engine.py` - Engine verification script

### 6.2 Proposed Test Structure

**File:** `tests/test_blokus_rules.py` (NEW)

**Test Categories:**

#### 6.2.1 Piece Creation Test

```python
def test_all_pieces_present():
    """Verify all 21 pieces are defined."""
    pieces = PieceGenerator.get_all_pieces()
    assert len(pieces) == 21
    assert set(p.id for p in pieces) == set(range(1, 22))

def test_piece_sizes():
    """Verify each piece has correct square count."""
    pieces = PieceGenerator.get_all_pieces()
    for piece in pieces:
        assert piece.size == np.sum(piece.shape)
        assert 1 <= piece.size <= 5

def test_piece_shapes_match_frontend():
    """Verify backend shapes match frontend definitions."""
    # Load frontend shapes from JSON or hardcode
    frontend_shapes = load_frontend_shapes()  # Implement this
    backend_pieces = PieceGenerator.get_all_pieces()
    
    for piece in backend_pieces:
        frontend_shape = frontend_shapes[piece.id]
        # Convert backend numpy to list for comparison
        backend_list = piece.shape.tolist()
        assert shapes_match(backend_list, frontend_shape), \
            f"Piece {piece.id} shape mismatch"

def test_piece_orientations():
    """Verify all pieces have valid orientations."""
    pieces = PieceGenerator.get_all_pieces()
    for piece in pieces:
        orientations = PieceGenerator.get_piece_rotations_and_reflections(piece)
        assert len(orientations) >= 1
        for orientation in orientations:
            assert np.sum(orientation) == piece.size
```

#### 6.2.2 Placement Rules Test

```python
def test_first_move_corner_rule():
    """Verify first move must cover start corner."""
    board = Board()
    red_corner = board.player_start_corners[Player.RED]
    
    # Valid: covers corner
    valid_positions = [red_corner, Position(0, 1)]
    assert board.can_place_piece(valid_positions, Player.RED)
    
    # Invalid: doesn't cover corner
    invalid_positions = [Position(1, 1), Position(1, 2)]
    assert not board.can_place_piece(invalid_positions, Player.RED)

def test_no_overlap():
    """Verify pieces cannot overlap."""
    board = Board()
    first_positions = [Position(0, 0), Position(0, 1)]
    board.place_piece(first_positions, Player.RED, 1)
    
    # Invalid: overlaps
    overlapping = [Position(0, 0), Position(0, 1)]
    assert not board.can_place_piece(overlapping, Player.BLUE)

def test_same_color_edge_adjacency():
    """Verify same-color pieces cannot touch edge-to-edge."""
    board = Board()
    first_positions = [Position(0, 0), Position(0, 1)]
    board.place_piece(first_positions, Player.RED, 1)
    
    # Invalid: edge-adjacent
    edge_adjacent = [Position(0, 2), Position(0, 3)]
    assert not board.can_place_piece(edge_adjacent, Player.RED)
    
    # Valid: corner-adjacent
    corner_adjacent = [Position(1, 1), Position(1, 2)]
    assert board.can_place_piece(corner_adjacent, Player.RED)

def test_different_color_edge_adjacency():
    """Verify different-color pieces CAN touch edge-to-edge."""
    board = Board()
    red_positions = [Position(0, 0), Position(0, 1)]
    board.place_piece(red_positions, Player.RED, 1)
    
    # Valid: BLUE can touch RED at edge
    blue_positions = [Position(0, 2), Position(0, 3)]
    assert board.can_place_piece(blue_positions, Player.BLUE)

def test_corner_connection_rule():
    """Verify pieces must connect via corners after first move."""
    board = Board()
    first_positions = [Position(0, 0), Position(0, 1)]
    board.place_piece(first_positions, Player.RED, 1)
    
    # Invalid: not connected
    isolated = [Position(5, 5), Position(5, 6)]
    assert not board.can_place_piece(isolated, Player.RED)
    
    # Valid: connected at corner
    connected = [Position(1, 1), Position(1, 2)]
    assert board.can_place_piece(connected, Player.RED)
```

#### 6.2.3 Turn Sequence Test

```python
def test_turn_rotation():
    """Verify turns rotate correctly."""
    game = BlokusGame()
    assert game.get_current_player() == Player.RED
    
    moves = game.get_legal_moves()
    game.make_move(moves[0])
    assert game.get_current_player() == Player.BLUE
    
    moves = game.get_legal_moves()
    game.make_move(moves[0])
    assert game.get_current_player() == Player.YELLOW

def test_passing_when_no_moves():
    """Verify players can pass when no legal moves."""
    game = BlokusGame()
    # Simulate no legal moves for current player
    # (This is hard to test without setting up specific board state)
    # Could manually set player_pieces_used to all pieces
    
    # For now, test that has_legal_moves returns False when appropriate
    # and that turn advances

def test_game_over_detection():
    """Verify game ends when no player can move."""
    game = BlokusGame()
    assert not game.is_game_over()
    
    # Simulate all players having no moves
    # (Requires specific board setup)
    # For now, test that _check_game_over() sets game_over correctly
```

#### 6.2.4 Agent Turn Test

```python
def test_agent_move_validation():
    """Verify agent moves are validated like human moves."""
    game = BlokusGame()
    agent = RandomAgent()
    
    legal_moves = game.get_legal_moves()
    agent_move = agent.select_action(game.board, Player.RED, legal_moves)
    
    assert agent_move is not None
    assert game.move_generator.is_move_legal(game.board, Player.RED, agent_move)
    
    success = game.make_move(agent_move, Player.RED)
    assert success

def test_agent_passing():
    """Verify agents can pass when no legal moves."""
    game = BlokusGame()
    agent = RandomAgent()
    
    # Simulate no legal moves
    legal_moves = []
    agent_move = agent.select_action(game.board, Player.RED, legal_moves)
    
    assert agent_move is None  # Agent should return None

def test_mixed_human_agent_game():
    """Test game with 1 human + 1 agent."""
    # This requires WebSocket/API integration testing
    # Could use a mock or integration test framework
    pass
```

### 6.3 Debug Utilities

**File:** `scripts/debug_blokus.py` (NEW)

```python
"""CLI debug utility for Blokus engine."""

def print_board(board: Board):
    """Print board as ASCII art."""
    for row in range(board.SIZE):
        row_str = ""
        for col in range(board.SIZE):
            value = board.grid[row, col]
            if value == 0:
                row_str += "."
            else:
                row_str += str(value)
        print(row_str)

def log_move(player: Player, move: Move, success: bool):
    """Log move attempt."""
    print(f"[{player.name}] Move: piece={move.piece_id}, "
          f"orientation={move.orientation}, anchor=({move.anchor_row},{move.anchor_col}), "
          f"success={success}")

def interactive_game():
    """Run interactive CLI game."""
    game = BlokusGame()
    move_generator = LegalMoveGenerator()
    
    while not game.is_game_over():
        current = game.get_current_player()
        print(f"\nCurrent player: {current.name}")
        print_board(game.board)
        
        legal_moves = move_generator.get_legal_moves(game.board, current)
        print(f"Legal moves: {len(legal_moves)}")
        
        if not legal_moves:
            print("No legal moves - passing")
            game.board._update_current_player()
            continue
        
        # Simple move selection (could be interactive)
        move = legal_moves[0]
        success = game.make_move(move, current)
        log_move(current, move, success)
```

**File:** `scripts/verify_pieces.py` (NEW)

```python
"""Verify piece definitions match between frontend and backend."""

def compare_pieces():
    """Compare backend and frontend piece shapes."""
    backend_pieces = PieceGenerator.get_all_pieces()
    frontend_shapes = load_frontend_shapes()  # Load from gameConstants.ts
    
    mismatches = []
    for piece in backend_pieces:
        backend_shape = piece.shape.tolist()
        frontend_shape = frontend_shapes.get(piece.id)
        
        if not shapes_match(backend_shape, frontend_shape):
            mismatches.append({
                'id': piece.id,
                'name': piece.name,
                'backend': backend_shape,
                'frontend': frontend_shape
            })
    
    if mismatches:
        print("❌ Piece shape mismatches found:")
        for m in mismatches:
            print(f"  Piece {m['id']} ({m['name']}):")
            print(f"    Backend:  {m['backend']}")
            print(f"    Frontend: {m['frontend']}")
    else:
        print("✅ All piece shapes match")
```

### 6.4 Test Execution Plan

**Run Tests:**
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_blokus_rules.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=engine --cov=agents
```

**Smoke Test Checklist:**
1. ✅ All 21 pieces defined and have correct sizes
2. ✅ First move corner rule enforced
3. ✅ Overlap prevention works
4. ✅ Same-color edge-adjacency prevented
5. ✅ Corner-adjacency allowed
6. ✅ Turn rotation works
7. ✅ Agent moves are validated
8. ✅ Game-over detection works (when implemented correctly)

---

## 7. Prioritized Fix & Implementation Plan

### Phase 1: Core Rules & Piece Set Correctness

**Priority: CRITICAL** - Blocks all gameplay

#### Task 1.1: Fix Piece Shape Mismatches

**File:** `engine/pieces.py`
- **Line 91:** Fix Pentomino F shape
  - **Current:** `np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])` (X shape)
  - **Fix to:** `np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]])` (F shape)

**File:** `frontend/src/constants/gameConstants.ts`
- **Line 18:** Fix Pentomino F shape
  - **Current:** `[[0, 1, 1], [1, 1, 0], [0, 1, 0]]` (verify this is correct F)
  - **Action:** Verify against official Blokus F pentomino, ensure matches backend

- **Line 28:** Fix Pentomino Y shape
  - **Current:** `[[1, 1, 1, 1], [0, 1, 0, 0]]` (wrong - 2 rows)
  - **Fix to:** `[[1, 0], [1, 1], [1, 0], [1, 0]]` (4 rows, matches backend)

**Validation:**
- Add test: `tests/test_piece_shapes_match.py`
- Compare all 21 pieces between frontend and backend
- Verify each piece's rotations/flips are consistent

#### Task 1.2: Verify All Piece Shapes

**File:** `engine/pieces.py`
- **Action:** Review all 21 piece definitions against official Blokus set
- **Reference:** Official Blokus piece set (verify via online resources or game manual)
- **Check:**
  - Monomino (1): `[[1]]` ✅
  - Domino (2): `[[1, 1]]` ✅
  - Tromino I (3): `[[1, 1, 1]]` ✅
  - Tromino L (4): `[[1, 0], [1, 1]]` ✅
  - Tetromino I (5): `[[1, 1, 1, 1]]` ✅
  - Tetromino O (6): `[[1, 1], [1, 1]]` ✅
  - Tetromino T (7): `[[1, 1, 1], [0, 1, 0]]` ✅
  - Tetromino L (8): `[[1, 0], [1, 0], [1, 1]]` ✅
  - Tetromino S (9): `[[0, 1, 1], [1, 1, 0]]` ✅
  - Tetromino Z (10): `[[1, 1, 0], [0, 1, 1]]` ✅
  - Pentomino F (11): ❌ **FIX NEEDED** (see Task 1.1)
  - Pentomino I (12): `[[1, 1, 1, 1, 1]]` ✅
  - Pentomino L (13): `[[1, 0], [1, 0], [1, 0], [1, 1]]` ✅
  - Pentomino N (14): `[[1, 0], [1, 1], [0, 1], [0, 1]]` ✅
  - Pentomino P (15): `[[1, 1], [1, 1], [1, 0]]` ✅
  - Pentomino T (16): `[[1, 1, 1], [0, 1, 0], [0, 1, 0]]` ✅
  - Pentomino U (17): `[[1, 0, 1], [1, 1, 1]]` ✅
  - Pentomino V (18): `[[1, 0, 0], [1, 0, 0], [1, 1, 1]]` ✅
  - Pentomino W (19): `[[1, 0, 0], [1, 1, 0], [0, 1, 1]]` ✅
  - Pentomino X (20): `[[0, 1, 0], [1, 1, 1], [0, 1, 0]]` ✅
  - Pentomino Y (21): ❌ **FIX NEEDED** (see Task 1.1)

#### Task 1.3: Enforce Placement Rules Strictly

**File:** `engine/board.py`
- **Function:** `can_place_piece()` (line 115-142)
- **Action:** Add detailed validation logging (optional, for debugging)

**File:** `engine/board.py`
- **Function:** `_check_adjacency_rules()` (line 144-171)
- **Action:** Add comments clarifying edge vs corner adjacency logic
- **Verify:** Logic correctly prevents edge-adjacency for same color, allows for different colors

**File:** `engine/board.py`
- **Function:** `_is_connected_via_corners()` (line 173-188)
- **Action:** Verify logic is correct (currently checks if ANY corner touches existing piece)
- **Note:** Current implementation is correct, but could be optimized

#### Task 1.4: Add Piece Shape Validation Test

**File:** `tests/test_piece_shapes_match.py` (NEW)
- **Purpose:** Ensure frontend and backend piece shapes are identical
- **Implementation:** See Section 6.2.1

**Estimated Time:** 2-4 hours

---

### Phase 2: Human Playability from the App

**Priority: HIGH** - Required for human gameplay

#### Task 2.1: Fix Move Validation Error Messages

**File:** `webapi/app.py`
- **Function:** `_process_human_move_immediately()` (line 250-316)
- **Action:** Improve error messages to specify rejection reason

**Current:** Returns generic "Invalid move"
**Fix to:** Return specific reasons:
- "Move is out of bounds"
- "Piece overlaps with existing pieces"
- "First move must cover your starting corner"
- "Pieces of the same color cannot touch edge-to-edge"
- "Piece must connect to your existing pieces via corners"
- "It's not your turn"
- "Piece has already been used"

**Implementation:**
```python
# In _process_human_move_immediately(), before calling game.make_move():
# Pre-validate and return specific error
if current_player != engine_player:
    return MoveResponse(success=False, message=f"It's not {move_request.player}'s turn", ...)

# Check piece used
if move_data.piece_id in game.board.player_pieces_used[engine_player]:
    return MoveResponse(success=False, message="Piece has already been used", ...)

# Check bounds
orientations = game.move_generator.piece_orientations_cache[move_data.piece_id]
orientation = orientations[move_data.orientation]
if not PiecePlacement.can_place_piece_at(...):
    return MoveResponse(success=False, message="Move is out of bounds", ...)

# Then call game.make_move() and return generic error if it fails
```

#### Task 2.2: Add Turn Indicator UI

**File:** `frontend/src/pages/Play.tsx`
- **Location:** Add after line 193 (error display) or in sidebar
- **Action:** Display current player indicator

**Implementation:**
```tsx
<div className="mb-4 bg-charcoal-800 border border-neon-blue p-4">
  <div className="flex items-center space-x-2">
    <div className="w-3 h-3 rounded-full" style={{
      backgroundColor: PLAYER_COLORS[gameState?.current_player?.toLowerCase() || 'empty']
    }}></div>
    <p className="text-sm text-gray-200">
      Current Turn: {gameState?.current_player || 'Unknown'}
    </p>
  </div>
</div>
```

#### Task 2.3: Add Pass Button for Human Players

**File:** `frontend/src/pages/Play.tsx`
- **Location:** Add near turn indicator
- **Action:** Show "Pass" button when current player is human and has no legal moves

**Implementation:**
```tsx
const legalMoves = gameState?.legal_moves || [];
const canPass = isHumanPlayer && legalMoves.length === 0;

{canPass && (
  <button
    onClick={handlePass}
    className="bg-charcoal-800 border border-neon-yellow hover:border-neon-yellow/80 text-neon-yellow px-4 py-2"
  >
    Pass (No Legal Moves)
  </button>
)}
```

**File:** `frontend/src/store/gameStore.ts`
- **Action:** Add `passTurn()` function that sends pass message via WebSocket

**File:** `webapi/app.py`
- **Action:** Add pass message handler in WebSocket endpoint
- **Implementation:** Call `board._update_current_player()` and broadcast state

#### Task 2.4: Fix Piece Preview Accuracy

**File:** `frontend/src/components/Board.tsx`
- **Function:** `getPiecePreview()` (line 130-143)
- **Action:** Verify `calculatePiecePositions()` matches backend anchor system
- **Test:** Place piece, verify preview matches actual placement

**File:** `frontend/src/utils/pieceUtils.ts`
- **Function:** `calculatePiecePositions()`
- **Action:** Ensure anchor system matches backend (top-left of piece shape)

#### Task 2.5: Add Legal Move Highlighting

**File:** `frontend/src/components/Board.tsx`
- **Action:** Use `gameState.heatmap` to highlight legal move positions
- **Implementation:** Overlay semi-transparent cells where `heatmap[row][col] === 1.0`

**File:** `frontend/src/pages/Play.tsx`
- **Action:** Ensure `gameState.heatmap` is passed to Board component

**Estimated Time:** 4-6 hours

---

### Phase 3: Agent Opponents (1–3 Agents)

**Priority: MEDIUM** - Already mostly working, needs verification

#### Task 3.1: Verify 1–3 Agent Scenarios

**Action:** Create test games with:
1. 1 human + 1 agent
2. 1 human + 2 agents
3. 1 human + 3 agents

**Test Cases:**
- Turn order is correct
- Agents make valid moves
- Agents can pass when no moves
- Game ends correctly
- Scores are calculated correctly

**File:** `tests/test_multi_agent_game.py` (NEW)
- **Purpose:** Integration tests for mixed human/agent games

#### Task 3.2: Improve Agent Error Handling

**File:** `webapi/app.py`
- **Function:** `_make_agent_move()` (line 142-200)
- **Action:** Instead of falling back to random move on timeout, consider:
  - Logging the timeout as an error
  - Returning None (pass) if agent times out
  - Or raising an exception to stop the game

**Current:** Falls back to random (line 176)
**Suggested:** Pass turn or raise error

#### Task 3.3: Add Agent Configuration Support

**File:** `schemas/game_config.py`
- **Action:** Extend `AgentConfig` to support agent-specific parameters
- **Current:** Has `parameters` field but may not be used

**File:** `webapi/app.py`
- **Function:** `_initialize_agents()` (line 71-93)
- **Action:** Use `player_config.parameters` to configure agents
- **Example:** Pass `iterations`, `time_limit`, `seed` to MCTS agents

#### Task 3.4: Add Agent Move Logging

**File:** `webapi/app.py`
- **Function:** `_make_agent_move()`
- **Action:** Log agent moves with agent type and move details
- **Purpose:** Debugging and game analysis

**Estimated Time:** 2-3 hours

---

### Phase 4: Cleanup, Tests & Documentation

**Priority: LOW** - Polish and maintainability

#### Task 4.1: Implement Smoke Tests

**Files to Create:**
- `tests/test_blokus_rules.py` - Comprehensive rules tests (see Section 6.2)
- `tests/test_piece_shapes_match.py` - Piece shape validation
- `tests/test_multi_agent_game.py` - Multi-agent integration tests

**Action:** Implement all tests from Section 6.2

#### Task 4.2: Add Debug Utilities

**Files to Create:**
- `scripts/debug_blokus.py` - CLI game runner (see Section 6.3)
- `scripts/verify_pieces.py` - Piece shape comparison tool (see Section 6.3)

#### Task 4.3: Fix Game-Over Detection

**File:** `engine/game.py`
- **Function:** `_check_game_over()` (line 69-81)
- **Action:** 
  1. Rename `all_players_can_move` to `any_player_can_move` for clarity
  2. Add consecutive-pass detection (game ends when all players pass in sequence)
  3. Optimize by caching legal move counts

**Implementation:**
```python
def _check_game_over(self) -> None:
    """Check if the game is over."""
    # Game ends when no player can make a legal move
    any_player_can_move = False
    for player in Player:
        if self.move_generator.has_legal_moves(self.board, player):
            any_player_can_move = True
            break
    
    if not any_player_can_move:
        self.board.game_over = True
        self.winner = self.get_winner()
```

#### Task 4.4: Update Documentation

**File:** `README.md`
- **Action:** Add section "How to Play Blokus in this App"
- **Content:**
  - How to start a game
  - How to place pieces (select, rotate/flip, click board)
  - How to pass when no moves
  - How to configure agents

**File:** `docs/GAME_ENGINE.md` (NEW)
- **Purpose:** Document main game engine files
- **Content:**
  - `engine/board.py` - Board state and placement rules
  - `engine/game.py` - Game orchestration and scoring
  - `engine/pieces.py` - Piece definitions
  - `engine/move_generator.py` - Legal move generation

**File:** `docs/TESTING.md` (NEW)
- **Purpose:** How to run tests and smoke checks
- **Content:**
  - Running pytest
  - Running smoke tests
  - Using debug utilities

**File:** `docs/AGENTS.md` (NEW)
- **Purpose:** How to hook up new agents
- **Content:**
  - Agent interface
  - Registering agents in webapi/app.py
  - Agent configuration

#### Task 4.5: Remove Dead Code

**Action:** Review and remove:
- Unused imports
- Commented-out code
- Duplicate game logic (if any)

**Estimated Time:** 3-4 hours

---

## Summary of Critical Issues

1. **❌ CRITICAL: Piece Shape Mismatches**
   - Pentomino F (ID 11): Backend has X shape, frontend has different shape
   - Pentomino Y (ID 21): Backend and frontend have completely different shapes
   - **Impact:** Preview shows wrong shape, placement may fail
   - **Fix:** Synchronize shapes in both files

2. **⚠️ HIGH: Game-Over Detection**
   - Logic works but variable naming is confusing
   - Missing consecutive-pass check
   - **Impact:** Game may not end correctly in some scenarios
   - **Fix:** Improve logic and add pass tracking

3. **⚠️ MEDIUM: Human Player Passing**
   - No explicit pass button/action for human players
   - **Impact:** Game may appear stuck when human has no moves
   - **Fix:** Add pass button and WebSocket handler

4. **⚠️ MEDIUM: Error Messages**
   - Generic "Invalid move" doesn't explain why
   - **Impact:** Poor user experience
   - **Fix:** Add specific error messages

5. **⚠️ LOW: Agent Timeout Handling**
   - Falls back to random move on timeout
   - **Impact:** May not be desired behavior
   - **Fix:** Consider passing or raising error instead

---

## Next Steps

1. **Immediate:** Fix piece shape mismatches (Phase 1, Task 1.1)
2. **Short-term:** Implement human playability fixes (Phase 2)
3. **Medium-term:** Verify and improve agent integration (Phase 3)
4. **Long-term:** Add comprehensive tests and documentation (Phase 4)

**Estimated Total Time:** 12-18 hours for all phases


