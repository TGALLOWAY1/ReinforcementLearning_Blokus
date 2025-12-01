# Blokus RL Performance Diagnostic Report

## Summary

This document summarizes the instrumentation and fixes added to diagnose performance issues and logging problems in the Blokus RL project.

## Issues Identified

### 1. Logging Issues
**Problem**: Backend logs showed "Player Unknown placed a piece at undefined,undefined"

**Root Cause**: 
- The frontend was trying to extract move information from WebSocket messages, but the backend wasn't including move details in the `move_made` broadcast messages
- The frontend was looking for `data.data?.move` or `data.data?.player`, but these fields weren't present in the message structure

**Fix**:
- Modified `_broadcast_game_state()` to accept and include `last_move` information in broadcast messages
- Updated both `_make_agent_move()` and `_process_human_move_immediately()` to pass move information when broadcasting
- Enhanced frontend logging to better extract and validate move data, with fallback logging if coordinates are missing

### 2. Performance Instrumentation

#### Backend Instrumentation

**Human Moves** (`_process_human_move_immediately`):
- Added timing for:
  - Total time (from WebSocket message receipt to broadcast completion)
  - `make_move` time (core engine move processing)
  - Broadcast time (serialization and WebSocket send)
- Added detailed logging of:
  - Raw incoming move payload
  - Player name and coordinates
  - Move validation results

**Agent Moves** (`_make_agent_move`):
- Added timing for:
  - Legal move generation
  - Agent selection (`select_action`)
  - Move application (`game.make_move`)
  - Broadcast
  - Total time
- Added logging of:
  - Number of legal moves found
  - Agent selection time
  - Move details (player, coordinates, piece_id, orientation)

**Engine Level** (`engine/game.py`):
- Added timing in `make_move()`:
  - Legal move check
  - Piece placement
  - Total core time
- Uses `logger.debug()` for detailed engine-level timing

**Legal Move Generation** (`engine/move_generator.py`):
- Added timing for:
  - Total legal move generation time
  - Per-piece timing (identifies slow pieces)
  - Logs slowest piece processing time

#### Frontend Instrumentation

**Board Rendering** (`components/Board.tsx`):
- Added `console.time/timeEnd` for board render cycles
- Added effect timing for gameState changes
- Measures render performance after each move

**Move Logging** (`store/gameStore.ts`):
- Enhanced move message parsing with detailed console logging
- Validates coordinates before logging
- Logs raw WebSocket message structure for debugging

**Move Sending** (`pages/Play.tsx`):
- Added detailed logging before sending moves:
  - gameId, player, pieceId, orientation, row, col
  - Ensures all values are defined before sending

## Logging Structure

### Backend Logs

All backend logs now use Python's `logging` module with structured format:
```
[HH:MM:SS] LEVEL: message
```

**Key Log Messages**:
- `Game created: {game_id}`
- `Connected to game {game_id}`
- `HUMAN MOVE START: game_id=..., player=..., raw_move=...`
- `HUMAN MOVE timing: total=...s, make_move=...s, broadcast=...s`
- `AGENT MOVE START: game_id=..., player=..., agent=...`
- `AGENT MOVE timing: total=...s, legal=...s, agent=...s, apply=...s, broadcast=...s`
- `Player {player_name} placed a piece at {row},{col} (piece_id=..., orientation=...)`
- `Legal move generation: {count} moves in {time}s for player=...`

### Frontend Logs

Frontend logs use `console.log/time/timeEnd`:
- `[UI] Sending move` - Before sending move to backend
- `[UI] Move message received` - When receiving move from backend
- `[UI] Board render` - Board render timing
- `[UI] Board effect (gameState change)` - Effect timing

## Expected Performance Data

After running games, you should see timing data like:

### Human Move Example:
```
[22:58:40] INFO: HUMAN MOVE START: game_id=..., player=RED, raw_move={'piece_id': 1, 'orientation': 0, 'anchor_row': 5, 'anchor_col': 5}
[22:58:40] INFO: HUMAN MOVE make_move: success=True in 0.0123s
[22:58:40] INFO: Player RED placed a piece at 5,5 (piece_id=1, orientation=0)
[22:58:40] INFO: HUMAN MOVE timing: total=0.0234s, make_move=0.0123s, broadcast=0.0111s
```

### Agent Move Example:
```
[22:58:45] INFO: AGENT MOVE START: game_id=..., player=BLUE, agent=RandomAgent
[22:58:45] INFO: AGENT MOVE legal_moves: 150 moves found in 0.2345s for player=BLUE
[22:58:45] INFO: AGENT MOVE agent_selection: move=Move(...) selected in 0.0012s
[22:58:45] INFO: AGENT MOVE apply: success=True in 0.0123s
[22:58:45] INFO: Player BLUE placed a piece at 0,19 (piece_id=2, orientation=1)
[22:58:45] INFO: AGENT MOVE timing: total=0.2500s, legal=0.2345s, agent=0.0012s, apply=0.0123s, broadcast=0.0020s
```

## Suspected Bottlenecks (To Be Confirmed with Data)

Based on the instrumentation, likely bottlenecks are:

1. **Legal Move Generation** - Generating all legal moves each turn can be expensive, especially as the board fills up
   - **Evidence**: Timing will show if `legal` time dominates agent moves
   - **Solution**: Cache legal moves, only regenerate when board changes

2. **Board Placement Validation** - `can_place_piece()` is called for every potential move
   - **Evidence**: If `make_move` time is high relative to simple operations
   - **Solution**: Optimize validation logic, use early exits

3. **JSON Serialization / Broadcast** - Large game state objects being serialized and sent
   - **Evidence**: If `broadcast` time is significant
   - **Solution**: Only send deltas, compress messages, optimize serialization

4. **Frontend Re-renders** - Board component re-rendering entire 20x20 grid
   - **Evidence**: If `[UI] Board render` times are high
   - **Solution**: Memoize cells, use React.memo, virtualize rendering

## Next Steps

1. **Run Test Games**:
   - Run a game with all random agents (2-4 players)
   - Run a game with one human + random agents
   - Collect timing data from logs

2. **Analyze Timing Data**:
   - Identify which stage takes the most time
   - Look for patterns (e.g., legal move generation gets slower as game progresses)
   - Compare human vs agent move timings

3. **Optimize Based on Findings**:
   - Focus optimization efforts on the identified bottlenecks
   - Measure improvement after each optimization

## Files Modified

### Backend:
- `webapi/app.py` - Added logging, timing instrumentation, fixed move broadcasting
- `engine/game.py` - Added timing to `make_move()`
- `engine/move_generator.py` - Added timing to `get_legal_moves()`

### Frontend:
- `frontend/src/store/gameStore.ts` - Enhanced move message parsing and logging
- `frontend/src/pages/Play.tsx` - Added move sending logs
- `frontend/src/components/Board.tsx` - Added render performance instrumentation

## Testing Recommendations

1. **Test with Random Agents Only**:
   ```bash
   # Create game with 4 random agents
   # Watch backend logs for timing data
   ```

2. **Test with Human Player**:
   ```bash
   # Create game with 1 human + 3 random agents
   # Make several moves and observe timing
   ```

3. **Check Browser Console**:
   - Look for `[UI]` prefixed logs
   - Check render times after each move
   - Verify move coordinates are logged correctly

4. **Check Backend Logs**:
   - Look for `HUMAN MOVE` and `AGENT MOVE` timing logs
   - Verify player names and coordinates are logged correctly
   - Check for any error messages

## Known Issues Fixed

✅ **Fixed**: "Player Unknown placed a piece at undefined,undefined" - Backend now includes move info in broadcasts
✅ **Fixed**: Missing timing data - Comprehensive timing instrumentation added
✅ **Fixed**: No visibility into performance bottlenecks - Detailed logging at each stage

## Remaining Work

- [x] Collect actual timing data from test runs
- [x] Identify specific bottlenecks based on data
- [x] Implement optimizations for identified bottlenecks
- [x] Re-measure performance after optimizations

## Phase 1 Optimization Results

See `PERFORMANCE_OPTIMIZATION_RESULTS.md` for detailed results.

**Summary**: Achieved ~82% performance improvement:
- Legal move generation: 1347ms → 242ms (82% faster)
- Move application: 1710ms → 268ms (84% faster)
- Human moves: 2126ms → 383ms (82% faster)
- Total per move: 3056ms → 511ms (83% faster)

The game is now **5-6x faster** and much more responsive, though still above the 100-150ms target. Further optimizations would require more invasive architectural changes.

