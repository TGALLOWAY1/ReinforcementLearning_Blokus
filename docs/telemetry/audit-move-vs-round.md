# Audit: Move vs Round (Game Turn) — Blokus RL

**Date:** 2026-03-01  
**Branch:** `43-introduce-the-concept-of-a-game-round-rather-than-just-player-turn`  
**Status:** Audit only — no functional code changes made.

---

## 1. Summary

### What "turn" currently means in practice

The codebase uses **"turn"** to mean exactly one thing: **a single player action** (place a piece or pass). There is **no concept of a "round"** anywhere in the engine, telemetry schemas, frontend store, or chart components. The two primary counters in use are:

| Symbol | Where | Increments when |
|---|---|---|
| `board.move_count` | `engine/board.py` | `place_piece()` succeeds — once per player placement |
| `turn_number` | `engine/game.py` → `game_history` | `make_move()` succeeds — once per player action |
| `turn_index` | `analytics/logging/schemas.py:StepLog` | Passed in from webapi as `game.get_move_count()` before the move |
| `turnNum` / `{ turn: idx+1 }` | `AnalysisDashboard.tsx` chart transforms | Once per `gameHistory` array entry (i.e. per move) |

The consequence: all per-player metrics lines in charts advance at different absolute x-positions. RED moves at x=1, BLUE at x=2, GREEN at x=3, YELLOW at x=4. Comparing their curves "at the same game phase" is misleading because they are not aligned to the same game state.

### Does any "round" concept already exist?

No. The only hint is `GameResultLog.seat_order: List[int]` in `analytics/logging/schemas.py`, which records the turn order as `[1, 2, 3, 4]` (static, hardcoded). No code computes `round_number = move_count // 4` or anything equivalent.

### Five minimum safe changes likely needed later

1. **Add `move_index` (0-based) and `round_number` (= `move_index // 4`, 0-based) fields** to `GameHistoryEntry` in `engine/game.py` and to `StepLog` in `analytics/logging/schemas.py` without removing any existing fields.
2. **Add `seat_index` (0–3) to `StepLog`** so telemetry can reconstruct which "slot" in the round this action occupied, derived deterministically from `player_id` and known seat order.
3. **Update charts x-axis from raw `turn` (move index) to `round_number`** in `AnalysisDashboard.tsx` `ModuleC_CornerChart`, `ModuleE_FrontierChart`, `ModuleF_UrgencyChart` — or add a toggle.
4. **Update `dashboardMetrics.ts:calculateWinProbability()`** (sampled once per move currently) to clearly document it is per-move, and optionally add a round-aligned variant that averages over a full round before emitting a data point.
5. **Add `round_number` field to `move_records`** in `webapi/app.py` to make it available for replay and the analysis API's x-axis.

---

## 2. Canonical Definitions (Proposed)

### Definitions

| Term | Proposed meaning |
|---|---|
| **Move** | A single player action: placing a piece OR passing. Equivalent to what the code currently calls a "turn". |
| **Round** (a.k.a. **Game Turn**) | A full cycle in which all 4 players (RED → BLUE → YELLOW → GREEN) have each had one move opportunity. One round = 4 moves (or fewer at game end when some players have passed out). |

### Index basis in the current codebase

- `board.move_count` — **0-based** counter; starts at 0, incremented to 1 after the first move.
- `game_history` list index — **0-based** Python list; `turn_number` field inside each entry is **1-based** (`len(self.game_history) + 1` before appending).
- `turn_index` in `StepLog` — **0-based** (captured as `game.get_move_count()` before the move).
- `turnNum` in chart transforms — **1-based** (`idx + 1`; `idx` comes from `Array.map`).
- `seat_order` — `[1, 2, 3, 4]` (Player.value integers, RED=1, BLUE=2, YELLOW=3, GREEN=4).

**Recommendation for new fields:** Use **0-based** `move_index` for the raw counter and **0-based** `round_index` = `move_index // 4`. This is consistent with `board.move_count` and `turn_index`.

---

## 3. Inventory of "Turn"-Related Entities

| Layer | File path | Symbol / Identifier | Current Meaning | Evidence | Risk | Notes |
|---|---|---|---|---|---|---|
| engine | `engine/board.py:62` | `Board.move_count` | PLAYER_ACTION_TURN | Initialized to 0, incremented by 1 in `place_piece()` on every successful placement. | medium | Does NOT increment on pass (pass calls `_update_current_player()` directly). |
| engine | `engine/board.py:547` | `Board._update_current_player()` | PLAYER_ACTION_TURN | Rotates `current_player` using `(idx+1) % 4`. Called after every place or pass. | medium | This is the authoritative "next turn" trigger. |
| engine | `engine/game.py:107` | `game_history[i]['turn_number']` | PLAYER_ACTION_TURN | Set as `len(self.game_history) + 1` (1-based), incremented per successful `make_move()`. | high | This is the primary save-game time key; used in test fixtures and `test_worker_bridge_save_load.py:76`. |
| engine | `engine/game.py:339` | `BlokusGame.get_move_count()` | PLAYER_ACTION_TURN | Returns `self.board.move_count`. Used as `turn_index` source in `webapi/app.py:358`. | medium | Functionally identical to `move_count`. |
| engine | `engine/game.py:344` | `BlokusGame.move_count` (property) | PLAYER_ACTION_TURN | Property alias for `self.board.move_count`. | low | Redundant with `get_move_count()`. |
| telemetry | `analytics/logging/schemas.py:12` | `StepLog.turn_index` | PLAYER_ACTION_TURN | Passed from `webapi/app.py:358` as `game.get_move_count()` captured *before* the move. 0-based. | high | Sort key for `reader.py:63`; used as x-axis in `get_analysis_summary`. |
| telemetry | `analytics/logging/schemas.py:35` | `GameResultLog.num_turns` | PLAYER_ACTION_TURN | Total count passed in by webapi at game end. Meaning is total moves made, not rounds. | medium | Misleadingly named — should be `num_moves`. |
| telemetry | `analytics/logging/schemas.py:37` | `GameResultLog.seat_order` | AMBIGUOUS | Set to `[p.value for p in Player]` = `[1,2,3,4]` — static, never varies. | low | Exists but never used to compute seat/round alignment anywhere. |
| telemetry | `analytics/logging/logger.py:53` | `StrategyLogger.on_step(turn_index=...)` | PLAYER_ACTION_TURN | Caller must provide the current move index. Logs one entry per player action. | medium | No awareness of rounds. |
| telemetry | `analytics/logging/reader.py:63` | `sort key: turn_index` | PLAYER_ACTION_TURN | Steps sorted by `turn_index` then `timestamp`. Used in `load_steps_for_game()`. | medium | Any new `round_index` field would need to be added here as a secondary sort or primary. |
| webapi | `webapi/app.py:358` | `turn_index = game.get_move_count()` | PLAYER_ACTION_TURN | Captured before move, passed to `_log_step`. Aliases `board.move_count`. | medium | |
| webapi | `webapi/app.py:385` | `move_records[i]['moveIndex']` | PLAYER_ACTION_TURN | Set as `game.get_move_count()` after the move. Used in replay endpoint sort. | medium | Passes are stored with `moveIndex: None`. Different from `turn_index` (pre-move vs post-move). |
| webapi | `webapi/app.py:384` | `move_records[i]['sequenceIndex']` | PLAYER_ACTION_TURN | Auto-incrementing event counter `_event_sequence`. Increments for passes too. | low | More reliable than `moveIndex` for ordering; closest thing to a global "event index". |
| webapi | `webapi/app.py:120` | `game_data['last_turn_started_at']` | PLAYER_ACTION_TURN | Records perf_counter timestamp of last turn start; used for timeout logic. | low | |
| webapi | `webapi/app.py:121` | `game_data['last_turn_player']` | PLAYER_ACTION_TURN | Records which player just moved. | low | |
| browser/worker | `browser_python/worker_bridge.py` | `game.game_history[i]['turn_number']` | PLAYER_ACTION_TURN | Comes from `engine/game.py:107`. 1-based player-action counter. | high | Used as save format key; tested in `test_worker_bridge_save_load.py`. |
| browser/worker | `browser_python/worker_bridge.py:343` | `state_response['move_count']` | PLAYER_ACTION_TURN | `game.get_move_count()` returned in every state update; consumed by frontend store. | medium | |
| frontend | `frontend/src/store/gameStore.ts:127` | `GameHistoryEntry.turn_number` | PLAYER_ACTION_TURN | TypeScript type matching `engine/game.py` history entry. 1-based move counter. | high | Changing this field name breaks save/load compatibility. |
| frontend | `frontend/src/store/gameStore.ts:160` | `currentSliderTurn` | PLAYER_ACTION_TURN | Zustand store state tracking which move the analysis slider is at. 1-based. | medium | Used as ReferenceLine x-value in charts. |
| frontend | `frontend/src/components/AnalysisDashboard.tsx:39` | `totalTurns` | PLAYER_ACTION_TURN | `gameHistory.length` — count of entries, one per player move. | medium | Labels slider max. Drives `Turn X / Y` UI label. |
| frontend | `frontend/src/components/AnalysisDashboard.tsx:85` | UI label `"Turn {N} / {M}"` | PLAYER_ACTION_TURN | Displayed to user; labeled "Turn" but is actually the move index. | medium | Visible to user — should become "Move N / M" (or "Round N" with a round count). |
| frontend | `frontend/src/components/AnalysisDashboard.tsx:533–576` | `turnNum = idx + 1` / `{ turn: turnNum }` in chart transforms | PLAYER_ACTION_TURN | Charts use raw `gameHistory` array index + 1 as x-axis. All 4 players plotted on the same x without alignment. | **high** | Root cause of per-player skew in all line charts. |
| frontend | `frontend/src/components/AnalysisDashboard.tsx:553,592,637` | `<XAxis dataKey="turn" />` | PLAYER_ACTION_TURN | Chart x-axis is per-move index. Domain is 1…N where N = total moves. | **high** | Must change to `round` (or add a toggle) to align per-player series. |
| frontend | `frontend/src/components/RightPanel.tsx:55` | `liveTurn = gameHistory.length` | PLAYER_ACTION_TURN | Used as `currentTurn` reference line position in live charts. | medium | |
| frontend | `frontend/src/utils/dashboardMetrics.ts:164` | `calculateWinProbability(board, metrics)` | AMBIGUOUS | Called once per state update (per move). Returns % share of score+territory+frontier. No time index attached. | **high** | Sampled once per move. RED samples first → inflated early score. Not round-aligned. |

---

## 4. Dataflow Map

### Browser/WebWorker path (primary game flow)

```
[BlokusGame.make_move()]
  └─► engine/game.py appends to game_history:
        { turn_number: N (1-based, per-move),
          player_to_move: "RED",
          action: { piece_id, orientation, anchor_row, anchor_col },
          board_state: [...], metrics: { corner_count, frontier_size, ... } }

  └─► engine/board.py: move_count += 1
  └─► engine/board.py: _update_current_player() → rotates RED→BLUE→YELLOW→GREEN

[WebWorkerGameBridge.get_state()]
  └─► Computes all_frontier_metrics, all_frontier_clusters
  └─► Backfills history entries missing frontier fields
  └─► Emits state_response:
        { move_count (per-move 0-based counter),
          current_player,
          game_history: [ { turn_number, player_to_move, action, ... }, ... ],
          frontier_metrics, frontier_clusters, ... }

[blokusWorker.ts] → postMessage({ type: 'state_update', data: state_response })

[gameStore.ts]
  └─► Receives state_update
  └─► Sets gameState.game_history (array of per-move entries)
  └─► Sets currentSliderTurn (defaults to history.length = latest move)

[AnalysisDashboard.tsx / RightPanel.tsx]
  └─► Reads gameHistory array
  └─► Chart transform: history.map((entry, idx) => ({ turn: idx+1, RED: ..., BLUE: ... }))
        ── x-axis = per-move counter (not round-aligned)
        ── All 4 player series plotted on same x without normalization
  └─► XAxis dataKey="turn" (per-move index)
  └─► ReferenceLine x={currentSliderTurn} (per-move index)
  └─► calculateWinProbability() called on each render with current board
        ── No time index attached to win probability samples
```

### Server-side (webapi / StrategyLogger) path

```
[GameManager.advance_turn()]
  └─► turn_index = game.get_move_count()  [captured BEFORE move; 0-based]
  └─► game.make_move(move, player)
  └─► move_records.append({
          sequenceIndex: incrementing event counter,
          moveIndex: game.get_move_count() [AFTER move; 1 higher],
          player: player_name,
          isPass: False })

[GameManager._log_step()]
  └─► StrategyLogger.on_step(turn_index=turn_index, player_id=...)
  └─► StepLog { turn_index (0-based, per-move), player_id, action, metrics }
  └─► Written to {game_id}/steps.jsonl

[on_game_end()]
  └─► GameResultLog { num_turns = total_moves_made, seat_order=[1,2,3,4] }
  └─► Written to results.jsonl

[webapi reader / analysis endpoints]
  └─► load_steps_for_game() sorts by (turn_index, timestamp)
  └─► get_analysis_summary() builds mobilityCurve with { turn_index, player_id }
        ── x-axis identified by per-move turn_index
        ── All players on same timeline (per-move, not per-round)
```

---

## 5. Compatibility Surface

Items that will break or require updating when `round_number` / `move_index` fields are added:

| Surface | Risk | Detail |
|---|---|---|
| `tests/test_worker_bridge_save_load.py:76` | **High** | `required = {"turn_number", "player_to_move", "action", "board_state", "metrics"}` — test asserts exact key set; adding `move_index` / `round_number` here would require updating the test. |
| `frontend/src/store/gameStore.ts: GameHistoryEntry` | **High** | TypeScript interface must be updated to include new fields; any downstream consumers narrowly typing the shape will error. |
| `analytics/logging/reader.py:63` | **Medium** | Sort key `(turn_index, timestamp)` — adding `round_index` doesn't break, but round-based queries would need a new sort or filter. |
| `analytics/logging/schemas.py: StepLog, GameResultLog` | **Medium** | Pydantic models — adding optional fields is backward-compatible; removing or renaming `turn_index` or `num_turns` would break existing `.jsonl` readers. |
| `AnalysisDashboard.tsx XAxis dataKey="turn"` | **High** | If chart data shape changes from `{ turn: N }` to `{ round: N }`, all three line charts break silently (no data displayed). |
| `AnalysisDashboard.tsx ReferenceLine x={currentSliderTurn}` | **High** | If x-axis switches to round numbers but `currentSliderTurn` remains in move units, the reference line will be misaligned. |
| `webapi/app.py: move_records sorting` | **Medium** | Replay endpoint sorts by `sequenceIndex` or `moveIndex`; adding a `roundIndex` field doesn't break but must be kept consistent with the game's canonical round calculation. |
| `engine/game.py game_history` save format | **High** | Save files created before this change won't have `round_number`; `load_game()` must handle the absence gracefully (treat as optional). |
| `dashboardMetrics.ts: calculateWinProbability()` | **Medium** | Currently stateless — called with current board only. If win-probability is to be plotted round-aligned, it needs a round index passed in and stored per-round in history. |
| `heuristic_agent.py:169` | **Low** | `game_progress = board.move_count / 100.0` — hard-coded denominator assumes ~100 total moves; adding a round concept doesn't break this, but round-based normalization (`move_count / 25.0` for ~25 rounds) would be more semantically correct. |

---

## 6. Seat / Order Facts

### How player order is represented

- **Engine**: `Player` enum defined in `engine/board.py:18-23` as `{RED=1, BLUE=2, YELLOW=3, GREEN=4}`. Order is fixed: RED starts first.
- **Rotation**: `Board._update_current_player()` (line 547-551) advances via `(current_index + 1) % 4` on the `list(Player)` which evaluates to `[RED, BLUE, YELLOW, GREEN]`. **This order is always fixed. There is no configurable rotation.**
- **Seat index**: Never stored explicitly. It is always implicit: `seat_index = (move_count % 4)`. Because order is fixed, `seat_index=0` always equals RED, `seat_index=1` always equals BLUE, etc.
- **`seat_order` in telemetry**: `GameResultLog.seat_order = [1, 2, 3, 4]` is hardcoded in `logger.py:153`. It is never dynamically computed and never consumed by any downstream code.
- **`player_id` in telemetry**: `StepLog.player_id = Player.value` (integer 1–4). Equivalent to `seat_index + 1`.

### Can telemetry reliably infer seat_index?

Yes, trivially:
```python
seat_index = player_id - 1  # RED=0, BLUE=1, YELLOW=2, GREEN=3
round_index = turn_index // 4
position_within_round = turn_index % 4
```
This is lossless and deterministic because player order is always fixed.

### Can round_number be derived from existing data?

Yes, from any of:
- `turn_index // 4` (StepLog)
- `turn_number // 4` (game_history entry, but 1-based so adjust: `(turn_number - 1) // 4`)
- `move_count // 4` (engine / worker state response)
- `gameHistory.length // 4` (frontend)

**No new data needs to be captured** to compute round number; the derivation is purely additive.

---

## 7. Suggested Verification Fixtures (for Later)

These scenarios should be logged and asserted after implementation, but should **not** be implemented now.

1. **Opening round verification** — Play exactly 4 moves (one per player, no passes). Assert that:
   - `game_history` entries 0–3 have `move_index` 0–3 and `round_number` = 0.
   - All 4 players have `seat_index` 0, 1, 2, 3 respectively.
   - Chart x-axis at round 0 shows the state *after* all 4 have moved.

2. **Pass in round 2** — Play 8 moves normally (rounds 0–1), then have BLUE pass on round 2.
   - Assert that the pass event gets `round_number=2`, `seat_index=1` (BLUE's slot).
   - Assert the round 2 chart data point is present and BLUE's metric value does not change from round 1.
   - Assert `num_turns` = 9 (8 placements + 1 pass) vs `round_count` = 2 (complete rounds) + 1 partial.

3. **Late-game multi-player pass** — Force a state where RED and BLUE have no legal moves but YELLOW and GREEN do. Assert:
   - `round_index` increments correctly even when some players pass.
   - `move_count // 4` still equals the expected round even with pass gaps.

4. **Full game round count** — Play a game to completion. Assert:
   - `num_turns` ≈ `len(game_history)`.
   - `max(round_number)` ≈ `num_turns // 4`.
   - Win probability sampled at each `round_number` shows a monotonically smoother curve than per-move sampling.

5. **Save/load round-trip** — Save a game after round 5, reload it, and assert:
   - `move_index` and `round_number` (if added to history) survive the save/load cycle.
   - Entries without these fields (saved by older code) load without error (fields default to None or computed on-the-fly).

---

## 8. Commit Plan for This Audit-Only Change

### Commit 1 — Add the audit document

```
docs: add move-vs-round terminology audit

Adds docs/telemetry/audit-move-vs-round.md — a surgical inventory of every
use of "turn" across the engine, telemetry, webapi, and frontend layers.

Identifies:
- board.move_count and turn_number both mean single-player-action (no round concept exists)
- Root cause of per-player chart skew: x-axis is per-move, not per-round
- Full compatibility surface for adding move_index / round_number fields later
- Seat/order facts showing round derivation is lossless from existing data

No functional code changes. Audit only.
```

---

*End of audit.*
