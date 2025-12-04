# Win Detection & Game Over Behavior Audit

**Date:** 2025-01-XX  
**Task:** WD-01 – Audit current win/game-over behavior

---

## Summary

This document maps out how game termination, final scores, and win detection currently flow from the engine → environment → training loop. The audit reveals that win detection is partially implemented in the engine but not propagated to the training loop.

---

## 1. Game Termination Detection

### Engine Level (`engine/game.py`)

**How the game ends:**
- **Location:** `BlokusGame._check_game_over()` (lines 88-112)
- **Logic:** The game ends when **no players have legal moves available**
  - Scans all players to check if any can make a legal move
  - If no player can move → sets `self.board.game_over = True`
  - Calls `self.get_winner()` to determine winner
- **Trigger:** Called after each successful move in `make_move()` (line 82)

**Key Code:**
```python
def _check_game_over(self) -> None:
    any_player_can_move = False
    for player in Player:
        if self.move_generator.has_legal_moves(self.board, player):
            any_player_can_move = True
            break
    
    if not any_player_can_move:
        self.board.game_over = True
        self.winner = self.get_winner()
```

**Other termination conditions:**
- **Max episode steps:** Handled at environment level (`envs/blokus_v0.py`, line 402)
- **Individual agent termination:** If an agent has no legal moves, it's terminated individually (line 414)

---

## 2. Final Score Computation

### Engine Level (`engine/game.py`)

**Where scores are computed:**
- **Location:** `BlokusGame.get_score(player)` (lines 136-150)
- **Base score:** `self.board.get_score(player)` (lines 551-566 in `board.py`)
  - Counts squares covered by player's pieces
  - +15 bonus if player used all 21 pieces
- **Bonus score:** `_calculate_bonus_score(player)` (lines 152-171)
  - Corner control bonus: +5 points per controlled corner (4 corners max = +20)
  - Center control bonus: +2 points per center square (4×4 center area)

**Score storage:**
- Scores are **computed on-demand** (not stored)
- `get_game_state()` (lines 210-221) includes scores in returned dict:
  ```python
  'scores': {player.name: self.get_score(player) for player in Player}
  ```

### Board Level (`engine/board.py`)

**Base score calculation:**
- **Location:** `Board.get_score(player)` (lines 551-566)
- **Formula:** `squares_covered + (15 if all_pieces_used else 0)`
- Uses numpy: `np.sum(self.grid == player.value)`

---

## 3. Winner Determination

### Engine Level (`engine/game.py`)

**Winner computation:**
- **Location:** `BlokusGame.get_winner()` (lines 114-134)
- **Logic:**
  1. Calculates scores for all players
  2. Finds player with highest score: `max(scores, key=scores.get)`
  3. Checks for ties: if multiple players have the max score → returns `None`
  4. Returns `Player` enum if unique winner, `None` if tie

**Winner storage:**
- **Instance variable:** `self.winner` (set in `_check_game_over()`, line 112)
- **Initial value:** `None` (set in `__init__`, line 27)
- **Reset:** Set to `None` in `reset_game()` (line 227)

**Key Code:**
```python
def get_winner(self) -> Optional[Player]:
    if not self.board.game_over:
        return None
    
    scores = {}
    for player in Player:
        scores[player] = self.get_score(player)
    
    winner = max(scores, key=scores.get)
    max_score = scores[winner]
    tied_players = [p for p, s in scores.items() if s == max_score]
    
    if len(tied_players) > 1:
        return None  # Tie
    
    return winner
```

---

## 4. Environment Info Dict (`envs/blokus_v0.py`)

### Current Info Dict Contents

**Location:** `BlokusEnv._get_info(agent)` (lines 246-313)

**What's included:**
- `legal_action_mask`: Boolean array of legal actions
- `legal_moves_count`: Number of legal moves
- `score`: Current score for the agent's player (via `self.game.get_score(player)`)
- `pieces_used`: Number of pieces used
- `pieces_remaining`: Number of pieces remaining
- `can_move`: Boolean indicating if agent has legal moves

**What's NOT included:**
- ❌ `game_over`: No flag indicating if game is over
- ❌ `winner`: No winner information
- ❌ `final_scores`: No final scores for all players
- ❌ `is_winner`: No boolean indicating if this agent won

### Terminal Step Info

**Location:** `BlokusEnv._check_termination_truncation()` (lines 386-419)

**What happens on terminal steps:**
- If `self.game.is_game_over()`:
  - Calculates final rewards (winner gets +100, tie gets +10)
  - Sets `self.terminations[agent] = True` for all agents
  - **BUT:** Does not add any special info dict keys for terminal state

**Key Code:**
```python
if self.game.is_game_over():
    winner = self.game.get_winner()
    for agent in self.agents:
        player = self._agent_to_player(agent)
        if winner == player:
            self.rewards[agent] += 100  # Winner bonus
        elif winner is None:
            self.rewards[agent] += 10   # Tie bonus
        self.terminations[agent] = True
```

**Note:** The winner is computed but only used for reward bonuses, not exposed in info dict.

---

## 5. Training Loop Win Detection (`training/trainer.py`)

### Current Implementation

**Location:** `TrainingCallback._on_episode_end()` (lines 441-519)

**Current behavior:**
- **Line 459:** TODO comment: `# TODO: Implement proper win detection based on game outcome`
- **Line 460:** `win = None  # Will be None for now since we don't have game outcome in callback`
- **Line 461-466:** Logs episode with `win=None`:
  ```python
  self.run_logger.log_episode(
      episode=episode_num,
      total_reward=self.env_current_reward[env_id],
      steps=self.env_current_length[env_id],
      win=win
  )
  ```

**Why win is None:**
- The callback receives `info` dict from SB3's `locals` (line 387)
- The info dict doesn't contain winner/game_over information
- The callback doesn't have direct access to the underlying `BlokusEnv` or `BlokusGame` instances

**Access to game state:**
- The callback can access `info` dict from `self.locals.get("infos", [])` (line 387)
- But the info dict only contains what `_get_info()` returns (score, legal_moves_count, etc.)
- No direct access to `env.game.winner` or `env.game.is_game_over()`

---

## 6. Proposed Canonical Win Definition

For training purposes, assuming the agent controls `player_0`:

**Win Definition:**
```python
def compute_win(player_0_score: int, other_scores: Dict[Player, int]) -> float:
    """
    Compute win value for player_0.
    
    Returns:
        - 1.0 if player_0 has strictly highest score
        - 0.0 if any other player has strictly higher score than player_0
        - 0.5 if player_0 ties for highest score (true tie)
    """
    max_score = max([player_0_score] + list(other_scores.values()))
    player_0_is_max = (player_0_score == max_score)
    
    # Count how many players have max score
    players_with_max = sum(1 for s in [player_0_score] + list(other_scores.values()) if s == max_score)
    
    if player_0_is_max and players_with_max == 1:
        return 1.0  # Player_0 wins
    elif not player_0_is_max:
        return 0.0  # Player_0 loses
    else:
        return 0.5  # Tie (player_0 tied for highest)
```

**Alternative (boolean):**
If boolean is preferred:
- `True` if player_0 has strictly highest score
- `False` if any other player has strictly higher score
- `None` if tie (or could use `False` for ties)

**Recommendation:** Use float (1.0/0.5/0.0) to distinguish wins from ties, which is useful for training metrics.

---

## 7. Plan for Implementation

### High-Level Approach

**Where to compute final scores and winner:**
1. **Engine level** (already done): `BlokusGame.get_winner()` and `get_score()` are implemented
2. **Environment level** (needs update): Add winner/final_scores to info dict on terminal steps
3. **Training loop** (needs update): Extract win from info dict in callback

### Implementation Steps

#### Step 1: Update Environment Info Dict (`envs/blokus_v0.py`)

**Modify `_get_info()` to include game-over information:**
```python
def _get_info(self, agent: str) -> Dict[str, Any]:
    player = self._agent_to_player(agent)
    
    info = {
        "legal_action_mask": ...,
        "legal_moves_count": ...,
        "score": self.game.get_score(player),
        "pieces_used": ...,
        "pieces_remaining": ...,
        "can_move": ...,
    }
    
    # Add game-over information
    if self.game.is_game_over():
        winner = self.game.winner
        all_scores = {p: self.game.get_score(p) for p in Player}
        
        info["game_over"] = True
        info["winner"] = winner.name if winner else None
        info["final_scores"] = {p.name: s for p, s in all_scores.items()}
        
        # For player_0 (the training agent)
        if player == Player.RED:  # player_0 = RED
            player_0_score = all_scores[Player.RED]
            other_scores = {p: s for p, s in all_scores.items() if p != Player.RED}
            info["is_winner"] = compute_win(player_0_score, other_scores)
    else:
        info["game_over"] = False
        info["winner"] = None
        info["final_scores"] = None
        info["is_winner"] = None
    
    return info
```

#### Step 2: Update Training Callback (`training/trainer.py`)

**Modify `_on_episode_end()` to extract win from info:**
```python
def _on_episode_end(self, env_id: int = 0):
    # ... existing code ...
    
    if self.run_logger:
        # Extract win from info dict if available
        # Get the last info dict for this environment
        # (Need to track last info per env, or extract from locals)
        win = None
        
        # Try to get win from info dict
        # Note: This requires tracking info dicts per env
        # or accessing the environment directly
        if hasattr(self, 'last_info') and self.last_info:
            win = self.last_info.get("is_winner")
        
        self.run_logger.log_episode(
            episode=episode_num,
            total_reward=self.env_current_reward[env_id],
            steps=self.env_current_length[env_id],
            win=win
        )
```

**Challenge:** The callback receives info dicts in `_on_step()`, but needs to access the terminal step's info in `_on_episode_end()`. Options:
1. Store last info dict per env in callback
2. Access environment directly: `self.model.env.envs[env_id].env.infos["player_0"]` (for VecEnv)
3. Extract from `self.locals["infos"]` in `_on_step()` when `done=True`

**Recommended approach:** Store last info dict per env when `done=True` in `_on_step()`, then use it in `_on_episode_end()`.

#### Step 3: Handle Edge Cases

- **Truncation:** If episode is truncated (max steps), `game_over` may be `False`. Should we still compute win based on current scores?
- **Multi-env:** Ensure win detection works correctly for each parallel environment
- **Type consistency:** Ensure `win` is always `float` (1.0/0.5/0.0) or `None`, not mixed types

### Files to Modify

1. **`envs/blokus_v0.py`**
   - Update `_get_info()` to include `game_over`, `winner`, `final_scores`, `is_winner`
   - Add helper function `_compute_win_for_player_0()` if needed

2. **`training/trainer.py`**
   - Update `TrainingCallback._on_step()` to track last info dict per env when `done=True`
   - Update `TrainingCallback._on_episode_end()` to extract `win` from stored info dict
   - Remove TODO comment and `win = None` hardcoding

3. **Testing**
   - Add test to verify info dict contains win information on terminal steps
   - Add test to verify training callback correctly extracts and logs win
   - Test tie scenarios (win = 0.5)

---

## 8. Current State Summary

| Component | Status | Details |
|-----------|--------|---------|
| **Game termination** | ✅ Implemented | `_check_game_over()` detects when no players can move |
| **Final scores** | ✅ Implemented | `get_score()` computes scores with bonuses |
| **Winner determination** | ✅ Implemented | `get_winner()` returns winner or `None` for ties |
| **Winner in engine** | ✅ Stored | `BlokusGame.winner` instance variable |
| **Winner in env info** | ❌ Missing | Not included in `_get_info()` return dict |
| **Win in training loop** | ❌ Missing | Hardcoded to `None` with TODO comment |
| **Win in MongoDB logs** | ⚠️ Partial | `log_episode()` accepts `win` but always receives `None` |

---

## 9. Exact Locations

### Where `win = None` is set:

**File:** `training/trainer.py`  
**Line:** 460  
**Context:**
```python
# Line 459: TODO comment
# TODO: Implement proper win detection based on game outcome

# Line 460: win is set to None
win = None  # Will be None for now since we don't have game outcome in callback
```

**Function:** `TrainingCallback._on_episode_end(env_id: int = 0)`  
**Called from:** `TrainingCallback._on_step()` when `done=True` (line 419)

---

## 10. Additional Notes

### Accessing Game State from Callback

The callback has limited access to game state:
- ✅ Can access `info` dict from `self.locals.get("infos", [])`
- ✅ Can access environment via `self.model.env` (but may be wrapped)
- ❌ Cannot directly access `BlokusEnv` or `BlokusGame` without unwrapping

**Unwrapping chain for VecEnv:**
```
self.model.env (VecEnv)
  → env.envs[env_id] (GymnasiumBlokusWrapper)
    → env.envs[env_id].env (BlokusEnv)
      → env.envs[env_id].env.game (BlokusGame)
```

**Unwrapping chain for single env:**
```
self.model.env (ActionMasker)
  → env.env (GymnasiumBlokusWrapper)
    → env.env.env (BlokusEnv)
      → env.env.env.game (BlokusGame)
```

### Alternative: Direct Environment Access

If info dict approach is problematic, could access environment directly:
```python
# In _on_episode_end()
if hasattr(self.model.env, 'num_envs'):
    # VecEnv
    blokus_env = self.model.env.envs[env_id].env
else:
    # Single env (unwrapped by ActionMasker)
    blokus_env = self.model.env.env.env

if blokus_env.game.is_game_over():
    winner = blokus_env.game.winner
    # Compute win for player_0...
```

**Recommendation:** Prefer info dict approach for cleaner separation of concerns, but direct access is a valid fallback.

---

## Conclusion

The engine fully implements game termination, scoring, and winner determination. The gap is in propagating this information to the training loop. The recommended fix is to:

1. Add game-over information to the environment's info dict
2. Extract win from info dict in the training callback
3. Log win to MongoDB for tracking win rates

This maintains clean separation between engine, environment, and training layers while enabling proper win detection in training metrics.

---

## 11. Implementation Status

**Status:** ✅ **FULLY IMPLEMENTED**

### Engine Level (WD-02)
- ✅ **GameResult dataclass**: Implemented in `engine/game.py` with `scores`, `winner_ids`, and `is_tie` fields
- ✅ **get_game_result() method**: Returns canonical game result using existing scoring logic
- ✅ **Refactored existing code**: `get_winner()` and `_check_game_over()` now use `get_game_result()`
- ✅ **Tests**: Comprehensive unit tests in `tests/test_game_result.py` (11 tests, all passing)

### Environment Level (WD-03)
- ✅ **Terminal step info**: `envs/blokus_v0.py` now includes `final_scores`, `winner_ids`, `is_tie`, and `player0_won` in info dict on terminal steps
- ✅ **Game result storage**: `_check_termination_truncation()` calls `get_game_result()` and stores it
- ✅ **Multi-agent support**: All agents receive consistent game result information
- ✅ **Tests**: Comprehensive tests in `tests/test_env_win_detection.py` (6 tests, all passing)

### Training Loop Level (WD-04)
- ✅ **Win detection**: `TrainingCallback._compute_win_from_info()` extracts win value from terminal step info
- ✅ **Win calculation**: 
  - `1.0` if `player0_won=True` and `is_tie=False` (player_0 wins uniquely)
  - `0.5` if `is_tie=True` and player_0 in `winner_ids` (player_0 ties)
  - `0.0` otherwise (player_0 loses)
- ✅ **MongoDB logging**: `run_logger.log_episode()` now receives win value (float) instead of `None`
- ✅ **Rolling win rate**: Updated to handle float win values (gives partial credit for ties)
- ✅ **Episode logging**: Win value included in episode completion logs
- ✅ **Error handling**: Warns if game result fields are missing (truncation edge case)

### Data Flow
```
Game Engine → Environment → Training Callback → MongoDB Logger
     ↓              ↓              ↓                    ↓
GameResult    info dict      win (float)        EpisodeMetric
```

### Files Modified
1. `engine/game.py`: Added `GameResult` and `get_game_result()`
2. `envs/blokus_v0.py`: Added game result to terminal step info
3. `training/trainer.py`: Extract and use win from info dict
4. `training/run_logger.py`: Updated to handle float win values
5. `tests/test_game_result.py`: Tests for engine-level game result
6. `tests/test_env_win_detection.py`: Tests for environment-level win detection

### Next Steps
- Monitor training runs to verify win detection works correctly in practice
- Consider adding win rate metrics to TensorBoard logging
- Potentially add win rate to evaluation scripts

---

## 12. How to Verify Win Detection

### Quick Smoke Test

Run the win detection smoke test from `docs/run_checklist.md` to confirm win values show up in logs:

```bash
PYTHONPATH=. python training/trainer.py --mode smoke --total-timesteps 2000
```

### What to Check

1. **Episode logs contain win values:**
   ```bash
   grep "Episode.*completed.*win=" runs/*/training.log
   ```
   Should show lines like: `Episode X completed: reward=..., length=..., win=1.0`

2. **No warnings about missing game result fields:**
   ```bash
   grep -i "missing.*final_scores\|missing.*winner_ids" runs/*/training.log
   ```
   Should be empty (no warnings)

3. **Win values are present:**
   - `win=1.0` - Player_0 won
   - `win=0.5` - Player_0 tied
   - `win=0.0` - Player_0 lost
   - `win=None` - Should not occur in normal gameplay

### Full Smoke Test

The full smoke test (15000 timesteps, 10 episodes) also validates win detection:

```bash
PYTHONPATH=. python training/trainer.py --mode smoke
```

This provides more episodes to verify win detection across different game outcomes.

### MongoDB Verification

If MongoDB is enabled, verify that win values are being logged:

```python
# Check MongoDB for episode metrics with win values
from webapi.db.mongo import get_database
import asyncio

async def check_win_logging():
    db = get_database()
    run = await db.training_runs.find_one({"status": "running"})
    if run:
        episodes = run.get("metrics", {}).get("episodes", [])
        wins = [e.get("win") for e in episodes if e.get("win") is not None]
        print(f"Episodes with win values: {len(wins)}")
        print(f"Win values: {wins[:10]}")  # First 10

asyncio.run(check_win_logging())
```

### Expected Behavior

- **All completed episodes** should have win values (1.0, 0.5, or 0.0)
- **No warnings** about missing game result fields
- **Win values distributed** across 1.0, 0.5, and 0.0 (depending on game outcomes)
- **Consistent logging** - win values appear in both console and log files

