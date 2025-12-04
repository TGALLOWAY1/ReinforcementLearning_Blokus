# Blokus Environment: "No Legal Moves" & Dead Agent Handling Audit

## Overview

This document audits how the Blokus environment (`envs/blokus_v0.py`) handles:
1. Situations where an agent has no legal moves
2. PettingZoo "dead agent" semantics
3. The interaction between AEC (Agent Environment Cycle) and Gymnasium wrapper patterns

## Environment Architecture

### Base Environment: `BlokusEnv` (AEC)

The core environment is a PettingZoo AEC (Agent Environment Cycle) environment:
- **Class**: `BlokusEnv(AECEnv)` (line 47)
- **Agents**: 4 players (`player_0`, `player_1`, `player_2`, `player_3`)
- **Turn Management**: Uses `agent_selector` to cycle through agents (line 104)
- **Current Agent**: Tracked in `self.agent_selection` (set in `reset()` at line 196)

### Wrapper: `GymnasiumBlokusWrapper`

For Gymnasium/Stable-Baselines3 compatibility:
- **Class**: `GymnasiumBlokusWrapper(gym.Env)` (line 507)
- **Focus Agent**: Always `player_0` (hardcoded at line 523: `self.agent_name = "player_0"`)
- **Purpose**: Converts AEC interface to single-agent Gymnasium interface

## Current Agent Selection (AEC-Style)

### How Current Agent is Determined

1. **Initialization**: `self._agent_selector = agent_selector(self.agents)` (line 104, 193)
2. **Reset**: `self.agent_selection = self._agent_selector.next()` (line 196)
3. **After Step**: `self.agent_selection = self._agent_selector.next()` (line 385)
4. **After Skip Turn**: `self.agent_selection = self._agent_selector.next()` (line 399)

The `agent_selector` cycles through all active agents in `self.agents`. When an agent is terminated, it may still be in `self.agents` but will be skipped by the selector if properly implemented.

## Detection of "No Legal Moves"

### Where Detection Happens

**Location**: `_get_info()` method (line 248)

```python
# Line 263: Get legal moves
legal_moves = self.move_generator.get_legal_moves(self.game.board, player)

# Line 267-273: Warning when no legal moves
if len(legal_moves) == 0:
    if MASK_DEBUG_LOGGING:
        _mask_logger.warning(
            f"BlokusEnv: NO LEGAL MOVES for {agent} (player {player.name}) "
            f"at step {self.step_count}. "
            f"Mask will be all False - this will break MaskablePPO!"
        )
```

### What Happens When No Legal Moves Are Detected

1. **Mask Construction**: An all-False mask is created (line 264: `legal_action_mask = np.zeros(...)`)
2. **Info Field**: `info["can_move"] = False` is set (line 314)
3. **Warning Logged**: Diagnostic warning is printed (lines 267-273)
4. **Termination Check**: `_check_termination_truncation()` is called after each step (line 382)

## Termination/Truncation Logic

### `_check_termination_truncation()` Method (line 402)

This method sets `self.terminations[agent] = True` in several scenarios:

1. **Game Over** (lines 405-418):
   - If `self.game.is_game_over()` returns True
   - All agents are terminated
   - Winner gets bonus reward (+100), tie gets (+10)

2. **Max Episode Steps** (lines 421-423):
   - If `self.step_count >= self.max_episode_steps`
   - All agents are truncated

3. **No Legal Moves** (lines 425-438):
   ```python
   # Check if any agent can't move
   agents_can_move = []
   for agent in self.agents:
       if not self.terminations[agent] and not self.truncations[agent]:
           can_move = self.infos[agent]["can_move"]
           if can_move:
               agents_can_move.append(agent)
           else:
               self.terminations[agent] = True  # ← Agent terminated here
   
   # If no agents can move, terminate all
   if not agents_can_move:
       for agent in self.agents:
           self.terminations[agent] = True
   ```

**Key Point**: When an agent has no legal moves (`can_move == False`), it is immediately terminated (line 433).

## Dead Agent Handling in `step()`

### `step()` Method (line 341)

```python
def step(self, action: int) -> None:
    """Execute one step in the environment."""
    if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
        self._was_dead_step(action)  # ← Line 344: Calls PettingZoo's dead agent handler
        return
    # ... rest of step logic
```

**Critical Issue**: If `self.agent_selection` points to a terminated agent, `_was_dead_step(action)` is called. This PettingZoo method raises `ValueError("when an agent is dead, the only valid action is None")` if `action is not None`.

## Benchmark Script Analysis

### Environment Construction

**File**: `scripts/benchmark_env.py`

```python
# Line 56: Creates Gymnasium wrapper
env = make_gymnasium_env(render_mode=None, max_episode_steps=config.max_steps_per_episode)
```

This returns a `GymnasiumBlokusWrapper` that wraps the AEC `BlokusEnv`.

### Step Loop Structure

```python
# Line 86: Step call
obs, reward, terminated, truncated, info = env.step(action)
```

**Problem**: The wrapper's `step()` method (line 566) calls `self.env.step(action)` directly, which:
1. Executes the step for `self.env.agent_selection` (the current AEC agent)
2. This may NOT be `player_0` (the wrapper's assumed agent)
3. If `agent_selection` is a terminated agent, `_was_dead_step(action)` raises ValueError

### Wrapper's `step()` Implementation

```python
def step(self, action: int):
    # Line 580: Calls AEC env's step() for current agent_selection
    self.env.step(action)
    
    # Line 583-587: Returns info for wrapper's agent_name (player_0)
    obs = self.env.observe(self.agent_name)
    reward = self.env.rewards[self.agent_name]
    terminated = self.env.terminations[self.agent_name]
    truncated = self.env.truncations[self.agent_name]
    info = self.env.infos[self.agent_name]
    
    return obs, reward, terminated, truncated, info
```

**Issue**: The wrapper assumes it controls `player_0`, but calls `step()` on whatever agent the AEC env has selected. This creates a mismatch:
- Wrapper thinks: "I'm controlling player_0"
- AEC env thinks: "Current agent is agent_selection (could be player_2, player_3, etc.)"
- If `agent_selection` is terminated, `_was_dead_step()` raises an error

## Diagnosis

### Root Cause

The error occurs due to **two interacting issues**:

1. **AEC-Gymnasium Mismatch**: The `GymnasiumBlokusWrapper` treats the environment as single-agent (always `player_0`), but the underlying AEC env cycles through all 4 agents. When the wrapper calls `env.step(action)`, it's executing for the AEC's current `agent_selection`, which may not be `player_0`.

2. **Dead Agent Step**: When an agent runs out of legal moves:
   - `_check_termination_truncation()` sets `self.terminations[agent] = True` (line 433)
   - The AEC env's `agent_selection` may still point to this terminated agent
   - When `step(action)` is called, it checks `if self.terminations[self.agent_selection]` (line 343)
   - This triggers `_was_dead_step(action)`, which raises ValueError if `action is not None`

### Sequence of Events in Benchmark

1. Step 62: `player_2` has no legal moves
2. `_check_termination_truncation()` sets `self.terminations["player_2"] = True`
3. Warning printed: "NO LEGAL MOVES for player_2"
4. AEC env's `agent_selection` may still be or become `player_2`
5. Benchmark calls `env.step(action)` (wrapper's step)
6. Wrapper calls `self.env.step(action)` (AEC env's step)
7. AEC env checks: `if self.terminations[self.agent_selection]` → True for `player_2`
8. `_was_dead_step(action)` is called with non-None action
9. PettingZoo raises: `ValueError("when an agent is dead, the only valid action is None")`

### Is This an Env Logic Issue or Benchmark Misuse?

**Both**:

1. **Env Logic Issue**: The wrapper doesn't properly handle the AEC agent selection. It should either:
   - Skip dead agents before calling `step()`
   - Only call `step()` when `agent_selection == agent_name`
   - Or properly advance the agent selector until a live agent is selected

2. **Benchmark Misuse**: The benchmark assumes a single-agent Gymnasium interface, but the wrapper doesn't fully abstract away the AEC nature. The benchmark should check if the current agent is terminated before calling `step()`.

### Additional Issues for MaskablePPO

The warning "Mask will be all False - this will break MaskablePPO!" (line 272) indicates that when an agent has no legal moves:
- The action mask is all False
- MaskablePPO cannot sample a valid action from an all-False mask
- This will cause training to fail

**Solution Needed**: When an agent has no legal moves, the environment should either:
1. Automatically skip the agent's turn (mark as terminated and advance)
2. Provide a special "pass" action that is always legal
3. Reset the episode when no agents can move

## Key Code Snippets

### `envs/blokus_v0.py` - Step Method

```python
# Lines 341-345
def step(self, action: int) -> None:
    """Execute one step in the environment."""
    if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
        self._was_dead_step(action)  # ← Raises ValueError if action is not None
        return
```

### `envs/blokus_v0.py` - No Legal Moves Detection

```python
# Lines 267-273
if len(legal_moves) == 0:
    if MASK_DEBUG_LOGGING:
        _mask_logger.warning(
            f"BlokusEnv: NO LEGAL MOVES for {agent} (player {player.name}) "
            f"at step {self.step_count}. "
            f"Mask will be all False - this will break MaskablePPO!"
        )
```

### `envs/blokus_v0.py` - Termination on No Moves

```python
# Lines 425-433
agents_can_move = []
for agent in self.agents:
    if not self.terminations[agent] and not self.truncations[agent]:
        can_move = self.infos[agent]["can_move"]
        if can_move:
            agents_can_move.append(agent)
        else:
            self.terminations[agent] = True  # ← Agent terminated when can_move == False
```

### `scripts/benchmark_env.py` - Environment Construction

```python
# Line 56
env = make_gymnasium_env(render_mode=None, max_episode_steps=config.max_steps_per_episode)
```

### `scripts/benchmark_env.py` - Step Loop

```python
# Line 86
obs, reward, terminated, truncated, info = env.step(action)
```

### `envs/blokus_v0.py` - Wrapper's Step Method

```python
# Lines 566-589
def step(self, action: int):
    # Execute action for current agent
    self.env.step(action)  # ← Calls AEC env's step() for agent_selection, not necessarily player_0
    
    # Get observation and info for our agent
    obs = self.env.observe(self.agent_name)  # ← Always player_0
    reward = self.env.rewards[self.agent_name]
    terminated = self.env.terminations[self.agent_name]
    truncated = self.env.truncations[self.agent_name]
    info = self.env.infos[self.agent_name]
    
    return obs, reward, terminated, truncated, info
```

## Recommendations

1. **Fix Wrapper**: The `GymnasiumBlokusWrapper.step()` should advance the AEC env until `agent_selection == agent_name` before calling `step()`, or skip dead agents.

2. **Handle Dead Agents**: Before calling `env.step(action)`, check if the current agent is terminated and handle appropriately (skip, reset, or pass None action).

3. **MaskablePPO Compatibility**: When no legal moves exist, either:
   - Automatically terminate and skip the agent
   - Add a "pass" action that's always legal
   - Reset the episode

4. **Agent Selector**: Ensure the `agent_selector` properly skips terminated agents, or manually filter them out before selection.

