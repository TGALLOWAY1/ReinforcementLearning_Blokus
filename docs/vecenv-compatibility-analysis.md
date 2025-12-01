# VecEnv Compatibility Analysis

## Environment API

### Core Environment Class

**File:** `envs/blokus_v0.py`

**Class Hierarchy:**
1. **`BlokusEnv`** (lines 41-421) - PettingZoo AEC environment (base)
2. **`GymnasiumBlokusWrapper`** (lines 439-491) - Gymnasium compatibility wrapper
3. **Returned by `make_gymnasium_env()`** (lines 494-506) - Factory function

**Actual Environment Used in Training:**
- `GymnasiumBlokusWrapper` instance (wrapped around `BlokusEnv`)
- Created via: `make_gymnasium_env(render_mode=None, max_episode_steps=config.max_steps_per_episode)`

### Gymnasium API Implementation

**`GymnasiumBlokusWrapper`** implements the standard Gymnasium API:

#### `reset(seed=None, options=None)` (lines 466-473)
- **Returns:** `(obs, info)` tuple
- **obs:** `np.ndarray` of shape `(obs_channels, height, width)` = `(30, 20, 20)`
  - Type: `spaces.Box(low=0, high=1, shape=(30, 20, 20), dtype=np.float32)`
- **info:** `Dict[str, Any]` containing:
  - `"legal_action_mask"`: boolean numpy array of shape `(action_space_size,)` ≈ `(36400,)`
  - `"legal_moves_count"`: int
  - `"score"`: int
  - `"pieces_used"`: int
  - `"pieces_remaining"`: int
  - `"can_move"`: bool

#### `step(action: int)` (lines 475-487)
- **Input:** `action` - integer in range `[0, action_space_size)`
- **Returns:** `(obs, reward, terminated, truncated, info)` tuple
- **obs:** Same as `reset()` - `np.ndarray` shape `(30, 20, 20)`
- **reward:** `float` - dense reward (score change) + sparse reward (game end bonuses)
- **terminated:** `bool` - True if game ended normally
- **truncated:** `bool` - True if episode truncated (max steps reached)
- **info:** Same dict structure as `reset()`

### Action and Observation Spaces

**Action Space:**
- **Type:** `gymnasium.spaces.Discrete`
- **Size:** `action_space_size` ≈ 36,400
- **Encoding:** Flattened mapping from `(piece_id, orientation, anchor_row, anchor_col)` to discrete action index
  - `piece_id`: 1-21 (Blokus pieces)
  - `orientation`: 0 to `len(orientations)-1` (varies by piece, up to 8)
  - `anchor_row, anchor_col`: 0-19 (20×20 board positions)

**Observation Space:**
- **Type:** `gymnasium.spaces.Box`
- **Shape:** `(30, 20, 20)` = `(channels, height, width)`
- **Channels:**
  - 0-4: Board channels (empty + 4 players)
  - 5-25: Remaining pieces (21 pieces)
  - 26-29: Last move (piece_id, orientation, row, col)
- **Dtype:** `np.float32`
- **Range:** `[0.0, 1.0]` (normalized values)

### Gymnasium API Compliance

✅ **Compliant:** The `GymnasiumBlokusWrapper` correctly implements:
- `reset()` returning `(obs, info)`
- `step(action)` returning `(obs, reward, terminated, truncated, info)`
- `action_space` and `observation_space` attributes
- `metadata` attribute
- `render()` method

**Note:** The environment is a **single-agent wrapper** around a multi-agent PettingZoo environment. It focuses on `"player_0"` only (line 448).

---

## Action Mask Function

### Mask Function Location

**File:** `training/trainer.py`  
**Function:** `mask_fn(env)` (lines 79-176)

### How Masks Are Computed

1. **Called by:** `ActionMasker` wrapper (from `sb3_contrib.common.wrappers`)
2. **Input:** `env` - The wrapped environment (ActionMasker passes the unwrapped env)
3. **Process:**
   ```python
   # Access underlying BlokusEnv
   blokus_env = env.env  # Unwrap: ActionMasker -> GymnasiumBlokusWrapper -> BlokusEnv
   agent_name = env.agent_name  # "player_0"
   
   # Extract mask from environment's info dict
   mask = blokus_env.infos[agent_name]["legal_action_mask"]
   mask = np.asarray(mask, dtype=np.bool_)
   ```
4. **Output:** Boolean numpy array of shape `(action_space.n,)` ≈ `(36400,)`
   - `True` = legal action
   - `False` = illegal action

### Mask Construction (in BlokusEnv)

**File:** `envs/blokus_v0.py`  
**Method:** `_get_info(agent: str)` (lines 207-264)

The mask is constructed by:
1. Getting legal moves from `LegalMoveGenerator.get_legal_moves()`
2. Creating a boolean array of size `action_space_size`
3. Marking legal actions as `True` by mapping moves to action indices
4. Storing in `self.infos[agent]["legal_action_mask"]`

### Single-Environment Assumptions

❌ **NOT VecEnv-Compatible:** The `mask_fn()` function makes several single-environment assumptions:

1. **Direct Dict Access:**
   ```python
   blokus_env.infos[agent_name]["legal_action_mask"]
   ```
   - Assumes `infos` is a single dict
   - In VecEnv, `infos` from `reset()`/`step()` is a **list of dicts** (one per environment)

2. **Single Agent Name:**
   ```python
   agent_name = env.agent_name  # "player_0"
   ```
   - Accesses `env.agent_name` directly
   - In VecEnv, each environment instance would have its own `agent_name`

3. **Single Environment Unwrapping:**
   ```python
   blokus_env = env.env  # Assumes single env structure
   ```
   - Assumes `env.env` is a single `GymnasiumBlokusWrapper`
   - In VecEnv, the structure would be different (VecEnv -> list of envs)

4. **Termination Check:**
   ```python
   if agent_name in blokus_env.terminations and blokus_env.terminations[agent_name]:
   ```
   - Accesses `terminations` dict directly
   - In VecEnv, termination info would be in the `info` dicts from `step()`

### Current Usage

**Applied in:** `training/trainer.py` line 494
```python
env = ActionMasker(env, mask_fn)
```

This wraps a **single** `GymnasiumBlokusWrapper` instance with `ActionMasker`.

---

## VecEnv Compatibility

### What is Already VecEnv-Friendly

✅ **Environment API:**
- `GymnasiumBlokusWrapper` implements the standard Gymnasium API
- Can be wrapped by SB3's `DummyVecEnv` or `SubprocVecEnv`
- Observation and action spaces are well-defined and consistent

✅ **MaskablePPO Support:**
- `MaskablePPO` from `sb3_contrib` supports vectorized environments
- Can handle batched action masks (one mask per environment)

✅ **ActionMasker Wrapper:**
- `ActionMasker` from `sb3_contrib.common.wrappers` can work with VecEnv
- However, the **mask function** must be adapted for VecEnv

### What Needs to Change for VecEnv

#### 1. Environment Vectorization

**Current:**
```python
env = make_gymnasium_env(render_mode=None, max_episode_steps=config.max_steps_per_episode)
env = ActionMasker(env, mask_fn)
```

**Required for VecEnv:**
```python
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

def make_env():
    env = make_gymnasium_env(render_mode=None, max_episode_steps=config.max_steps_per_episode)
    return env

# Option 1: DummyVecEnv (sequential, same process)
env = DummyVecEnv([make_env for _ in range(n_envs)])

# Option 2: SubprocVecEnv (parallel, separate processes)
env = SubprocVecEnv([make_env for _ in range(n_envs)])

# Then wrap with ActionMasker
env = ActionMasker(env, mask_fn_vecenv)  # Need VecEnv-compatible mask function
```

#### 2. Mask Function Adaptation

**Current `mask_fn()` (Single-Env):**
```python
def mask_fn(env):
    blokus_env = env.env
    agent_name = env.agent_name
    mask = blokus_env.infos[agent_name]["legal_action_mask"]
    return np.asarray(mask, dtype=np.bool_)
```

**Required `mask_fn_vecenv()` (VecEnv-Compatible):**
```python
def mask_fn_vecenv(env):
    """
    Extract action masks for vectorized environments.
    
    Args:
        env: VecEnv instance (wrapped with ActionMasker)
        
    Returns:
        np.ndarray of shape (n_envs, action_space.n) - batch of masks
    """
    # When ActionMasker wraps a VecEnv, it passes the VecEnv itself
    # We need to extract masks from each sub-environment
    
    masks = []
    for i in range(env.num_envs):
        # Get the sub-environment
        sub_env = env.envs[i]  # Unwrap VecEnv to get list of envs
        # Unwrap ActionMasker if present, then GymnasiumBlokusWrapper
        if hasattr(sub_env, 'env'):  # ActionMasker wrapper
            wrapped_env = sub_env.env
        else:
            wrapped_env = sub_env
            
        # Access underlying BlokusEnv
        blokus_env = wrapped_env.env  # GymnasiumBlokusWrapper -> BlokusEnv
        agent_name = wrapped_env.agent_name  # "player_0"
        
        # Extract mask
        mask = blokus_env.infos[agent_name]["legal_action_mask"]
        masks.append(np.asarray(mask, dtype=np.bool_))
    
    # Stack into batch: (n_envs, action_space.n)
    return np.stack(masks, axis=0)
```

**Alternative Approach (Using Info from VecEnv):**
```python
def mask_fn_vecenv(env):
    """
    Extract action masks from VecEnv's info dicts.
    
    This approach uses the info dicts returned by VecEnv.step()/reset(),
    which are already batched.
    """
    # ActionMasker should provide access to the last info dicts
    # This requires checking how ActionMasker handles VecEnv internally
    
    # If ActionMasker stores last infos:
    if hasattr(env, '_last_infos'):
        masks = []
        for info in env._last_infos:
            mask = info["legal_action_mask"]
            masks.append(np.asarray(mask, dtype=np.bool_))
        return np.stack(masks, axis=0)
    else:
        # Fallback: extract from sub-environments
        # (implementation as above)
        ...
```

#### 3. ActionMasker Behavior with VecEnv

**Key Question:** How does `ActionMasker` handle VecEnv?

- **Expected Behavior:** `ActionMasker` should call `mask_fn(env)` where `env` is the VecEnv
- **Mask Function Must Return:** 
  - For single env: `np.ndarray` of shape `(action_space.n,)`
  - For VecEnv: `np.ndarray` of shape `(n_envs, action_space.n)`

**Verification Needed:**
- Check `sb3_contrib` source code or documentation for `ActionMasker` VecEnv behavior
- Test whether `ActionMasker` automatically handles VecEnv or requires special mask function

#### 4. Training Code Changes

**Current (`training/trainer.py`):**
```python
env = make_gymnasium_env(...)
env = ActionMasker(env, mask_fn)
model = MaskablePPO(..., env=env)
```

**Required for VecEnv:**
```python
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return make_gymnasium_env(...)

# Create vectorized environment
n_envs = 4  # or config parameter
env = DummyVecEnv([make_env for _ in range(n_envs)])

# Wrap with ActionMasker (using VecEnv-compatible mask function)
env = ActionMasker(env, mask_fn_vecenv)

# Model creation remains the same
model = MaskablePPO(..., env=env)
```

#### 5. TrainingCallback Adaptation

**Current (`training/trainer.py` lines 219-230):**
```python
# SB3 provides these as lists (for vectorized envs), but we use single env
rewards = self.locals.get("rewards", [])
dones = self.locals.get("dones", [])
infos = self.locals.get("infos", [])

# Extract values (handle both list and single value)
reward = rewards[0] if rewards else 0.0
done = dones[0] if dones else False
info = infos[0] if infos else {}
```

**Note:** The callback already handles lists (for VecEnv compatibility), but currently only processes the first environment (`[0]`). For full VecEnv support, the callback would need to:
- Process all environments in the batch
- Aggregate metrics across environments
- Handle per-environment episode tracking

#### 6. Configuration Updates

**Add to `TrainingConfig`:**
- `n_envs: int = 1` - Number of parallel environments
- `vec_env_type: Literal["dummy", "subproc"] = "dummy"` - VecEnv type

---

## Summary of Compatibility Status

| Component | Single-Env Status | VecEnv Status | Changes Required |
|-----------|------------------|---------------|------------------|
| `GymnasiumBlokusWrapper` | ✅ Works | ✅ Compatible | None (standard Gym API) |
| `BlokusEnv` | ✅ Works | ✅ Compatible | None (wrapped by GymnasiumBlokusWrapper) |
| `make_gymnasium_env()` | ✅ Works | ✅ Compatible | None (factory function) |
| `mask_fn()` | ✅ Works | ❌ **NOT Compatible** | **Must adapt for VecEnv** |
| `ActionMasker` wrapper | ✅ Works | ⚠️ **Unknown** | **Verify VecEnv support** |
| `MaskablePPO` | ✅ Works | ✅ Compatible | None (supports VecEnv) |
| `TrainingCallback` | ✅ Works | ⚠️ **Partially Compatible** | **Process all envs in batch** |
| Training code | ✅ Works | ❌ **NOT Compatible** | **Add VecEnv creation** |

### Critical Changes Required

1. **Adapt `mask_fn()` for VecEnv:**
   - Handle batched info dicts (list instead of single dict)
   - Return mask batch of shape `(n_envs, action_space.n)`
   - Access sub-environments correctly

2. **Verify `ActionMasker` VecEnv behavior:**
   - Check if it automatically handles VecEnv
   - Confirm expected mask function signature for VecEnv
   - Test with simple VecEnv example

3. **Update training code:**
   - Add VecEnv creation (DummyVecEnv or SubprocVecEnv)
   - Use VecEnv-compatible mask function
   - Add configuration for `n_envs`

4. **Testing:**
   - Test with `n_envs=2` first (smoke test)
   - Verify masks are correctly extracted for all environments
   - Ensure training loop handles batched observations/rewards correctly

---

## Recommended Implementation Order

1. **Research Phase:**
   - Check `sb3_contrib` documentation/examples for `ActionMasker` + `VecEnv`
   - Verify expected mask function signature for VecEnv

2. **Prototype Phase:**
   - Create `mask_fn_vecenv()` function
   - Test with `DummyVecEnv([make_env, make_env])` (2 envs)
   - Verify mask extraction works correctly

3. **Integration Phase:**
   - Add `n_envs` config parameter
   - Update `train()` function to create VecEnv
   - Use VecEnv-compatible mask function

4. **Validation Phase:**
   - Run smoke test with `n_envs=2`
   - Compare training metrics with single-env baseline
   - Verify no performance regressions

