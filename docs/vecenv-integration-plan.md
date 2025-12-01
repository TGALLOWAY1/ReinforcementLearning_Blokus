# VecEnv Integration Plan

## Where to Introduce num_envs

### TrainingConfig Changes

**File:** `training/config.py`

**Location:** Add to `TrainingConfig` dataclass (after line 81, before `__post_init__`)

```python
num_envs: int = 1
vec_env_type: Literal["dummy", "subproc"] = "dummy"
```

**Rationale:**
- `num_envs=1` maintains backward compatibility (single-env by default)
- `vec_env_type` allows choosing between `DummyVecEnv` (sequential, same process) and `SubprocVecEnv` (parallel, separate processes)
- Default to `"dummy"` for robustness and easier debugging

**CLI Argument Addition:**

**Location:** `create_arg_parser()` function (after line 343, before `return parser`)

```python
parser.add_argument(
    "--num-envs",
    type=int,
    default=1,
    help="Number of parallel environments (default: 1, use VecEnv if > 1)"
)
parser.add_argument(
    "--vec-env-type",
    type=str,
    choices=["dummy", "subproc"],
    default="dummy",
    help="VecEnv type: 'dummy' for sequential (same process), 'subproc' for parallel (default: dummy)"
)
```

**Config Parsing:**

**Location:** `parse_args_to_config()` function (after line 402, before `config.__post_init__()`)

```python
if args.num_envs is not None:
    config.num_envs = args.num_envs
if args.vec_env_type:
    config.vec_env_type = args.vec_env_type
```

**Logging:**

**Location:** `TrainingConfig.log_config()` method (after line 189, before closing `=`)

```python
logger.info(f"Number of Environments: {self.num_envs}")
logger.info(f"VecEnv Type: {self.vec_env_type}")
```

### Entrypoint Changes in trainer.py

**Current Code Location:** `training/trainer.py`, lines 481-494

**Current Flow:**
```python
# Line 482-485: Create single environment
env = make_gymnasium_env(
    render_mode=None,
    max_episode_steps=config.max_steps_per_episode
)

# Line 488-491: Reset environment
if config.env_seed is not None:
    env.reset(seed=config.env_seed)
else:
    env.reset()

# Line 494: Wrap with ActionMasker
env = ActionMasker(env, mask_fn)
```

**Replacement Strategy:**
- Replace lines 481-494 with a call to a new factory function
- Factory will handle both single-env and VecEnv cases
- Factory will handle ActionMasker wrapping internally

**New Code Location:** Replace lines 481-494 with:
```python
from training.env_factory import make_training_env

env = make_training_env(config)
```

---

## VecEnv Factory Design

### Module Location

**New File:** `training/env_factory.py`

**Rationale:**
- Centralizes environment creation logic
- Keeps `trainer.py` clean and focused
- Makes it easy to test environment creation separately
- Follows single responsibility principle

### Factory Functions

#### 1. `make_single_env(config, seed=None) -> GymnasiumBlokusWrapper`

**Purpose:** Create a single environment instance, ready for ActionMasker wrapping.

**Signature:**
```python
def make_single_env(
    config: TrainingConfig,
    seed: Optional[int] = None
) -> GymnasiumBlokusWrapper:
```

**Implementation Plan:**
1. Call `make_gymnasium_env(render_mode=None, max_episode_steps=config.max_steps_per_episode)`
2. Reset with seed if provided
3. Return unwrapped `GymnasiumBlokusWrapper` (ActionMasker will be applied later)

**Usage:** Used internally by `make_training_env()` to create individual env instances.

#### 2. `make_training_env(config) -> Union[ActionMasker, VecEnv]`

**Purpose:** Main factory function that returns an environment ready for MaskablePPO training.

**Signature:**
```python
def make_training_env(
    config: TrainingConfig
) -> Union[ActionMasker, VecEnv]:
```

**Implementation Logic:**
```python
if config.num_envs == 1:
    # Single environment path (backward compatible)
    env = make_single_env(config, seed=config.env_seed)
    env = ActionMasker(env, mask_fn)
    return env
else:
    # VecEnv path
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    
    def make_env_fn(rank: int):
        """Factory function for creating individual env instances."""
        # Use different seeds for each env if env_seed is provided
        # This ensures diversity across parallel envs
        env_seed = None
        if config.env_seed is not None:
            env_seed = config.env_seed + rank
        
        env = make_single_env(config, seed=env_seed)
        return env
    
    # Create list of env factory functions
    env_fns = [lambda rank=i: make_env_fn(rank) for i in range(config.num_envs)]
    
    # Choose VecEnv type
    if config.vec_env_type == "subproc":
        vec_env = SubprocVecEnv(env_fns)
    else:  # "dummy"
        vec_env = DummyVecEnv(env_fns)
    
    # Wrap VecEnv with ActionMasker (using unified mask_fn)
    vec_env = ActionMasker(vec_env, mask_fn)
    
    return vec_env
```

**Key Design Decisions:**
1. **Seed Diversity:** Each parallel env gets `env_seed + rank` to ensure different initial states
2. **Default to DummyVecEnv:** More robust, easier to debug, no process overhead
3. **ActionMasker Applied After VecEnv:** This is the correct order - VecEnv first, then ActionMasker
4. **Return Type:** Union type indicates it can return either single env or VecEnv

**Dependencies:**
- Import `make_gymnasium_env` from `envs.blokus_v0`
- Import `ActionMasker` from `sb3_contrib.common.wrappers`
- Import `DummyVecEnv`, `SubprocVecEnv` from `stable_baselines3.common.vec_env`
- Import `mask_fn` from `training.trainer` (or move it to a shared location)

---

## Unified mask_fn Design (Single + VecEnv)

### Current Implementation

**File:** `training/trainer.py`, lines 79-176

**Current Assumptions:**
- Single environment only
- Accesses `env.env` (unwrapped GymnasiumBlokusWrapper)
- Accesses `env.agent_name` directly
- Accesses `blokus_env.infos[agent_name]["legal_action_mask"]` directly

### Refactored Design

**Location:** Keep in `training/trainer.py` (same file, refactor existing function)

**New Function Signature:**
```python
def mask_fn(env) -> np.ndarray:
    """
    Unified mask function that works for both single env and VecEnv.
    
    Args:
        env: Either:
            - ActionMasker-wrapped GymnasiumBlokusWrapper (single env)
            - ActionMasker-wrapped VecEnv (vectorized envs)
    
    Returns:
        - For single env: np.ndarray of shape (action_space.n,)
        - For VecEnv: np.ndarray of shape (num_envs, action_space.n)
    """
```

### Detection Strategy

**Method 1: Check for `num_envs` attribute (Recommended)**
```python
# VecEnv has num_envs attribute
if hasattr(env, 'num_envs'):
    # VecEnv path
    num_envs = env.num_envs
    masks = []
    for i in range(num_envs):
        sub_env = env.envs[i]  # Get sub-environment
        # Unwrap ActionMasker if present
        if hasattr(sub_env, 'env'):
            wrapped_env = sub_env.env
        else:
            wrapped_env = sub_env
        # Extract mask from underlying BlokusEnv
        blokus_env = wrapped_env.env
        agent_name = wrapped_env.agent_name
        mask = blokus_env.infos[agent_name]["legal_action_mask"]
        masks.append(np.asarray(mask, dtype=np.bool_))
    return np.stack(masks, axis=0)  # Shape: (num_envs, action_space.n)
else:
    # Single env path (existing logic)
    blokus_env = env.env
    agent_name = env.agent_name
    mask = blokus_env.infos[agent_name]["legal_action_mask"]
    return np.asarray(mask, dtype=np.bool_)  # Shape: (action_space.n,)
```

**Method 2: Type checking (Alternative)**
```python
from stable_baselines3.common.vec_env import VecEnv

if isinstance(env, VecEnv):
    # VecEnv path
    ...
else:
    # Single env path
    ...
```

**Recommendation:** Use Method 1 (`hasattr(env, 'num_envs')`) because:
- More robust (works even if VecEnv interface changes)
- Doesn't require importing VecEnv base class
- Clearer intent

### Refactoring Steps

1. **Extract Common Mask Extraction Logic:**
   ```python
   def _extract_mask_from_env(env) -> np.ndarray:
       """Extract mask from a single GymnasiumBlokusWrapper."""
       blokus_env = env.env  # Unwrap to BlokusEnv
       agent_name = env.agent_name
       # ... existing mask extraction logic ...
       return mask
   ```

2. **Update Main `mask_fn()`:**
   - Add VecEnv detection at the start
   - Branch to VecEnv path or single-env path
   - Reuse existing logic for single-env case
   - For VecEnv, loop over `env.envs` and extract mask from each

3. **Handle Edge Cases:**
   - Terminated agents (fallback mask)
   - Empty masks (no legal actions)
   - Shape validation for both single and batched masks

4. **Update Diagnostic Logging:**
   - Log whether single-env or VecEnv path was taken
   - For VecEnv, log mask stats per environment
   - Maintain existing diagnostic behavior for single-env

### Code Structure

```python
def mask_fn(env):
    """
    Unified mask function for single env and VecEnv.
    """
    # Detect VecEnv
    if hasattr(env, 'num_envs'):
        return _mask_fn_vecenv(env)
    else:
        return _mask_fn_single(env)

def _mask_fn_single(env):
    """Extract mask from single environment (existing logic)."""
    # ... existing implementation ...

def _mask_fn_vecenv(env):
    """Extract masks from vectorized environment."""
    num_envs = env.num_envs
    masks = []
    for i in range(num_envs):
        sub_env = env.envs[i]
        # Unwrap ActionMasker -> GymnasiumBlokusWrapper -> BlokusEnv
        if hasattr(sub_env, 'env'):
            wrapped_env = sub_env.env
        else:
            wrapped_env = sub_env
        blokus_env = wrapped_env.env
        agent_name = wrapped_env.agent_name
        
        # Extract mask (reuse logic from _mask_fn_single)
        mask = _extract_mask_from_wrapped_env(wrapped_env, blokus_env, agent_name)
        masks.append(mask)
    
    return np.stack(masks, axis=0)  # Shape: (num_envs, action_space.n)
```

---

## TrainingCallback Multi-Env Plan

### Current Implementation

**File:** `training/trainer.py`, lines 179-380

**Current State:**
- Tracks single episode: `current_episode_reward`, `current_episode_length`
- Only processes `rewards[0]`, `dones[0]`, `infos[0]`
- Single lists: `episode_rewards`, `episode_lengths`

### Required Changes

#### 1. Add Per-Environment State

**Location:** `TrainingCallback.__init__()` (after line 205)

**New State Variables:**
```python
# Per-environment tracking
self.num_envs = 1  # Will be set from config or detected from env
self.env_episode_rewards = {}  # Dict: {env_id: list of episode rewards}
self.env_episode_lengths = {}  # Dict: {env_id: list of episode lengths}
self.env_current_reward = {}  # Dict: {env_id: current episode reward}
self.env_current_length = {}  # Dict: {env_id: current episode length}
self.env_episode_count = {}  # Dict: {env_id: episode count}
```

**Initialization:**
```python
# Initialize per-env state (will be populated in _on_step)
for i in range(self.num_envs):
    self.env_episode_rewards[i] = []
    self.env_episode_lengths[i] = []
    self.env_current_reward[i] = 0.0
    self.env_current_length[i] = 0
    self.env_episode_count[i] = 0
```

#### 2. Detect Number of Environments

**Location:** `TrainingCallback.__init__()` or first `_on_step()` call

**Strategy:**
```python
# Option 1: Pass num_envs from config
def __init__(self, config, run_logger=None, model=None, verbose=0, num_envs=1):
    ...
    self.num_envs = num_envs

# Option 2: Detect from model's environment
def __init__(self, config, run_logger=None, model=None, verbose=0):
    ...
    if model and hasattr(model.env, 'num_envs'):
        self.num_envs = model.env.num_envs
    else:
        self.num_envs = 1
```

**Recommendation:** Pass `num_envs` explicitly from `train()` function for clarity.

#### 3. Update `_on_step()` Method

**Location:** `training/trainer.py`, lines 207-264

**Current Logic:**
```python
rewards = self.locals.get("rewards", [])
dones = self.locals.get("dones", [])
infos = self.locals.get("infos", [])

reward = rewards[0] if rewards else 0.0
done = dones[0] if dones else False
info = infos[0] if infos else {}
```

**New Logic:**
```python
rewards = self.locals.get("rewards", [])
dones = self.locals.get("dones", [])
infos = self.locals.get("infos", [])

# Process all environments in batch
for env_id in range(self.num_envs):
    reward = rewards[env_id] if env_id < len(rewards) else 0.0
    done = dones[env_id] if env_id < len(dones) else False
    info = infos[env_id] if env_id < len(infos) else {}
    
    # Update per-env tracking
    self.env_current_reward[env_id] += reward
    self.env_current_length[env_id] += 1
    
    # Sanity checks (per env)
    if self.config.enable_sanity_checks:
        self._sanity_check(reward, obs, action, info, env_id)
    
    # Check episode termination (per env)
    if done:
        self._on_episode_end(env_id)
    
    # Update step count (only once, not per env)
    if env_id == 0:
        self.step_count += 1
```

**Note:** `step_count` should increment once per batch, not per environment.

#### 4. Update `_on_episode_end()` Method

**Location:** `training/trainer.py`, lines 266-339

**New Signature:**
```python
def _on_episode_end(self, env_id: int = 0):
    """Handle episode end for a specific environment."""
```

**New Logic:**
```python
def _on_episode_end(self, env_id: int = 0):
    """Handle episode end for a specific environment."""
    # Update per-env episode count
    self.env_episode_count[env_id] += 1
    episode_num = self.env_episode_count[env_id]
    
    # Store episode metrics
    self.env_episode_rewards[env_id].append(self.env_current_reward[env_id])
    self.env_episode_lengths[env_id].append(self.env_current_length[env_id])
    
    # Log to MongoDB (per env)
    if self.run_logger:
        win = None  # TODO: Implement proper win detection
        self.run_logger.log_episode(
            episode=episode_num,
            total_reward=self.env_current_reward[env_id],
            steps=self.env_current_length[env_id],
            win=win,
            env_id=env_id  # Add env_id to distinguish parallel envs
        )
    
    # Log episode completion (only for first env or if verbose)
    if env_id == 0 or (self.config.mode == "smoke" or self.config.logging_verbosity >= 1):
        logger.info(
            f"Episode {episode_num} (env {env_id}) completed: "
            f"reward={self.env_current_reward[env_id]:.2f}, "
            f"length={self.env_current_length[env_id]}"
        )
    
    # Reset per-env tracking
    self.env_current_reward[env_id] = 0.0
    self.env_current_length[env_id] = 0
    
    # Check episode limit (aggregate across all envs)
    total_episodes = sum(self.env_episode_count.values())
    if self.config.max_episodes is not None and total_episodes >= self.config.max_episodes:
        logger.info(f"Reached max_episodes limit ({self.config.max_episodes}), stopping training")
        return False
    
    # Checkpoint saving (based on aggregate episode count)
    if (self.model and 
        self.config.checkpoint_interval_episodes and 
        total_episodes % self.config.checkpoint_interval_episodes == 0):
        # ... existing checkpoint logic ...
```

#### 5. Update Training Summary

**Location:** `training/trainer.py`, lines 731-743

**Current Logic:**
```python
logger.info(f"Total episodes: {callback.episode_count}")
logger.info(f"Total steps: {callback.step_count}")
if callback.episode_rewards:
    logger.info(f"Average reward: {np.mean(callback.episode_rewards):.2f}")
    ...
```

**New Logic:**
```python
# Aggregate across all environments
all_episode_rewards = []
all_episode_lengths = []
for env_id in range(callback.num_envs):
    all_episode_rewards.extend(callback.env_episode_rewards[env_id])
    all_episode_lengths.extend(callback.env_episode_lengths[env_id])

total_episodes = sum(callback.env_episode_count.values())
logger.info(f"Total episodes: {total_episodes}")
logger.info(f"Total steps: {callback.step_count}")
if all_episode_rewards:
    logger.info(f"Average reward: {np.mean(all_episode_rewards):.2f}")
    logger.info(f"Best episode reward: {np.max(all_episode_rewards):.2f}")
    logger.info(f"Worst episode reward: {np.min(all_episode_rewards):.2f}")
    
    # Optionally, show per-env stats
    if callback.num_envs > 1:
        logger.info("Per-environment statistics:")
        for env_id in range(callback.num_envs):
            env_rewards = callback.env_episode_rewards[env_id]
            if env_rewards:
                logger.info(
                    f"  Env {env_id}: {len(env_rewards)} episodes, "
                    f"avg reward: {np.mean(env_rewards):.2f}"
                )
```

#### 6. Backward Compatibility

**Maintain Single-Env Behavior:**
- When `num_envs == 1`, behavior should be identical to current implementation
- `episode_count` can be computed as `sum(self.env_episode_count.values())` for compatibility
- Keep `episode_rewards` and `episode_lengths` as aggregated lists for backward compatibility

**Optional Compatibility Layer:**
```python
@property
def episode_count(self):
    """Backward compatibility: total episodes across all envs."""
    return sum(self.env_episode_count.values())

@property
def episode_rewards(self):
    """Backward compatibility: all episode rewards flattened."""
    return [r for rewards in self.env_episode_rewards.values() for r in rewards]

@property
def episode_lengths(self):
    """Backward compatibility: all episode lengths flattened."""
    return [l for lengths in self.env_episode_lengths.values() for l in lengths]
```

### Summary of TrainingCallback Changes

| Component | Current | New |
|-----------|---------|-----|
| **State** | Single episode tracking | Per-env dicts |
| **`_on_step()`** | Process `[0]` only | Loop over all envs |
| **`_on_episode_end()`** | No env_id parameter | Takes `env_id` parameter |
| **Episode counting** | Single counter | Per-env counters |
| **Training Summary** | Single list stats | Aggregate across all envs |
| **Backward compat** | N/A | Properties for single-env compatibility |

---

## Implementation Order

1. **Phase 1: Config & Factory (No Behavior Change)**
   - Add `num_envs` and `vec_env_type` to `TrainingConfig`
   - Add CLI arguments
   - Create `training/env_factory.py` with `make_single_env()` and `make_training_env()`
   - Update `train()` to use factory (still `num_envs=1` by default)

2. **Phase 2: Unified mask_fn (Backward Compatible)**
   - Refactor `mask_fn()` to detect VecEnv
   - Add `_mask_fn_vecenv()` helper
   - Test with single env (should work unchanged)
   - Test with `num_envs=2` (smoke test)

3. **Phase 3: TrainingCallback Multi-Env (Backward Compatible)**
   - Add per-env state to `TrainingCallback`
   - Update `_on_step()` to process all envs
   - Update `_on_episode_end()` to take `env_id`
   - Update Training Summary to aggregate
   - Test with single env (should work unchanged)

4. **Phase 4: Integration Testing**
   - Smoke test with `num_envs=1` (verify no regressions)
   - Smoke test with `num_envs=2` (verify VecEnv works)
   - Full test with `num_envs=4` (verify performance)

---

## Testing Strategy

### Smoke Tests

1. **Single-Env Backward Compatibility:**
   ```bash
   python training/trainer.py --mode smoke --num-envs 1
   ```
   - Verify output matches current behavior exactly
   - Verify Training Summary format unchanged

2. **Multi-Env Basic Functionality:**
   ```bash
   python training/trainer.py --mode smoke --num-envs 2
   ```
   - Verify no errors
   - Verify masks are extracted correctly
   - Verify episodes complete for both envs

3. **Multi-Env with SubprocVecEnv:**
   ```bash
   python training/trainer.py --mode smoke --num-envs 2 --vec-env-type subproc
   ```
   - Verify parallel execution works
   - Verify no process-related errors

### Validation Checks

- [ ] Single-env training produces identical results to current implementation
- [ ] Multi-env training completes without errors
- [ ] Action masks are correctly extracted for all environments
- [ ] Episode tracking works correctly per environment
- [ ] Training Summary aggregates statistics correctly
- [ ] Checkpoint saving works with multi-env
- [ ] MongoDB logging works with multi-env (if applicable)

