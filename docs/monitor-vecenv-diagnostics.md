# Monitor + VecEnv + Reset/Step Diagnostics

## Overview

This document diagnoses issues related to Stable-Baselines3's Monitor wrapper interaction with vectorized environments (VecEnv) and the reset/step method signatures.

**Date:** 2025-12-01  
**Branch:** M4---implement-multiple-envs-and-parallelize  
**Status:** Diagnostic Analysis (No Code Changes)

---

## Monitor Usage Locations

### Current Implementation

**Finding:** Monitor is **NOT manually applied** in our codebase. SB3 automatically wraps environments with Monitor when creating a model.

**Location of Automatic Wrapping:**
- SB3's `BaseAlgorithm.__init__()` (called by `MaskablePPO.__init__()`) automatically wraps the environment
- This happens in `training/trainer.py` at line 749: `model = MaskablePPO(**model_kwargs)`
- The `env` parameter passed to MaskablePPO is wrapped by SB3's internal `_patch_env()` function

**Code Flow:**
1. `training/trainer.py:672` - `env = make_training_env(config, mask_fn)` creates the environment
2. `training/trainer.py:728` - `env` is passed to `MaskablePPO(**model_kwargs)`
3. SB3 internally calls `_patch_env(env)` which wraps with Monitor if not already wrapped

**No Manual Monitor Usage Found:**
- ❌ No `from stable_baselines3.common.monitor import Monitor` imports in our code
- ❌ No explicit `Monitor(env)` wrapping in `training/trainer.py`
- ❌ No explicit `Monitor(env)` wrapping in `training/env_factory.py`
- ✅ Monitor is applied automatically by SB3

---

## Wrapper Order (Single-Env vs VecEnv)

### Single-Env Mode (`num_envs == 1`)

**Wrapper Stack (Outermost → Innermost):**

```
ActionMasker
  └── Monitor (applied by SB3 automatically)
      └── GymnasiumBlokusWrapper
          └── BlokusEnv (PettingZoo AECEnv)
```

**Creation Flow:**
1. `env_factory.py:68` - `make_single_env()` returns `GymnasiumBlokusWrapper`
2. `env_factory.py:69` - `ActionMasker(env, mask_fn)` wraps it
3. `trainer.py:728` - Passed to `MaskablePPO(env=env, ...)`
4. SB3's `_patch_env()` wraps with `Monitor(ActionMasker(GymnasiumBlokusWrapper(...)))`

**Final Wrapper Chain:**
```
Monitor(ActionMasker(GymnasiumBlokusWrapper(BlokusEnv)))
```

### VecEnv Mode (`num_envs > 1`)

**Wrapper Stack (Outermost → Innermost):**

```
ActionMasker
  └── VecEnv (DummyVecEnv or SubprocVecEnv)
      └── [Monitor (applied by SB3 to each sub-env)]
          └── GymnasiumBlokusWrapper
              └── BlokusEnv (PettingZoo AECEnv)
```

**Creation Flow:**
1. `env_factory.py:97` - Creates list of factory functions: `env_fns = [make_env_fn(rank=i) for i in range(config.num_envs)]`
2. `env_factory.py:103` - `DummyVecEnv(env_fns)` or `SubprocVecEnv(env_fns)` creates VecEnv
   - Each `env_fns[i]()` returns `GymnasiumBlokusWrapper` (no ActionMasker yet)
3. `env_factory.py:106` - `ActionMasker(vec_env, mask_fn)` wraps the VecEnv
4. `trainer.py:728` - Passed to `MaskablePPO(env=env, ...)`
5. SB3's `_patch_env()` wraps each sub-env in the VecEnv with Monitor

**Final Wrapper Chain:**
```
ActionMasker(
  DummyVecEnv([
    Monitor(GymnasiumBlokusWrapper(BlokusEnv)),  # envs[0]
    Monitor(GymnasiumBlokusWrapper(BlokusEnv)),  # envs[1]
    ...
  ])
)
```

**Key Difference:**
- **Single-env:** Monitor wraps ActionMasker (outermost)
- **VecEnv:** Monitor wraps each individual GymnasiumBlokusWrapper (inside VecEnv), ActionMasker wraps the VecEnv (outermost)

---

## Reset/Step Signatures

### Expected Signatures (Gymnasium Standard)

**reset():**
```python
def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[ObsType, Dict[str, Any]]:
```

**step():**
```python
def step(self, action: ActionType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
    # Returns: (obs, reward, terminated, truncated, info)
```

### Current Implementations

#### 1. BlokusEnv.reset() (PettingZoo AECEnv)
**File:** `envs/blokus_v0.py:139`

```python
def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None, **kwargs) -> None:
```

**Issues:**
- ✅ Accepts `seed` and `options` as keyword arguments
- ✅ Accepts `**kwargs` for compatibility
- ❌ **Returns `None` instead of `(obs, info)` tuple** - This is a PettingZoo AECEnv convention, not Gymnasium

#### 2. GymnasiumBlokusWrapper.reset()
**File:** `envs/blokus_v0.py:469`

```python
def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None, **kwargs):
    self.env.reset(seed=seed, options=options, **kwargs)
    obs = self.env.observe(self.agent_name)
    info = self.env.infos[self.agent_name]
    return obs, info
```

**Status:**
- ✅ Accepts `seed` and `options` as keyword arguments
- ✅ Accepts `**kwargs` for compatibility
- ✅ Returns `(obs, info)` tuple (correct Gymnasium signature)

#### 3. Monitor.reset() (SB3)
**Source:** `stable_baselines3.common.monitor.Monitor`

```python
def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
    # ...
    return self.env.reset(**kwargs)
```

**Behavior:**
- Accepts `**kwargs` (including `seed` and `options`)
- Calls `self.env.reset(**kwargs)` on the wrapped environment
- Returns the result from `self.env.reset()`

#### 4. DummyVecEnv.reset()
**Source:** `stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv`

```python
def reset(self) -> VecEnvObs:
    for env_idx in range(self.num_envs):
        obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=self._seeds[env_idx])
        # ...
```

**Behavior:**
- Calls `self.envs[env_idx].reset(seed=self._seeds[env_idx])` on each sub-env
- Passes `seed` as a **keyword argument**
- Expects each sub-env to return `(obs, info)` tuple

#### 5. GymnasiumBlokusWrapper.step()
**File:** `envs/blokus_v0.py:481`

```python
def step(self, action: int):
    self.env.step(action)
    obs = self.env.observe(self.agent_name)
    reward = self.env.rewards[self.agent_name]
    terminated = self.env.terminations[self.agent_name]
    truncated = self.env.truncations[self.agent_name]
    info = self.env.infos[self.agent_name]
    return obs, reward, terminated, truncated, info
```

**Status:**
- ✅ Returns correct 5-tuple: `(obs, reward, terminated, truncated, info)`

---

## Identified Problems

### Problem 1: Monitor.reset() → GymnasiumBlokusWrapper.reset() Chain

**Location:** `stable_baselines3/common/monitor.py:83` (Monitor.reset) → `envs/blokus_v0.py:469` (GymnasiumBlokusWrapper.reset)

**Issue:**
When SB3 wraps each sub-env in VecEnv with Monitor, Monitor.reset() calls:
```python
return self.env.reset(**kwargs)  # Monitor calls this
```

Where `self.env` is `GymnasiumBlokusWrapper`. The error trace shows:
```
File ".../gymnasium/core.py:414, in reset
    return self.env.reset(seed=seed, options=options)
TypeError: reset() got an unexpected keyword argument 'seed'
```

**Root Cause Analysis:**
- ✅ **Confirmed:** Monitor inherits from `gymnasium.Wrapper` (verified via MRO inspection)
- ✅ **Confirmed:** `gymnasium.Wrapper.reset()` signature: `reset(self, *, seed=..., options=...)` (keyword-only arguments)
- ✅ **Confirmed:** `gymnasium.Wrapper.reset()` calls: `self.env.reset(seed=seed, options=options)` (explicit keyword arguments)
- ✅ **Confirmed:** `GymnasiumBlokusWrapper.reset()` signature: `reset(self, seed=..., options=..., **kwargs)` (should accept keyword arguments)
- ❌ **Issue:** Despite correct signatures, the call fails with "reset() got an unexpected keyword argument 'seed'"

**Method Resolution Order (MRO):**
```
Monitor → gymnasium.Wrapper → gymnasium.Env → Generic → object
```

**Call Chain:**
1. `DummyVecEnv.reset()` calls `self.envs[env_idx].reset(seed=self._seeds[env_idx])`
2. `Monitor.reset(**kwargs)` receives `seed=...` in kwargs
3. Monitor inherits from `gymnasium.Wrapper`, so Monitor.reset() likely calls `super().reset(**kwargs)` or directly `self.env.reset(**kwargs)`
4. `gymnasium.Wrapper.reset()` extracts `seed` and `options` from kwargs and calls `self.env.reset(seed=seed, options=options)`
5. `self.env` is `GymnasiumBlokusWrapper`, which should accept these arguments...

**Hypothesis:**
The error occurs in `gymnasium.Wrapper.reset()` when it calls `self.env.reset(seed=seed, options=options)`. Despite `GymnasiumBlokusWrapper.reset()` having the correct signature, there may be a Python 3.7 compatibility issue with how keyword arguments are handled through the inheritance chain.

**Investigation Needed:**
- Check if Monitor inherits from `gymnasium.Wrapper`
- Verify the actual method being called in the error trace
- Check if there's a signature mismatch in Python 3.7

### Problem 2: DummyVecEnv.reset() Calls Sub-Envs with seed= Keyword

**Location:** `stable_baselines3/common/vec_env/dummy_vec_env.py:76`

**Code:**
```python
obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=self._seeds[env_idx])
```

**Issue:**
- DummyVecEnv calls `reset(seed=...)` on each sub-env
- Each sub-env should be `Monitor(GymnasiumBlokusWrapper(...))` after SB3 wrapping
- Monitor.reset() should accept `**kwargs` and pass through
- But the error suggests Monitor is calling something that doesn't accept `seed` as a keyword

**Possible Causes:**
1. Monitor is not properly wrapping GymnasiumBlokusWrapper
2. Monitor.reset() is calling through a different method chain
3. There's a Python 3.7 compatibility issue with keyword arguments

### Problem 3: Wrapper Order Mismatch Between Single-Env and VecEnv

**Single-Env:**
```
Monitor(ActionMasker(GymnasiumBlokusWrapper))
```

**VecEnv:**
```
ActionMasker(DummyVecEnv([Monitor(GymnasiumBlokusWrapper), ...]))
```

**Issue:**
- In single-env mode, Monitor wraps ActionMasker (outermost)
- In VecEnv mode, Monitor wraps each GymnasiumBlokusWrapper (inside VecEnv)
- This asymmetry might cause different behavior

**Impact:**
- Single-env training works (Monitor wraps ActionMasker)
- VecEnv training fails (Monitor wraps GymnasiumBlokusWrapper directly)

### Problem 4: BlokusEnv.reset() Returns None (PettingZoo Convention)

**Location:** `envs/blokus_v0.py:139`

**Issue:**
- `BlokusEnv.reset()` returns `None` (PettingZoo AECEnv convention)
- `GymnasiumBlokusWrapper.reset()` correctly returns `(obs, info)` tuple
- This is handled correctly, but worth noting for debugging

---

## Error Trace Analysis

### Actual Error from Tests

```
File ".../stable_baselines3/common/vec_env/dummy_vec_env.py:76, in reset
    obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=self._seeds[env_idx])
File ".../stable_baselines3/common/monitor.py:83, in reset
    return self.env.reset(**kwargs)
File ".../gymnasium/core.py:414, in reset
    return self.env.reset(seed=seed, options=options)
TypeError: reset() got an unexpected keyword argument 'seed'
```

### Interpretation

1. **DummyVecEnv.reset()** calls `self.envs[env_idx].reset(seed=...)`
   - `self.envs[env_idx]` should be `Monitor(GymnasiumBlokusWrapper(...))`

2. **Monitor.reset()** receives `seed=...` in `**kwargs` and calls `self.env.reset(**kwargs)`
   - `self.env` should be `GymnasiumBlokusWrapper`

3. **gymnasium/core.py:414** is the `Wrapper` base class's reset method
   - This suggests Monitor inherits from `gymnasium.Wrapper`
   - The Wrapper base class calls `self.env.reset(seed=seed, options=options)`
   - But `self.env.reset()` doesn't accept `seed` as a keyword argument

**Key Insight:**
The error occurs in `gymnasium.Wrapper.reset()`, not in our code. This suggests:
- Monitor inherits from `gymnasium.Wrapper`
- `gymnasium.Wrapper.reset()` extracts `seed` and `options` from `**kwargs`
- It then calls `self.env.reset(seed=seed, options=options)` with explicit keyword arguments
- But `self.env.reset()` (GymnasiumBlokusWrapper) should accept these...

**Wait:** The signature shows `reset(self, seed=..., options=..., **kwargs)`, which SHOULD accept `seed` as a keyword argument. This is puzzling.

---

## Next Steps for Investigation

1. ✅ **Verify Monitor's Inheritance:** COMPLETED
   - Confirmed: Monitor inherits from `gymnasium.Wrapper`
   - MRO: `Monitor → gymnasium.Wrapper → gymnasium.Env → Generic → object`

2. ✅ **Inspect gymnasium.Wrapper.reset() Implementation:** COMPLETED
   - Signature: `reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None)`
   - Implementation: `return self.env.reset(seed=seed, options=options)`
   - Uses keyword-only arguments (`*` in signature)

3. **Check Python 3.7 Compatibility:**
   - Verify that method signatures with `**kwargs` work correctly in Python 3.7
   - Test if there's a difference in how keyword arguments are handled
   - Check if Python 3.7's type annotation syntax (`int | None`) causes issues

4. **Test Direct Calls:**
   - ✅ Verified: `GymnasiumBlokusWrapper.reset(seed=42)` works when called directly
   - ⚠️ Need to test: `Monitor(GymnasiumBlokusWrapper(...)).reset(seed=42)` in isolation
   - Compare with what happens in VecEnv context

5. **Check SB3's _patch_env() Behavior:**
   - Understand exactly how SB3 wraps environments
   - Check if there's a difference between single-env and VecEnv wrapping
   - Verify if Monitor is applied differently in VecEnv vs single-env

6. **Investigate Method Resolution:**
   - Check if Monitor overrides reset() and how it calls super()
   - Verify the actual method being called at runtime using introspection
   - Check if there's a descriptor or property interfering with method calls

---

## Summary

### Current State
- ✅ `GymnasiumBlokusWrapper` properly inherits from `gymnasium.Env`
- ✅ `reset()` and `step()` methods have correct signatures
- ✅ Direct calls to `reset(seed=42)` work correctly
- ❌ VecEnv tests fail with `TypeError: reset() got an unexpected keyword argument 'seed'`

### Root Cause Hypothesis
The error occurs in `gymnasium.Wrapper.reset()` when it calls `self.env.reset(seed=seed, options=options)`. Despite `GymnasiumBlokusWrapper.reset()` having the correct signature, something in the method resolution or Python 3.7's handling of keyword arguments is causing the failure.

### Most Likely Issues
1. **Method Resolution Order (MRO) Issue:** Monitor or gymnasium.Wrapper might be calling a different method than expected
2. **Python 3.7 Compatibility:** There might be a subtle difference in how Python 3.7 handles keyword arguments with `**kwargs`
3. **Wrapper Chain Issue:** The actual wrapper chain at runtime might differ from what we expect

### Recommended Fix Strategy
1. Verify the actual wrapper chain at runtime using introspection
2. Check if there's a way to ensure Monitor calls the correct method
3. Consider explicitly handling the reset call in a way that's more compatible with SB3's expectations
4. Test with a newer Python version to rule out Python 3.7-specific issues

---

## File References

- `envs/blokus_v0.py:139` - `BlokusEnv.reset()` signature
- `envs/blokus_v0.py:469` - `GymnasiumBlokusWrapper.reset()` signature
- `envs/blokus_v0.py:481` - `GymnasiumBlokusWrapper.step()` signature
- `training/env_factory.py:19-44` - `make_single_env()` function
- `training/env_factory.py:47-108` - `make_training_env()` function
- `training/trainer.py:672` - Environment creation
- `training/trainer.py:728` - Environment passed to MaskablePPO
- `training/trainer.py:749` - MaskablePPO model creation (triggers Monitor wrapping)
- `tests/test_vecenv_training.py:76-147` - VecEnv test that fails

