# Action Masking Diagnostics

This document describes the diagnostic logging added to diagnose the MaskableCategorical Simplex constraint violation error.

## Overview

Diagnostic logging has been added to track:
- Mask shape, dtype, and value semantics
- "No legal moves" states
- Mask construction and extraction points
- Environment state when masks are created

## Files Modified

### 1. `envs/blokus_v0.py`

**Added:**
- Header comments explaining action indexing scheme and mask construction
- Diagnostic logging in `_get_info()` method where masks are created
- Logging for:
  - When no legal moves are found (critical warning)
  - Mask shape, dtype, sum (number of legal actions)
  - Sample of legal action indices
  - Mismatch between legal moves count and mapped actions

**Key Logging Points:**
- `_get_info()`: Logs mask properties when created, especially when mask is empty or in first 10 steps
- Warns when `legal_moves == 0` (mask will be all False)

### 2. `training/trainer.py`

**Added:**
- Diagnostic logging in `mask_fn()` function where masks are extracted
- Logging for:
  - Mask shape, dtype, sum
  - Action space size
  - Legal moves count from environment
  - Sample legal action indices
  - Critical error when mask is empty (will break MaskablePPO)

**Key Logging Points:**
- `mask_fn()`: Logs mask properties for first 20 calls, or when mask is empty/shape mismatch
- Critical error logged when `mask.sum() == 0` (no legal actions)

## Enabling/Disabling Diagnostics

Both files have a flag at the top:

```python
MASK_DEBUG_LOGGING = True  # Set to False to disable diagnostic logging
```

Set to `False` to disable all diagnostic logging once the issue is resolved.

## What to Look For

When running a smoke test, watch for:

1. **Empty Masks**: 
   - Look for: `"NO LEGAL ACTIONS PRESENT IN MASK - this will break MaskablePPO!"`
   - This indicates the environment has no legal moves but mask_fn is still being called

2. **Shape Mismatches**:
   - Look for: `"SHAPE MISMATCH - mask.shape=... != action_space.n=..."`
   - This indicates the mask size doesn't match the action space

3. **Mask Properties**:
   - Check: `mask.shape`, `mask.dtype`, `mask.sum()`
   - Should be: `(36400,)`, `bool`, and `> 0` (typically 50-200 for Blokus)

4. **Legal Moves Count**:
   - Compare `legal_moves_count` from environment vs `mask.sum()`
   - These should match (or mask.sum() should be <= legal_moves_count if some moves don't map)

## Running Diagnostics

Run a short smoke test:

```bash
python training/trainer.py --mode smoke --max-episodes 5
```

The logs will show:
- Mask creation in `_get_info()` (first 10 steps)
- Mask extraction in `mask_fn()` (first 20 calls)
- Any empty masks or shape mismatches (always logged)

## Expected Output

Normal case (legal moves present):
```
INFO - envs.blokus_v0.mask_diagnostics - BlokusEnv._get_info(player_0): legal_moves=58, mapped_to_mask=58, mask.sum()=58, mask.shape=(36400,), mask.dtype=bool, action_space_size=36400
INFO - training.trainer.mask_fn_diagnostics - mask_fn(player_0) call #1: mask.shape=(36400,), mask.dtype=bool, mask.sum()=58, action_space.n=36400, env.infos[player_0]['legal_moves_count']=58
```

Problem case (no legal moves):
```
WARNING - envs.blokus_v0.mask_diagnostics - BlokusEnv: NO LEGAL MOVES for player_0 (player RED) at step 42. Mask will be all False - this will break MaskablePPO!
ERROR - training.trainer.mask_fn_diagnostics - mask_fn(player_0): NO LEGAL ACTIONS PRESENT IN MASK - this will break MaskablePPO! env.infos[player_0]['can_move']=False, env.infos[player_0]['legal_moves_count']=0
```

## Next Steps

Once diagnostics identify the issue:
1. If empty masks are the problem: Fix mask_fn to handle "no legal moves" case
2. If shape mismatches: Fix action space or mask generation
3. If dtype issues: Ensure masks are boolean, not float
4. If mapping issues: Check move_to_action dictionary construction

