# Training Run Checklist

Step-by-step guide to validate your setup and run training on the Blokus RL project.

---

## Prerequisites

### 1. Environment Setup

**Python Environment:**
```bash
# Ensure you're in the project root
cd /path/to/blokus_rl

# Install dependencies (if not already done)
pip install -r requirements.txt
# Or using pyproject.toml
pip install -e .
```

**Verify Installation:**
```bash
# Check Python version (3.9+ required)
python --version

# Verify key packages
python -c "import torch; import stable_baselines3; import gymnasium; print('✓ Dependencies OK')"
```

**No Virtual Environment Required:**
- The project can run with system Python or a virtual environment
- If using a virtual environment, activate it before running commands

### 2. Environment Variables (Optional)

The following environment variables can be set to control optimizations (all enabled by default):

```bash
# Move generation optimizations (default: enabled)
export BLOKUS_USE_FRONTIER_MOVEGEN=1      # Use frontier-based generation (default: 1)
export BLOKUS_USE_BITBOARD_LEGALITY=1     # Use bitboard legality checks (default: 1)
export BLOKUS_USE_HEURISTIC_ANCHORS=0     # Use heuristic anchors (default: 0, disabled for safety)

# Debug logging (default: disabled)
export BLOKUS_MOVEGEN_DEBUG=0             # Enable move generation timing logs
export BLOKUS_PROFILE_MOVEGEN=0           # Enable episode-level profiling
```

**Note:** You don't need to set these - defaults are optimized. Only set if you want to disable optimizations for debugging.

---

## Step 1: Run Environment Performance Benchmark

### Command

```bash
PYTHONPATH=. python scripts/benchmark_env.py
```

### Expected Output

You should see two benchmark sections:

#### Environment Step() Benchmark
```
Environment Step() Benchmark
================================================================================
Number of steps: 10,000
Seed: 42

Running benchmark...

Results:
  Total steps: 10,000
  Total time: 12.34s
  Steps per second: 810.37
  Episodes completed: 25
  Average steps per episode: 400.0
  Total reward: 1234.56
  Average reward per step: 0.1235
```

#### Move Generation Benchmark
```
Move Generation Benchmark
================================================================================
Number of iterations: 1,000
Seed: 42

Running benchmark...

Results:
  Total iterations: 1,000
  Total time: 5.67s
  Average time per call: 5.67ms
  Calls per second: 176.37
  Min time: 2.34ms
  Max time: 15.67ms
  Median (p50) time: 5.12ms
  p95 time: 12.45ms
  p99 time: 14.89ms
  Total moves generated: 45,678
  Average moves per call: 45.7
```

### Interpreting Results

#### Environment Steps per Second

**Good Performance:**
- **>500 steps/s** - Excellent, ready for training
- **200-500 steps/s** - Acceptable, training will work but may be slower
- **<200 steps/s** - Slow, investigate performance issues

**What This Means:**
- Steps/s directly impacts training throughput
- Higher is better - more data collected per second
- If low, training will take longer

#### Move Generation Performance

**Good Performance:**
- **<10ms average** - Excellent, won't bottleneck training
- **10-50ms average** - Acceptable, may slow down environment steps
- **>50ms average** - Slow, may indicate optimization issues

**What This Means:**
- Move generation is called every step to build action masks
- High move generation time can bottleneck environment steps
- If both environment and move generation are slow, check system resources

### Customizing the Benchmark

```bash
# More steps for better accuracy
PYTHONPATH=. python scripts/benchmark_env.py --num-steps 50000

# More move generation iterations
PYTHONPATH=. python scripts/benchmark_env.py --movegen-iterations 2000

# Only environment benchmark
PYTHONPATH=. python scripts/benchmark_env.py --skip-movegen

# Only move generation benchmark
PYTHONPATH=. python scripts/benchmark_env.py --skip-env
```

### If Benchmark Fails

**"No module named 'envs'" or similar import errors:**
```bash
# Ensure PYTHONPATH is set
PYTHONPATH=. python scripts/benchmark_env.py

# Or add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/benchmark_env.py
```

**Very slow benchmark (<50 steps/s):**
- Check CPU usage (should be high during benchmark)
- Verify optimizations are enabled (check environment variables)
- Check for background processes consuming resources
- Try running with fewer steps first: `--num-steps 1000`

---

## Step 2: Run Smoke Test Training

### Command

```bash
PYTHONPATH=. python training/trainer.py --config training/config_smoke.yaml
```

**Alternative (using mode flag):**
```bash
PYTHONPATH=. python training/trainer.py --mode smoke
```

### What Happens

The smoke test will:
1. Create a timestamped run directory: `runs/YYYYMMDD_HHMMSS_smoke/`
2. Set up file logging: `runs/YYYYMMDD_HHMMSS_smoke/training.log`
3. Run ~15,000 training steps across 10 episodes
4. Save checkpoints every 5 episodes
5. Log episode stats and speed metrics every 10 episodes
6. Complete in a few minutes

### Expected Output

#### 1. Run Directory Creation
```
2025-01-15 14:30:22 - training.trainer - INFO - Created run directory: runs/20250115_143022_smoke
2025-01-15 14:30:22 - training.trainer - INFO - Log file: runs/20250115_143022_smoke/training.log
```

#### 2. Configuration Summary
```
================================================================================
Starting Training Run
================================================================================
Run directory: runs/20250115_143022_smoke
Timestamp: 2025-01-15 14:30:22

Reproducibility Information:
  Code Version: 1.0.0
  Git Hash: abc123def456...
  Git Branch: main
  Python Version: 3.10.5
  PyTorch Version: 2.0.0
  Stable-Baselines3 Version: 2.0.0

================================================================================
Training Configuration
================================================================================
Mode: SMOKE
Max Episodes: 10
Total Timesteps: 15000
...
```

#### 3. Episode Completion Logs (with Win Detection)
```
2025-01-15 14:30:25 - training.trainer - INFO - Episode 1 completed: reward=12.34, length=45, win=1.0
2025-01-15 14:30:28 - training.trainer - INFO - Episode 2 completed: reward=15.67, length=52, win=0.0
2025-01-15 14:30:31 - training.trainer - INFO - Episode 3 completed: reward=18.90, length=48, win=0.5
...
```

**Win Values:**
- `win=1.0` - Player_0 won (highest score, no tie)
- `win=0.5` - Player_0 tied for highest score
- `win=0.0` - Player_0 lost (another player had higher score)
- `win=None` - Win information unavailable (should not occur in normal gameplay)

#### 4. Periodic Statistics (Every 10 Episodes)
```
2025-01-15 14:31:10 - training.trainer - INFO - Episodes 1-10 (last 10): reward=14.23±2.45, length=48.2±5.1, speed=125.5 steps/s (125.5 env steps/s)
```

#### 5. Checkpoint Creation
```
2025-01-15 14:31:05 - training.trainer - INFO - Saved checkpoint at episode 5
2025-01-15 14:31:15 - training.trainer - INFO - Saved checkpoint at episode 10
```

#### 6. Training Summary
```
================================================================================
Training Summary
================================================================================
Total episodes: 10
Total steps: 482
Total environment steps: 482
Total time: 45.2s (0.8 minutes)
Average speed: 10.7 steps/s (10.7 env steps/s)

Episode Statistics:
  Average reward: 14.23 ± 2.45
  Average episode length: 48.2 ± 5.1
  Best episode reward: 18.90
  Worst episode reward: 10.15
```

### What to Verify

✅ **Check These Items:**

1. **Run directory created:**
   ```bash
   ls -la runs/
   # Should see: runs/YYYYMMDD_HHMMSS_smoke/
   ```

2. **Log file exists:**
   ```bash
   ls -la runs/*/training.log
   # Should see the log file
   ```

3. **Checkpoints saved:**
   ```bash
   ls -la checkpoints/ppo_agent/*/ep*.zip
   # Should see at least one checkpoint (ep000005.zip or ep000010.zip)
   ```

4. **Win detection working:**
   ```bash
   # Check for win values in episode logs
   grep "Episode.*completed.*win=" runs/*/training.log
   # Should see lines like: Episode X completed: reward=..., length=..., win=1.0
   
   # Verify no warnings about missing game result fields
   grep -i "missing.*final_scores\|missing.*winner_ids" runs/*/training.log
   # Should be empty (no warnings)
   ```

5. **No errors in console or log:**
   - No "Non-finite reward detected" errors
   - No "No legal actions available" errors (except possibly at game end)
   - No exceptions or tracebacks
   - No warnings about missing game result fields (win detection should work)

6. **Reasonable performance:**
   - Speed metrics show >50 steps/s (system-dependent)
   - Episode rewards in expected range (typically 5-30 for Blokus)
   - Episodes complete successfully
   - Win values present in logs (1.0, 0.5, or 0.0)

### Inspecting Logs

**View the log file:**
```bash
# Find the most recent run directory
LATEST_RUN=$(ls -td runs/*/ | head -1)

# View the log
cat ${LATEST_RUN}training.log

# Or follow it in real-time (if training is still running)
tail -f ${LATEST_RUN}training.log
```

**Search for specific information:**
```bash
# Check for errors
grep -i "error\|exception\|traceback" ${LATEST_RUN}training.log

# Check episode completions
grep "Episode.*completed" ${LATEST_RUN}training.log

# Check speed metrics
grep "speed=" ${LATEST_RUN}training.log

# Check checkpoint saves
grep "Saved checkpoint" ${LATEST_RUN}training.log
```

**Check checkpoint location:**
```bash
# Find checkpoints for the run
RUN_ID=$(grep "run_id" ${LATEST_RUN}training.log | head -1 | grep -o 'run_[a-f0-9-]*' | head -1)
ls -la checkpoints/ppo_agent/${RUN_ID}/
```

---

## Step 2.5: Win Detection Smoke Test

### Purpose

This smoke test specifically validates that win detection and logging are working correctly. It's a subset of the full smoke test, focusing on verifying that:
- Episodes complete without errors
- Win values (1.0, 0.5, 0.0) appear in logs
- No warnings about missing game result fields
- Win information flows correctly from engine → environment → training → logs

### Command

```bash
PYTHONPATH=. python training/trainer.py --mode smoke --total-timesteps 2000
```

**Note:** This uses a smaller timestep count (2000) for faster validation. The full smoke test (15000 timesteps) also validates win detection.

### What to Look For

#### ✅ Expected: Win Values in Episode Logs

Look for episode completion lines that include win values:

```bash
# Check for win values in logs
grep "Episode.*completed.*win=" runs/*/training.log
```

**Expected output:**
```
2025-01-15 14:30:25 - training.trainer - INFO - Episode 1 completed: reward=12.34, length=45, win=1.0
2025-01-15 14:30:28 - training.trainer - INFO - Episode 2 completed: reward=15.67, length=52, win=0.0
2025-01-15 14:30:31 - training.trainer - INFO - Episode 3 completed: reward=18.90, length=48, win=0.5
```

**Win value meanings:**
- `win=1.0` - Player_0 (the training agent) won with highest score
- `win=0.5` - Player_0 tied for highest score
- `win=0.0` - Player_0 lost (another player had higher score)
- `win=None` - Win information unavailable (should not occur in normal gameplay)

#### ✅ Expected: No Warnings About Missing Game Result Fields

```bash
# Check for warnings about missing game result fields
grep -i "missing.*final_scores\|missing.*winner_ids\|no info dict available" runs/*/training.log
```

**Expected output:** Empty (no warnings)

**If warnings appear:**
- This indicates win detection failed for some episodes
- May occur if episodes are truncated before game completion
- Should be rare in normal gameplay

#### ✅ Expected: Episodes Complete Successfully

```bash
# Count completed episodes
grep -c "Episode.*completed" runs/*/training.log
```

**Expected:** Should see multiple episodes (typically 3-5 for 2000 timesteps)

### Quick Verification Script

```bash
# Find the most recent run
LATEST_RUN=$(ls -td runs/*/ | head -1)

# Check for win values
echo "=== Win Values in Logs ==="
grep "Episode.*completed.*win=" ${LATEST_RUN}training.log | tail -5

# Check for warnings
echo ""
echo "=== Warnings About Missing Game Result ==="
grep -i "missing.*final_scores\|missing.*winner_ids\|no info dict available" ${LATEST_RUN}training.log

# Count episodes with win values
echo ""
echo "=== Episode Count with Win Values ==="
grep -c "Episode.*completed.*win=" ${LATEST_RUN}training.log || echo "0 (no win values found - this is a problem!)"
```

### What This Validates

✅ **Win Detection Pipeline:**
1. Engine computes `GameResult` when game ends
2. Environment exposes `final_scores`, `winner_ids`, `is_tie` in terminal step info
3. Training callback extracts win value from info dict
4. Win value is logged to console and file
5. Win value is passed to MongoDB logger (if enabled)

✅ **Data Flow:**
```
Game Engine → Environment Info Dict → Training Callback → Logs/MongoDB
     ↓                ↓                        ↓                ↓
GameResult    final_scores,          _compute_win_from_info()  win=1.0/0.5/0.0
              winner_ids, is_tie
```

### Troubleshooting

**If win values are missing (`win=None` in all episodes):**
1. Check if episodes are completing normally (games reaching end state)
2. Verify environment is calling `get_game_result()` on game over
3. Check for truncation (episodes ending due to max steps before game completion)
4. Review logs for any errors in win detection logic

**If warnings about missing fields appear:**
1. Check if episodes are being truncated (max_steps_per_episode too low)
2. Verify `_check_termination_truncation()` is being called
3. Check that `game_result` is being set in environment

**If win values are always 0.0:**
- This may be normal if the agent is untrained and losing all games
- Check that `player0_won` flag is being computed correctly in environment
- Verify that player_0 (RED) is actually the training agent

### Integration with Full Smoke Test

The full smoke test (Step 2) also validates win detection, but with more episodes. This focused test (Step 2.5) is useful for:
- Quick validation after code changes
- Debugging win detection issues
- Verifying the win detection pipeline works end-to-end

Both tests use the same win detection code path - smoke mode does not skip win detection.

---

## Step 3: Verify Smoke Test Results

### Quick Verification Checklist

Run these commands to verify everything worked:

```bash
# 1. Find the most recent run
LATEST_RUN=$(ls -td runs/*/ | head -1)
echo "Latest run: ${LATEST_RUN}"

# 2. Check log file exists and has content
wc -l ${LATEST_RUN}training.log
# Should show >100 lines

# 3. Verify episodes completed
grep -c "Episode.*completed" ${LATEST_RUN}training.log
# Should show 10 episodes

# 4. Check for errors
grep -i "error\|exception" ${LATEST_RUN}training.log | wc -l
# Should be 0 (or very few, only expected warnings)

# 5. Verify checkpoints
ls -la checkpoints/ppo_agent/*/ep*.zip | wc -l
# Should show at least 1 checkpoint

# 6. Check final summary
tail -20 ${LATEST_RUN}training.log | grep -A 10 "Training Summary"
# Should show episode statistics and speed metrics
```

### Healthy Run Indicators

✅ **All of these should be true:**
- Run directory exists: `runs/YYYYMMDD_HHMMSS_smoke/`
- Log file has content (>100 lines)
- 10 episodes completed (check: `grep -c "Episode.*completed"`)
- At least 1 checkpoint saved
- No errors in log (check: `grep -i error`)
- Speed metrics present (check: `grep "speed="`)
- Training summary at end of log

---

## If Something Looks Wrong

### Common Issues and Quick Fixes

#### 1. "No module named 'training'" or Import Errors

**Symptom:**
```
ModuleNotFoundError: No module named 'training'
```

**Fix:**
```bash
# Always use PYTHONPATH
PYTHONPATH=. python training/trainer.py --config training/config_smoke.yaml

# Or set it permanently for the session
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python training/trainer.py --config training/config_smoke.yaml
```

#### 2. Very Slow Performance (<10 steps/s)

**Symptom:**
- Training runs but is extremely slow
- Speed metrics show <10 steps/s

**Check:**
```bash
# 1. Verify optimizations are enabled (default: enabled)
echo $BLOKUS_USE_FRONTIER_MOVEGEN      # Should be 1 or unset (defaults to 1)
echo $BLOKUS_USE_BITBOARD_LEGALITY     # Should be 1 or unset (defaults to 1)

# 2. Run benchmark first to establish baseline
PYTHONPATH=. python scripts/benchmark_env.py

# 3. Check system resources
top  # or htop, check CPU usage

# 4. Check for background processes
ps aux | grep python
```

**Fix:**
- Ensure optimizations are enabled (don't set env vars to 0)
- Close other resource-intensive applications
- Check if system is CPU-bound or memory-bound

#### 3. "No legal actions available" Errors

**Symptom:**
```
BlokusEnv: NO LEGAL MOVES for player_0 (player RED) at step X
```

**Check:**
```bash
# Check log for context
grep -B 5 -A 5 "NO LEGAL MOVES" runs/*/training.log
```

**Fix:**
- This can happen at game end (normal)
- If it happens early in episodes, may indicate environment issue
- Check if it's consistent or intermittent
- If consistent, check environment reset logic

#### 4. "Non-finite reward detected" Errors

**Symptom:**
```
Non-finite reward detected: nan
# or
Non-finite reward detected: inf
```

**Check:**
```bash
# Find where it occurs
grep -B 10 "Non-finite reward" runs/*/training.log
```

**Fix:**
- This indicates a bug in reward calculation
- Check score calculation in environment
- Verify no division by zero
- Check if it's consistent or intermittent

#### 5. No Checkpoints Saved

**Symptom:**
- Training completes but no checkpoints found

**Check:**
```bash
# Verify checkpoint interval is set
grep "checkpoint_interval" runs/*/training.log | head -1

# Check for checkpoint errors
grep -i "checkpoint" runs/*/training.log | grep -i "error\|fail"
```

**Fix:**
- Verify `checkpoint_interval_episodes` is set in config (should be 5 for smoke test)
- Check `checkpoint_dir` is writable: `ls -ld checkpoints/`
- Check disk space: `df -h .`

#### 6. Training Stops Early

**Symptom:**
- Training stops before completing all episodes

**Check:**
```bash
# Check for max_episodes limit
grep "max_episodes\|Reached max_episodes" runs/*/training.log

# Check for exceptions
grep -i "exception\|traceback" runs/*/training.log
```

**Fix:**
- Verify `max_episodes` setting (should be 10 for smoke test)
- Check for exceptions in log
- Verify `total_timesteps` is sufficient

### Files to Inspect

**Primary Log File:**
```bash
# Most recent run log
LATEST_RUN=$(ls -td runs/*/ | head -1)
cat ${LATEST_RUN}training.log
```

**Checkpoint Directory:**
```bash
# List all checkpoints
ls -la checkpoints/ppo_agent/*/ep*.zip

# Check checkpoint metadata
cat checkpoints/ppo_agent/*/ep*_metadata.json
```

**TensorBoard Logs:**
```bash
# List TensorBoard log directories
ls -la logs/

# View in TensorBoard (if installed)
tensorboard --logdir logs/
```

### Key Log Lines to Check

**Start of training:**
```bash
grep "Starting Training Run" runs/*/training.log
```

**Configuration:**
```bash
grep -A 20 "Training Configuration" runs/*/training.log
```

**Episode completions:**
```bash
grep "Episode.*completed" runs/*/training.log
```

**Speed metrics:**
```bash
grep "speed=" runs/*/training.log
```

**Errors:**
```bash
grep -i "error\|exception\|warning" runs/*/training.log
```

**Final summary:**
```bash
grep -A 15 "Training Summary" runs/*/training.log
```

---

## Scaling Up to Real Training Runs

### Configuration Files

**Full Training Config:**
- **File:** `training/config_full.yaml`
- **Default:** 1M timesteps, checkpoint every 100 episodes
- **Usage:** `python training/trainer.py --config training/config_full.yaml`

**Custom Config:**
- Create your own YAML/JSON config file
- Override any parameters as needed
- See `docs/training_entrypoints.md` for all options

### Recommended Training Progression

#### 1. Start Small (Validation)
```bash
# Smoke test (already done)
PYTHONPATH=. python training/trainer.py --config training/config_smoke.yaml
```

#### 2. Medium Run (Initial Training)
```bash
# 100K steps, checkpoint every 25 episodes
PYTHONPATH=. python training/trainer.py \
  --mode full \
  --total-timesteps 100000 \
  --checkpoint-interval-episodes 25 \
  --seed 42
```

#### 3. Full Training Run
```bash
# 1M+ steps, checkpoint every 100 episodes
PYTHONPATH=. python training/trainer.py \
  --mode full \
  --total-timesteps 1000000 \
  --checkpoint-interval-episodes 100 \
  --seed 42
```

#### 4. Large-Scale Training
```bash
# 10M+ steps with multiple environments
PYTHONPATH=. python training/trainer.py \
  --mode full \
  --total-timesteps 10000000 \
  --num-envs 8 \
  --vec-env-type dummy \
  --checkpoint-interval-episodes 200 \
  --seed 42
```

### Increasing Parallel Environments

**Single Environment (Default):**
```bash
python training/trainer.py --mode full --total-timesteps 1000000 --num-envs 1
```

**Multiple Environments (Faster Data Collection):**
```bash
# 4 parallel environments
python training/trainer.py --mode full --total-timesteps 1000000 --num-envs 4

# 8 parallel environments
python training/trainer.py --mode full --total-timesteps 1000000 --num-envs 8
```

**VecEnv Types:**
- `--vec-env-type dummy` (default): Sequential execution, same process
- `--vec-env-type subproc`: Parallel execution, separate processes

**Trade-offs:**
- More envs = faster data collection, but more memory/CPU
- Start with `num-envs=4` and increase if system can handle it
- Monitor CPU usage and memory during training

### Monitoring Training Progress

**During Training:**
```bash
# Follow log in real-time
LATEST_RUN=$(ls -td runs/*/ | head -1)
tail -f ${LATEST_RUN}training.log

# Check TensorBoard (in another terminal)
tensorboard --logdir logs/
# Then open http://localhost:6006
```

**After Training:**
```bash
# View final summary
LATEST_RUN=$(ls -td runs/*/ | head -1)
tail -30 ${LATEST_RUN}training.log

# Check all checkpoints
ls -lh checkpoints/ppo_agent/*/ep*.zip

# Count episodes
grep -c "Episode.*completed" ${LATEST_RUN}training.log
```

### Performance Considerations

**Before Scaling Up:**

1. **Verify baseline performance:**
   ```bash
   PYTHONPATH=. python scripts/benchmark_env.py --num-steps 50000
   ```
   - Should get >200 steps/s for single env
   - If lower, investigate before scaling up

2. **Test with multiple envs:**
   ```bash
   # Quick test with 4 envs
   PYTHONPATH=. python training/trainer.py \
     --mode smoke \
     --num-envs 4 \
     --total-timesteps 5000
   ```
   - Monitor CPU and memory usage
   - Verify speed improvement

3. **Monitor system resources:**
   ```bash
   # In another terminal, monitor during training
   top  # or htop
   # Watch CPU and memory usage
   ```

**Scaling Guidelines:**

- **Single env:** Good for debugging, slower training
- **4 envs:** Good balance, recommended starting point
- **8 envs:** Faster training, requires more CPU/memory
- **16+ envs:** May hit system limits, test first

**Speed Targets:**
- Single env: >200 steps/s
- 4 envs: >500 env steps/s (125 steps/s × 4)
- 8 envs: >800 env steps/s (100 steps/s × 8)

### Stability Checks

**Before Long Training:**

1. **Run smoke test successfully** (already done)
2. **Run medium test (100K steps):**
   ```bash
   PYTHONPATH=. python training/trainer.py \
     --mode full \
     --total-timesteps 100000 \
     --checkpoint-interval-episodes 25
   ```
3. **Verify:**
   - No crashes or exceptions
   - Checkpoints save correctly
   - Speed metrics remain stable
   - No memory leaks (memory usage doesn't grow unbounded)

**If Medium Test Passes:**
- Proceed to full training (1M+ steps)
- Consider using multiple environments for faster training
- Monitor first few hours of training closely

---

## Quick Reference

### Essential Commands

```bash
# Benchmark
PYTHONPATH=. python scripts/benchmark_env.py

# Smoke test
PYTHONPATH=. python training/trainer.py --config training/config_smoke.yaml

# Full training
PYTHONPATH=. python training/trainer.py --mode full --total-timesteps 1000000

# View latest log
LATEST_RUN=$(ls -td runs/*/ | head -1)
cat ${LATEST_RUN}training.log

# Check checkpoints
ls -la checkpoints/ppo_agent/*/ep*.zip
```

### Performance Targets

- **Environment:** >200 steps/s (single env), >500 env steps/s (4 envs)
- **Move Generation:** <10ms average, >100 calls/s
- **Training Speed:** >50 steps/s minimum, >100 steps/s preferred

### File Locations

- **Logs:** `runs/YYYYMMDD_HHMMSS_<experiment>/training.log`
- **Checkpoints:** `checkpoints/ppo_agent/<run_id>/ep<episode>.zip`
- **TensorBoard:** `logs/PPO_*/events.out.tfevents.*`
- **Configs:** `training/config_smoke.yaml`, `training/config_full.yaml`

---

**Last Updated:** 2025-01-XX

