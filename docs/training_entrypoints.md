# Training Entrypoints & Configuration Guide

This document describes how to launch RL training runs in the Blokus RL project, including all configuration options and entry points.

---

## Quick Start

### Basic Training Run

The simplest way to start training:

```bash
# From project root
PYTHONPATH=. python training/trainer.py --mode smoke
```

Or using the module syntax:

```bash
python -m training.trainer --mode smoke
```

### Recommended First Steps

1. **Run a smoke test** (quick verification - **recommended first**):
   ```bash
   python training/trainer.py --mode smoke
   ```
   Or using the smoke test config:
   ```bash
   python training/trainer.py --config training/config_smoke.yaml
   ```

2. **Run a full training session**:
   ```bash
   python training/trainer.py --mode full --total-timesteps 1000000
   ```

3. **Use a config file**:
   ```bash
   python training/trainer.py --config training/config_smoke.yaml
   ```

---

## Smoke Test Run

### Quick Validation Command

Run a quick end-to-end validation to verify the training pipeline works correctly:

```bash
# Using smoke test config (recommended)
PYTHONPATH=. python training/trainer.py --config training/config_smoke.yaml

# Or using CLI mode flag
PYTHONPATH=. python training/trainer.py --mode smoke
```

### What the Smoke Test Does

The smoke test configuration (`training/config_smoke.yaml`) is optimized for quick validation:

- **~15,000 total timesteps** - Runs in a few minutes
- **10 episodes** - Enough to see periodic stats and checkpoint creation
- **Checkpoint every 5 episodes** - Ensures at least one checkpoint is created
- **INFO-level logging** - Shows episode stats and speed metrics without excessive verbosity
- **Sanity checks enabled** - Catches NaN/Inf issues immediately
- **Same environment setup** - Uses the same env configuration as full training

### Expected Output

A healthy smoke test run should show:

#### 1. **Run Directory Creation**
```
2025-01-15 14:30:22 - training.trainer - INFO - Created run directory: runs/20250115_143022_smoke
2025-01-15 14:30:22 - training.trainer - INFO - Log file: runs/20250115_143022_smoke/training.log
```

#### 2. **Configuration Summary**
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

#### 3. **Episode Completion Logs**
```
2025-01-15 14:30:25 - training.trainer - INFO - Episode 1 completed: reward=12.34, length=45
2025-01-15 14:30:28 - training.trainer - INFO - Episode 2 completed: reward=15.67, length=52
...
```

#### 4. **Periodic Statistics (Every 10 Episodes)**
```
2025-01-15 14:31:10 - training.trainer - INFO - Episodes 1-10 (last 10): reward=14.23±2.45, length=48.2±5.1, speed=125.5 steps/s (125.5 env steps/s)
```

#### 5. **Checkpoint Creation**
```
2025-01-15 14:31:05 - training.trainer - INFO - Saved checkpoint at episode 5
2025-01-15 14:31:15 - training.trainer - INFO - Saved checkpoint at episode 10
```

#### 6. **Training Summary**
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

### What to Look For (Healthy Run)

✅ **Good Signs:**
- Run directory created successfully
- Episodes complete without errors
- Episode rewards are reasonable (typically 5-30 for Blokus)
- Speed metrics show >50 steps/s (system-dependent, but should be reasonable)
- No NaN or Inf errors in logs
- At least one checkpoint saved
- Training summary shows completed episodes

❌ **Warning Signs:**
- **"No legal actions available"** errors - May indicate environment issue
- **"Non-finite reward detected"** - Indicates reward calculation problem
- **Very low steps/s (<10)** - May indicate performance issue
- **No checkpoints saved** - Check checkpoint configuration
- **Training stops early** - Check max_episodes or max_steps_per_episode limits

### Smoke Test Configuration

The smoke test config (`training/config_smoke.yaml`) includes:

```yaml
mode: smoke
max_episodes: 10
total_timesteps: 15000
checkpoint_interval_episodes: 5  # Saves checkpoint every 5 episodes
logging_verbosity: 1  # INFO level
enable_sanity_checks: true
```

### Customizing the Smoke Test

You can override any parameter:

```bash
# More episodes
python training/trainer.py --config training/config_smoke.yaml --max-episodes 20

# Different checkpoint interval
python training/trainer.py --config training/config_smoke.yaml --checkpoint-interval-episodes 3

# More verbose logging
python training/trainer.py --config training/config_smoke.yaml --verbosity 2
```

### Smoke Test Workflow

**Before starting long training:**

1. **Run smoke test:**
   ```bash
   PYTHONPATH=. python training/trainer.py --config training/config_smoke.yaml
   ```

2. **Verify output:**
   - Check that run directory was created: `runs/YYYYMMDD_HHMMSS_smoke/`
   - Check log file: `runs/YYYYMMDD_HHMMSS_smoke/training.log`
   - Verify no errors in console or log
   - Confirm checkpoint was saved: `checkpoints/ppo_agent/<run_id>/ep000005.zip`

3. **Check performance:**
   - Speed metrics should be reasonable (>50 steps/s typically)
   - Episode rewards should be in expected range
   - No NaN/Inf warnings

4. **If smoke test passes, proceed to full training:**
   ```bash
   python training/trainer.py --mode full --total-timesteps 1000000
   ```

### Troubleshooting Smoke Test

**"No module named 'training'"**
```bash
# Use PYTHONPATH
PYTHONPATH=. python training/trainer.py --config training/config_smoke.yaml
```

**"Config file not found"**
```bash
# Use absolute path or ensure you're in project root
python training/trainer.py --config $(pwd)/training/config_smoke.yaml
```

**Smoke test runs but no checkpoints**
- Check `checkpoint_interval_episodes` is set (should be 5 in smoke config)
- Verify `checkpoint_dir` exists and is writable
- Check logs for checkpoint errors

**Very slow smoke test**
- Run performance benchmark first: `python scripts/benchmark_env.py`
- Check system resources (CPU, memory)
- Verify optimizations are enabled (frontier/bitboard)

---

## Main Training Script

**File:** `training/trainer.py`

**Entry Point:** `main()` function (called when script is run directly)

**Function:** `train(config: TrainingConfig)` - Core training logic

### How to Run

```bash
# Direct execution
python training/trainer.py [arguments]

# As module
python -m training.trainer [arguments]

# With PYTHONPATH (if needed)
PYTHONPATH=. python training/trainer.py [arguments]
```

---

## Configuration Methods

Configuration can be provided in three ways (in order of precedence):

1. **CLI Arguments** (highest priority) - Overrides everything
2. **Config File** (via `--config`) - Overrides defaults
3. **Built-in Defaults** - Based on mode (smoke/full)

### Example: Using Config File

```bash
# Smoke test config
python training/trainer.py --config training/config_smoke.yaml

# Full training config
python training/trainer.py --config training/config_full.yaml

# Config file + CLI overrides
python training/trainer.py --config training/config_full.yaml --total-timesteps 2000000
```

### Example: Using Agent Config

```bash
# Use agent hyperparameter config
python training/trainer.py --agent-config config/agents/ppo_agent_v1.yaml

# Combine with training config
python training/trainer.py \
  --config training/config_full.yaml \
  --agent-config config/agents/ppo_agent_v1.yaml
```

---

## Important CLI Arguments

### Mode Selection

- `--mode {smoke,full}` (default: `full`)
  - **smoke**: Quick verification (5 episodes, 10K timesteps, DEBUG logging)
  - **full**: Production training (unlimited episodes, configurable timesteps)

### Training Parameters

- `--total-timesteps N` (default: 100000)
  - Total number of environment steps to train
  - In smoke mode: defaults to 10000

- `--n-steps N` (default: 2048)
  - Steps to collect per PPO update
  - In smoke mode: defaults to 512

- `--batch-size N` (default: 64)
  - Batch size for PPO updates

- `--lr, --learning-rate FLOAT` (default: 3e-4)
  - Learning rate for optimizer

### Environment Configuration

- `--num-envs N` (default: 1)
  - Number of parallel environments
  - `1` = single environment
  - `>1` = vectorized environment (DummyVecEnv or SubprocVecEnv)
  - **Note:** More envs = faster data collection, but more memory/CPU

- `--vec-env-type {dummy,subproc}` (default: `dummy`)
  - **dummy**: Sequential execution in same process (good for CPU-bound)
  - **subproc**: Parallel execution in separate processes (good for I/O-bound)
  - **Note:** SubprocVecEnv requires picklable factory functions (already implemented)

- `--max-steps-per-episode N` (default: 1000)
  - Maximum steps before episode truncation
  - In smoke mode: defaults to 100

### Episode Limits

- `--max-episodes N` (default: None)
  - Maximum number of episodes to run
  - `None` = unlimited (use `total_timesteps` instead)
  - In smoke mode: defaults to 5

### Logging & Output

- `--checkpoint-dir PATH` (default: `checkpoints`)
  - Directory to save model checkpoints
  - Structure: `checkpoints/<agent_id>/<run_id>/ep<episode>.zip`

- `--tensorboard-log-dir PATH` (default: `./logs`)
  - Directory for TensorBoard logs
  - Files: `logs/PPO_*/events.out.tfevents.*`

- `--checkpoint-interval-episodes N` (default: 50)
  - Save checkpoint every N episodes
  - `None` = only save at end of training

- `--keep-last-n-checkpoints N` (default: 3)
  - Number of recent checkpoints to keep (older ones auto-deleted)

- `--verbosity {0,1,2}` (default: 1)
  - `0` = ERROR only
  - `1` = INFO (default)
  - `2` = DEBUG (verbose)

- `--log-action-details` (flag, default: False)
  - Log detailed action information at each step
  - Auto-enabled in smoke mode

### Reproducibility

- `--seed N` (default: None)
  - Random seed (sets all seeds: env, agent, numpy, torch)
  - If not set, training is non-deterministic

- `--env-seed N` (default: None)
  - Separate seed for environment
  - Overrides `--seed` for env only

- `--agent-seed N` (default: None)
  - Separate seed for agent/RL algorithm
  - Overrides `--seed` for agent only

### Resume & Checkpoints

- `--resume-from-checkpoint PATH`
  - Path to checkpoint file (`.zip`) to resume training from
  - Restores model weights, optimizer state, and training config
  - Example: `--resume-from-checkpoint checkpoints/ppo_agent/run_123/ep000050.zip`

### Agent Configuration

- `--agent-config PATH`
  - Path to agent hyperparameter config file (YAML/JSON)
  - Located in `config/agents/`
  - Defines: learning rate, gamma, network architecture, PPO params
  - Example: `--agent-config config/agents/ppo_agent_v1.yaml`

### Debugging

- `--disable-sanity-checks` (flag)
  - Disable NaN/Inf detection and other sanity checks
  - Not recommended for production

---

## Environment Construction

### How Environments Are Created

The environment construction flow:

1. **Config Parsing** (`training/config.py`)
   - Loads `TrainingConfig` from CLI args or config file
   - Determines `num_envs` and `vec_env_type`

2. **Environment Factory** (`training/env_factory.py`)
   - `make_training_env(config, mask_fn)` creates environment(s)
   - If `num_envs=1`: Single `GymnasiumBlokusWrapper` wrapped with `ActionMasker`
   - If `num_envs>1`: `DummyVecEnv` or `SubprocVecEnv` containing multiple wrapped envs

3. **Environment Wrapping Chain**:
   ```
   BlokusEnv (PettingZoo AECEnv)
     ↓
   GymnasiumBlokusWrapper (Gymnasium compatibility)
     ↓
   ActionMasker (Action masking for MaskablePPO)
     ↓
   [Optional: VecEnv wrapper if num_envs > 1]
     ↓
   Monitor (SB3 automatic wrapper)
   ```

### Environment Details

- **Action Space:** Discrete(36400) - ~36,400 possible actions
- **Observation Space:** Box(30, 20, 20) - 30 channels × 20×20 board
- **Agents:** 4 players (currently training focuses on `player_0`)
- **Rewards:** Dense (score delta per move)
- **Termination:** Game over or max steps per episode

### Seeding

Seeds are set in `training/seeds.py`:

- **Environment seed:** Controls game state randomness
- **Agent seed:** Controls RL algorithm randomness (PyTorch, NumPy)
- **Separate seeds:** Allows independent control of env vs agent randomness

**Location:** `training/trainer.py` lines 663-668

```python
set_seed(
    seed=config.random_seed,
    env_seed=config.env_seed,
    agent_seed=config.agent_seed,
    log=True
)
```

---

## Logging Setup

### Python Logging

**Location:** `training/trainer.py` lines 67-72

- **Format:** Timestamp, logger name, level, message
- **Level:** Controlled by `--verbosity` (0=ERROR, 1=INFO, 2=DEBUG)
- **Output:** Console (stdout)

### TensorBoard Logging

**Location:** `training/trainer.py` line 730

- **Automatic:** Enabled via SB3's `tensorboard_log` parameter
- **Directory:** `--tensorboard-log-dir` (default: `./logs`)
- **Metrics:** Policy loss, value loss, entropy, learning rate, etc.

### MongoDB Logging

**Location:** `training/run_logger.py`

- **Optional:** Gracefully degrades if MongoDB unavailable
- **Data:** Training run metadata, episode metrics, checkpoints
- **Creation:** `training/trainer.py` lines 643-660

---

## Checkpointing Setup

### Checkpoint Saving

**Location:** `training/checkpoints.py`

- **Frequency:** Every N episodes (via `--checkpoint-interval-episodes`)
- **Format:** SB3 `.zip` files + companion `_metadata.json`
- **Structure:** `checkpoints/<agent_id>/<run_id>/ep<episode>.zip`
- **Cleanup:** Auto-deletes old checkpoints (keeps last N)

### Checkpoint Loading

**Location:** `training/trainer.py` lines 693-714

- **Resume:** `--resume-from-checkpoint PATH`
- **Restores:** Model weights, optimizer state, training config
- **Continues:** From saved episode number

---

## Example Commands

### Smoke Test (Quick Verification)

```bash
# Minimal smoke test
python training/trainer.py --mode smoke

# Smoke test with custom seed
python training/trainer.py --mode smoke --seed 42

# Smoke test with verbose logging
python training/trainer.py --mode smoke --verbosity 2 --log-action-details
```

### Full Training

```bash
# Basic full training
python training/trainer.py --mode full --total-timesteps 1000000

# Full training with custom checkpoint interval
python training/trainer.py --mode full --total-timesteps 1000000 --checkpoint-interval-episodes 100

# Full training with multiple environments
python training/trainer.py --mode full --total-timesteps 1000000 --num-envs 4

# Full training with agent config
python training/trainer.py \
  --mode full \
  --total-timesteps 1000000 \
  --agent-config config/agents/ppo_agent_v1.yaml
```

### Using Config Files

```bash
# Smoke test config
python training/trainer.py --config training/config_smoke.yaml

# Full training config
python training/trainer.py --config training/config_full.yaml

# Config file with overrides
python training/trainer.py \
  --config training/config_full.yaml \
  --total-timesteps 2000000 \
  --checkpoint-interval-episodes 50
```

### Resume Training

```bash
# Resume from checkpoint
python training/trainer.py \
  --resume-from-checkpoint checkpoints/ppo_agent/run_abc123/ep000050.zip \
  --total-timesteps 2000000
```

### Hyperparameter Sweep

```bash
# Run sweep (uses run_sweep.py, not trainer.py)
python training/run_sweep.py config/agents/ppo_agent_sweep_*.yaml
```

---

## Debug Modes

### Enable Detailed Logging

```bash
# Maximum verbosity
python training/trainer.py --mode smoke --verbosity 2 --log-action-details
```

This will log:
- Every step in first 3 episodes
- Action selections and rewards
- Legal moves count
- Score updates
- All DEBUG-level messages

### Move Generation Profiling

Enable move generation timing:

```bash
BLOKUS_PROFILE_MOVEGEN=1 python training/trainer.py --mode smoke
```

This logs:
- Total time spent in move generation per episode
- Average and max time per call
- Call count

### Move Generation Debug Logging

Enable detailed move generation logs:

```bash
BLOKUS_MOVEGEN_DEBUG=1 python training/trainer.py --mode smoke
```

This logs:
- Per-call timing (ms)
- Piece-level timing
- Frontier size
- Legal moves count

### Disable Sanity Checks

```bash
# Not recommended, but useful for debugging
python training/trainer.py --mode smoke --disable-sanity-checks
```

---

## Configuration Files

### Training Config Files

Located in `training/`:

- **`config_smoke.yaml`**: Smoke test configuration
- **`config_full.yaml`**: Full training configuration
- **`config_smoke.json`**: Smoke test (JSON format)

### Agent Config Files

Located in `config/agents/`:

- **`ppo_agent_v1.yaml`**: Base PPO agent config
- **`ppo_agent_sweep_lr_high.yaml`**: High learning rate variant
- **`ppo_agent_sweep_lr_low.yaml`**: Low learning rate variant
- **`ppo_agent_sweep_gamma_high.yaml`**: High gamma variant

See `config/agents/README.md` for details.

### Config File Format

YAML example:

```yaml
mode: full
total_timesteps: 1000000
n_steps: 2048
learning_rate: 3e-4
batch_size: 64
num_envs: 4
vec_env_type: dummy
checkpoint_interval_episodes: 100
random_seed: 42
checkpoint_dir: checkpoints
tensorboard_log_dir: ./logs
```

JSON example:

```json
{
  "mode": "full",
  "total_timesteps": 1000000,
  "n_steps": 2048,
  "learning_rate": 3e-4,
  "batch_size": 64,
  "num_envs": 4,
  "vec_env_type": "dummy",
  "checkpoint_interval_episodes": 100,
  "random_seed": 42,
  "checkpoint_dir": "checkpoints",
  "tensorboard_log_dir": "./logs"
}
```

---

## Environment Variables

### Move Generation Flags

- **`BLOKUS_USE_FRONTIER_MOVEGEN`** (default: `1`)
  - `1` = Use frontier-based move generation (fast)
  - `0` = Use naive full-board scan (slow, for debugging)

- **`BLOKUS_USE_BITBOARD_LEGALITY`** (default: `1`)
  - `1` = Use bitboard legality checks (fast)
  - `0` = Use grid-based legality checks (slower, for debugging)

- **`BLOKUS_USE_HEURISTIC_ANCHORS`** (default: `0`)
  - `1` = Use heuristic anchor selection (faster, with fallback)
  - `0` = Use all anchors (slower, exact mode)

- **`BLOKUS_MOVEGEN_DEBUG`** (default: `0`)
  - `1` = Enable move generation timing logs
  - `0` = Disable timing logs

- **`BLOKUS_PROFILE_MOVEGEN`** (default: `0`)
  - `1` = Enable episode-level move generation profiling
  - `0` = Disable profiling

---

## Output Locations

### Checkpoints

- **Directory:** `checkpoints/` (or `--checkpoint-dir`)
- **Structure:** `checkpoints/<agent_id>/<run_id>/ep<episode>.zip`
- **Metadata:** `checkpoints/<agent_id>/<run_id>/ep<episode>_metadata.json`
- **Config:** `checkpoints/training_config.yaml`

### TensorBoard Logs

- **Directory:** `./logs/` (or `--tensorboard-log-dir`)
- **Files:** `logs/PPO_*/events.out.tfevents.*`
- **View:** `tensorboard --logdir ./logs`

### MongoDB

- **Database:** MongoDB (if available)
- **Collection:** `training_runs`
- **Records:** TrainingRun documents with episode metrics

---

## Common Workflows

### 1. Quick Verification

```bash
python training/trainer.py --mode smoke --seed 42
```

### 2. Short Training Run

```bash
python training/trainer.py \
  --mode full \
  --total-timesteps 100000 \
  --checkpoint-interval-episodes 25 \
  --seed 42
```

### 3. Production Training

```bash
python training/trainer.py \
  --mode full \
  --total-timesteps 10000000 \
  --num-envs 8 \
  --vec-env-type dummy \
  --checkpoint-interval-episodes 100 \
  --agent-config config/agents/ppo_agent_v1.yaml \
  --seed 42
```

### 4. Resume Training

```bash
python training/trainer.py \
  --resume-from-checkpoint checkpoints/ppo_agent/run_abc123/ep001000.zip \
  --total-timesteps 20000000
```

### 5. Debug Run

```bash
BLOKUS_MOVEGEN_DEBUG=1 \
BLOKUS_PROFILE_MOVEGEN=1 \
python training/trainer.py \
  --mode smoke \
  --verbosity 2 \
  --log-action-details
```

---

## Troubleshooting

### "No module named 'training'"

**Solution:** Run from project root with `PYTHONPATH=.`:

```bash
PYTHONPATH=. python training/trainer.py --mode smoke
```

### "Config file not found"

**Solution:** Use absolute path or relative to project root:

```bash
# Relative (from project root)
python training/trainer.py --config training/config_smoke.yaml

# Absolute
python training/trainer.py --config /full/path/to/config.yaml
```

### "MongoDB connection failed"

**Solution:** Training will continue without MongoDB logging. Check:
- MongoDB is running
- Connection string in `webapi/db/mongo.py` is correct
- Network connectivity

### "Checkpoint not found" (when resuming)

**Solution:** Verify checkpoint path:

```bash
# List checkpoints
ls -la checkpoints/ppo_agent/*/ep*.zip

# Use full path
python training/trainer.py \
  --resume-from-checkpoint $(pwd)/checkpoints/ppo_agent/run_123/ep000050.zip
```

---

## Logging & Run Outputs

### Log File Location

Every training run automatically creates a timestamped run directory:

```
runs/<YYYYMMDD>_<HHMMSS>_<experiment_name>/
```

Where:
- `YYYYMMDD_HHMMSS` is the timestamp when training started
- `experiment_name` defaults to the training mode (`smoke` or `full`)

The log file is saved as:
```
runs/<timestamp>_<experiment_name>/training.log
```

### Log Contents

The log file contains:
- **Run start information:** Timestamp, run directory, git hash/branch
- **Configuration summary:** All training parameters, hyperparameters, environment settings
- **Reproducibility info:** Git commit hash, Python version, package versions
- **Episode summaries:** Reward, length, speed metrics (every 10 episodes)
- **Training summary:** Final statistics, average speeds, episode statistics

### Log Format

Logs use a consistent format:
```
YYYY-MM-DD HH:MM:SS - logger.name - LEVEL - message
```

Example:
```
2025-01-15 14:30:22 - training.trainer - INFO - Episode 10 completed: reward=15.23, length=45
```

### Changing Log Level

Use the `--verbosity` flag:

```bash
# ERROR only
python training/trainer.py --verbosity 0

# INFO (default)
python training/trainer.py --verbosity 1

# DEBUG (verbose)
python training/trainer.py --verbosity 2
```

### Logging Frequency

- **Episode completion:** Every episode (INFO level)
- **Periodic stats:** Every 10 episodes (INFO level)
  - Includes: average reward ± std, average length ± std, speed metrics
- **Checkpoints:** Every N episodes (configurable via `--checkpoint-interval-episodes`)
- **Detailed step info:** First 3 episodes, first 10 steps (smoke-test mode only, DEBUG level)

### Speed Metrics

Speed metrics are logged every 10 episodes:
- **Steps per second:** Training steps per second
- **Environment steps per second:** Total environment steps per second (accounts for vectorization)

Example:
```
Episodes 1-10 (last 10): reward=15.23±2.45, length=45.2±5.1, speed=12.5 steps/s (50.0 env steps/s)
```

### Console vs File Logging

Logs are written to **both**:
- **Console (stdout):** Real-time monitoring
- **Log file:** Persistent record for later analysis

Both use the same format and level.

### Viewing Logs

```bash
# View log file
cat runs/20250115_143022_smoke/training.log

# Follow log in real-time (if training is running)
tail -f runs/20250115_143022_smoke/training.log

# Search for specific patterns
grep "Episode.*completed" runs/20250115_143022_smoke/training.log

# Count episodes
grep -c "Episode.*completed" runs/20250115_143022_smoke/training.log
```

## Performance Benchmarks

### Running the Benchmark

The benchmark script measures environment step performance and move generation speed:

```bash
# Run both benchmarks (default)
PYTHONPATH=. python scripts/benchmark_env.py

# Run with custom parameters
PYTHONPATH=. python scripts/benchmark_env.py --num-steps 50000 --movegen-iterations 1000

# Run only environment benchmark
PYTHONPATH=. python scripts/benchmark_env.py --skip-movegen

# Run only move generation benchmark
PYTHONPATH=. python scripts/benchmark_env.py --skip-env
```

### Benchmark Options

- `--num-steps N`: Number of environment steps to run (default: 10,000)
- `--movegen-iterations N`: Number of move generation calls (default: 1,000)
- `--seed N`: Random seed for reproducibility (default: 42)
- `--skip-env`: Skip environment step benchmark
- `--skip-movegen`: Skip move generation benchmark

### Expected Output

The benchmark outputs two sections:

#### 1. Environment Step() Benchmark

Measures how fast the environment can process steps:

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

**Key Metrics:**
- **Steps per second**: Primary performance metric - higher is better
- **Episodes completed**: Number of full episodes in the benchmark
- **Average steps per episode**: Typical episode length

**Typical Values:**
- Good: >500 steps/s
- Acceptable: 200-500 steps/s
- Slow: <200 steps/s (may indicate performance issues)

#### 2. Move Generation Benchmark

Measures how fast legal moves can be generated:

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

**Key Metrics:**
- **Calls per second**: How many move generation calls per second
- **Average time per call**: Average milliseconds per call
- **Percentiles (p50, p95, p99)**: Distribution of call times
- **Average moves per call**: Typical number of legal moves found

**Typical Values:**
- Good: <10ms average, >100 calls/s
- Acceptable: 10-50ms average, 20-100 calls/s
- Slow: >50ms average, <20 calls/s (may indicate optimization needed)

### Interpreting Results

**For Training:**
- Environment step speed directly impacts training throughput
- If steps/s is low, training will be slow
- Target: >500 steps/s for efficient training

**For Move Generation:**
- Move generation is called every step to build action masks
- High move generation time can bottleneck environment steps
- Target: <10ms average for smooth training

**Performance Bottlenecks:**
- If environment steps/s is low but move generation is fast: Check observation generation, reward calculation
- If move generation is slow: Consider enabling frontier/bitboard optimizations (default), or check board state complexity
- If both are slow: Check system resources (CPU, memory)

### Before Long Training Runs

Run the benchmark before starting long training to:
1. **Baseline performance**: Establish expected steps/s for your system
2. **Detect regressions**: Compare after code changes
3. **System validation**: Ensure performance is acceptable
4. **Resource planning**: Estimate training time based on steps/s

**Example Workflow:**
```bash
# 1. Run benchmark
PYTHONPATH=. python scripts/benchmark_env.py --num-steps 50000

# 2. Verify performance is acceptable
# (e.g., >500 steps/s for environment)

# 3. Start training
python training/trainer.py --mode full --total-timesteps 1000000
```

### Troubleshooting

**Low steps/s:**
- Check if frontier/bitboard optimizations are enabled (default: enabled)
- Verify system resources (CPU usage, memory)
- Check for background processes consuming resources
- Consider using fewer parallel environments if CPU-bound

**High move generation time:**
- Ensure `BLOKUS_USE_FRONTIER_MOVEGEN=1` (default)
- Ensure `BLOKUS_USE_BITBOARD_LEGALITY=1` (default)
- Check if board states are unusually complex (many pieces placed)
- Verify no debug logging is enabled (`BLOKUS_MOVEGEN_DEBUG=0`)

---

## Related Documentation

- **Training Architecture:** `docs/training-architecture.md`
- **Checkpoints:** `docs/checkpoints.md`
- **Current State:** `docs/rl_current_state.md`
- **Agent Configs:** `config/agents/README.md`
- **Training README:** `training/README.md`

---

**Last Updated:** 2025-01-XX

