# Training Documentation

This directory contains the training infrastructure for Blokus RL agents using Stable-Baselines3's MaskablePPO.

## Overview

The training system supports two modes:
- **Smoke-test mode**: Quick verification runs with small episode counts and detailed logging
- **Full mode**: Production training with longer runs and optimized logging

## Quick Start

### Smoke Test (Recommended First Step)

Run a quick smoke test to verify the environment, agent, and training pipeline work end-to-end:

```bash
# Using CLI arguments
python training/trainer.py --mode smoke

# Using config file
python training/trainer.py --config training/config_smoke.yaml

# With custom parameters
python training/trainer.py --mode smoke --max-episodes 10 --seed 42
```

### Full Training

Run a full training session:

```bash
# Using CLI arguments
python training/trainer.py --mode full --total-timesteps 1000000

# Using config file
python training/trainer.py --config training/config_full.yaml

# With custom parameters
python training/trainer.py --mode full --total-timesteps 1000000 --checkpoint-interval 100
```

## Configuration

### Configuration Modes

#### Smoke-Test Mode (`--mode smoke`)

Smoke-test mode is designed for quick verification. It automatically:
- Sets `max_episodes` to 5 (if not specified)
- Caps `max_steps_per_episode` to 100
- Limits `total_timesteps` to 10,000
- Reduces `n_steps` to 512
- Enables DEBUG-level logging (`logging_verbosity=2`)
- Enables detailed action logging
- Enables all sanity checks

**Use smoke-test mode to:**
- Verify the environment works correctly
- Check that agents can select actions
- Ensure rewards are computed properly
- Validate that the training loop runs without errors
- Debug configuration issues

#### Full Mode (`--mode full`)

Full mode uses production-ready defaults:
- No episode limit (uses `total_timesteps` instead)
- `max_steps_per_episode` = 1000
- `total_timesteps` = 100,000 (configurable)
- `n_steps` = 2048
- INFO-level logging
- Sanity checks enabled but less verbose

### Configuration Sources

Configuration can be provided in three ways (in order of precedence):

1. **CLI Arguments**: Highest priority, overrides config file
2. **Config File**: YAML or JSON file (see examples below)
3. **Defaults**: Built-in defaults based on mode

### Example Config Files

#### Smoke-Test Config (`config_smoke.yaml`)

```yaml
mode: smoke
max_episodes: 5
max_steps_per_episode: 100
total_timesteps: 10000
n_steps: 512
learning_rate: 3e-4
batch_size: 64
logging_verbosity: 2
random_seed: 42
checkpoint_dir: checkpoints
tensorboard_log_dir: ./logs
enable_sanity_checks: true
log_action_details: true
```

#### Full Training Config (`config_full.yaml`)

```yaml
mode: full
max_episodes: null
max_steps_per_episode: 1000
total_timesteps: 1000000
n_steps: 2048
learning_rate: 3e-4
batch_size: 64
logging_verbosity: 1
random_seed: 42
checkpoint_dir: checkpoints
tensorboard_log_dir: ./logs
checkpoint_interval: 100
enable_sanity_checks: true
log_action_details: false
```

## CLI Arguments

### Mode and Basic Parameters

- `--mode {smoke,full}`: Training mode (default: `full`)
- `--config PATH`: Path to YAML or JSON config file
- `--max-episodes N`: Maximum number of episodes (None = unlimited)
- `--max-steps-per-episode N`: Maximum steps per episode (default: 1000)
- `--total-timesteps N`: Total timesteps for training (default: 100000)
- `--n-steps N`: Steps to collect per update (default: 2048)
- `--lr, --learning-rate FLOAT`: Learning rate (default: 3e-4)
- `--batch-size N`: Batch size (default: 64)

### Logging and Debugging

- `--verbosity {0,1,2}`: Logging level (0=ERROR, 1=INFO, 2=DEBUG)
- `--log-action-details`: Log detailed action information

### Seeds

- `--seed N`: Random seed (sets all seeds)
- `--env-seed N`: Separate seed for environment
- `--agent-seed N`: Separate seed for agent

### Checkpoints and Logging

- `--checkpoint-dir PATH`: Directory for checkpoints (default: `checkpoints`)
- `--tensorboard-log-dir PATH`: Directory for TensorBoard logs (default: `./logs`)
- `--checkpoint-interval-episodes N`: Save checkpoint every N episodes (default: 50, None = only at end)
- `--keep-last-n-checkpoints N`: Number of recent checkpoints to keep (default: 3)
- `--resume-from-checkpoint PATH`: Path to checkpoint file to resume training from

See `docs/checkpoints.md` for complete checkpointing documentation.

### Sanity Checks

- `--disable-sanity-checks`: Disable NaN/Inf detection and other checks

## Smoke-Test Workflow

### Step 1: Run Smoke Test

```bash
python training/trainer.py --mode smoke
```

### Step 2: Verify Output

Look for these indicators of success:

1. **Configuration logged**: The effective config should be printed at the start
2. **Seeds initialized**: Seed values should be logged
3. **Episodes complete**: Should see "Episode X completed" messages
4. **No errors**: No exceptions or error messages
5. **Training summary**: Final statistics should be printed

Example successful output:
```
2025-01-XX XX:XX:XX - training.trainer - INFO - ================================================================================
2025-01-XX XX:XX:XX - training.trainer - INFO - Training Configuration
2025-01-XX XX:XX:XX - training.trainer - INFO - ================================================================================
2025-01-XX XX:XX:XX - training.trainer - INFO - Mode: SMOKE
2025-01-XX XX:XX:XX - training.trainer - INFO - Max Episodes: 5
...
2025-01-XX XX:XX:XX - training.trainer - INFO - Episode 1 completed: reward=15.23, length=45
...
2025-01-XX XX:XX:XX - training.trainer - INFO - Training Summary
2025-01-XX XX:XX:XX - training.trainer - INFO - Total episodes: 5
...
```

### Step 3: Check Logs

In smoke-test mode, detailed logs are written at DEBUG level. Check for:
- Action selections and rewards at each step (first few episodes)
- Legal moves count
- Score updates
- No NaN or Inf values

### Step 4: Verify Checkpoints

After training, check that:
- Model checkpoint is saved in `checkpoints/ppo_blokus`
- Training config is saved in `checkpoints/training_config.yaml`
- TensorBoard logs are in `./logs/`

### Step 5: Run Full Training

Once smoke test passes, run full training:

```bash
python training/trainer.py --mode full --total-timesteps 1000000
```

## Sanity Checks

The training system includes automatic sanity checks:

- **NaN/Inf Detection**: Checks rewards, observations, and actions for non-finite values
- **Action Validity**: Verifies actions are within valid ranges
- **Episode Termination**: Ensures episodes terminate correctly

In smoke-test mode, sanity check failures raise exceptions immediately. In full mode, they log warnings but continue training.

## Reproducibility

### Seed Control

Seeds can be set in three ways:

1. **Single seed** (sets all):
   ```bash
   python training/trainer.py --seed 42
   ```

2. **Separate seeds**:
   ```bash
   python training/trainer.py --seed 42 --env-seed 100 --agent-seed 200
   ```

3. **Config file**:
   ```yaml
   random_seed: 42
   env_seed: 100
   agent_seed: 200
   ```

### Saved Configuration

After training, the effective configuration is saved to:
```
checkpoints/training_config.yaml
```

This allows you to reproduce exact training runs by loading the saved config.

## Troubleshooting

### Common Issues

1. **"No legal actions available"**
   - This can happen at game end - check if episode terminated correctly
   - In smoke-test mode, this will raise an error for debugging

2. **"Non-finite reward detected"**
   - Check reward calculation in environment
   - Verify score computation is correct
   - In smoke-test mode, this will stop training immediately

3. **Training stops early**
   - Check `max_episodes` limit
   - Verify `max_steps_per_episode` isn't too low
   - Check for exceptions in logs

4. **Config file not found**
   - Use absolute path or relative to project root
   - Ensure file extension is `.yaml`, `.yml`, or `.json`

### Debug Mode

For maximum verbosity, use:
```bash
python training/trainer.py --mode smoke --verbosity 2 --log-action-details
```

This will log every step in the first few episodes, making it easy to trace issues.

## File Structure

```
training/
├── __init__.py
├── config.py              # Configuration system
├── seeds.py               # Seed initialization utilities
├── trainer.py             # Main training script
├── config_smoke.yaml      # Example smoke-test config
├── config_full.yaml       # Example full training config
├── config_smoke.json      # Example smoke-test config (JSON)
└── README.md              # This file
```

## Integration with Stable-Baselines3

The training system uses Stable-Baselines3's MaskablePPO with:
- Action masking to ensure only legal moves are selected
- Custom callback for episode limits and logging
- TensorBoard integration for monitoring
- Checkpoint saving for model persistence

See [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/) for more details on the PPO algorithm and hyperparameters.

