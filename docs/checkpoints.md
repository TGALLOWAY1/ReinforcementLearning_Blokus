# Checkpointing System Documentation

## Overview

The checkpointing system provides automatic periodic saving of training state, allowing you to:
- Resume training from any saved checkpoint
- Recover from crashes or interruptions
- Analyze model performance at different training stages
- Share trained models with others

## Checkpoint Structure

### Directory Layout

Checkpoints are organized in a structured directory hierarchy:

```
checkpoints/
├── <agent_id>/
│   ├── <run_id>/
│   │   ├── ep000050.zip          # Episode 50 checkpoint
│   │   ├── ep000050_metadata.json
│   │   ├── ep000100.zip          # Episode 100 checkpoint
│   │   ├── ep000100_metadata.json
│   │   └── ...
```

### Checkpoint Contents

Each checkpoint consists of:

1. **Model File** (`.zip`):
   - Model weights (neural network parameters)
   - Optimizer state (learning rate, momentum, etc.)
   - Training step/episode counters
   - Stable-Baselines3 internal state

2. **Metadata File** (`_metadata.json`):
   - Episode number when checkpoint was saved
   - Training run ID
   - Training configuration (hyperparameters)
   - Timestamp
   - Any additional state

### Example Metadata

```json
{
  "episode": 100,
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "config": {
    "mode": "full",
    "total_timesteps": 1000000,
    "learning_rate": 0.0003,
    ...
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "extra_state": {}
}
```

## Configuration

### Checkpoint Interval

Control how often checkpoints are saved:

```bash
# Save checkpoint every 50 episodes (default)
python training/trainer.py --checkpoint-interval-episodes 50

# Save checkpoint every 100 episodes
python training/trainer.py --checkpoint-interval-episodes 100

# Disable periodic checkpointing (only save at end)
python training/trainer.py --checkpoint-interval-episodes 0
```

Or in config file:

```yaml
checkpoint_interval_episodes: 50  # Save every 50 episodes
```

### Keeping Checkpoints

Control how many recent checkpoints to keep:

```bash
# Keep last 3 checkpoints (default)
python training/trainer.py --keep-last-n-checkpoints 3

# Keep last 10 checkpoints
python training/trainer.py --keep-last-n-checkpoints 10
```

Older checkpoints are automatically deleted to save disk space.

### Checkpoint Directory

Specify where checkpoints are saved:

```bash
python training/trainer.py --checkpoint-dir checkpoints
```

## Resuming Training

### Basic Resume

To resume training from a checkpoint:

```bash
python training/trainer.py --resume-from-checkpoint checkpoints/ppo_agent/run123/ep000100.zip
```

This will:
1. Load the model weights and optimizer state
2. Restore training configuration
3. Continue from the saved episode number
4. Create a new TrainingRun record (referencing the original checkpoint)

### Resume with Overrides

You can override configuration when resuming:

```bash
python training/trainer.py \
  --resume-from-checkpoint checkpoints/ppo_agent/run123/ep000100.zip \
  --total-timesteps 2000000 \
  --learning-rate 0.0001
```

### Finding Checkpoints

List all checkpoints for a run:

```python
from training.checkpoints import list_checkpoints

checkpoints = list_checkpoints(
    checkpoint_dir="checkpoints",
    run_id="550e8400-e29b-41d4-a716-446655440000",
    agent_id="ppo_agent"
)

for checkpoint_path in checkpoints:
    print(checkpoint_path)
```

## Training Run Integration

### Automatic Logging

When checkpoints are saved during training:
- Checkpoint path is automatically logged to MongoDB
- Episode number is recorded
- Checkpoint appears in Training History UI

### Viewing Checkpoints

1. Navigate to Training History: `http://localhost:5173/training`
2. Click on a training run
3. Scroll to "Checkpoints" section
4. See all saved checkpoints with resume commands

### Resume from UI

The Training Run detail page shows:
- All checkpoints with episode numbers
- File paths
- Copyable resume commands

Click "Copy" to copy the resume command to clipboard.

## Usage Examples

### Example 1: Basic Training with Checkpointing

```bash
# Start training with automatic checkpointing every 50 episodes
python training/trainer.py \
  --mode full \
  --total-timesteps 1000000 \
  --checkpoint-interval-episodes 50 \
  --keep-last-n-checkpoints 3
```

### Example 2: Resume After Crash

```bash
# Training crashed at episode 150
# Resume from last checkpoint (episode 100)
python training/trainer.py \
  --resume-from-checkpoint checkpoints/ppo_agent/run123/ep000100.zip \
  --total-timesteps 1000000
```

### Example 3: Fine-tuning with Different Learning Rate

```bash
# Resume from checkpoint but change learning rate
python training/trainer.py \
  --resume-from-checkpoint checkpoints/ppo_agent/run123/ep000200.zip \
  --learning-rate 0.0001 \
  --total-timesteps 2000000
```

### Example 4: Config File

```yaml
# config_resume.yaml
mode: full
total_timesteps: 2000000
checkpoint_interval_episodes: 50
keep_last_n_checkpoints: 5
resume_from_checkpoint: checkpoints/ppo_agent/run123/ep000100.zip
learning_rate: 0.0001
```

```bash
python training/trainer.py --config config_resume.yaml
```

## API Access

### List Checkpoints for a Run

```python
import requests

run_id = "550e8400-e29b-41d4-a716-446655440000"
response = requests.get(f"http://localhost:8000/api/training-runs/{run_id}")
run = response.json()

for checkpoint in run.get("checkpoint_paths", []):
    print(f"Episode {checkpoint['episode']}: {checkpoint['path']}")
```

## Limitations and Notes

### What Gets Restored

✅ **Restored:**
- Model weights (neural network parameters)
- Optimizer state (learning rate, momentum, etc.)
- Training step/episode counters
- Training configuration

❌ **Not Restored:**
- Replay buffer (if applicable) - training continues with fresh buffer
- Random number generator state - new random state after resume
- Environment state - environment is reset

### Best Practices

1. **Regular Checkpoints**: Save checkpoints frequently enough to minimize lost progress
   - For long training: every 50-100 episodes
   - For short training: every 10-20 episodes

2. **Keep Multiple Checkpoints**: Keep at least 3-5 recent checkpoints
   - Allows recovery from corrupted checkpoints
   - Enables comparison across training stages

3. **Monitor Disk Space**: Checkpoints can be large (10-100MB each)
   - Use `keep_last_n_checkpoints` to limit storage
   - Archive old checkpoints if needed

4. **Verify Checkpoints**: Test loading checkpoints before deleting old ones
   ```python
   from training.checkpoints import load_checkpoint
   model, config, _ = load_checkpoint("path/to/checkpoint.zip", env=env)
   ```

5. **Document Checkpoints**: Note which checkpoints correspond to important milestones
   - Best performance
   - Training milestones
   - Before/after hyperparameter changes

## Troubleshooting

### Checkpoint Not Found

**Error**: `FileNotFoundError: Checkpoint not found: ...`

**Solutions**:
- Verify checkpoint path is correct
- Check that checkpoint file exists
- Ensure path is absolute or relative to current directory

### Cannot Load Checkpoint

**Error**: `RuntimeError: Failed to load model from checkpoint`

**Solutions**:
- Verify checkpoint file is not corrupted
- Ensure Stable-Baselines3 version matches
- Check that environment is compatible

### Out of Disk Space

**Error**: Checkpoint saving fails due to disk space

**Solutions**:
- Reduce `keep_last_n_checkpoints`
- Increase `checkpoint_interval_episodes` to save less frequently
- Clean up old checkpoints manually
- Use a different `checkpoint_dir` with more space

### Resume Episode Count Wrong

**Issue**: Episode count doesn't match checkpoint episode

**Solutions**:
- Check metadata file for correct episode number
- Verify checkpoint filename matches episode number
- Manually set episode count if needed

## Related Documentation

- [Training Configuration](./training/README.md): Training config system
- [Training History](./training-history.md): Viewing training runs and checkpoints
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/): Model save/load details

