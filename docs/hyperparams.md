# Hyperparameter Configuration and Sweeps

## Overview

The hyperparameter configuration system provides a structured way to define, version, and test agent hyperparameters before committing to long training runs.

## Agent Configuration Files

### Location

Agent configuration files are stored in `config/agents/`:

```
config/agents/
├── README.md
├── ppo_agent_v1.yaml              # Base PPO agent config
├── ppo_agent_sweep_lr_high.yaml   # High learning rate variant
├── ppo_agent_sweep_lr_low.yaml    # Low learning rate variant
└── ppo_agent_sweep_gamma_high.yaml # High gamma variant
```

### Configuration Structure

Each agent config file defines:

```yaml
# Agent identification
agent_id: ppo_agent
name: PPO Agent v1
algorithm: MaskablePPO
version: 1
sweep_variant: null  # Optional, for sweep variants

# Core hyperparameters
learning_rate: 3.0e-4
gamma: 0.99
n_steps: 2048
batch_size: 64
n_epochs: 10

# Network architecture
network:
  policy: MlpPolicy
  net_arch:
    - 256
    - 256
  activation_fn: tanh

# PPO-specific parameters
ppo:
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5

# Description
description: "Base PPO agent with standard hyperparameters"
```

### Key Hyperparameters

**Core Parameters:**
- `learning_rate`: Learning rate for optimizer (default: 3e-4)
- `gamma`: Discount factor for future rewards (default: 0.99)
- `n_steps`: Steps to collect before update (default: 2048)
- `batch_size`: Batch size for training (default: 64)
- `n_epochs`: Number of optimization epochs per update (default: 10)

**Network Architecture:**
- `policy`: Policy type (e.g., "MlpPolicy")
- `net_arch`: List of hidden layer sizes (e.g., [256, 256])
- `activation_fn`: Activation function ("tanh", "relu", "elu")

**PPO-Specific:**
- `clip_range`: PPO clipping range (default: 0.2)
- `ent_coef`: Entropy coefficient for exploration (default: 0.01)
- `vf_coef`: Value function loss coefficient (default: 0.5)
- `max_grad_norm`: Maximum gradient norm for clipping (default: 0.5)

## Using Agent Configs

### Single Training Run

Use a specific agent config:

```bash
python training/trainer.py \
  --agent-config config/agents/ppo_agent_v1.yaml \
  --mode full \
  --total-timesteps 1000000
```

### Override Individual Parameters

You can still override individual hyperparameters via CLI:

```bash
python training/trainer.py \
  --agent-config config/agents/ppo_agent_v1.yaml \
  --learning-rate 1e-3 \
  --total-timesteps 1000000
```

CLI arguments take precedence over agent config values.

## Hyperparameter Sweeps

### Quick Sanity Checks

Before running long training jobs, test multiple hyperparameter configurations with short runs:

```bash
# Run sweep across all sweep variants
python training/run_sweep.py config/agents/ppo_agent_sweep_*.yaml --episodes 100

# Run with more episodes
python training/run_sweep.py config/agents/ppo_agent_sweep_*.yaml --episodes 500

# Use full mode instead of smoke
python training/run_sweep.py config/agents/ppo_agent_sweep_*.yaml --episodes 200 --mode full
```

### Sweep Options

- `config_pattern`: Glob pattern for config files (required)
- `--episodes N`: Episodes per config (default: 100)
- `--mode {smoke,full}`: Training mode (default: smoke)
- `--base-config PATH`: Base training config file (optional)

### Example Sweep

```bash
# Test learning rate variants
python training/run_sweep.py \
  config/agents/ppo_agent_sweep_lr_*.yaml \
  --episodes 200 \
  --mode smoke
```

This will:
1. Find all configs matching the pattern
2. Run each config for 200 episodes
3. Create separate TrainingRun records for each
4. Print a summary of results

### Interpreting Sweep Results

After running a sweep:

1. **View in Training History**: Navigate to `http://localhost:5173/training`
2. **Compare Learning Curves**: Look at reward progression charts
3. **Check Stability**: Look for consistent upward trends
4. **Compare Final Metrics**: Check average rewards and win rates

**Good signs:**
- Steady upward trend in rewards
- Low variance between episodes
- Consistent performance across configs

**Warning signs:**
- High variance or unstable learning
- No improvement over episodes
- Negative or zero rewards

## Creating New Configs

### Base Config

Create a new versioned config:

```yaml
# config/agents/ppo_agent_v2.yaml
agent_id: ppo_agent
name: PPO Agent v2
algorithm: MaskablePPO
version: 2

learning_rate: 3.0e-4
gamma: 0.99
# ... other parameters
```

### Sweep Variant

Create a variant for testing:

```yaml
# config/agents/ppo_agent_sweep_lr_high.yaml
agent_id: ppo_agent
name: PPO Agent (LR High)
algorithm: MaskablePPO
version: 1
sweep_variant: lr_high  # Important: marks this as a sweep variant

learning_rate: 1.0e-3  # Different from base
gamma: 0.99
# ... other parameters same as base
```

### Naming Conventions

- **Base configs**: `{agent_id}_v{version}.yaml`
- **Sweep variants**: `{agent_id}_sweep_{variant}.yaml`
- **Descriptive names**: Use clear, descriptive variant names (e.g., `lr_high`, `gamma_low`, `large_net`)

## Training History Integration

### Viewing Configs

In the Training History UI:

1. **List View**: Shows config short name (e.g., `ppo_agent_lr_high`)
2. **Detail View**: Shows full config name, version, and key hyperparameters
3. **Filtering**: Can filter by agent_id (configs are linked via agent_id)

### Config Summary Card

The detail view includes a "Config Summary" card showing:
- Config name and version
- Key hyperparameters (learning rate, gamma, batch size, etc.)
- Network architecture
- Full config path

## Workflow: Calibration and Scaling

### Step 1: Create Sweep Variants

Create config files for hyperparameters you want to test:

```bash
# Copy base config
cp config/agents/ppo_agent_v1.yaml config/agents/ppo_agent_sweep_lr_high.yaml

# Edit to change learning rate
# learning_rate: 1.0e-3
```

### Step 2: Run Short Sweep

Test all variants with short runs:

```bash
python training/run_sweep.py \
  config/agents/ppo_agent_sweep_*.yaml \
  --episodes 100 \
  --mode smoke
```

### Step 3: Analyze Results

1. Open Training History: `http://localhost:5173/training`
2. Compare learning curves
3. Look for:
   - Which configs show promise
   - Which configs are unstable
   - Which configs converge fastest

### Step 4: Select Best Config

Based on sweep results:
- Choose config with best learning curve
- Check for stability and consistency
- Note any hyperparameters that seem optimal

### Step 5: Long Training Run

Run full training with selected config:

```bash
python training/trainer.py \
  --agent-config config/agents/ppo_agent_sweep_lr_high.yaml \
  --mode full \
  --total-timesteps 1000000
```

## Best Practices

### Config Management

1. **Version Control**: Keep configs in version control
2. **Documentation**: Add descriptions to configs
3. **Naming**: Use clear, consistent naming
4. **Incremental Changes**: Test one hyperparameter at a time in sweeps

### Sweep Strategy

1. **Start Small**: Test with 50-100 episodes first
2. **Expand Gradually**: Increase episodes if results look promising
3. **Compare Apples to Apples**: Use same base config for all variants
4. **Document Findings**: Note which configs work best

### Hyperparameter Selection

**Learning Rate:**
- Too high: Unstable training, loss spikes
- Too low: Slow convergence
- Good: Steady improvement, stable loss

**Gamma:**
- Higher (0.99-0.999): Long-term planning
- Lower (0.9-0.95): Short-term focus
- Default 0.99 works well for most cases

**Network Architecture:**
- Larger networks: More capacity, slower training
- Smaller networks: Faster training, less capacity
- Start with [256, 256] and adjust based on performance

## Troubleshooting

### Config Not Found

**Error**: `FileNotFoundError: Agent config file not found`

**Solutions**:
- Check file path is correct
- Ensure file extension is `.yaml`, `.yml`, or `.json`
- Use absolute path if relative path doesn't work

### Invalid Config Format

**Error**: `ValueError: Config file must contain a dictionary/object`

**Solutions**:
- Ensure YAML/JSON is valid
- Check that root level is a dictionary/object
- Verify required fields are present

### Sweep Fails

**Error**: Some configs in sweep fail

**Solutions**:
- Check individual config files are valid
- Verify all configs use same algorithm
- Check for missing dependencies

## Related Documentation

- [Training Configuration](./training/README.md): Training config system
- [Training History](./training-history.md): Viewing training runs
- [Checkpoints](./checkpoints.md): Saving and resuming training

