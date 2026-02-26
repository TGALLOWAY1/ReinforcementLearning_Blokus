# Agent Configuration Files

This directory contains agent hyperparameter configuration files.

## Structure

Each agent config file defines:
- Agent identifier and algorithm
- Hyperparameters (learning rate, gamma, network architecture, etc.)
- Version information

## File Naming

- `ppo_agent_v1.yaml` - Base PPO agent config version 1
- `ppo_agent_v2.yaml` - Updated PPO agent config
- `ppo_agent_sweep_lr_high.yaml` - High learning rate variant for sweeps
- `ppo_agent_sweep_lr_low.yaml` - Low learning rate variant for sweeps

## Usage

```bash
# Use specific agent config
python training/trainer.py --agent-config config/agents/ppo_agent_v1.yaml

# Run hyperparameter sweep
python training/run_sweep.py config/agents/ppo_agent_sweep_*.yaml
```

