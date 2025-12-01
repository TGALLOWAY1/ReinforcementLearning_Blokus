# Quick Start: Agent Configs and Sweeps

## Using an Agent Config

```bash
python training/trainer.py --agent-config config/agents/ppo_agent_v1.yaml
```

## Running a Sweep

```bash
# Quick sweep (100 episodes per config)
python training/run_sweep.py config/agents/ppo_agent_sweep_*.yaml --episodes 100

# Longer sweep (500 episodes)
python training/run_sweep.py config/agents/ppo_agent_sweep_*.yaml --episodes 500
```

## Viewing Results

1. Open Training History: `http://localhost:5173/training`
2. Compare learning curves
3. Check config names in the "Agent / Config" column
4. Click on runs to see detailed hyperparameters

## Creating a New Config

1. Copy an existing config:
   ```bash
   cp config/agents/ppo_agent_v1.yaml config/agents/ppo_agent_sweep_lr_custom.yaml
   ```

2. Edit the file and change hyperparameters

3. Test it:
   ```bash
   python training/trainer.py --agent-config config/agents/ppo_agent_sweep_lr_custom.yaml --mode smoke
   ```

See `docs/hyperparams.md` for complete documentation.

