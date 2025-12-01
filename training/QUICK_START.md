# Quick Start Guide

## Smoke Test (30 seconds)

Verify everything works before running long training jobs:

```bash
python training/trainer.py --mode smoke
```

**What to look for:**
- ✅ Configuration printed at start
- ✅ "Episode X completed" messages (should see 5 episodes)
- ✅ "Training Summary" at the end
- ✅ No errors or exceptions
- ✅ Model saved to `checkpoints/ppo_blokus`

## Full Training

Once smoke test passes:

```bash
python training/trainer.py --mode full --total-timesteps 1000000
```

## Common Commands

```bash
# Smoke test with custom seed
python training/trainer.py --mode smoke --seed 42

# Smoke test with config file
python training/trainer.py --config training/config_smoke.yaml

# Full training with checkpointing every 100 episodes
python training/trainer.py --mode full --total-timesteps 1000000 --checkpoint-interval 100

# Debug mode (maximum verbosity)
python training/trainer.py --mode smoke --verbosity 2 --log-action-details
```

## Troubleshooting

**Import errors?**
```bash
pip install -r requirements.txt
```

**Config file not found?**
- Use absolute path or path relative to project root
- Ensure file extension is `.yaml`, `.yml`, or `.json`

**Training stops early?**
- Check `--max-episodes` limit
- Verify `--max-steps-per-episode` isn't too low
- Check logs for exceptions

See `training/README.md` for complete documentation.

