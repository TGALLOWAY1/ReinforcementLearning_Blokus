# RL Training Guide

## Smoke Test Mode

Smoke test mode is a quick verification mode designed to validate that the environment and PPO training pipeline work correctly before running full training sessions. It uses reduced episode counts, shorter episode limits, and high verbosity logging to quickly catch issues.

**What it does:**
- Runs a short training session (default: 5 episodes, 10,000 timesteps)
- Enables detailed logging and action-level debugging
- Performs comprehensive sanity checks (NaN/Inf detection, action validation)
- Verifies environment integration with MaskablePPO
- Tests action masking and legal move generation

**Example command:**
```bash
python training/trainer.py --mode smoke
```

You can also customize smoke test parameters:
```bash
python training/trainer.py --mode smoke --max-episodes 3 --total-timesteps 5000
```

## Large Action Space Categorical Quirk

Blokus uses a massive discrete action space of 36,400 possible actions (21 pieces × orientations × 20×20 board positions). When MaskablePPO applies softmax to logits over this large action space, floating-point rounding errors cause the resulting probabilities to sum to approximately 0.99998 instead of exactly 1.0. PyTorch's `Distribution.validate_args=True` enforces a strict Simplex constraint that requires probabilities to sum to exactly 1.0, which causes spurious `ValueError` exceptions even though the distribution is valid for practical purposes. To prevent these false-positive errors, distribution validation is disabled by default via the `disable_distribution_validation` config option. This is safe because SB3-contrib's MaskablePPO is designed to handle these numerical precision issues, and the diagnostic logging still catches real problems like NaNs or negative probabilities.

