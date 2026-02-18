"""Checkpoint integrity test for MaskablePPO self-play pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from rl.train import TrainConfig, _make_vec_env, _save_checkpoint, _load_checkpoint


def _extract_action(action) -> int:
    if isinstance(action, (list, tuple, np.ndarray)):
        return int(np.asarray(action).flatten()[0])
    return int(action)

def _resolve_test_device() -> str:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def test_checkpoint_integrity_roundtrip():
    from torch.distributions.distribution import Distribution
    Distribution.set_default_validate_args(False)

    config = TrainConfig(
        seed=123,
        training_stage=2,
        num_envs=1,
        vec_env_type="dummy",
        opponents=["random", "random", "random"],
        total_timesteps=1000,
        n_steps=128,
        batch_size=32,
        gamma=0.99,
    )
    env = _make_vec_env(config)
    device = _resolve_test_device()
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        verbose=0,
        device=device,
    )

    obs = env.reset()
    masks = get_action_masks(env)
    action_before, _ = model.predict(obs, action_masks=masks, deterministic=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "checkpoint_0.zip"
        _save_checkpoint(model, checkpoint_path, step=0, config=config)
        reloaded = _load_checkpoint(str(checkpoint_path), env)
        action_after, _ = reloaded.predict(obs, action_masks=masks, deterministic=True)

    action_before = _extract_action(action_before)
    action_after = _extract_action(action_after)

    assert action_before == action_after, "Loaded checkpoint should produce same deterministic action"

    if masks is not None:
        mask = np.asarray(masks)
        if mask.ndim == 2:
            mask = mask[0]
        assert mask[action_after], "Reloaded action should be legal under mask"

    env.close()
