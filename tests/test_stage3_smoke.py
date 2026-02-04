"""Stage 3 end-to-end smoke test (short run)."""

import tempfile
from pathlib import Path

from sb3_contrib import MaskablePPO
import torch

from rl.train import TrainConfig, _make_vec_env, _save_checkpoint, train


def _seed_stage2_checkpoints(seed_dir: Path, count: int = 3) -> None:
    config = TrainConfig(
        seed=11,
        training_stage=2,
        num_envs=1,
        vec_env_type="dummy",
        opponents=["random", "random", "random"],
        total_timesteps=1000,
        n_steps=128,
        batch_size=32,
        gamma=0.99,
        checkpoint_dir=str(seed_dir),
    )
    env = _make_vec_env(config)
    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
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
    for idx in range(count):
        ckpt_path = seed_dir / f"checkpoint_{idx}.zip"
        _save_checkpoint(model, ckpt_path, step=idx, config=config)
    env.close()


def test_stage3_smoke():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        seed_dir = tmp_path / "seed"
        stage3_dir = tmp_path / "stage3"
        seed_dir.mkdir(parents=True, exist_ok=True)
        stage3_dir.mkdir(parents=True, exist_ok=True)

        _seed_stage2_checkpoints(seed_dir, count=3)

        config = TrainConfig(
            seed=13,
            training_stage=3,
            total_timesteps=64,
            n_steps=32,
            batch_size=16,
            num_envs=1,
            vec_env_type="dummy",
            eval_interval_steps=32,
            eval_matches=1,
            checkpoint_interval_steps=32,
            checkpoint_dir=str(stage3_dir),
            log_dir=str(stage3_dir / "logs"),
            league_db=str(stage3_dir / "league.db"),
        )
        config.device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        config.stage3_league.seed_dir = str(seed_dir)
        config.stage3_league.league_dir = str(stage3_dir)
        config.stage3_league.save_every_steps = 32
        config.stage3_league.min_checkpoints = 3

        train(config)

        registry_path = Path(config.stage3_league.league_dir) / config.stage3_league.registry_filename
        state_path = Path(config.stage3_league.league_dir) / config.stage3_league.state_filename
        assert registry_path.exists(), "Stage 3 registry should be created"
        assert state_path.exists(), "Stage 3 state file should be created"
