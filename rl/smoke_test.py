"""
Smoke test: tiny training run and Elo updates.
"""

from __future__ import annotations

import tempfile

from rl.train import TrainConfig, train


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = TrainConfig(
            seed=7,
            total_timesteps=2000,
            n_steps=256,
            batch_size=64,
            num_envs=1,
            eval_interval_steps=1000,
            eval_matches=10,
            checkpoint_interval_steps=2000,
            checkpoint_dir=f"{tmp_dir}/checkpoints",
            log_dir=f"{tmp_dir}/logs",
            league_db=f"{tmp_dir}/league.db",
        )
        train(config)


if __name__ == "__main__":
    main()
