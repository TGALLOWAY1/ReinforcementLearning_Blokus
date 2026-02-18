"""Lightweight benchmark sanity test (not a strict perf gate)."""

from benchmarks.bench_selfplay_league import run_benchmark
from rl.train import TrainConfig


def test_bench_selfplay_league_smoke():
    stage2_config = TrainConfig(
        seed=1,
        training_stage=2,
        num_envs=1,
        vec_env_type="dummy",
        opponents=["random", "random", "random"],
        total_timesteps=1000,
        n_steps=128,
        batch_size=32,
    )
    stage3_config = TrainConfig(
        seed=1,
        training_stage=3,
        num_envs=1,
        vec_env_type="dummy",
        opponents=[],
        total_timesteps=1000,
        n_steps=128,
        batch_size=32,
    )

    results = run_benchmark(stage2_config, stage3_config, steps=200, device="cpu")

    assert "stage2" in results and "stage3" in results
    assert results["stage2"]["steps_per_sec"] > 0
    assert results["stage3"]["steps_per_sec"] > 0
