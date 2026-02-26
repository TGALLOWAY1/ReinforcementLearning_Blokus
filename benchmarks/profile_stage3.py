"""Profile Stage 3 self-play rollout to identify hot paths."""

from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path

from torch.distributions.distribution import Distribution

Distribution.set_default_validate_args(False)

from benchmarks.bench_selfplay_league import (
    _make_model,
    _prepare_stage3_sampler,
    _resolve_device,
    _rollout_steps,
)
from rl.train import _make_vec_env, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Stage 3 rollout")
    parser.add_argument("--config", default="configs/stage3_selfplay_gpu.yaml")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sort", type=str, default="cumtime")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    config = load_config(args.config)
    config.device = _resolve_device(config.device)

    sampler = _prepare_stage3_sampler(config, Path("/tmp/blokus_stage3_profile"), device=config.device)
    env = _make_vec_env(config, opponent_sampler=sampler)
    model = _make_model(env, config, device=config.device)

    prof = cProfile.Profile()
    prof.enable()
    _rollout_steps(env, model, args.steps)
    prof.disable()

    env.close()

    stats = pstats.Stats(prof).sort_stats(args.sort)
    stats.print_stats(args.limit)


if __name__ == "__main__":
    main()
