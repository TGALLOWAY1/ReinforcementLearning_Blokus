"""Scan Stage 3 throughput across vec env modes and num_envs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from torch.distributions.distribution import Distribution
Distribution.set_default_validate_args(False)

from benchmarks.bench_selfplay_league import (
    _prepare_stage3_sampler,
    _make_model,
    _rollout_steps,
    _accelerator_metrics,
    _resolve_device,
)
from rl.train import load_config, _make_vec_env


def _run_once(config_path: str, num_envs: int, vecenv: str, steps: int, opponent_device: str) -> Dict[str, Any]:
    config = load_config(config_path)
    config.device = _resolve_device(config.device)
    config.num_envs = num_envs
    config.vec_env_type = vecenv
    config.stage3_league.vecenv_mode = vecenv
    config.stage3_league.opponent_device = opponent_device

    sampler = _prepare_stage3_sampler(config, Path("/tmp/blokus_stage3_scan"), device=config.device)
    env = _make_vec_env(config, opponent_sampler=sampler)
    model = _make_model(env, config, device=config.device)

    metrics = _rollout_steps(env, model, steps)
    metrics.update(_accelerator_metrics())
    env.close()

    return {
        "num_envs": num_envs,
        "vec_env_type": vecenv,
        "opponent_device": opponent_device,
        "steps": steps,
        "metrics": metrics,
    }


def _default_opponent_device(vecenv: str, resolved_device: str) -> str:
    if vecenv == "subproc":
        return "cpu"
    return resolved_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Stage 3 throughput across env settings")
    parser.add_argument("--config", default="configs/stage3_selfplay_gpu.yaml")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--num-envs", type=str, default="2,4,8")
    parser.add_argument("--vecenvs", type=str, default="dummy,subproc")
    args = parser.parse_args()

    num_envs_list = [int(x.strip()) for x in args.num_envs.split(",") if x.strip()]
    vecenvs = [x.strip() for x in args.vecenvs.split(",") if x.strip()]

    resolved_device = _resolve_device(load_config(args.config).device)
    results: List[Dict[str, Any]] = []
    for vecenv in vecenvs:
        for num_envs in num_envs_list:
            opponent_device = _default_opponent_device(vecenv, resolved_device)
            results.append(_run_once(args.config, num_envs, vecenv, args.steps, opponent_device))

    best = max(results, key=lambda r: r["metrics"]["steps_per_sec"])
    summary = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "config": args.config,
        "results": results,
        "best": best,
    }

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stage3_env_scan_{summary['timestamp']}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
