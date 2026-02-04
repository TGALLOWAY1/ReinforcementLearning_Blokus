"""Benchmark Stage 2 (MCTS baselines) vs Stage 3 (checkpoint-only league) rollout throughput."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from rl.train import TrainConfig, load_config, _make_vec_env, _save_checkpoint
from league.pdl import LeagueManager, CheckpointOpponentSampler

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

try:
    import pynvml
except Exception:  # pragma: no cover
    pynvml = None


def _resolve_device(requested: str) -> str:
    if requested is None:
        requested = "auto"
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested


def _make_model(env, config: TrainConfig, device: str) -> MaskablePPO:
    resolved = _resolve_device(device)
    return MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=min(config.n_steps, 256),
        batch_size=min(config.batch_size, 128),
        gamma=config.gamma,
        verbose=0,
        device=resolved,
    )


def _rollout_steps(env, model: MaskablePPO, steps: int) -> Dict[str, Any]:
    obs = env.reset()
    num_envs = getattr(env, "num_envs", 1)

    step_count = 0
    predict_times = []
    episode_lengths: List[float] = []
    start = time.time()
    while step_count < steps:
        masks = get_action_masks(env)
        t0 = time.perf_counter()
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        t1 = time.perf_counter()
        predict_times.append(t1 - t0)

        obs, rewards, dones, infos = env.step(action)
        if isinstance(infos, (list, tuple)):
            for info in infos:
                if isinstance(info, dict) and "episode" in info:
                    episode_lengths.append(info["episode"].get("l", 0))
        elif isinstance(infos, dict) and "episode" in infos:
            episode_lengths.append(infos["episode"].get("l", 0))
        if isinstance(dones, (list, np.ndarray)):
            if np.any(dones):
                obs = env.reset()
        elif dones:
            obs = env.reset()

        step_count += num_envs

    elapsed = time.time() - start
    steps_per_sec = step_count / max(elapsed, 1e-6)
    return {
        "steps": step_count,
        "elapsed_sec": elapsed,
        "steps_per_sec": steps_per_sec,
        "step_time_ms_mean": (elapsed / max(step_count, 1)) * 1000.0,
        "predict_ms_mean": float(np.mean(predict_times) * 1000.0),
        "predict_ms_p95": float(np.percentile(predict_times, 95) * 1000.0),
        "episode_len_mean": float(np.mean(episode_lengths)) if episode_lengths else None,
    }


def _cpu_metrics(start_cpu, end_cpu, elapsed: float) -> Dict[str, Any]:
    if psutil is None:
        return {"cpu_util_pct": None}
    try:
        cpu_time = (end_cpu.user + end_cpu.system) - (start_cpu.user + start_cpu.system)
        cpu_cores = psutil.cpu_count(logical=True) or 1
        cpu_util = 100.0 * cpu_time / max(elapsed, 1e-6) / cpu_cores
        return {"cpu_util_pct": cpu_util}
    except Exception:
        return {"cpu_util_pct": None}


def _accelerator_metrics() -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    metrics = {
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "accelerator": "cpu",
        "gpu_device": None,
        "gpu_util_pct": None,
        "gpu_mem_allocated_mb": None,
        "gpu_mem_reserved_mb": None,
        "mps_mem_allocated_mb": None,
    }
    if cuda_available:
        metrics["accelerator"] = "cuda"
        metrics["gpu_device"] = torch.cuda.get_device_name(0)
        metrics["gpu_mem_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        metrics["gpu_mem_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 ** 2)
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics["gpu_util_pct"] = float(util.gpu)
            except Exception:
                metrics["gpu_util_pct"] = None
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
    elif mps_available:
        metrics["accelerator"] = "mps"
        if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            try:
                metrics["mps_mem_allocated_mb"] = torch.mps.current_allocated_memory() / (1024 ** 2)
            except Exception:
                metrics["mps_mem_allocated_mb"] = None
    return metrics


def _prepare_stage3_sampler(config: TrainConfig, tmp_dir: Path, device: str) -> CheckpointOpponentSampler:
    seed_config = copy.deepcopy(config)
    seed_config.training_stage = 2
    seed_config.opponents = ["random", "random", "random"]
    seed_env = _make_vec_env(seed_config)
    seed_model = _make_model(seed_env, seed_config, device=device)
    ckpt_path = tmp_dir / "checkpoint_0.zip"
    _save_checkpoint(seed_model, ckpt_path, step=0, config=seed_config, extra_metadata={"source": "benchmark_seed"})
    seed_env.close()

    config.stage3_league.league_dir = str(tmp_dir)
    config.checkpoint_dir = str(tmp_dir)

    league_manager = LeagueManager(config.stage3_league, config.checkpoint_dir)
    league_manager.register_checkpoint(ckpt_path, step=0, metadata={"source": "benchmark_seed"})
    league_manager.write_state(0, config.total_timesteps)

    sampler = CheckpointOpponentSampler(
        registry_path=league_manager.registry_path,
        state_path=league_manager.state_path,
        sampling_config=config.stage3_league.sampling,
        window_schedule=config.stage3_league.window_schedule,
        lru_cache_size=config.stage3_league.lru_cache_size,
        opponent_device=config.stage3_league.opponent_device,
        seed=config.seed,
    )
    sampler.refresh()
    return sampler


def run_benchmark(
    stage2_config: TrainConfig,
    stage3_config: TrainConfig,
    steps: int,
    device: str,
    stage2_vecenv: Optional[str] = None,
    stage3_vecenv: Optional[str] = None,
    stage3_opponent_device: Optional[str] = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"steps": steps}

    # Stage 2 baseline
    if psutil:
        proc = psutil.Process(os.getpid())
        cpu_start = proc.cpu_times()
    else:
        cpu_start = None

    if stage2_vecenv:
        stage2_config.vec_env_type = stage2_vecenv
    env_stage2 = _make_vec_env(stage2_config)
    model_stage2 = _make_model(env_stage2, stage2_config, device=device)
    stage2_metrics = _rollout_steps(env_stage2, model_stage2, steps)
    env_stage2.close()

    if psutil and cpu_start is not None:
        stage2_metrics.update(_cpu_metrics(cpu_start, proc.cpu_times(), stage2_metrics["elapsed_sec"]))

    stage2_metrics.update(_accelerator_metrics())
    results["stage2"] = stage2_metrics

    # Stage 3 self-play league
    tmp_dir = Path("/tmp/blokus_stage3_bench")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if stage3_vecenv:
        stage3_config.vec_env_type = stage3_vecenv
    if stage3_opponent_device:
        stage3_config.stage3_league.opponent_device = stage3_opponent_device
    sampler = _prepare_stage3_sampler(stage3_config, tmp_dir, device=device)
    if psutil:
        proc = psutil.Process(os.getpid())
        cpu_start = proc.cpu_times()
    else:
        cpu_start = None

    env_stage3 = _make_vec_env(stage3_config, opponent_sampler=sampler)
    model_stage3 = _make_model(env_stage3, stage3_config, device=device)
    stage3_metrics = _rollout_steps(env_stage3, model_stage3, steps)
    env_stage3.close()

    if psutil and cpu_start is not None:
        stage3_metrics.update(_cpu_metrics(cpu_start, proc.cpu_times(), stage3_metrics["elapsed_sec"]))

    stage3_metrics.update(_accelerator_metrics())
    results["stage3"] = stage3_metrics

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Stage 2 vs Stage 3 rollout throughput")
    parser.add_argument("--stage2-config", default="configs/v1_rl_vs_mcts.yaml")
    parser.add_argument("--stage3-config", default="configs/stage3_selfplay.yaml")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stage2-vecenv", choices=["dummy", "subproc"], default=None)
    parser.add_argument("--stage3-vecenv", choices=["dummy", "subproc"], default=None)
    parser.add_argument("--stage3-opponent-device", choices=["auto", "cpu", "cuda"], default=None)
    args = parser.parse_args()

    from torch.distributions.distribution import Distribution
    Distribution.set_default_validate_args(False)

    stage2_config = load_config(args.stage2_config)
    stage3_config = load_config(args.stage3_config)

    stage2_config.device = _resolve_device(stage2_config.device)
    stage3_config.device = _resolve_device(stage3_config.device)

    results = run_benchmark(
        stage2_config,
        stage3_config,
        args.steps,
        args.device,
        stage2_vecenv=args.stage2_vecenv,
        stage3_vecenv=args.stage3_vecenv,
        stage3_opponent_device=args.stage3_opponent_device,
    )

    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"selfplay_league_bench_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("Benchmark results:")
    print(json.dumps(results, indent=2))
    if "stage2" in results and "stage3" in results:
        s2 = results["stage2"]["steps_per_sec"]
        s3 = results["stage3"]["steps_per_sec"]
        if s2 > 0:
            ratio = s3 / s2
            print(f"Stage 3 / Stage 2 throughput ratio: {ratio:.2f}x")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
