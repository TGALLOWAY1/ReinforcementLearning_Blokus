"""
Self-play training pipeline for Blokus using MaskablePPO.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from agents.registry import RLPolicyAgent, build_baseline_agent, AgentSpec, AgentProtocol
from league.league import League, build_league_agents
from league.pdl import Stage3LeagueConfig, LeagueManager, CheckpointOpponentSampler, LeagueCheckpoint
from rl.env_wrapper import SelfPlayBlokusEnv

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    seed: int = 0
    training_stage: int = 2
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    gamma: float = 0.99
    num_envs: int = 1
    vec_env_type: str = "dummy"
    max_episode_steps: int = 1000
    opponents: List[str] = field(default_factory=lambda: ["heuristic", "random", "random"])
    eval_interval_steps: int = 20_000
    eval_matches: int = 10
    checkpoint_interval_steps: int = 50_000
    checkpoint_dir: str = "checkpoints/rl"
    log_dir: str = "logs/rl"
    league_db: str = "league.db"
    tensorboard: bool = False
    resume_path: Optional[str] = None
    stage3_league: Stage3LeagueConfig = field(default_factory=Stage3LeagueConfig)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = TrainConfig()
    for key, value in raw.items():
        if key == "stage3_league" and isinstance(value, dict):
            config.stage3_league = Stage3LeagueConfig.from_dict(value)
        elif hasattr(config, key):
            setattr(config, key, value)
    return config


def _config_to_dict(config: TrainConfig) -> Dict[str, Any]:
    data = asdict(config)
    return data


def make_env(config: TrainConfig, seed_offset: int = 0, opponent_sampler: Optional[CheckpointOpponentSampler] = None):
    def _init():
        if config.training_stage == 3:
            if opponent_sampler is None:
                raise ValueError("Stage 3 requires a checkpoint opponent sampler")
            opponents = None
            sampler = opponent_sampler
        else:
            opponents = _build_opponents(config, config.seed + seed_offset)
            sampler = None
        env = SelfPlayBlokusEnv(
            max_episode_steps=config.max_episode_steps,
            opponents=opponents,
            opponent_sampler=sampler,
            seed=config.seed + seed_offset,
        )
        env = Monitor(env)
        env = ActionMasker(env, lambda e: e.get_wrapper_attr("get_action_mask")())
        return env

    return _init


def _build_opponents(config: TrainConfig, seed: int):
    opponent_agents = {}
    player_names = ["player_1", "player_2", "player_3"]
    for name, agent_type in zip(player_names, config.opponents):
        opponent_agents[name] = build_baseline_agent(agent_type, seed=seed)
    return opponent_agents


def _make_vec_env(config: TrainConfig, opponent_sampler: Optional[CheckpointOpponentSampler] = None):
    if config.num_envs <= 1:
        return DummyVecEnv([make_env(config, seed_offset=0, opponent_sampler=opponent_sampler)])
    env_fns = [make_env(config, seed_offset=i, opponent_sampler=opponent_sampler) for i in range(config.num_envs)]
    if config.vec_env_type == "subproc":
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def _save_checkpoint(
    model: MaskablePPO,
    path: Path,
    step: int,
    config: TrainConfig,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    rng_state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    with open(path.with_suffix(".rng.pkl"), "wb") as handle:
        pickle.dump(rng_state, handle)

    metadata = {
        "step": int(step),
        "timestamp": datetime.utcnow().isoformat(),
        "training_stage": int(config.training_stage),
        "total_timesteps": int(config.total_timesteps),
        "checkpoint_path": str(path),
        "config": _config_to_dict(config),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    with open(path.with_suffix(".meta.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)


def _load_checkpoint(model_path: str, env, tensorboard_log: Optional[str] = None):
    model = MaskablePPO.load(model_path, env=env)
    if tensorboard_log:
        model.tensorboard_log = tensorboard_log
    rng_path = Path(model_path).with_suffix(".rng.pkl")
    if rng_path.exists():
        with open(rng_path, "rb") as handle:
            rng_state = pickle.load(handle)
        random.setstate(rng_state["random"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
    return model


def _load_checkpoint_metadata(model_path: str) -> Dict[str, Any]:
    meta_path = Path(model_path).with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _infer_step_from_checkpoint_path(model_path: str) -> Optional[int]:
    name = Path(model_path).stem
    if name.startswith("checkpoint_") and name.replace("checkpoint_", "").isdigit():
        return int(name.replace("checkpoint_", ""))
    if name.startswith("ep") and name.replace("ep", "").isdigit():
        return int(name.replace("ep", ""))
    meta = _load_checkpoint_metadata(model_path)
    if "step" in meta:
        try:
            return int(meta["step"])
        except Exception:
            return None
    return None


def _get_episode_lengths(env) -> List[float]:
    """Collect episode lengths from Monitor wrappers in (possibly vectorized) env."""
    episode_lengths = []
    if hasattr(env, "envs"):
        for sub_env in env.envs:
            monitor = sub_env
            while hasattr(monitor, "env"):
                if hasattr(monitor, "episode_lengths"):
                    episode_lengths.extend(monitor.episode_lengths)
                    break
                monitor = monitor.env
    elif hasattr(env, "episode_lengths"):
        episode_lengths.extend(env.episode_lengths)
    return episode_lengths


def _estimate_games_per_sec(env) -> float:
    episode_lengths = _get_episode_lengths(env)
    if not episode_lengths:
        return 0.0
    avg_length = float(np.mean(episode_lengths))
    steps_per_sec = getattr(env, "steps_per_sec", 0.0)
    return steps_per_sec / max(avg_length, 1.0)


# Typical RL steps per completed game (one step = one RL move). Used when SubprocVecEnv
# doesn't expose episode_lengths from workers.
ESTIMATED_STEPS_PER_GAME = 250


def _estimate_games_played(env, steps_done: int) -> Optional[int]:
    """Estimate total training games: from monitor episode counts, or steps/avg_length, or steps/ESTIMATED_STEPS_PER_GAME."""
    lengths = _get_episode_lengths(env)
    if lengths:
        return len(lengths)
    avg = getattr(env, "_last_avg_episode_length", None)
    if avg and avg > 0:
        return int(steps_done / avg)
    # SubprocVecEnv: workers don't share episode_lengths, so fall back to steps-based estimate
    return int(steps_done / ESTIMATED_STEPS_PER_GAME)


def _select_eval_pool(
    entries: List[LeagueCheckpoint],
    size: int,
    strategy: str = "old_mid_recent",
) -> List[LeagueCheckpoint]:
    if len(entries) < size:
        raise ValueError(f"Need at least {size} checkpoints for eval pool, got {len(entries)}")
    entries = sorted(entries, key=lambda e: e.step)
    if strategy == "old_mid_recent":
        if size == 1:
            return [entries[-1]]
        indices = np.linspace(0, len(entries) - 1, size).round().astype(int)
        return [entries[i] for i in indices]
    if strategy == "recent":
        return entries[-size:]
    rng = random.Random(0)
    return rng.sample(entries, size)


def evaluate_and_update_league(
    model: MaskablePPO,
    league: League,
    config: TrainConfig,
    step: int,
) -> Dict[str, Any]:
    # 4-player league: RL + 3 opponents (e.g. mcts, random, random). Opponents must have length 3.
    if len(config.opponents) != 3:
        raise ValueError(
            "League evaluation expects exactly 3 opponents for 4-player matches "
            "(e.g. opponents: [mcts, random, random])"
        )
    agents, specs, ordered_4p_names = build_league_agents(
        config.opponents, config.seed, model, "rl_checkpoint"
    )
    if len(ordered_4p_names) != 4:
        raise ValueError("build_league_agents must return 4 names for 4-player league")
    for spec in specs:
        league.register_agent(spec)
    ordered_agents = [agents[n] for n in ordered_4p_names]

    results = {"wins": 0, "losses": 0, "draws": 0}
    rl_name = "rl_checkpoint"
    for match_id in range(config.eval_matches):
        seed = config.seed + step + match_id
        # Shuffle player order so RL is not always player_0 (fairness)
        indices = list(range(4))
        random.shuffle(indices)
        names = [ordered_4p_names[i] for i in indices]
        ags = [ordered_agents[i] for i in indices]
        match = league.play_match(names, ags, seed=seed, max_moves=config.max_episode_steps)
        if match.winner == rl_name:
            results["wins"] += 1
        elif match.winner is None:
            results["draws"] += 1
        else:
            results["losses"] += 1
        league.update_elo_after_4p_match(match, seed)

    rl_agent_id = league.db.get_agent_id(rl_name)
    current_elo = league.db.get_rating(rl_agent_id) if rl_agent_id else 1200.0
    results["elo"] = current_elo

    return results


def evaluate_and_update_league_stage3(
    model: MaskablePPO,
    league: League,
    config: TrainConfig,
    league_manager: LeagueManager,
    step: int,
    eval_entries: Optional[List[LeagueCheckpoint]] = None,
) -> Dict[str, Any]:
    if eval_entries is not None:
        entries = list(eval_entries)
    else:
        entries = league_manager.sample_opponents(num_opponents=3, step=step, seed=config.seed + step)
    if len(entries) != 3:
        raise ValueError("Stage 3 evaluation requires exactly 3 checkpoint opponents")

    agents: Dict[str, AgentProtocol] = {}
    specs: List[AgentSpec] = []

    rl_name = "rl_checkpoint"
    agents[rl_name] = RLPolicyAgent(model)
    specs.append(AgentSpec(name=rl_name, agent_type="rl", version=str(step)))

    opponent_agents = league_manager.build_opponent_agents(entries)
    opponent_names: List[str] = []
    for idx, (entry, opponent) in enumerate(zip(entries, opponent_agents)):
        name = f"ckpt_{entry.step}_{idx}"
        opponent_names.append(name)
        agents[name] = opponent
        specs.append(
            AgentSpec(name=name, agent_type="checkpoint", version=str(entry.step), checkpoint_path=entry.path)
        )

    ordered_4p_names = [rl_name] + opponent_names
    ordered_agents = [agents[n] for n in ordered_4p_names]

    for spec in specs:
        league.register_agent(spec)

    results = {"wins": 0, "losses": 0, "draws": 0}
    for match_id in range(config.eval_matches):
        seed = config.seed + step + match_id
        indices = list(range(4))
        random.shuffle(indices)
        names = [ordered_4p_names[i] for i in indices]
        ags = [ordered_agents[i] for i in indices]
        match = league.play_match(names, ags, seed=seed, max_moves=config.max_episode_steps)
        if match.winner == rl_name:
            results["wins"] += 1
        elif match.winner is None:
            results["draws"] += 1
        else:
            results["losses"] += 1
        league.update_elo_after_4p_match(match, seed)

    rl_agent_id = league.db.get_agent_id(rl_name)
    current_elo = league.db.get_rating(rl_agent_id) if rl_agent_id else 1200.0
    results["elo"] = current_elo

    return results


def train(config: TrainConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Reduce noise from SB3 and other libs; we log progress ourselves
    logging.getLogger("stable_baselines3").setLevel(logging.WARNING)
    logging.getLogger("sb3_contrib").setLevel(logging.WARNING)
    set_seeds(config.seed)

    # Disable PyTorch distribution validation for large action space (Simplex rounding errors)
    from torch.distributions.distribution import Distribution
    Distribution.set_default_validate_args(False)

    resume_step = 0
    if config.resume_path:
        resume_metadata = _load_checkpoint_metadata(config.resume_path)
        resume_step = resume_metadata.get("step", 0) or _infer_step_from_checkpoint_path(config.resume_path) or 0
        if resume_step:
            logger.info("Resume metadata step: %s", resume_step)

    league_manager: Optional[LeagueManager] = None
    opponent_sampler: Optional[CheckpointOpponentSampler] = None
    if config.training_stage == 3:
        if config.stage3_league.vecenv_mode:
            config.vec_env_type = config.stage3_league.vecenv_mode
        if config.stage3_league.opponent_device == "auto" and config.vec_env_type == "subproc":
            config.stage3_league.opponent_device = "cpu"
            logger.info("Stage 3 opponent_device auto-resolved to cpu for SubprocVecEnv")
        if config.stage3_league.window_schedule.schedule_steps <= 0:
            config.stage3_league.window_schedule.schedule_steps = config.total_timesteps
        league_manager = LeagueManager(config.stage3_league, config.checkpoint_dir)
        if config.stage3_league.discover_on_start:
            seed_dir = config.stage3_league.resolve_seed_dir()
            extra_dirs = [seed_dir] if seed_dir else None
            league_manager.discover_checkpoints(extra_dirs=extra_dirs)
        if config.resume_path and config.stage3_league.strict_resume:
            rng_path = Path(config.resume_path).with_suffix(".rng.pkl")
            if not rng_path.exists():
                raise RuntimeError(
                    f"Stage 3 strict resume requires RNG state at {rng_path}. "
                    "Provide the matching .rng.pkl or disable strict_resume."
                )
            if resume_step <= 0:
                raise RuntimeError(
                    "Stage 3 strict resume could not infer checkpoint step. "
                    "Ensure the checkpoint filename includes step or .meta.json has 'step'."
                )
        if config.opponents:
            logger.info("Stage 3 ignores config.opponents (checkpoint-only league)")
        if config.stage3_league.save_every_steps and (
            config.stage3_league.save_every_steps % config.checkpoint_interval_steps != 0
        ):
            logger.warning(
                "stage3_league.save_every_steps (%s) is not aligned with checkpoint_interval_steps (%s); "
                "league registration occurs only when checkpoints are saved.",
                config.stage3_league.save_every_steps,
                config.checkpoint_interval_steps,
            )
        if config.resume_path and resume_step:
            if not any(entry.path == config.resume_path for entry in league_manager.registry.entries):
                league_manager.register_checkpoint(
                    checkpoint_path=Path(config.resume_path),
                    step=resume_step,
                    metadata={"source": "resume"},
                )
        if len(league_manager.registry.entries) < config.stage3_league.min_checkpoints:
            raise RuntimeError(
                "Stage 3 requires existing checkpoints. "
                f"Found {len(league_manager.registry.entries)} but need at least "
                f"{config.stage3_league.min_checkpoints}. "
                "Point stage3_league.league_dir to a directory with prior checkpoints "
                "or resume from a saved checkpoint."
            )
        league_manager.write_state(resume_step, config.total_timesteps)
        league_dir = config.stage3_league.resolve_league_dir(config.checkpoint_dir)
        opponent_sampler = CheckpointOpponentSampler(
            registry_path=league_dir / config.stage3_league.registry_filename,
            state_path=league_dir / config.stage3_league.state_filename,
            sampling_config=config.stage3_league.sampling,
            window_schedule=config.stage3_league.window_schedule,
            lru_cache_size=config.stage3_league.lru_cache_size,
            opponent_device=config.stage3_league.opponent_device,
            seed=config.seed,
        )
        opponent_sampler.refresh()

    env = _make_vec_env(config, opponent_sampler=opponent_sampler)
    tensorboard_log = config.log_dir if config.tensorboard else None

    if config.resume_path:
        logger.info("Resuming from checkpoint: %s", config.resume_path)
        model = _load_checkpoint(config.resume_path, env, tensorboard_log=tensorboard_log)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            verbose=0,
            tensorboard_log=tensorboard_log,
        )

    league = League(db_path=config.league_db)
    log_path = Path(config.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    metrics_path = log_path / "train_metrics.jsonl"

    if config.training_stage == 3:
        logger.info(
            "Training Stage 3: %s steps | checkpoint league (%s entries) | league_dir %s | league_db %s",
            config.total_timesteps,
            len(league_manager.registry.entries) if league_manager else 0,
            config.stage3_league.resolve_league_dir(config.checkpoint_dir),
            config.league_db,
        )
        if config.stage3_league.seed_dir:
            logger.info("Stage 3 seed_dir: %s", config.stage3_league.seed_dir)
        logger.info(
            "Stage 3 sampling: recent/mid/old=%.2f/%.2f/%.2f | window_frac=%.2f→%.2f (%s)",
            config.stage3_league.sampling.recent_band_pct,
            config.stage3_league.sampling.mid_band_pct,
            config.stage3_league.sampling.old_band_pct,
            config.stage3_league.window_schedule.start_window_frac,
            config.stage3_league.window_schedule.end_window_frac,
            config.stage3_league.window_schedule.schedule_type,
        )
    else:
        logger.info(
            "Training: %s steps | opponents %s | checkpoint_dir %s | league_db %s",
            config.total_timesteps,
            config.opponents,
            config.checkpoint_dir,
            config.league_db,
        )

    eval_pool_entries: Optional[List[LeagueCheckpoint]] = None
    if config.training_stage == 3 and league_manager is not None:
        eval_pool_entries = _select_eval_pool(
            league_manager.registry.entries,
            size=config.stage3_league.eval_pool_size,
            strategy=config.stage3_league.eval_pool_strategy,
        )
        logger.info(
            "Stage 3 eval pool (%s): %s",
            config.stage3_league.eval_pool_strategy,
            [entry.step for entry in eval_pool_entries],
        )

    start_time = time.time()
    steps_done = int(resume_step) if resume_step else 0
    while steps_done < config.total_timesteps:
        remaining = config.total_timesteps - steps_done
        chunk = min(config.eval_interval_steps, remaining)
        chunk_start = time.time()
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        chunk_time = time.time() - chunk_start
        steps_done += chunk
        steps_per_sec = chunk / max(chunk_time, 1e-6)
        env.steps_per_sec = steps_per_sec

        games_per_sec = _estimate_games_per_sec(env)
        lengths = _get_episode_lengths(env)
        if lengths:
            env._last_avg_episode_length = float(np.mean(lengths))
        games_est = _estimate_games_played(env, steps_done)
        if config.training_stage == 3:
            if league_manager is None:
                raise RuntimeError("Stage 3 requires a LeagueManager")
            eval_results = evaluate_and_update_league_stage3(
                model,
                league,
                config,
                league_manager,
                steps_done,
                eval_entries=eval_pool_entries,
            )
        else:
            eval_results = evaluate_and_update_league(model, league, config, steps_done)

        checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint_{steps_done}.zip"
        if steps_done % config.checkpoint_interval_steps == 0 or steps_done >= config.total_timesteps:
            _save_checkpoint(model, checkpoint_path, steps_done, config, extra_metadata={"eval": eval_results})
            logger.info("Checkpoint saved: %s", checkpoint_path)
            if config.training_stage == 3 and league_manager is not None:
                league_save_interval = config.stage3_league.save_every_steps or config.checkpoint_interval_steps
                if steps_done % league_save_interval == 0 or steps_done >= config.total_timesteps:
                    league_manager.register_checkpoint(
                        checkpoint_path=checkpoint_path,
                        step=steps_done,
                        metadata={"eval": eval_results},
                        elo=eval_results.get("elo"),
                    )

        if config.training_stage == 3 and league_manager is not None:
            league_manager.write_state(steps_done, config.total_timesteps)

        metrics = {
            "step": steps_done,
            "elapsed_sec": time.time() - start_time,
            "steps_per_sec": steps_per_sec,
            "games_per_sec_est": games_per_sec,
            "games_est": games_est,
            "eval": eval_results,
        }
        if config.training_stage == 3 and league_manager is not None:
            metrics["league_size"] = len(league_manager.registry.entries)
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")

        if config.training_stage == 3 and opponent_sampler is not None and config.vec_env_type == "dummy":
            sampling_stats = opponent_sampler.sampling_stats()
            cache_stats = opponent_sampler.cache_stats()
            logger.info(
                "Stage 3 sampling stats: bands=%s pct=%s recent_step_mean=%s cache_hit_rate=%.2f load_time=%.3fs",
                sampling_stats.get("band_counts"),
                sampling_stats.get("band_pct"),
                sampling_stats.get("recent_steps_mean"),
                cache_stats.get("hit_rate", 0.0),
                cache_stats.get("load_time_sec", 0.0),
            )
            opponent_sampler.reset_stats()
            opponent_sampler.reset_cache_stats()

        progress_pct = 100.0 * steps_done / config.total_timesteps
        games_str = f"~{games_est} train games" if games_est is not None else ""
        if config.training_stage == 3 and league_manager is not None:
            logger.info(
                "%.1f%% | %s / %s steps%s | %.1f steps/s | league_size %s | W/L/D %s/%s/%s (%s eval) | Elo %.0f",
                progress_pct,
                steps_done,
                config.total_timesteps,
                f" | {games_str}" if games_str else "",
                steps_per_sec,
                len(league_manager.registry.entries),
                eval_results["wins"],
                eval_results["losses"],
                eval_results["draws"],
                config.eval_matches,
                eval_results.get("elo", 1200.0),
            )
        else:
            logger.info(
                "%.1f%% | %s / %s steps%s | %.1f steps/s | league W/L/D %s/%s/%s (%s eval) | Elo %.0f",
                progress_pct,
                steps_done,
                config.total_timesteps,
                f" | {games_str}" if games_str else "",
                steps_per_sec,
                eval_results["wins"],
                eval_results["losses"],
                eval_results["draws"],
                config.eval_matches,
                eval_results.get("elo", 1200.0),
            )

    elapsed = time.time() - start_time
    logger.info("Training complete: %s steps in %.1f min | final Elo %.0f", steps_done, elapsed / 60.0, eval_results.get("elo", 1200.0))
    league.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Blokus self-play agent (MaskablePPO)")
    parser.add_argument("--config", type=str, default="configs/overnight.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config)
