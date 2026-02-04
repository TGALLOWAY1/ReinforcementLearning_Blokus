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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from agents.registry import RLPolicyAgent, build_baseline_agent, AgentSpec
from league.league import League, build_league_agents
from rl.env_wrapper import SelfPlayBlokusEnv

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    seed: int = 0
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


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = TrainConfig()
    for key, value in raw.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def make_env(config: TrainConfig, seed_offset: int = 0):
    def _init():
        env = SelfPlayBlokusEnv(
            max_episode_steps=config.max_episode_steps,
            opponents=_build_opponents(config, config.seed + seed_offset),
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


def _make_vec_env(config: TrainConfig):
    if config.num_envs <= 1:
        return DummyVecEnv([make_env(config, seed_offset=0)])
    env_fns = [make_env(config, seed_offset=i) for i in range(config.num_envs)]
    if config.vec_env_type == "subproc":
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def _save_checkpoint(model: MaskablePPO, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    rng_state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    with open(path.with_suffix(".rng.pkl"), "wb") as handle:
        pickle.dump(rng_state, handle)


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


def train(config: TrainConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Reduce noise from SB3 and other libs; we log progress ourselves
    logging.getLogger("stable_baselines3").setLevel(logging.WARNING)
    logging.getLogger("sb3_contrib").setLevel(logging.WARNING)
    set_seeds(config.seed)

    # Disable PyTorch distribution validation for large action space (Simplex rounding errors)
    from torch.distributions.distribution import Distribution
    Distribution.set_default_validate_args(False)

    env = _make_vec_env(config)
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

    logger.info(
        "Training: %s steps | opponents %s | checkpoint_dir %s | league_db %s",
        config.total_timesteps,
        config.opponents,
        config.checkpoint_dir,
        config.league_db,
    )

    start_time = time.time()
    steps_done = 0
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
        eval_results = evaluate_and_update_league(model, league, config, steps_done)

        checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint_{steps_done}.zip"
        if steps_done % config.checkpoint_interval_steps == 0 or steps_done >= config.total_timesteps:
            _save_checkpoint(model, checkpoint_path)
            logger.info("Checkpoint saved: %s", checkpoint_path)

        metrics = {
            "step": steps_done,
            "elapsed_sec": time.time() - start_time,
            "steps_per_sec": steps_per_sec,
            "games_per_sec_est": games_per_sec,
            "games_est": games_est,
            "eval": eval_results,
        }
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")

        progress_pct = 100.0 * steps_done / config.total_timesteps
        games_str = f"~{games_est} train games" if games_est is not None else ""
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
