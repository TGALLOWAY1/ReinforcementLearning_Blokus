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
from league.league import League
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
        env = ActionMasker(env, lambda e: e.get_action_mask())
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


def _load_checkpoint(model_path: str, env):
    model = MaskablePPO.load(model_path, env=env)
    rng_path = Path(model_path).with_suffix(".rng.pkl")
    if rng_path.exists():
        with open(rng_path, "rb") as handle:
            rng_state = pickle.load(handle)
        random.setstate(rng_state["random"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
    return model


def _estimate_games_per_sec(env) -> float:
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
    if not episode_lengths:
        return 0.0
    avg_length = float(np.mean(episode_lengths))
    steps_per_sec = getattr(env, "steps_per_sec", 0.0)
    return steps_per_sec / max(avg_length, 1.0)


def evaluate_and_update_league(
    model: MaskablePPO,
    league: League,
    config: TrainConfig,
    step: int,
) -> Dict[str, Any]:
    baseline_types = list({*config.opponents, "mcts"})
    agents = {}
    for agent_type in baseline_types:
        name = f"{agent_type}_baseline"
        agents[name] = build_baseline_agent(agent_type, seed=config.seed)
        league.register_agent(AgentSpec(name=name, agent_type=agent_type))
    agents["rl_checkpoint"] = RLPolicyAgent(model)
    league.register_agent(AgentSpec(name="rl_checkpoint", agent_type="rl", version=str(step)))

    results = {"wins": 0, "losses": 0, "draws": 0}
    for match_id in range(config.eval_matches):
        opponent_type = random.choice(baseline_types)
        opponent_name = f"{opponent_type}_baseline"
        rl_name = "rl_checkpoint"
        seed = config.seed + step + match_id
        match = league.play_match(
            rl_name,
            opponent_name,
            agents[rl_name],
            agents[opponent_name],
            seed=seed,
        )
        if match.winner == rl_name:
            results["wins"] += 1
            result = 1.0
        elif match.winner is None:
            results["draws"] += 1
            result = 0.5
        else:
            results["losses"] += 1
            result = 0.0
        league.update_elo(rl_name, opponent_name, result, seed)

    return results


def train(config: TrainConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_seeds(config.seed)

    env = _make_vec_env(config)
    tensorboard_log = config.log_dir if config.tensorboard else None

    if config.resume_path:
        model = _load_checkpoint(config.resume_path, env)
    else:
        model = MaskablePPO(
            "CnnPolicy",
            env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            verbose=1,
            tensorboard_log=tensorboard_log,
        )

    league = League(db_path=config.league_db)
    log_path = Path(config.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    metrics_path = log_path / "train_metrics.jsonl"

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

        eval_results = evaluate_and_update_league(model, league, config, steps_done)
        checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint_{steps_done}.zip"
        if steps_done % config.checkpoint_interval_steps == 0 or steps_done >= config.total_timesteps:
            _save_checkpoint(model, checkpoint_path)

        metrics = {
            "step": steps_done,
            "elapsed_sec": time.time() - start_time,
            "steps_per_sec": steps_per_sec,
            "games_per_sec_est": _estimate_games_per_sec(env),
            "eval": eval_results,
        }
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")

        logger.info(
            "Step %s | %s steps/s | eval W/L/D: %s/%s/%s",
            steps_done,
            f"{steps_per_sec:.1f}",
            eval_results["wins"],
            eval_results["losses"],
            eval_results["draws"],
        )

    league.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Blokus self-play agent (MaskablePPO)")
    parser.add_argument("--config", type=str, default="configs/overnight.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config)
