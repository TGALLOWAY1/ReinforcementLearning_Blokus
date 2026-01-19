"""
League orchestration: scheduling, matches, and Elo updates.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np

from envs.blokus_v0 import BlokusEnv
from agents.registry import AgentProtocol, AgentSpec, build_baseline_agent, RLPolicyAgent
from league.db import LeagueDB
from league.elo import EloConfig, update_ratings


@dataclass
class MatchResult:
    winner: Optional[str]
    scores: Dict[str, int]
    moves_made: int


class League:
    def __init__(self, db_path: str = "league.db", elo_config: Optional[EloConfig] = None):
        self.db = LeagueDB(db_path)
        self.elo_config = elo_config or EloConfig()

    def register_agent(self, spec: AgentSpec, initial_elo: float = 1200.0) -> int:
        return self.db.add_agent(
            name=spec.name,
            agent_type=spec.agent_type,
            version=spec.version,
            checkpoint_path=spec.checkpoint_path,
            initial_elo=initial_elo,
        )

    def play_match(
        self,
        agent1_name: str,
        agent2_name: str,
        agent1: AgentProtocol,
        agent2: AgentProtocol,
        seed: Optional[int] = None,
        max_moves: int = 1000,
    ) -> MatchResult:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        env = BlokusEnv(max_episode_steps=max_moves)
        env.reset(seed=seed)

        proxy = MatchEnvProxy(env)
        agent_map = {
            "player_0": agent1,
            "player_1": agent2,
            "player_2": agent2,
            "player_3": agent1,
        }
        moves_made = 0
        winner = None

        while env.agents and moves_made < max_moves:
            agent_name = env.agent_selection
            if env.terminations.get(agent_name, False) or env.truncations.get(agent_name, False):
                env.step(None)
                moves_made += 1
                continue
            obs = env.observe(agent_name)
            info = env.infos[agent_name]
            legal_mask = info["legal_action_mask"]
            action = agent_map[agent_name].act(obs, legal_mask, env=proxy)
            env.step(action)
            moves_made += 1
            if all(env.terminations.get(agent, False) or env.truncations.get(agent, False) for agent in env.agents):
                break

        scores = {
            agent1_name: env.game.get_score(env._agent_to_player("player_0")),
            agent2_name: env.game.get_score(env._agent_to_player("player_1")),
        }
        if scores[agent1_name] > scores[agent2_name]:
            winner = agent1_name
        elif scores[agent2_name] > scores[agent1_name]:
            winner = agent2_name
        else:
            winner = None

        return MatchResult(winner=winner, scores=scores, moves_made=moves_made)

    def update_elo(self, agent1_name: str, agent2_name: str, result: float, seed: Optional[int]) -> None:
        agent1_id = self.db.get_agent_id(agent1_name)
        agent2_id = self.db.get_agent_id(agent2_name)
        if agent1_id is None or agent2_id is None:
            raise ValueError("Both agents must be registered before updating Elo.")

        rating1 = self.db.get_rating(agent1_id)
        rating2 = self.db.get_rating(agent2_id)
        new_r1, new_r2 = update_ratings(rating1, rating2, result, self.elo_config)
        self.db.update_rating(agent1_id, new_r1)
        self.db.update_rating(agent2_id, new_r2)
        self.db.record_match(agent1_id, agent2_id, result, seed)

    def leaderboard(self, limit: Optional[int] = None):
        return self.db.leaderboard(limit=limit)

    def close(self):
        self.db.close()


class MatchEnvProxy:
    """
    Provides a minimal wrapper for AgentProtocol adapters during league matches.
    """

    def __init__(self, env: BlokusEnv):
        self.env = env
        self.move_generator = env.move_generator
        self._cache = {}

    def get_legal_moves(self, player_name: str):
        player = self.env._agent_to_player(player_name)
        key = (self.env.step_count, player.value)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        moves = self.move_generator.get_legal_moves(self.env.game.board, player)
        self._cache[key] = moves
        return moves

def build_league_agents(
    baseline_types: List[str],
    seed: Optional[int],
    model=None,
    model_name: Optional[str] = None,
) -> Tuple[Dict[str, AgentProtocol], List[AgentSpec]]:
    agents: Dict[str, AgentProtocol] = {}
    specs: List[AgentSpec] = []

    for agent_type in baseline_types:
        name = f"{agent_type}_baseline"
        agents[name] = build_baseline_agent(agent_type, seed=seed)
        specs.append(AgentSpec(name=name, agent_type=agent_type))

    if model is not None and model_name is not None:
        agents[model_name] = RLPolicyAgent(model)
        specs.append(AgentSpec(name=model_name, agent_type="rl", checkpoint_path=model_name))

    return agents, specs
