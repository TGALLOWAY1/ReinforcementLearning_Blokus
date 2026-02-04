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
        agent_names: List[str],
        agents: List[AgentProtocol],
        seed: Optional[int] = None,
        max_moves: int = 1000,
    ) -> MatchResult:
        """
        Play a single 4-player Blokus match. Each of the 4 agents controls one player.
        Winner is the single player with the highest score (individual, not team).

        Args:
            agent_names: Length-4 list of agent display names (player_0 .. player_3).
            agents: Length-4 list of AgentProtocol instances in same order.
            seed: Optional RNG seed for reproducibility.
            max_moves: Maximum steps per game.

        Returns:
            MatchResult with winner (or None if tie for first), all 4 scores, and moves_made.
        """
        if len(agent_names) != 4 or len(agents) != 4:
            raise ValueError("play_match requires exactly 4 agent names and 4 agents")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        env = BlokusEnv(max_episode_steps=max_moves)
        env.reset(seed=seed)

        proxy = MatchEnvProxy(env)
        agent_map = {f"player_{i}": agents[i] for i in range(4)}
        moves_made = 0

        while env.agents and moves_made < max_moves:
            env_agent = env.agent_selection
            if env.terminations.get(env_agent, False) or env.truncations.get(env_agent, False):
                env.step(None)
                moves_made += 1
                continue
            obs = env.observe(env_agent)
            info = env.infos[env_agent]
            legal_mask = info["legal_action_mask"]
            action = agent_map[env_agent].act(obs, legal_mask, env=proxy)
            env.step(action)
            moves_made += 1
            if all(env.terminations.get(a, False) or env.truncations.get(a, False) for a in env.agents):
                break

        scores = {
            agent_names[i]: env.game.get_score(env._agent_to_player(f"player_{i}"))
            for i in range(4)
        }
        max_score = max(scores.values())
        winners = [name for name, s in scores.items() if s == max_score]
        winner = winners[0] if len(winners) == 1 else None

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

    def update_elo_after_4p_match(
        self, match_result: MatchResult, seed: Optional[int] = None
    ) -> None:
        """
        Update Elo ratings after a 4-player match using pairwise results.

        For every pair of players (i, j):
        - If score_i > score_j: result 1.0 (i wins) -> update_elo(i, j, 1.0).
        - If score_i < score_j: result 0.0 (j wins from i's perspective) -> update_elo(i, j, 0.0).
        - If score_i == score_j: result 0.5 (draw) -> update_elo(i, j, 0.5).

        Ties still change Elo: a high-rated player tying a low-rated player was expected to win,
        so the high-rated player loses points and the low-rated gains.
        """
        names = list(match_result.scores.keys())
        scores = match_result.scores
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name_i, name_j = names[i], names[j]
                s_i, s_j = scores[name_i], scores[name_j]
                if s_i > s_j:
                    result = 1.0
                elif s_i < s_j:
                    result = 0.0
                else:
                    result = 0.5
                self.update_elo(name_i, name_j, result, seed)

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
) -> Tuple[Dict[str, AgentProtocol], List[AgentSpec], List[str]]:
    """
    Build agents for 4-player league matches. Expects baseline_types to list the
    three non-RL opponents (e.g. [mcts, random, random]); RL is added when model/model_name provided.

    Returns:
        agents: name -> AgentProtocol
        specs: AgentSpec for each agent
        ordered_4p_names: length-4 list of names in player order [player_0, player_1, player_2, player_3]
                          (RL first when present, then baselines in order)
    """
    agents: Dict[str, AgentProtocol] = {}
    specs: List[AgentSpec] = []
    ordered_baseline_names: List[str] = []

    for i, agent_type in enumerate(baseline_types):
        name = f"{agent_type}_baseline_{i}"
        agents[name] = build_baseline_agent(agent_type, seed=seed)
        specs.append(AgentSpec(name=name, agent_type=agent_type))
        ordered_baseline_names.append(name)

    if model is not None and model_name is not None:
        agents[model_name] = RLPolicyAgent(model)
        specs.append(AgentSpec(name=model_name, agent_type="rl", checkpoint_path=model_name))
        ordered_4p_names = [model_name] + ordered_baseline_names
    else:
        ordered_4p_names = ordered_baseline_names

    return agents, specs, ordered_4p_names
