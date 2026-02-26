"""
Self-play Gymnasium wrapper for Blokus with configurable opponents.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from agents.registry import AgentProtocol, build_baseline_agent
from envs.blokus_v0 import BlokusEnv
from mcts.zobrist import ZobristHash

logger = logging.getLogger(__name__)


class SelfPlayBlokusEnv(gym.Env):
    """
    Gymnasium-compatible single-agent wrapper that plays against opponent agents.

    The learning agent always controls player_0. All other players are driven by
    the provided opponent agents via their AgentProtocol interface.
    """

    metadata = {"render_modes": [None, "human"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
        opponents: Optional[Dict[str, AgentProtocol]] = None,
        opponent_sampler: Optional[Callable[[Optional[int]], Dict[str, AgentProtocol]]] = None,
        cache_size: int = 2048,
        seed: Optional[int] = None,
    ):
        self.env = BlokusEnv(render_mode=render_mode, max_episode_steps=max_episode_steps)
        self.agent_name = "player_0"
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.render_mode = render_mode
        self._opponent_sampler = opponent_sampler
        self._episode_count = 0
        if self._opponent_sampler is not None:
            self._opponents = self._opponent_sampler(seed)
        else:
            self._opponents = opponents or self._default_opponents(seed)
        self._zobrist = ZobristHash(seed=seed)
        self._move_cache: "OrderedDict[Tuple[int, int], Tuple[list, list]]" = OrderedDict()
        self._cache_size = cache_size

    def _default_opponents(self, seed: Optional[int]) -> Dict[str, AgentProtocol]:
        return {
            "player_1": build_baseline_agent("heuristic", seed=seed),
            "player_2": build_baseline_agent("random", seed=seed),
            "player_3": build_baseline_agent("random", seed=seed),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.env.reset(seed=seed, options=options)
        self._move_cache.clear()
        if self._opponent_sampler is not None:
            self._opponents = self._opponent_sampler(seed)
        self._episode_count += 1
        obs = self.env.observe(self.agent_name)
        info = self.env.infos[self.agent_name]
        return obs, info

    def step(self, action: int):
        self.env.step(action)
        self._advance_opponents()

        obs = self.env.observe(self.agent_name)
        reward = self.env.rewards[self.agent_name]
        terminated = self.env.terminations[self.agent_name]
        truncated = self.env.truncations[self.agent_name]
        info = self.env.infos[self.agent_name]
        return obs, reward, terminated, truncated, info

    def _advance_opponents(self) -> None:
        max_iterations = len(self.env.agents) * 2
        iterations = 0

        while self.env.agent_selection != self.agent_name and iterations < max_iterations:
            current_agent = self.env.agent_selection
            if self.env.terminations.get(current_agent, False) or self.env.truncations.get(current_agent, False):
                self.env.step(None)
                iterations += 1
                continue

            opponent = self._opponents.get(current_agent)
            if opponent is None:
                self.env.step(None)
                iterations += 1
                continue

            obs = self.env.observe(current_agent)
            legal_mask = self.env.infos[current_agent]["legal_action_mask"]
            action = opponent.act(obs, legal_mask, env=self)
            if action is None:
                self.env.step(None)
            else:
                self.env.step(int(action))
            iterations += 1

    def get_action_mask(self) -> np.ndarray:
        return self.env.infos[self.agent_name]["legal_action_mask"]

    def get_legal_moves(self, player_name: str):
        player = self.env._agent_to_player(player_name)
        board_hash = self._zobrist.hash_board(self.env.game.board)
        cache_key = (board_hash, player.value)
        cached = self._move_cache.get(cache_key)
        if cached is not None:
            self._move_cache.move_to_end(cache_key)
            return cached[0]

        legal_moves = self.env.move_generator.get_legal_moves(self.env.game.board, player)
        action_ids = [
            self.env.move_to_action.get((move.piece_id, move.orientation, move.anchor_row, move.anchor_col))
            for move in legal_moves
        ]
        self._move_cache[cache_key] = (legal_moves, action_ids)
        if len(self._move_cache) > self._cache_size:
            self._move_cache.popitem(last=False)
        return legal_moves

    def get_action_ids(self, player_name: str):
        player = self.env._agent_to_player(player_name)
        board_hash = self._zobrist.hash_board(self.env.game.board)
        cache_key = (board_hash, player.value)
        cached = self._move_cache.get(cache_key)
        if cached is not None:
            self._move_cache.move_to_end(cache_key)
            return cached[1]

        self.get_legal_moves(player_name)
        return self._move_cache.get(cache_key, ([], []))[1]

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
