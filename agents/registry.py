"""
Agent registry and unified act() protocol for Blokus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from agents.fast_mcts_agent import FastMCTSAgent
from mcts.mcts_agent import MCTSAgent


class AgentProtocol(Protocol):
    """Unified interface: act(observation, legal_mask) -> action index."""

    def act(self, observation: np.ndarray, legal_mask: np.ndarray, env=None) -> Optional[int]:
        ...


@dataclass
class AgentSpec:
    name: str
    agent_type: str
    version: Optional[str] = None
    checkpoint_path: Optional[str] = None


class RandomAgentAdapter:
    def __init__(self, seed: Optional[int] = None):
        self.agent = RandomAgent(seed=seed)

    def act(self, observation: np.ndarray, legal_mask: np.ndarray, env=None) -> Optional[int]:
        if legal_mask is None or not legal_mask.any():
            return None
        action_ids = np.flatnonzero(legal_mask)
        if action_ids.size == 0:
            return None
        return int(np.random.choice(action_ids))


class HeuristicAgentAdapter:
    def __init__(self, seed: Optional[int] = None):
        self.agent = HeuristicAgent(seed=seed)

    def act(self, observation: np.ndarray, legal_mask: np.ndarray, env=None) -> Optional[int]:
        if env is None:
            return None
        base_env = env.env if hasattr(env, "env") else env
        player_name = base_env.agent_selection
        legal_moves = env.get_legal_moves(player_name) if hasattr(env, "get_legal_moves") else base_env.move_generator.get_legal_moves(
            base_env.game.board, base_env._agent_to_player(player_name)
        )
        if not legal_moves:
            return None
        player = base_env._agent_to_player(player_name)
        move = self.agent.select_action(base_env.game.board, player, legal_moves)
        if move is None:
            return None
        return base_env.move_to_action.get((move.piece_id, move.orientation, move.anchor_row, move.anchor_col))


class MCTSAgentAdapter:
    def __init__(self, seed: Optional[int] = None, iterations: int = 200, time_limit: Optional[float] = None):
        self.agent = MCTSAgent(iterations=iterations, time_limit=time_limit, seed=seed)

    def act(self, observation: np.ndarray, legal_mask: np.ndarray, env=None) -> Optional[int]:
        if env is None:
            return None
        base_env = env.env if hasattr(env, "env") else env
        player_name = base_env.agent_selection
        legal_moves = env.get_legal_moves(player_name) if hasattr(env, "get_legal_moves") else base_env.move_generator.get_legal_moves(
            base_env.game.board, base_env._agent_to_player(player_name)
        )
        if not legal_moves:
            return None
        player = base_env._agent_to_player(player_name)
        move = self.agent.select_action(base_env.game.board, player, legal_moves)
        if move is None:
            return None
        return base_env.move_to_action.get((move.piece_id, move.orientation, move.anchor_row, move.anchor_col))


class FastMCTSAgentAdapter:
    def __init__(self, seed: Optional[int] = None, time_limit: float = 0.05):
        self.agent = FastMCTSAgent(time_limit=time_limit, seed=seed)

    def act(self, observation: np.ndarray, legal_mask: np.ndarray, env=None) -> Optional[int]:
        if env is None:
            return None
        base_env = env.env if hasattr(env, "env") else env
        player_name = base_env.agent_selection
        legal_moves = env.get_legal_moves(player_name) if hasattr(env, "get_legal_moves") else base_env.move_generator.get_legal_moves(
            base_env.game.board, base_env._agent_to_player(player_name)
        )
        if not legal_moves:
            return None
        player = base_env._agent_to_player(player_name)
        move = self.agent.select_action(base_env.game.board, player, legal_moves)
        if move is None:
            return None
        return base_env.move_to_action.get((move.piece_id, move.orientation, move.anchor_row, move.anchor_col))


class RLPolicyAgent:
    def __init__(self, model):
        self.model = model

    def act(self, observation: np.ndarray, legal_mask: np.ndarray, env=None) -> Optional[int]:
        if legal_mask is None or not legal_mask.any():
            return None
        action, _ = self.model.predict(observation, action_masks=legal_mask, deterministic=True)
        return int(action)


def build_baseline_agent(agent_type: str, seed: Optional[int] = None) -> AgentProtocol:
    agent_type = agent_type.lower()
    if agent_type == "random":
        return RandomAgentAdapter(seed=seed)
    if agent_type == "heuristic":
        return HeuristicAgentAdapter(seed=seed)
    if agent_type == "mcts":
        return MCTSAgentAdapter(seed=seed)
    if agent_type == "fast_mcts":
        return FastMCTSAgentAdapter(seed=seed)
    raise ValueError(f"Unknown agent type: {agent_type}")
