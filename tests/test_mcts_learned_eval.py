from __future__ import annotations

import math

from engine.game import BlokusGame
from mcts.mcts_agent import MCTSAgent


class _FakeLearnedEvaluator:
    def __init__(self, artifact_path: str, *, potential_mode: str = "prob") -> None:
        self.artifact_path = artifact_path
        self.potential_mode = potential_mode

    def predict_player_win_probability(self, board, player) -> float:
        base = 0.25 + (0.1 * (player.value - 1))
        move_component = min(board.move_count * 0.01, 0.2)
        return float(max(0.01, min(0.99, base + move_component)))

    def potential(self, board, player) -> float:
        prob = self.predict_player_win_probability(board, player)
        if self.potential_mode == "logit":
            prob = min(max(prob, 1e-6), 1 - 1e-6)
            return float(math.log(prob / (1.0 - prob)))
        return prob


def _initial_state():
    game = BlokusGame()
    player = game.get_current_player()
    legal_moves = game.get_legal_moves(player)
    return game, player, legal_moves


def test_leaf_evaluation_records_calls(monkeypatch):
    monkeypatch.setattr(
        "mcts.mcts_agent.LearnedWinProbabilityEvaluator",
        _FakeLearnedEvaluator,
    )
    game, player, legal_moves = _initial_state()
    agent = MCTSAgent(
        iterations=8,
        seed=123,
        use_transposition_table=False,
        learned_model_path="models/fake.pkl",
        leaf_evaluation_enabled=True,
    )
    move = agent.select_action(game.board, player, legal_moves)
    stats = agent.get_action_info()["stats"]
    assert move is not None
    assert stats["leaf_eval_calls"] > 0
    assert stats["evaluator_errors"] == 0


def test_progressive_bias_records_updates(monkeypatch):
    monkeypatch.setattr(
        "mcts.mcts_agent.LearnedWinProbabilityEvaluator",
        _FakeLearnedEvaluator,
    )
    game, player, legal_moves = _initial_state()
    agent = MCTSAgent(
        iterations=8,
        seed=123,
        use_transposition_table=False,
        learned_model_path="models/fake.pkl",
        progressive_bias_enabled=True,
        progressive_bias_weight=0.5,
    )
    move = agent.select_action(game.board, player, legal_moves)
    stats = agent.get_action_info()["stats"]
    assert move is not None
    assert stats["progressive_bias_updates"] > 0
    assert stats["evaluator_errors"] == 0


def test_potential_shaping_applies_on_truncated_rollout(monkeypatch):
    monkeypatch.setattr(
        "mcts.mcts_agent.LearnedWinProbabilityEvaluator",
        _FakeLearnedEvaluator,
    )
    game, player, legal_moves = _initial_state()
    agent = MCTSAgent(
        iterations=8,
        seed=123,
        use_transposition_table=False,
        learned_model_path="models/fake.pkl",
        potential_shaping_enabled=True,
        potential_mode="logit",
        max_rollout_moves=1,
    )
    move = agent.select_action(game.board, player, legal_moves)
    stats = agent.get_action_info()["stats"]
    assert move is not None
    assert len(stats["potential_shaping_terms"]) > 0
    assert stats["evaluator_errors"] == 0
