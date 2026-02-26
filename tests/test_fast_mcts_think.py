import time

from agents.fast_mcts_agent import FastMCTSAgent
from engine.game import BlokusGame


def test_fast_mcts_returns_legal_move():
    game = BlokusGame()
    player = game.get_current_player()
    legal_moves = game.get_legal_moves(player)
    agent = FastMCTSAgent(iterations=2000, time_limit=1.0)

    result = agent.think(game.board, player, legal_moves, 200)
    move = result["move"]

    assert move is not None
    assert any(
        m.piece_id == move.piece_id and m.orientation == move.orientation and m.anchor_row == move.anchor_row and m.anchor_col == move.anchor_col
        for m in legal_moves
    )
    assert "stats" in result
    assert result["stats"]["nodesEvaluated"] >= 1


def test_fast_mcts_honors_time_budget_tolerance():
    game = BlokusGame()
    player = game.get_current_player()
    legal_moves = game.get_legal_moves(player)
    agent = FastMCTSAgent(iterations=100000, time_limit=5.0)

    start = time.perf_counter()
    result = agent.think(game.board, player, legal_moves, 150)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert result["move"] is not None
    assert elapsed_ms <= 700  # generous CI tolerance
