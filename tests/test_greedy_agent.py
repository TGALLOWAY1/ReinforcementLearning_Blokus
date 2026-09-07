"""Deterministic greedy baseline: argmax of the fixed move heuristic with a
canonical tie-break. It must not depend on the seed, so it is a fixed
yardstick rather than a sampler (unlike HeuristicAgent's softmax)."""

from agents.greedy_agent import GREEDY_VERSION, GreedyAgent
from agents.heuristic_agent import HeuristicAgent
from analytics.tournament.arena_runner import AgentConfig, build_agent
from engine.board import Board, Player
from engine.move_generator import LegalMoveGenerator


def _midgame_board(seed: int = 3, plies: int = 12) -> Board:
    gen = LegalMoveGenerator()
    board = Board()
    agent = HeuristicAgent(seed=seed)
    order = list(Player)
    for ply in range(plies):
        player = order[ply % 4]
        moves = gen.get_legal_moves(board, player)
        mv = agent.select_action(board, player, moves)
        board.place_piece(mv.get_positions(gen.piece_orientations_cache[mv.piece_id]), player, mv.piece_id)
    return board


def test_greedy_is_seed_independent_and_repeatable():
    gen = LegalMoveGenerator()
    board = _midgame_board()
    moves = gen.get_legal_moves(board, Player.RED)
    picks = {GreedyAgent(seed=s).select_action(board, Player.RED, moves).to_dict()["piece_id"]
             for s in (1, 2, 3)}
    keys = [tuple(GreedyAgent(seed=s).select_action(board, Player.RED, moves).to_dict().values())
            for s in (1, 2, 3, 4)]
    assert len(set(keys)) == 1
    assert len(picks) == 1


def test_greedy_picks_the_highest_heuristic_score_with_canonical_tie_break():
    gen = LegalMoveGenerator()
    board = _midgame_board()
    moves = gen.get_legal_moves(board, Player.RED)
    agent = GreedyAgent()
    chosen = agent.select_action(board, Player.RED, moves)
    scores = [agent._evaluate_move(board, Player.RED, m) for m in moves]
    best = max(scores)
    tied = [m for m, s in zip(moves, scores) if s == best]
    expected = min(tied, key=lambda m: (m.piece_id, m.orientation, m.anchor_row, m.anchor_col))
    assert chosen.to_dict() == expected.to_dict()


def test_greedy_is_versioned_and_buildable_by_the_arena():
    assert GREEDY_VERSION == "greedy_v1"
    adapter = build_agent(AgentConfig.from_dict({"name": "greedy", "type": "greedy"}), seed=7)
    gen = LegalMoveGenerator()
    board = Board()
    mv, _stats = adapter.choose_move(board, Player.RED, gen.get_legal_moves(board, Player.RED), None)
    assert mv is not None
