"""Arena adapter hooks needed by protocol v3 anchors: non-MCTS agents can
report per-move stats, and adapters can be closed explicitly (external
engines must not rely on garbage collection to exit)."""

from analytics.tournament.arena_runner import _SelectActionAdapter
from engine.board import Board, Player
from engine.move_generator import LegalMoveGenerator


class _Dummy:
    def __init__(self):
        self.closed = False
        self.stats = {"moves": 0, "mismatches": 0}

    def select_action(self, board, player, legal_moves):
        self.stats["moves"] += 1
        return legal_moves[0]

    def get_action_info(self):
        return {"name": "dummy", "stats": dict(self.stats)}

    def close(self):
        self.closed = True


def test_adapter_passes_through_agent_stats_and_close():
    agent = _Dummy()
    adapter = _SelectActionAdapter(agent)
    board = Board()
    moves = LegalMoveGenerator().get_legal_moves(board, Player.RED)
    _mv, stats = adapter.choose_move(board, Player.RED, moves, None)
    assert stats["_agent_stats"] == {"moves": 1, "mismatches": 0}
    adapter.close()
    assert agent.closed
