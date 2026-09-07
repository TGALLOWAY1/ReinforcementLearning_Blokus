"""Deterministic greedy baseline.

Plays the argmax of the fixed move heuristic (the same scoring function
``HeuristicAgent`` samples from) with a canonical tie-break, so its play
depends only on the position — never on a seed. This makes it a fixed,
versioned yardstick for protocol v3 evaluations; ``HeuristicAgent`` remains a
temperature-1 sampler and is not suitable as a yardstick.
"""

from __future__ import annotations

from typing import List, Optional

from agents.heuristic_agent import HeuristicAgent
from engine.board import Board, Player
from engine.move_generator import Move

GREEDY_VERSION = "greedy_v1"


class GreedyAgent(HeuristicAgent):
    """Argmax of the fixed heuristic; ties broken by canonical move order."""

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed=seed)  # seed is accepted for interface parity; unused

    def select_action(self, board: Board, player: Player, legal_moves: List[Move]) -> Optional[Move]:
        if not legal_moves:
            return None
        best: Optional[Move] = None
        best_key = None
        for move in legal_moves:
            score = self._evaluate_move(board, player, move)
            key = (-score, move.piece_id, move.orientation, move.anchor_row, move.anchor_col)
            if best_key is None or key < best_key:
                best_key = key
                best = move
        return best

    def get_action_info(self):
        info = super().get_action_info()
        info.update({"name": "GreedyAgent", "type": "greedy", "version": GREEDY_VERSION,
                     "description": "Deterministic argmax of the fixed move heuristic"})
        return info
