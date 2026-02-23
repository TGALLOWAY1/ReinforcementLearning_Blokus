"""
Deploy/runtime-only gameplay agent protocol for web turns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

from engine.board import Board, Player
from engine.move_generator import Move


class GameplayAgentProtocol(Protocol):
    """
    Minimal gameplay contract for web runtime agents.
    """

    def choose_move(
        self,
        board: Board,
        player: Player,
        legal_moves: List[Move],
        time_budget_ms: int,
    ) -> Tuple[Optional[Move], Dict[str, Any]]:
        ...
