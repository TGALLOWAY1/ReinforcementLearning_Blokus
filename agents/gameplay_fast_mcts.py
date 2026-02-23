"""
Gameplay adapter that exposes FastMCTSAgent via choose_move().
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agents.fast_mcts_agent import FastMCTSAgent
from engine.board import Board, Player
from engine.move_generator import Move


class GameplayFastMCTSAgent:
    def __init__(
        self,
        *,
        iterations: int = 5000,
        exploration_constant: float = 1.414,
        seed: Optional[int] = None,
    ) -> None:
        self._agent = FastMCTSAgent(
            iterations=iterations,
            time_limit=1.0,
            exploration_constant=exploration_constant,
            seed=seed,
        )

    def choose_move(
        self,
        board: Board,
        player: Player,
        legal_moves: List[Move],
        time_budget_ms: int,
    ) -> Tuple[Optional[Move], Dict[str, Any]]:
        result = self._agent.think(board, player, legal_moves, time_budget_ms)
        return result.get("move"), dict(result.get("stats") or {})
