"""
Random agent for Blokus that picks uniformly from legal actions.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from engine.board import Board, Player
from engine.move_generator import LegalMoveGenerator, Move
from engine.pieces import PieceGenerator


class RandomAgent:
    """
    Random agent that selects moves uniformly from legal actions.
    
    This agent serves as a baseline for comparison with more sophisticated
    agents and algorithms.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random agent.
        
        Args:
            seed: Random seed for reproducible behavior
        """
        self.rng = np.random.RandomState(seed)
        self.move_generator = LegalMoveGenerator()
        self.piece_generator = PieceGenerator()
        
    def select_action(self, board: Board, player: Player, legal_moves: List[Move]) -> Optional[Move]:
        """
        Select a random legal move.
        
        Args:
            board: Current board state
            player: Player making the move
            legal_moves: List of legal moves available
            
        Returns:
            Selected move, or None if no legal moves available
        """
        if not legal_moves:
            return None
            
        # Select uniformly from legal moves
        move_idx = self.rng.randint(0, len(legal_moves))
        return legal_moves[move_idx]
        
    def get_action_info(self) -> Dict[str, Any]:
        """Get information about the agent."""
        return {
            "name": "RandomAgent",
            "type": "random",
            "description": "Selects moves uniformly from legal actions"
        }
        
    def reset(self):
        """Reset agent state (no-op for random agent)."""
        pass
        
    def set_seed(self, seed: int):
        """Set random seed for reproducible behavior."""
        self.rng = np.random.RandomState(seed)
