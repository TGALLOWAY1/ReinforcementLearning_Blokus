"""
Heuristic agent for Blokus with strategic preferences.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from engine.board import Board, Player, Position
from engine.move_generator import LegalMoveGenerator, Move
from engine.pieces import PieceGenerator


class HeuristicAgent:
    """
    Heuristic agent with strategic preferences:
    - Prefer large pieces
    - Create new corners for future moves
    - Avoid edges early in the game
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize heuristic agent.
        
        Args:
            seed: Random seed for reproducible behavior
        """
        self.rng = np.random.RandomState(seed)
        self.move_generator = LegalMoveGenerator()
        self.piece_generator = PieceGenerator()
        
        # Heuristic weights
        self.piece_size_weight = 1.0
        self.corner_creation_weight = 2.0
        self.edge_avoidance_weight = -1.5
        self.center_preference_weight = 0.5
        
    def select_action(self, board: Board, player: Player, legal_moves: List[Move]) -> Optional[Move]:
        """
        Select a move based on heuristics.
        
        Args:
            board: Current board state
            player: Player making the move
            legal_moves: List of legal moves available
            
        Returns:
            Selected move, or None if no legal moves available
        """
        if not legal_moves:
            return None
            
        # Score each legal move
        move_scores = []
        for move in legal_moves:
            score = self._evaluate_move(board, player, move)
            move_scores.append(score)
            
        # Convert scores to probabilities (softmax)
        scores_array = np.array(move_scores)
        probabilities = self._softmax(scores_array, temperature=1.0)
        
        # Select move based on probabilities
        move_idx = self.rng.choice(len(legal_moves), p=probabilities)
        return legal_moves[move_idx]
        
    def _evaluate_move(self, board: Board, player: Player, move: Move) -> float:
        """
        Evaluate a move based on heuristics.
        
        Args:
            board: Current board state
            player: Player making the move
            move: Move to evaluate
            
        Returns:
            Heuristic score for the move
        """
        score = 0.0
        
        # Get piece information
        piece = self.piece_generator.get_piece_by_id(move.piece_id)
        orientations = self.move_generator.piece_orientations_cache[move.piece_id]
        orientation = orientations[move.orientation]
        
        # 1. Piece size preference (larger pieces are better)
        piece_size = piece.size
        score += self.piece_size_weight * piece_size
        
        # 2. Corner creation potential
        corner_score = self._evaluate_corner_creation(board, player, move, orientation)
        score += self.corner_creation_weight * corner_score
        
        # 3. Edge avoidance (avoid edges early in game)
        edge_score = self._evaluate_edge_avoidance(board, player, move, orientation)
        score += self.edge_avoidance_weight * edge_score
        
        # 4. Center preference
        center_score = self._evaluate_center_preference(move)
        score += self.center_preference_weight * center_score
        
        return score
        
    def _evaluate_corner_creation(self, board: Board, player: Player, move: Move, orientation: np.ndarray) -> float:
        """
        Evaluate how many new corners this move creates.
        
        Args:
            board: Current board state
            player: Player making the move
            move: Move to evaluate
            orientation: Piece orientation
            
        Returns:
            Corner creation score
        """
        # Get piece positions
        piece_positions = self._get_piece_positions(move, orientation)
        
        # Count new corners created
        new_corners = 0
        for pos in piece_positions:
            # Check if this position creates new corner opportunities
            corner_adjacent = board.get_corner_adjacent_positions(pos)
            for adj_pos in corner_adjacent:
                if board.is_empty(adj_pos):
                    # Check if this corner is not edge-adjacent to any player pieces
                    edge_adjacent = board.get_edge_adjacent_positions(adj_pos)
                    safe_corner = True
                    for edge_pos in edge_adjacent:
                        if board.get_player_at(edge_pos) == player:
                            safe_corner = False
                            break
                    if safe_corner:
                        new_corners += 1
                        
        return new_corners
        
    def _evaluate_edge_avoidance(self, board: Board, player: Player, move: Move, orientation: np.ndarray) -> float:
        """
        Evaluate edge avoidance (penalize moves near edges early in game).
        
        Args:
            board: Current board state
            player: Player making the move
            move: Move to evaluate
            orientation: Piece orientation
            
        Returns:
            Edge avoidance score (negative = good, positive = bad)
        """
        # Get piece positions
        piece_positions = self._get_piece_positions(move, orientation)
        
        # Count positions near edges
        edge_positions = 0
        for pos in piece_positions:
            # Distance from edges
            min_distance = min(
                pos.row, pos.col,
                board.SIZE - 1 - pos.row,
                board.SIZE - 1 - pos.col
            )
            if min_distance <= 2:  # Near edge
                edge_positions += 1
                
        # Early game penalty for edge moves
        game_progress = board.move_count / 100.0  # Normalize by expected game length
        if game_progress < 0.3:  # Early game
            return edge_positions
        else:
            return edge_positions * 0.5  # Reduced penalty later
            
    def _evaluate_center_preference(self, move: Move) -> float:
        """
        Evaluate preference for center positions.
        
        Args:
            move: Move to evaluate
            
        Returns:
            Center preference score
        """
        # Distance from center
        center_row = 9.5  # Board center
        center_col = 9.5
        
        distance_from_center = np.sqrt(
            (move.anchor_row - center_row) ** 2 + 
            (move.anchor_col - center_col) ** 2
        )
        
        # Closer to center is better
        max_distance = np.sqrt(2 * (9.5 ** 2))  # Maximum distance from center
        normalized_distance = distance_from_center / max_distance
        
        return 1.0 - normalized_distance  # Higher score for closer to center
        
    def _get_piece_positions(self, move: Move, orientation: np.ndarray) -> List[Position]:
        """
        Get positions that a piece would occupy.
        
        Args:
            move: Move containing anchor position
            orientation: Piece orientation
            
        Returns:
            List of positions the piece would occupy
        """
        positions = []
        rows, cols = orientation.shape
        
        for i in range(rows):
            for j in range(cols):
                if orientation[i, j] == 1:
                    pos = Position(move.anchor_row + i, move.anchor_col + j)
                    positions.append(pos)
                    
        return positions
        
    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Apply softmax with temperature to convert scores to probabilities.
        
        Args:
            x: Input scores
            temperature: Temperature parameter (higher = more random)
            
        Returns:
            Probabilities
        """
        # Apply temperature scaling
        x_scaled = x / temperature
        
        # Subtract max for numerical stability
        x_max = np.max(x_scaled)
        x_shifted = x_scaled - x_max
        
        # Compute softmax
        exp_x = np.exp(x_shifted)
        probabilities = exp_x / np.sum(exp_x)
        
        return probabilities
        
    def get_action_info(self) -> Dict[str, Any]:
        """Get information about the agent."""
        return {
            "name": "HeuristicAgent",
            "type": "heuristic",
            "description": "Strategic agent with piece size, corner, and edge preferences",
            "weights": {
                "piece_size": self.piece_size_weight,
                "corner_creation": self.corner_creation_weight,
                "edge_avoidance": self.edge_avoidance_weight,
                "center_preference": self.center_preference_weight
            }
        }
        
    def reset(self):
        """Reset agent state (no-op for heuristic agent)."""
        pass
        
    def set_seed(self, seed: int):
        """Set random seed for reproducible behavior."""
        self.rng = np.random.RandomState(seed)
        
    def set_weights(self, weights: Dict[str, float]):
        """
        Set heuristic weights.
        
        Args:
            weights: Dictionary of weight names and values
        """
        if "piece_size" in weights:
            self.piece_size_weight = weights["piece_size"]
        if "corner_creation" in weights:
            self.corner_creation_weight = weights["corner_creation"]
        if "edge_avoidance" in weights:
            self.edge_avoidance_weight = weights["edge_avoidance"]
        if "center_preference" in weights:
            self.center_preference_weight = weights["center_preference"]
