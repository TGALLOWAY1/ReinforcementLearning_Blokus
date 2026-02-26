"""
Utility functions for generating test game states.
"""

import random
from typing import Tuple

from engine.board import Board, Player
from engine.move_generator import LegalMoveGenerator


def generate_random_valid_state(num_moves: int, seed: int = 0) -> Tuple[Board, Player]:
    """
    Generate a random but valid board state by playing random legal moves.
    
    Uses only the naive move generator and grid-based legality to ensure
    we're using the trusted reference implementation.
    
    Args:
        num_moves: Number of moves to make
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (board, current_player) representing the final state
    """
    random.seed(seed)
    
    board = Board()
    generator = LegalMoveGenerator()
    
    moves_made = 0
    max_attempts = num_moves * 10  # Prevent infinite loops
    
    for attempt in range(max_attempts):
        if moves_made >= num_moves:
            break
            
        player = board.current_player
        
        # Use naive generator (trusted reference)
        moves = generator._get_legal_moves_naive(board, player)
        
        if not moves:
            # No legal moves for current player, try next player
            board._update_current_player()
            continue
        
        # Choose a random move
        move = random.choice(moves)
        orientations = generator.piece_orientations_cache[move.piece_id]
        positions = move.get_positions(orientations)
        
        success = board.place_piece(positions, player, move.piece_id)
        if success:
            moves_made += 1
    
    return board, board.current_player

