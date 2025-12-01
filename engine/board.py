

"""
Blokus Board implementation with 20x20 grid and game state management.
"""

import numpy as np
from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class Player(Enum):
    """Player enumeration."""
    RED = 1
    BLUE = 2
    YELLOW = 3
    GREEN = 4


@dataclass
class Position:
    """Represents a position on the board."""
    row: int
    col: int
    
    def __hash__(self):
        return hash((self.row, self.col))
    
    def __eq__(self, other):
        return self.row == other.row and self.col == other.col


class Board:
    """
    Blokus game board implementation.
    
    The board is a 20x20 grid where:
    - 0 represents empty space
    - 1-4 represent players (RED, BLUE, YELLOW, GREEN)
    """
    
    SIZE = 20
    
    def __init__(self):
        self.grid = np.zeros((self.SIZE, self.SIZE), dtype=int)
        self.player_start_corners = {
            Player.RED: Position(0, 0),
            Player.BLUE: Position(0, self.SIZE - 1),
            Player.YELLOW: Position(self.SIZE - 1, self.SIZE - 1),
            Player.GREEN: Position(self.SIZE - 1, 0),
        }
        self.player_pieces_used = {player: set() for player in Player}
        self.player_first_move = {player: True for player in Player}
        self.game_over = False
        self.current_player = Player.RED
        self.move_count = 0
    
    def is_valid_position(self, pos: Position) -> bool:
        """Check if position is within board bounds."""
        return 0 <= pos.row < self.SIZE and 0 <= pos.col < self.SIZE
    
    def get_cell(self, pos: Position) -> int:
        """Get the value at a position."""
        if not self.is_valid_position(pos):
            return -1  # Invalid position
        return self.grid[pos.row, pos.col]
    
    def set_cell(self, pos: Position, value: int) -> None:
        """Set the value at a position."""
        if self.is_valid_position(pos):
            self.grid[pos.row, pos.col] = value
    
    def is_empty(self, pos: Position) -> bool:
        """Check if a position is empty."""
        return self.get_cell(pos) == 0
    
    def get_player_at(self, pos: Position) -> Optional[Player]:
        """Get the player at a position, or None if empty."""
        value = self.get_cell(pos)
        if value == 0:
            return None
        return Player(value)
    
    def get_adjacent_positions(self, pos: Position) -> List[Position]:
        """Get all adjacent positions (including diagonals)."""
        positions = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_pos = Position(pos.row + dr, pos.col + dc)
                if self.is_valid_position(new_pos):
                    positions.append(new_pos)
        return positions
    
    def get_edge_adjacent_positions(self, pos: Position) -> List[Position]:
        """Get positions that share an edge (not diagonal)."""
        positions = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = Position(pos.row + dr, pos.col + dc)
            if self.is_valid_position(new_pos):
                positions.append(new_pos)
        return positions
    
    def get_corner_adjacent_positions(self, pos: Position) -> List[Position]:
        """Get positions that are diagonally adjacent (corner touching)."""
        positions = []
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            new_pos = Position(pos.row + dr, pos.col + dc)
            if self.is_valid_position(new_pos):
                positions.append(new_pos)
        return positions
    
    def can_place_piece(self, piece_positions: List[Position], player: Player) -> bool:
        """
        Check if a piece can be placed at the given positions.
        
        Rules:
        1. First move must cover player's start corner
        2. Pieces of same color must touch only at corners
        3. No edge-to-edge adjacency with same color
        
        OPTIMIZED: Uses direct grid access to avoid Position object overhead.
        """
        if not piece_positions:
            return False
        
        player_value = player.value
        grid = self.grid  # Direct reference for faster access
        
        # Fast bounds and overlap check using direct grid access
        for pos in piece_positions:
            r, c = pos.row, pos.col
            # Bounds check
            if r < 0 or r >= self.SIZE or c < 0 or c >= self.SIZE:
                return False
            # Overlap check - direct grid access
            if grid[r, c] != 0:
                return False
        
        # Rule 1: First move must cover player's start corner
        if self.player_first_move[player]:
            start_corner = self.player_start_corners[player]
            if start_corner not in piece_positions:
                return False
        
        # Rule 2 & 3: Adjacency rules (optimized)
        if not self._check_adjacency_rules_fast(piece_positions, player_value, grid):
            return False
        
        return True
    
    def _check_adjacency_rules(self, piece_positions: List[Position], player: Player) -> bool:
        """
        Check adjacency rules for piece placement.
        
        - Pieces of same color must touch only at corners
        - No edge-to-edge adjacency with same color
        
        DEPRECATED: Use _check_adjacency_rules_fast for better performance.
        """
        return self._check_adjacency_rules_fast(piece_positions, player.value, self.grid)
    
    def _check_adjacency_rules_fast(self, piece_positions: List[Position], player_value: int, grid: np.ndarray) -> bool:
        """
        Optimized adjacency rules check using direct grid access.
        
        - Pieces of same color must touch only at corners
        - No edge-to-edge adjacency with same color
        """
        has_corner_connection = False
        
        # Check each position in the piece
        for pos in piece_positions:
            r, c = pos.row, pos.col
            
            # Check edge adjacency (not allowed with same color) - direct grid access
            # Top, bottom, left, right
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                    if grid[nr, nc] == player_value:
                        return False  # Edge adjacency not allowed
            
            # Check corner adjacency (allowed with same color) - direct grid access
            # Diagonals
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                    if grid[nr, nc] == player_value:
                        has_corner_connection = True  # Corner connection found
        
        # Additional check: piece must be connected to existing pieces via corners (if not first move)
        if not self.player_first_move[Player(player_value)]:
            if not has_corner_connection:
                return False
        
        return True
    
    def _is_connected_via_corners(self, piece_positions: List[Position], player: Player) -> bool:
        """
        Check if the piece is connected to existing pieces of the same color via corners.
        
        DEPRECATED: This is now handled in _check_adjacency_rules_fast for better performance.
        """
        # For first move, this is handled by the start corner rule
        if self.player_first_move[player]:
            return True
        
        player_value = player.value
        grid = self.grid
        
        # Check if any position in the piece touches an existing piece at a corner
        for pos in piece_positions:
            r, c = pos.row, pos.col
            # Check diagonals - direct grid access
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                    if grid[nr, nc] == player_value:
                        return True
        
        return False
    
    def place_piece(self, piece_positions: List[Position], player: Player, piece_id: int) -> bool:
        """
        Place a piece on the board.
        
        OPTIMIZED: Uses direct grid access for faster placement.
        
        Returns True if placement was successful, False otherwise.
        """
        if not self.can_place_piece(piece_positions, player):
            return False
        
        # Place the piece using direct grid access (faster than set_cell)
        player_value = player.value
        for pos in piece_positions:
            self.grid[pos.row, pos.col] = player_value
        
        # Record the piece as used
        self.player_pieces_used[player].add(piece_id)
        
        # Mark first move as completed
        self.player_first_move[player] = False
        
        # Update game state
        self.move_count += 1
        self._update_current_player()
        
        return True
    
    def _update_current_player(self) -> None:
        """Update the current player for the next turn."""
        players = list(Player)
        current_index = players.index(self.current_player)
        self.current_player = players[(current_index + 1) % len(players)]
    
    def get_score(self, player: Player) -> int:
        """
        Calculate score for a player.
        
        Score is based on:
        - Number of squares covered by pieces
        - Bonus for using all pieces
        """
        # Count squares covered by player
        squares_covered = np.sum(self.grid == player.value)
        
        # Bonus for using all pieces (21 pieces total)
        if len(self.player_pieces_used[player]) == 21:
            squares_covered += 15  # Bonus for using all pieces
        
        return squares_covered
    
    def get_winner(self) -> Optional[Player]:
        """Determine the winner based on scores."""
        if not self.game_over:
            return None
        
        scores = {player: self.get_score(player) for player in Player}
        return max(scores, key=scores.get)
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        # Game is over when no player can make a legal move
        # This is a simplified check - in practice, you'd need to check all possible moves
        return self.game_over
    
    def copy(self) -> 'Board':
        """Create a deep copy of the board."""
        new_board = Board()
        new_board.grid = self.grid.copy()
        new_board.player_pieces_used = {k: v.copy() for k, v in self.player_pieces_used.items()}
        new_board.player_first_move = self.player_first_move.copy()
        new_board.game_over = self.game_over
        new_board.current_player = self.current_player
        new_board.move_count = self.move_count
        return new_board
    
    def __str__(self) -> str:
        """String representation of the board."""
        result = []
        for row in range(self.SIZE):
            row_str = ""
            for col in range(self.SIZE):
                value = self.grid[row, col]
                if value == 0:
                    row_str += "."
                else:
                    row_str += str(value)
            result.append(row_str)
        return "\n".join(result)
