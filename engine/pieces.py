"""
Blokus piece definitions with all 21 polyominoes and their rotations/reflections.
"""

import numpy as np
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum


@dataclass
class Piece:
    """Represents a Blokus piece."""
    id: int
    name: str
    shape: np.ndarray  # 2D array representing the piece
    size: int  # Number of squares in the piece
    
    def __post_init__(self):
        """Validate piece after initialization."""
        if self.shape.ndim != 2:
            raise ValueError("Piece shape must be 2D")
        if np.sum(self.shape) != self.size:
            raise ValueError("Piece shape sum must equal size")


class PieceType(Enum):
    """Enumeration of all Blokus piece types."""
    MONOMINO = 1      # 1 square
    DOMINO = 2        # 2 squares
    TROMINO_I = 3     # 3 squares in line
    TROMINO_L = 4     # 3 squares in L shape
    TETROMINO_I = 5   # 4 squares in line
    TETROMINO_O = 6   # 2x2 square
    TETROMINO_T = 7   # T shape
    TETROMINO_L = 8   # L shape
    TETROMINO_S = 9   # S shape
    TETROMINO_Z = 10  # Z shape
    PENTOMINO_F = 11  # F shape
    PENTOMINO_I = 12  # 5 squares in line
    PENTOMINO_L = 13  # L shape
    PENTOMINO_N = 14  # N shape
    PENTOMINO_P = 15  # P shape
    PENTOMINO_T = 16  # T shape
    PENTOMINO_U = 17  # U shape
    PENTOMINO_V = 18  # V shape
    PENTOMINO_W = 19  # W shape
    PENTOMINO_X = 20  # X shape
    PENTOMINO_Y = 21  # Y shape


class PieceGenerator:
    """Generates all Blokus pieces with rotations and reflections."""
    
    @staticmethod
    def get_all_pieces() -> List[Piece]:
        """Get all 21 Blokus pieces."""
        pieces = []
        
        # Monomino (1 square)
        pieces.append(Piece(1, "Monomino", np.array([[1]]), 1))
        
        # Domino (2 squares)
        pieces.append(Piece(2, "Domino", np.array([[1, 1]]), 2))
        
        # Tromino I (3 squares in line)
        pieces.append(Piece(3, "Tromino I", np.array([[1, 1, 1]]), 3))
        
        # Tromino L (3 squares in L shape)
        pieces.append(Piece(4, "Tromino L", np.array([[1, 0], [1, 1]]), 3))
        
        # Tetromino I (4 squares in line)
        pieces.append(Piece(5, "Tetromino I", np.array([[1, 1, 1, 1]]), 4))
        
        # Tetromino O (2x2 square)
        pieces.append(Piece(6, "Tetromino O", np.array([[1, 1], [1, 1]]), 4))
        
        # Tetromino T (T shape)
        pieces.append(Piece(7, "Tetromino T", np.array([[1, 1, 1], [0, 1, 0]]), 4))
        
        # Tetromino L (L shape)
        pieces.append(Piece(8, "Tetromino L", np.array([[1, 0], [1, 0], [1, 1]]), 4))
        
        # Tetromino S (S shape)
        pieces.append(Piece(9, "Tetromino S", np.array([[0, 1, 1], [1, 1, 0]]), 4))
        
        # Tetromino Z (Z shape)
        pieces.append(Piece(10, "Tetromino Z", np.array([[1, 1, 0], [0, 1, 1]]), 4))
        
        # Pentomino F (F shape)
        pieces.append(Piece(11, "Pentomino F", np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]]), 5))
        
        # Pentomino I (5 squares in line)
        pieces.append(Piece(12, "Pentomino I", np.array([[1, 1, 1, 1, 1]]), 5))
        
        # Pentomino L (L shape)
        pieces.append(Piece(13, "Pentomino L", np.array([[1, 0], [1, 0], [1, 0], [1, 1]]), 5))
        
        # Pentomino N (N shape)
        pieces.append(Piece(14, "Pentomino N", np.array([[1, 0], [1, 1], [0, 1], [0, 1]]), 5))
        
        # Pentomino P (P shape)
        pieces.append(Piece(15, "Pentomino P", np.array([[1, 1], [1, 1], [1, 0]]), 5))
        
        # Pentomino T (T shape)
        pieces.append(Piece(16, "Pentomino T", np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]]), 5))
        
        # Pentomino U (U shape)
        pieces.append(Piece(17, "Pentomino U", np.array([[1, 0, 1], [1, 1, 1]]), 5))
        
        # Pentomino V (V shape)
        pieces.append(Piece(18, "Pentomino V", np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]]), 5))
        
        # Pentomino W (W shape)
        pieces.append(Piece(19, "Pentomino W", np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]]), 5))
        
        # Pentomino X (X shape)
        pieces.append(Piece(20, "Pentomino X", np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), 5))
        
        # Pentomino Y (Y shape)
        pieces.append(Piece(21, "Pentomino Y", np.array([[1, 0], [1, 1], [1, 0], [1, 0]]), 5))
        
        return pieces
    
    @staticmethod
    def get_piece_by_id(piece_id: int) -> Optional[Piece]:
        """Get a piece by its ID."""
        pieces = PieceGenerator.get_all_pieces()
        for piece in pieces:
            if piece.id == piece_id:
                return piece
        return None
    
    @staticmethod
    def get_piece_rotations_and_reflections(piece: Piece) -> List[np.ndarray]:
        """
        Get all rotations and reflections of a piece.
        
        Returns a list of 2D numpy arrays representing all possible orientations.
        """
        orientations = []
        current_shape = piece.shape.copy()
        
        # Add original shape
        orientations.append(current_shape.copy())
        
        # Add 90, 180, 270 degree rotations
        for _ in range(3):
            current_shape = np.rot90(current_shape)
            orientations.append(current_shape.copy())
        
        # Add reflections
        reflected = np.fliplr(piece.shape)
        orientations.append(reflected.copy())
        
        # Add reflected rotations
        for _ in range(3):
            reflected = np.rot90(reflected)
            orientations.append(reflected.copy())
        
        # Remove duplicates
        unique_orientations = []
        for orientation in orientations:
            if not any(np.array_equal(orientation, existing) for existing in unique_orientations):
                unique_orientations.append(orientation)
        
        return unique_orientations
    
    @staticmethod
    def get_all_piece_orientations() -> List[Tuple[Piece, np.ndarray]]:
        """
        Get all pieces with all their orientations.
        
        Returns a list of tuples (piece, orientation_shape).
        """
        all_orientations = []
        pieces = PieceGenerator.get_all_pieces()
        
        for piece in pieces:
            orientations = PieceGenerator.get_piece_rotations_and_reflections(piece)
            for orientation in orientations:
                all_orientations.append((piece, orientation))
        
        return all_orientations


class PiecePlacement:
    """Helper class for piece placement calculations."""
    
    @staticmethod
    def get_piece_positions(shape: np.ndarray, anchor_row: int, anchor_col: int) -> List[Tuple[int, int]]:
        """
        Get the board positions that a piece shape would occupy when placed at anchor position.
        
        Args:
            shape: 2D numpy array representing the piece
            anchor_row: Row position of the anchor (top-left of piece)
            anchor_col: Column position of the anchor (top-left of piece)
        
        Returns:
            List of (row, col) tuples representing occupied positions
        """
        positions = []
        rows, cols = shape.shape
        
        for i in range(rows):
            for j in range(cols):
                if shape[i, j] == 1:
                    positions.append((anchor_row + i, anchor_col + j))
        
        return positions
    
    @staticmethod
    def can_place_piece_at(board_shape: Tuple[int, int], piece_shape: np.ndarray, 
                          anchor_row: int, anchor_col: int) -> bool:
        """
        Check if a piece can be placed at the given anchor position without going out of bounds.
        
        Args:
            board_shape: (height, width) of the board
            piece_shape: 2D numpy array representing the piece
            anchor_row: Row position of the anchor
            anchor_col: Column position of the anchor
        
        Returns:
            True if piece can be placed without going out of bounds
        """
        board_height, board_width = board_shape
        piece_height, piece_width = piece_shape.shape
        
        # Check if piece would go out of bounds
        if (anchor_row < 0 or anchor_col < 0 or 
            anchor_row + piece_height > board_height or 
            anchor_col + piece_width > board_width):
            return False
        
        return True
    
    @staticmethod
    def get_valid_anchor_positions(board_shape: Tuple[int, int], piece_shape: np.ndarray) -> List[Tuple[int, int]]:
        """
        Get all valid anchor positions for placing a piece on the board.
        
        Args:
            board_shape: (height, width) of the board
            piece_shape: 2D numpy array representing the piece
        
        Returns:
            List of (row, col) tuples representing valid anchor positions
        """
        valid_positions = []
        board_height, board_width = board_shape
        piece_height, piece_width = piece_shape.shape
        
        for row in range(board_height - piece_height + 1):
            for col in range(board_width - piece_width + 1):
                if PiecePlacement.can_place_piece_at(board_shape, piece_shape, row, col):
                    valid_positions.append((row, col))
        
        return valid_positions
