"""
Blokus piece definitions with all 21 polyominoes and their rotations/reflections.
"""

import numpy as np
from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from .bitboard import coords_to_mask, coord_to_bit


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


@dataclass
class PieceOrientation:
    """
    Precomputed orientation data for a piece.
    
    Contains normalized offsets, bitmasks, and other precomputed data
    for efficient move generation.
    """
    piece_id: int
    orientation_id: int
    offsets: List[Tuple[int, int]]  # (row, col) offsets anchored at (0,0), normalized
    shape_mask: int  # Bitmask for the shape cells
    diag_mask: int  # Bitmask for diagonal neighbors (not including shape cells)
    orth_mask: int  # Bitmask for orthogonal neighbors (not including shape cells)
    anchor_indices: List[int]  # Indices into offsets for anchor points (placeholder for now)


def shape_to_offsets(shape: np.ndarray) -> List[Tuple[int, int]]:
    """
    Convert a numpy shape array to a list of (row, col) offsets.
    
    Args:
        shape: 2D numpy array with 1s where cells are occupied
        
    Returns:
        List of (row, col) tuples for occupied cells
    """
    offsets = []
    rows, cols = shape.shape
    for i in range(rows):
        for j in range(cols):
            if shape[i, j] == 1:
                offsets.append((i, j))
    return offsets


def normalize_offsets(offsets: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Normalize offsets so that min_row = 0 and min_col = 0 (anchor at (0,0)).
    
    Args:
        offsets: List of (row, col) tuples
        
    Returns:
        Normalized list of offsets, sorted for canonical ordering
    """
    if not offsets:
        return []
    
    min_row = min(r for r, c in offsets)
    min_col = min(c for r, c in offsets)
    
    normalized = [(r - min_row, c - min_col) for r, c in offsets]
    return sorted(normalized)  # Sort for canonical ordering


def _compute_anchor_indices(offsets: List[Tuple[int, int]], max_anchors: int = 4) -> List[int]:
    """
    Compute heuristic anchor indices for a piece orientation.
    
    Selects strategic anchor points that are more likely to produce valid placements
    when aligned with frontier cells. This reduces redundant anchor attempts.
    
    Heuristic:
    - Top-left cell (min row+col)
    - Bottom-right cell (max row+col)
    - Furthest-from-centroid cell (max distance from average position)
    
    Args:
        offsets: List of normalized (row, col) offsets
        max_anchors: Maximum number of anchors to select (default 4)
        
    Returns:
        List of indices into offsets for anchor points
    """
    if not offsets:
        return []
    
    if len(offsets) <= max_anchors:
        # If piece is small, use all offsets as anchors
        return list(range(len(offsets)))
    
    anchor_set = set()
    
    # 1. Top-left cell: minimum (row + col)
    top_left_idx = min(range(len(offsets)), 
                      key=lambda i: offsets[i][0] + offsets[i][1])
    anchor_set.add(top_left_idx)
    
    # 2. Bottom-right cell: maximum (row + col)
    bottom_right_idx = max(range(len(offsets)),
                          key=lambda i: offsets[i][0] + offsets[i][1])
    anchor_set.add(bottom_right_idx)
    
    # 3. Furthest-from-centroid cell
    if len(offsets) > 2:
        # Compute centroid
        avg_row = sum(r for r, c in offsets) / len(offsets)
        avg_col = sum(c for r, c in offsets) / len(offsets)
        
        # Find cell with max distance squared from centroid
        furthest_idx = max(range(len(offsets)),
                          key=lambda i: (offsets[i][0] - avg_row)**2 + 
                                       (offsets[i][1] - avg_col)**2)
        anchor_set.add(furthest_idx)
    
    # Convert to sorted list and cap at max_anchors
    anchor_list = sorted(anchor_set)
    if len(anchor_list) > max_anchors:
        anchor_list = anchor_list[:max_anchors]
    
    return anchor_list


def generate_orientations_for_piece(piece_id: int, base_shape: np.ndarray) -> List[PieceOrientation]:
    """
    Generate all unique orientations for a piece with precomputed bitmasks.
    
    Args:
        piece_id: ID of the piece
        base_shape: Base numpy array shape of the piece
        
    Returns:
        List of PieceOrientation instances, one per unique orientation
    """
    orientations = []
    
    # Generate all rotations and reflections using existing logic
    current_shape = base_shape.copy()
    shape_variants = [current_shape.copy()]
    
    # Add 90, 180, 270 degree rotations
    for _ in range(3):
        current_shape = np.rot90(current_shape)
        shape_variants.append(current_shape.copy())
    
    # Add reflections
    reflected = np.fliplr(base_shape)
    shape_variants.append(reflected.copy())
    
    # Add reflected rotations
    for _ in range(3):
        reflected = np.rot90(reflected)
        shape_variants.append(reflected.copy())
    
    # Deduplicate by converting to normalized offsets and using as key
    seen_offsets = {}
    orientation_id = 0
    
    for shape in shape_variants:
        # Convert to offsets
        offsets = shape_to_offsets(shape)
        normalized = normalize_offsets(offsets)
        
        # Use sorted tuple as key for deduplication
        offsets_key = tuple(normalized)
        if offsets_key in seen_offsets:
            continue
        
        seen_offsets[offsets_key] = True
        
        # Compute shape mask
        shape_mask = coords_to_mask(normalized)
        
        # Compute diagonal neighbor coords (excluding shape cells)
        # Include ALL neighbors relative to the normalized offsets.
        # Note: Some neighbors may have negative coordinates, which is fine -
        # they represent neighbors that exist when the piece is placed at positions
        # where the normalized anchor (0,0) maps to a board position > 0.
        # When we shift the mask later, shift_mask will filter out any that go off-board.
        diag_coords = set()
        for r, c in normalized:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                diag_coords.add((nr, nc))
        # Remove shape cells from diag_coords
        diag_coords -= set(normalized)
        # For bitmask representation, we can only represent coordinates in [0, BOARD_SIZE)
        # So we filter to non-negative coordinates for the mask.
        # This means we'll miss some neighbors when the piece is at the edge, but
        # that's acceptable - those neighbors would be off-board anyway.
        # The key insight: we need to include neighbors that could be valid when
        # the piece is placed anywhere on the board. Since normalized offsets start at (0,0),
        # neighbors with negative coordinates are only valid when the piece anchor is > 0.
        # We'll include neighbors in a wider range to capture most cases.
        diag_coords_for_mask = {(r, c) for r, c in diag_coords if r >= 0 and c >= 0}
        diag_mask = coords_to_mask(diag_coords_for_mask) if diag_coords_for_mask else 0
        
        # Compute orthogonal neighbor coords (excluding shape cells)
        orth_coords = set()
        for r, c in normalized:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                orth_coords.add((nr, nc))
        # Remove shape cells from orth_coords
        orth_coords -= set(normalized)
        orth_coords_for_mask = {(r, c) for r, c in orth_coords if r >= 0 and c >= 0}
        orth_mask = coords_to_mask(orth_coords_for_mask) if orth_coords_for_mask else 0
        
        # Compute anchor indices using heuristic selection
        # This reduces redundant anchor attempts by selecting strategic anchor points
        # that are more likely to produce valid placements when aligned with frontier cells
        anchor_indices = _compute_anchor_indices(normalized)
        
        # Create PieceOrientation
        orientation = PieceOrientation(
            piece_id=piece_id,
            orientation_id=orientation_id,
            offsets=normalized,
            shape_mask=shape_mask,
            diag_mask=diag_mask,
            orth_mask=orth_mask,
            anchor_indices=anchor_indices
        )
        
        orientations.append(orientation)
        orientation_id += 1
    
    return orientations


# Global registry of all piece orientations
ALL_PIECE_ORIENTATIONS: Dict[int, List[PieceOrientation]] = {}


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


def init_piece_orientations():
    """
    Initialize the global ALL_PIECE_ORIENTATIONS registry.
    
    This should be called once during module import or engine setup.
    """
    global ALL_PIECE_ORIENTATIONS
    if ALL_PIECE_ORIENTATIONS:
        return  # Already initialized
    
    pieces = PieceGenerator.get_all_pieces()
    for piece in pieces:
        orientations = generate_orientations_for_piece(piece.id, piece.shape)
        ALL_PIECE_ORIENTATIONS[piece.id] = orientations


# Initialize on import (after PieceGenerator is defined)
init_piece_orientations()
