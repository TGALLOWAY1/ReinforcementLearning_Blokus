"""
Bitboard utilities for Blokus board representation.

This module provides helper functions for converting between board coordinates
and bit masks, enabling efficient bit-level operations on the board state.
"""

from typing import List, Tuple, Optional, Iterable

# Board dimensions - imported from Board class to maintain consistency
# For now, we'll use the constant directly, but this could be made configurable
BOARD_WIDTH = 20
BOARD_HEIGHT = 20
NUM_CELLS = BOARD_WIDTH * BOARD_HEIGHT


def coord_to_index(row: int, col: int) -> int:
    """
    Convert board coordinates to a linear index.
    
    Args:
        row: Row coordinate (0-based)
        col: Column coordinate (0-based)
        
    Returns:
        Linear index in [0, NUM_CELLS)
    """
    return row * BOARD_WIDTH + col


def index_to_coord(index: int) -> Tuple[int, int]:
    """
    Convert a linear index back to board coordinates.
    
    Args:
        index: Linear index in [0, NUM_CELLS)
        
    Returns:
        Tuple of (row, col)
    """
    row = index // BOARD_WIDTH
    col = index % BOARD_WIDTH
    return (row, col)


def coord_to_bit(row: int, col: int) -> int:
    """
    Convert board coordinates to a bit mask with a single bit set.
    
    Args:
        row: Row coordinate (0-based)
        col: Column coordinate (0-based)
        
    Returns:
        Integer with bit set at position corresponding to (row, col)
    """
    index = coord_to_index(row, col)
    return 1 << index


def coords_to_mask(coords: Iterable[Tuple[int, int]]) -> int:
    """
    Convert a collection of coordinates to a bitmask.
    
    Args:
        coords: Iterable of (row, col) tuples
        
    Returns:
        Bitmask with bits set for each coordinate
    """
    mask = 0
    for row, col in coords:
        mask |= coord_to_bit(row, col)
    return mask


def mask_to_coords(mask: int) -> List[Tuple[int, int]]:
    """
    Convert a bitmask back into a list of coordinates.
    
    Useful for debugging and testing.
    
    Args:
        mask: Bitmask with bits set
        
    Returns:
        List of (row, col) tuples for all set bits
    """
    coords = []
    index = 0
    while mask != 0:
        if mask & 1:
            coords.append(index_to_coord(index))
        mask >>= 1
        index += 1
    return coords


def shift_mask(mask: int, d_row: int, d_col: int, strict: bool = True) -> Optional[int]:
    """
    Shift all bits in a mask by (d_row, d_col).
    
    Args:
        mask: Bitmask to shift
        d_row: Row offset (can be negative)
        d_col: Column offset (can be negative)
        strict: If True, returns None if ANY bit would go off-board.
                If False, filters out off-board bits and returns the remaining mask.
        
    Returns:
        Shifted bitmask, or None if strict=True and any bit would go off-board
    """
    # Convert mask to coords, shift them, and rebuild mask
    coords = mask_to_coords(mask)
    shifted_coords = []
    
    for row, col in coords:
        new_row = row + d_row
        new_col = col + d_col
        
        # Check bounds
        if new_row < 0 or new_row >= BOARD_HEIGHT:
            if strict:
                return None
            else:
                continue  # Skip this coordinate
        if new_col < 0 or new_col >= BOARD_WIDTH:
            if strict:
                return None
            else:
                continue  # Skip this coordinate
        
        # Check for row wrapping (if we shifted across a row boundary incorrectly)
        # This shouldn't happen with our bounds check, but we verify the index is still valid
        new_index = coord_to_index(new_row, new_col)
        if new_index < 0 or new_index >= NUM_CELLS:
            if strict:
                return None
            else:
                continue  # Skip this coordinate
        
        shifted_coords.append((new_row, new_col))
    
    return coords_to_mask(shifted_coords) if shifted_coords else 0

