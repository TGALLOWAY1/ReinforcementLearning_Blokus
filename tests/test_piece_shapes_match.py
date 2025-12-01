"""
Tests to verify piece shapes match expected definitions, especially for
Pentomino F (ID 11) and Pentomino Y (ID 21).
"""

import unittest
import numpy as np
from engine.pieces import PieceGenerator


def print_piece_shape(piece):
    """
    Helper function to print a piece's shape as a list-of-lists for debugging.
    
    Args:
        piece: Piece object with a shape attribute
    """
    shape_list = piece.shape.tolist()
    print(f"\nPiece ID {piece.id} ({piece.name}):")
    print(f"  Shape: {shape_list}")
    print(f"  Size: {piece.size}")
    print(f"  Shape sum: {np.sum(piece.shape)}")
    return shape_list


class TestPieceShapes(unittest.TestCase):
    """Test piece shape definitions and validation."""
    
    def test_exactly_21_pieces(self):
        """Verify there are exactly 21 pieces."""
        pieces = PieceGenerator.get_all_pieces()
        self.assertEqual(len(pieces), 21, "Should have exactly 21 pieces")
    
    def test_piece_sizes(self):
        """Verify each piece's size equals np.sum(piece.shape) and is between 1 and 5."""
        pieces = PieceGenerator.get_all_pieces()
        
        for piece in pieces:
            shape_sum = np.sum(piece.shape)
            self.assertEqual(
                piece.size, 
                shape_sum,
                f"Piece {piece.id} ({piece.name}): size={piece.size} but shape sum={shape_sum}"
            )
            self.assertGreaterEqual(
                piece.size, 1,
                f"Piece {piece.id} ({piece.name}): size must be at least 1"
            )
            self.assertLessEqual(
                piece.size, 5,
                f"Piece {piece.id} ({piece.name}): size must be at most 5"
            )
    
    def test_pentomino_f_shape(self):
        """Verify Pentomino F (ID 11) has the correct F shape."""
        pieces = PieceGenerator.get_all_pieces()
        piece_11 = next(p for p in pieces if p.id == 11)
        
        # Expected F pentomino shape:
        # [0, 1, 1]
        # [1, 1, 0]
        # [0, 1, 0]
        expected_shape = np.array([
            [0, 1, 1],
            [1, 1, 0],
            [0, 1, 0]
        ])
        
        self.assertEqual(
            piece_11.name, 
            "Pentomino F",
            f"Piece 11 should be named 'Pentomino F', got '{piece_11.name}'"
        )
        self.assertEqual(
            piece_11.size, 
            5,
            f"Piece 11 should have size 5, got {piece_11.size}"
        )
        self.assertTrue(
            np.array_equal(piece_11.shape, expected_shape),
            f"Piece 11 shape mismatch.\n"
            f"Expected:\n{expected_shape}\n"
            f"Got:\n{piece_11.shape}\n"
            f"Shape as list: {piece_11.shape.tolist()}"
        )
    
    def test_pentomino_y_shape(self):
        """Verify Pentomino Y (ID 21) has the correct Y shape."""
        pieces = PieceGenerator.get_all_pieces()
        piece_21 = next(p for p in pieces if p.id == 21)
        
        # Expected Y pentomino shape:
        # [1, 0]
        # [1, 1]
        # [1, 0]
        # [1, 0]
        expected_shape = np.array([
            [1, 0],
            [1, 1],
            [1, 0],
            [1, 0]
        ])
        
        self.assertEqual(
            piece_21.name, 
            "Pentomino Y",
            f"Piece 21 should be named 'Pentomino Y', got '{piece_21.name}'"
        )
        self.assertEqual(
            piece_21.size, 
            5,
            f"Piece 21 should have size 5, got {piece_21.size}"
        )
        self.assertTrue(
            np.array_equal(piece_21.shape, expected_shape),
            f"Piece 21 shape mismatch.\n"
            f"Expected:\n{expected_shape}\n"
            f"Got:\n{piece_21.shape}\n"
            f"Shape as list: {piece_21.shape.tolist()}"
        )
    
    def test_all_pieces_have_valid_shapes(self):
        """Verify all pieces have valid 2D shapes with correct dimensions."""
        pieces = PieceGenerator.get_all_pieces()
        
        for piece in pieces:
            # Check shape is 2D
            self.assertEqual(
                piece.shape.ndim, 
                2,
                f"Piece {piece.id} ({piece.name}): shape must be 2D, got {piece.shape.ndim}D"
            )
            
            # Check shape has at least one row and one column
            rows, cols = piece.shape.shape
            self.assertGreater(rows, 0, f"Piece {piece.id} must have at least one row")
            self.assertGreater(cols, 0, f"Piece {piece.id} must have at least one column")
            
            # Check shape contains only 0s and 1s
            unique_values = np.unique(piece.shape)
            self.assertTrue(
                np.all(np.isin(unique_values, [0, 1])),
                f"Piece {piece.id} ({piece.name}): shape must contain only 0s and 1s, got {unique_values}"
            )


if __name__ == '__main__':
    unittest.main()

