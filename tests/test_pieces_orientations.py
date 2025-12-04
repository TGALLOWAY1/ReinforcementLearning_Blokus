"""
Tests for piece orientation precomputation and correctness.
"""

import unittest
from engine.pieces import (
    PieceGenerator, PieceOrientation, ALL_PIECE_ORIENTATIONS,
    generate_orientations_for_piece, normalize_offsets, shape_to_offsets
)
from engine.bitboard import mask_to_coords, coords_to_mask


class TestPieceOrientations(unittest.TestCase):
    """Test piece orientation precomputation."""
    
    def test_all_pieces_have_orientations(self):
        """Test that all pieces have at least one orientation."""
        pieces = PieceGenerator.get_all_pieces()
        
        for piece in pieces:
            self.assertIn(piece.id, ALL_PIECE_ORIENTATIONS,
                         f"Piece {piece.id} ({piece.name}) missing from ALL_PIECE_ORIENTATIONS")
            orientations = ALL_PIECE_ORIENTATIONS[piece.id]
            self.assertGreater(len(orientations), 0,
                             f"Piece {piece.id} ({piece.name}) has no orientations")
    
    def test_no_duplicate_orientations(self):
        """Test that there are no duplicate orientation shapes for each piece."""
        for piece_id, orientations in ALL_PIECE_ORIENTATIONS.items():
            seen_shapes = set()
            for orientation in orientations:
                # Use sorted offsets tuple as shape signature
                shape_key = tuple(sorted(orientation.offsets))
                self.assertNotIn(shape_key, seen_shapes,
                               f"Piece {piece_id} has duplicate orientation {orientation.orientation_id}")
                seen_shapes.add(shape_key)
    
    def test_orientation_area_consistency(self):
        """Test that all orientations of a piece have the same area (number of cells)."""
        for piece_id, orientations in ALL_PIECE_ORIENTATIONS.items():
            if not orientations:
                continue
            
            expected_size = len(orientations[0].offsets)
            for orientation in orientations:
                self.assertEqual(len(orientation.offsets), expected_size,
                               f"Piece {piece_id} orientation {orientation.orientation_id} "
                               f"has inconsistent size: {len(orientation.offsets)} vs {expected_size}")
    
    def test_domino_orientations(self):
        """Test that domino (2 cells) has expected number of orientations."""
        domino_id = 2
        orientations = ALL_PIECE_ORIENTATIONS[domino_id]
        
        # Domino should have 2 orientations (horizontal and vertical)
        self.assertEqual(len(orientations), 2,
                        f"Domino should have 2 orientations, got {len(orientations)}")
        
        # Check that orientations are different
        offsets_0 = tuple(sorted(orientations[0].offsets))
        offsets_1 = tuple(sorted(orientations[1].offsets))
        self.assertNotEqual(offsets_0, offsets_1, "Domino orientations should be different")
    
    def test_monomino_orientations(self):
        """Test that monomino (1 cell) has exactly 1 orientation."""
        monomino_id = 1
        orientations = ALL_PIECE_ORIENTATIONS[monomino_id]
        
        self.assertEqual(len(orientations), 1,
                        f"Monomino should have 1 orientation, got {len(orientations)}")
        self.assertEqual(orientations[0].offsets, [(0, 0)],
                        "Monomino should have offset at (0, 0)")
    
    def test_orientation_masks_non_zero(self):
        """Test that shape_mask is non-zero and properly computed."""
        for piece_id, orientations in ALL_PIECE_ORIENTATIONS.items():
            for orientation in orientations:
                self.assertNotEqual(orientation.shape_mask, 0,
                                  f"Piece {piece_id} orientation {orientation.orientation_id} "
                                  f"should have non-zero shape_mask")
                
                # Verify shape_mask matches offsets
                mask_coords = mask_to_coords(orientation.shape_mask)
                self.assertEqual(set(mask_coords), set(orientation.offsets),
                               f"Piece {piece_id} orientation {orientation.orientation_id} "
                               f"shape_mask doesn't match offsets")
    
    def test_diag_mask_no_overlap_with_shape(self):
        """Test that diag_mask doesn't overlap with shape_mask."""
        for piece_id, orientations in ALL_PIECE_ORIENTATIONS.items():
            for orientation in orientations:
                overlap = orientation.shape_mask & orientation.diag_mask
                self.assertEqual(overlap, 0,
                               f"Piece {piece_id} orientation {orientation.orientation_id} "
                               f"diag_mask overlaps with shape_mask")
    
    def test_orth_mask_no_overlap_with_shape(self):
        """Test that orth_mask doesn't overlap with shape_mask."""
        for piece_id, orientations in ALL_PIECE_ORIENTATIONS.items():
            for orientation in orientations:
                overlap = orientation.shape_mask & orientation.orth_mask
                self.assertEqual(overlap, 0,
                               f"Piece {piece_id} orientation {orientation.orientation_id} "
                               f"orth_mask overlaps with shape_mask")
    
    def test_offsets_normalized(self):
        """Test that all orientations have normalized offsets (min_row=0, min_col=0)."""
        for piece_id, orientations in ALL_PIECE_ORIENTATIONS.items():
            for orientation in orientations:
                if not orientation.offsets:
                    continue
                
                rows = [r for r, c in orientation.offsets]
                cols = [c for r, c in orientation.offsets]
                
                self.assertEqual(min(rows), 0,
                               f"Piece {piece_id} orientation {orientation.orientation_id} "
                               f"offsets not normalized: min_row={min(rows)}")
                self.assertEqual(min(cols), 0,
                               f"Piece {piece_id} orientation {orientation.orientation_id} "
                               f"offsets not normalized: min_col={min(cols)}")
    
    def test_generate_orientations_function(self):
        """Test generate_orientations_for_piece function directly."""
        from engine.pieces import PieceGenerator
        pieces = PieceGenerator.get_all_pieces()
        
        # Test with monomino
        monomino = next(p for p in pieces if p.id == 1)
        orientations = generate_orientations_for_piece(monomino.id, monomino.shape)
        
        self.assertEqual(len(orientations), 1)
        self.assertEqual(orientations[0].offsets, [(0, 0)])
        self.assertNotEqual(orientations[0].shape_mask, 0)
    
    def test_normalize_offsets(self):
        """Test normalize_offsets function."""
        offsets = [(2, 3), (2, 4), (3, 3)]
        normalized = normalize_offsets(offsets)
        
        self.assertEqual(normalized, [(0, 0), (0, 1), (1, 0)])
        
        # Test with already normalized
        offsets2 = [(0, 0), (0, 1)]
        normalized2 = normalize_offsets(offsets2)
        self.assertEqual(normalized2, [(0, 0), (0, 1)])
    
    def test_shape_to_offsets(self):
        """Test shape_to_offsets function."""
        import numpy as np
        shape = np.array([[1, 1], [1, 0]])
        offsets = shape_to_offsets(shape)
        
        self.assertEqual(set(offsets), {(0, 0), (0, 1), (1, 0)})


if __name__ == "__main__":
    unittest.main()

