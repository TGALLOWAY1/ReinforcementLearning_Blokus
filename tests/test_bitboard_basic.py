"""
Basic tests for bitboard functionality and consistency.
"""

import unittest
from engine.board import Board, Player, Position
from engine.bitboard import (
    coord_to_index, index_to_coord, coord_to_bit, coords_to_mask,
    mask_to_coords, shift_mask, NUM_CELLS
)


class TestBitboardUtilities(unittest.TestCase):
    """Test bitboard utility functions."""
    
    def test_coord_to_index(self):
        """Test coordinate to index conversion."""
        self.assertEqual(coord_to_index(0, 0), 0)
        self.assertEqual(coord_to_index(0, 1), 1)
        self.assertEqual(coord_to_index(1, 0), 20)
        self.assertEqual(coord_to_index(19, 19), 399)
    
    def test_index_to_coord(self):
        """Test index to coordinate conversion."""
        self.assertEqual(index_to_coord(0), (0, 0))
        self.assertEqual(index_to_coord(1), (0, 1))
        self.assertEqual(index_to_coord(20), (1, 0))
        self.assertEqual(index_to_coord(399), (19, 19))
    
    def test_coord_to_bit(self):
        """Test coordinate to bit conversion."""
        self.assertEqual(coord_to_bit(0, 0), 1)
        self.assertEqual(coord_to_bit(0, 1), 2)
        self.assertEqual(coord_to_bit(1, 0), 1 << 20)
        self.assertEqual(coord_to_bit(19, 19), 1 << 399)
    
    def test_coords_to_mask(self):
        """Test coordinates to mask conversion."""
        coords = [(0, 0), (0, 1), (1, 0)]
        mask = coords_to_mask(coords)
        
        # Check that all bits are set
        self.assertNotEqual(mask, 0)
        self.assertTrue(mask & coord_to_bit(0, 0))
        self.assertTrue(mask & coord_to_bit(0, 1))
        self.assertTrue(mask & coord_to_bit(1, 0))
    
    def test_mask_to_coords(self):
        """Test mask to coordinates conversion."""
        coords = [(0, 0), (0, 1), (1, 0)]
        mask = coords_to_mask(coords)
        result = mask_to_coords(mask)
        
        # Should contain all original coords (order may differ)
        self.assertEqual(set(result), set(coords))
    
    def test_shift_mask(self):
        """Test mask shifting."""
        # Shift a single bit
        mask = coord_to_bit(5, 5)
        shifted = shift_mask(mask, 1, 1)
        self.assertIsNotNone(shifted)
        self.assertEqual(mask_to_coords(shifted), [(6, 6)])
        
        # Shift multiple bits
        coords = [(5, 5), (5, 6)]
        mask = coords_to_mask(coords)
        shifted = shift_mask(mask, 1, 0)
        self.assertIsNotNone(shifted)
        self.assertEqual(set(mask_to_coords(shifted)), {(6, 5), (6, 6)})
        
        # Shift off board (should return None)
        mask = coord_to_bit(19, 19)
        shifted = shift_mask(mask, 1, 0)
        self.assertIsNone(shifted)
        
        # Shift negative
        mask = coord_to_bit(5, 5)
        shifted = shift_mask(mask, -1, -1)
        self.assertIsNotNone(shifted)
        self.assertEqual(mask_to_coords(shifted), [(4, 4)])


class TestBitboardConsistency(unittest.TestCase):
    """Test bitboard consistency with grid state."""
    
    def test_empty_board_bitboard_zero(self):
        """Test that new board has zero bitboard state."""
        board = Board()
        
        self.assertEqual(board.occupied_bits, 0, "New board should have no occupied bits")
        for player in Player:
            self.assertEqual(board.player_bits[player], 0, 
                           f"{player.name} should have no bits set on new board")
        
        # Consistency check should pass
        board.assert_bitboard_consistent()
    
    def test_single_piece_placement_updates_bitboard(self):
        """Test that placing a single piece updates bitboard correctly."""
        board = Board()
        
        # Place a single square piece (monomino) at RED's starting corner
        positions = [Position(0, 0)]
        success = board.place_piece(positions, Player.RED, 1)
        self.assertTrue(success, "Should be able to place piece")
        
        # Check that occupied_bits has bit set
        self.assertNotEqual(board.occupied_bits, 0, "occupied_bits should be non-zero")
        self.assertTrue(board.occupied_bits & coord_to_bit(0, 0), 
                       "Bit for (0,0) should be set in occupied_bits")
        
        # Check that player_bits[RED] has bit set
        self.assertNotEqual(board.player_bits[Player.RED], 0, 
                           "player_bits[RED] should be non-zero")
        self.assertTrue(board.player_bits[Player.RED] & coord_to_bit(0, 0),
                       "Bit for (0,0) should be set in player_bits[RED]")
        
        # Other players should have no bits
        for player in [Player.BLUE, Player.YELLOW, Player.GREEN]:
            self.assertEqual(board.player_bits[player], 0,
                           f"{player.name} should have no bits set")
        
        # Consistency check should pass
        board.assert_bitboard_consistent()
    
    def test_multiple_pieces_bitboard_consistency(self):
        """Test bitboard consistency after placing multiple pieces."""
        board = Board()
        
        # Place first piece for RED
        positions1 = [Position(0, 0), Position(0, 1)]
        board.place_piece(positions1, Player.RED, 2)
        
        # Place first piece for BLUE
        positions2 = [Position(0, 19), Position(0, 18)]
        board.place_piece(positions2, Player.BLUE, 2)
        
        # Check consistency
        board.assert_bitboard_consistent()
        
        # Verify bits are set correctly
        red_mask = coords_to_mask([(0, 0), (0, 1)])
        blue_mask = coords_to_mask([(0, 19), (0, 18)])
        
        self.assertEqual(board.player_bits[Player.RED], red_mask)
        self.assertEqual(board.player_bits[Player.BLUE], blue_mask)
        self.assertEqual(board.occupied_bits, red_mask | blue_mask)
    
    def test_bitboard_consistency_after_multiple_moves(self):
        """Test bitboard consistency after multiple moves."""
        board = Board()
        from engine.move_generator import LegalMoveGenerator
        generator = LegalMoveGenerator()
        
        # Make several moves
        for _ in range(6):
            player = board.current_player
            moves = generator.get_legal_moves(board, player)
            if not moves:
                break
            
            move = moves[0]
            orientations = generator.piece_orientations_cache[move.piece_id]
            positions = move.get_positions(orientations)
            board.place_piece(positions, player, move.piece_id)
            
            # Check consistency after each move
            board.assert_bitboard_consistent()
    
    def test_bitboard_copy_consistency(self):
        """Test that copied board maintains bitboard state."""
        board = Board()
        
        # Place some pieces
        positions1 = [Position(0, 0), Position(0, 1)]
        board.place_piece(positions1, Player.RED, 2)
        
        positions2 = [Position(0, 19), Position(0, 18)]
        board.place_piece(positions2, Player.BLUE, 2)
        
        # Copy board
        copied = board.copy()
        
        # Check that bitboard state is copied
        self.assertEqual(copied.occupied_bits, board.occupied_bits)
        for player in Player:
            self.assertEqual(copied.player_bits[player], board.player_bits[player])
        
        # Check consistency of copied board
        copied.assert_bitboard_consistent()


if __name__ == "__main__":
    unittest.main()

