


"""
Comprehensive tests for the Blokus game engine.
"""

import unittest

import numpy as np

from engine.board import Board, Player, Position
from engine.game import BlokusGame
from engine.move_generator import LegalMoveGenerator, Move
from engine.pieces import PieceGenerator


class TestBoard(unittest.TestCase):
    """Test the Board class."""
    
    def test_board_initialization(self):
        """Test board initialization."""
        board = Board()
        self.assertEqual(board.grid.shape, (20, 20))
        self.assertTrue(np.all(board.grid == 0))
        self.assertEqual(board.current_player, Player.RED)
        self.assertEqual(board.move_count, 0)
        self.assertFalse(board.game_over)
    
    def test_start_corners(self):
        """Test that start corners are correctly set."""
        board = Board()
        assert board.player_start_corners[Player.RED] == Position(0, 0)
        assert board.player_start_corners[Player.BLUE] == Position(0, 19)
        assert board.player_start_corners[Player.YELLOW] == Position(19, 19)
        assert board.player_start_corners[Player.GREEN] == Position(19, 0)
    
    def test_position_validation(self):
        """Test position validation."""
        board = Board()
        
        # Valid positions
        assert board.is_valid_position(Position(0, 0))
        assert board.is_valid_position(Position(19, 19))
        assert board.is_valid_position(Position(10, 10))
        
        # Invalid positions
        assert not board.is_valid_position(Position(-1, 0))
        assert not board.is_valid_position(Position(0, -1))
        assert not board.is_valid_position(Position(20, 0))
        assert not board.is_valid_position(Position(0, 20))
    
    def test_cell_operations(self):
        """Test cell get/set operations."""
        board = Board()
        pos = Position(5, 5)
        
        # Initially empty
        assert board.is_empty(pos)
        assert board.get_cell(pos) == 0
        
        # Set and get
        board.set_cell(pos, Player.RED.value)
        assert board.get_cell(pos) == Player.RED.value
        assert not board.is_empty(pos)
        assert board.get_player_at(pos) == Player.RED
    
    def test_adjacent_positions(self):
        """Test adjacent position calculations."""
        board = Board()
        pos = Position(5, 5)
        
        # Edge adjacent (4 positions)
        edge_adjacent = board.get_edge_adjacent_positions(pos)
        assert len(edge_adjacent) == 4
        expected_edge = [Position(4, 5), Position(6, 5), Position(5, 4), Position(5, 6)]
        assert set(edge_adjacent) == set(expected_edge)
        
        # Corner adjacent (4 positions)
        corner_adjacent = board.get_corner_adjacent_positions(pos)
        assert len(corner_adjacent) == 4
        expected_corner = [Position(4, 4), Position(4, 6), Position(6, 4), Position(6, 6)]
        assert set(corner_adjacent) == set(expected_corner)
        
        # All adjacent (8 positions)
        all_adjacent = board.get_adjacent_positions(pos)
        assert len(all_adjacent) == 8
        assert set(all_adjacent) == set(edge_adjacent + corner_adjacent)


class TestPiecePlacement:
    """Test piece placement rules."""
    
    def test_first_move_corner_rule(self):
        """Test that first move must cover player's start corner."""
        board = Board()
        
        # Valid first move for RED (covers corner 0,0)
        valid_positions = [Position(0, 0), Position(0, 1)]
        assert board.can_place_piece(valid_positions, Player.RED)
        
        # Invalid first move for RED (doesn't cover corner 0,0)
        invalid_positions = [Position(1, 1), Position(1, 2)]
        assert not board.can_place_piece(invalid_positions, Player.RED)
    
    def test_adjacency_rules(self):
        """Test adjacency rules for piece placement."""
        board = Board()
        
        # Place first piece for RED
        first_positions = [Position(0, 0), Position(0, 1)]
        assert board.place_piece(first_positions, Player.RED, 1)
        
        # Valid second move (touches at corner)
        valid_positions = [Position(1, 1), Position(1, 2)]
        assert board.can_place_piece(valid_positions, Player.RED)
        
        # Invalid second move (touches at edge)
        invalid_positions = [Position(0, 2), Position(0, 3)]
        assert not board.can_place_piece(invalid_positions, Player.RED)
    
    def test_edge_adjacency_rule(self):
        """Test that pieces of same color cannot touch edge-to-edge."""
        board = Board()
        
        # Place first piece
        first_positions = [Position(0, 0), Position(0, 1)]
        board.place_piece(first_positions, Player.RED, 1)
        
        # Try to place piece that would touch edge-to-edge
        invalid_positions = [Position(0, 2), Position(0, 3)]
        assert not board.can_place_piece(invalid_positions, Player.RED)
        
        # Try to place piece that touches only at corner
        valid_positions = [Position(1, 1), Position(1, 2)]
        assert board.can_place_piece(valid_positions, Player.RED)
    
    def test_corner_adjacency_rule(self):
        """Test that pieces of same color can touch at corners."""
        board = Board()
        
        # Place first piece
        first_positions = [Position(0, 0), Position(0, 1)]
        board.place_piece(first_positions, Player.RED, 1)
        
        # Place second piece that touches at corner
        second_positions = [Position(1, 1), Position(1, 2)]
        assert board.can_place_piece(second_positions, Player.RED, 2)
        
        # Place third piece that touches second piece at corner
        third_positions = [Position(2, 2), Position(2, 3)]
        assert board.can_place_piece(third_positions, Player.RED, 3)


class TestPieceDefinitions:
    """Test piece definitions and orientations."""
    
    def test_piece_generation(self):
        """Test that all 21 pieces are generated correctly."""
        pieces = PieceGenerator.get_all_pieces()
        assert len(pieces) == 21
        
        # Check that all pieces have unique IDs
        piece_ids = [piece.id for piece in pieces]
        assert len(set(piece_ids)) == 21
        assert set(piece_ids) == set(range(1, 22))
    
    def test_piece_sizes(self):
        """Test that pieces have correct sizes."""
        pieces = PieceGenerator.get_all_pieces()
        
        # Check specific pieces
        monomino = next(p for p in pieces if p.id == 1)
        assert monomino.size == 1
        
        domino = next(p for p in pieces if p.id == 2)
        assert domino.size == 2
        
        # Check that all pieces have correct sizes
        for piece in pieces:
            assert piece.size == np.sum(piece.shape)
    
    def test_piece_orientations(self):
        """Test piece rotations and reflections."""
        pieces = PieceGenerator.get_all_pieces()
        
        for piece in pieces:
            orientations = PieceGenerator.get_piece_rotations_and_reflections(piece)
            
            # Should have at least 1 orientation
            assert len(orientations) >= 1
            
            # All orientations should have same size
            for orientation in orientations:
                assert np.sum(orientation) == piece.size
    
    def test_specific_pieces(self):
        """Test specific piece shapes."""
        pieces = PieceGenerator.get_all_pieces()
        
        # Test monomino
        monomino = next(p for p in pieces if p.id == 1)
        assert np.array_equal(monomino.shape, np.array([[1]]))
        
        # Test domino
        domino = next(p for p in pieces if p.id == 2)
        assert np.array_equal(domino.shape, np.array([[1, 1]]))
        
        # Test tetromino O (2x2 square)
        tetromino_o = next(p for p in pieces if p.id == 6)
        expected_shape = np.array([[1, 1], [1, 1]])
        assert np.array_equal(tetromino_o.shape, expected_shape)


class TestMoveGenerator:
    """Test the legal move generator."""
    
    def test_move_generator_initialization(self):
        """Test move generator initialization."""
        generator = LegalMoveGenerator()
        assert len(generator.all_pieces) == 21
        assert len(generator.piece_orientations_cache) == 21
    
    def test_legal_moves_empty_board(self):
        """Test legal moves on empty board."""
        board = Board()
        generator = LegalMoveGenerator()
        
        # First player (RED) should have moves that cover corner (0,0)
        moves = generator.get_legal_moves(board, Player.RED)
        assert len(moves) > 0
        
        # All moves should be legal
        for move in moves:
            assert generator.is_move_legal(board, Player.RED, move)
    
    def test_legal_moves_after_first_move(self):
        """Test legal moves after first move is made."""
        board = Board()
        generator = LegalMoveGenerator()
        
        # Make first move
        first_moves = generator.get_legal_moves(board, Player.RED)
        assert len(first_moves) > 0
        
        # Place first piece
        first_move = first_moves[0]
        orientations = generator.piece_orientations_cache[first_move.piece_id]
        positions = first_move.get_positions(orientations)
        board.place_piece(positions, Player.RED, first_move.piece_id)
        
        # Check that piece was placed
        assert first_move.piece_id in board.player_pieces_used[Player.RED]
        
        # Get moves for next player
        next_moves = generator.get_legal_moves(board, Player.BLUE)
        assert len(next_moves) > 0
    
    def test_move_validation(self):
        """Test move validation."""
        board = Board()
        generator = LegalMoveGenerator()
        
        # Get a legal move
        moves = generator.get_legal_moves(board, Player.RED)
        assert len(moves) > 0
        
        legal_move = moves[0]
        assert generator.is_move_legal(board, Player.RED, legal_move)
        
        # Create an invalid move
        invalid_move = Move(1, 0, -1, -1)  # Out of bounds
        assert not generator.is_move_legal(board, Player.RED, invalid_move)


class TestScoring:
    """Test scoring system."""
    
    def test_basic_scoring(self):
        """Test basic scoring rules."""
        board = Board()
        
        # Place some pieces
        positions1 = [Position(0, 0), Position(0, 1)]
        board.place_piece(positions1, Player.RED, 1)
        
        positions2 = [Position(1, 1), Position(1, 2)]
        board.place_piece(positions2, Player.RED, 2)
        
        # Check score
        score = board.get_score(Player.RED)
        assert score >= 4  # At least 4 squares covered
    
    def test_bonus_scoring(self):
        """Test bonus scoring for using all pieces."""
        board = Board()
        
        # Simulate using all pieces
        for i in range(1, 22):
            board.player_pieces_used[Player.RED].add(i)
        
        score = board.get_score(Player.RED)
        assert score >= 15  # Bonus for using all pieces
    
    def test_corner_bonus(self):
        """Test corner control bonus."""
        board = Board()
        
        # Place piece in corner
        positions = [Position(0, 0)]
        board.place_piece(positions, Player.RED, 1)
        
        score = board.get_score(Player.RED)
        assert score >= 6  # 1 for square + 5 for corner bonus


class TestGameEngine:
    """Test the main game engine."""
    
    def test_game_initialization(self):
        """Test game initialization."""
        game = BlokusGame()
        assert game.board.current_player == Player.RED
        assert game.move_count == 0
        assert not game.is_game_over()
        assert game.winner is None
    
    def test_making_moves(self):
        """Test making moves in the game."""
        game = BlokusGame()
        
        # Get legal moves
        moves = game.get_legal_moves()
        assert len(moves) > 0
        
        # Make first move
        first_move = moves[0]
        success = game.make_move(first_move)
        assert success
        
        # Check game state
        assert game.move_count == 1
        assert game.board.current_player != Player.RED  # Should have switched
    
    def test_invalid_moves(self):
        """Test that invalid moves are rejected."""
        game = BlokusGame()
        
        # Create invalid move
        invalid_move = Move(1, 0, -1, -1)
        success = game.make_move(invalid_move)
        assert not success
    
    def test_game_state(self):
        """Test game state tracking."""
        game = BlokusGame()
        
        state = game.get_game_state()
        assert 'board' in state
        assert 'current_player' in state
        assert 'move_count' in state
        assert 'game_over' in state
        assert 'scores' in state
    
    def test_game_reset(self):
        """Test game reset functionality."""
        game = BlokusGame()
        
        # Make a move
        moves = game.get_legal_moves()
        if moves:
            game.make_move(moves[0])
        
        # Reset game
        game.reset_game()
        assert game.move_count == 0
        assert game.board.current_player == Player.RED
        assert game.winner is None


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_complete_game_flow(self):
        """Test a complete game flow."""
        game = BlokusGame()
        
        # Make several moves
        for _ in range(10):  # Make 10 moves
            moves = game.get_legal_moves()
            if not moves:
                break
            
            move = moves[0]
            success = game.make_move(move)
            assert success
        
        # Check that game state is consistent
        state = game.get_game_state()
        assert state['move_count'] > 0
    
    def test_piece_exhaustion(self):
        """Test behavior when pieces are exhausted."""
        game = BlokusGame()
        
        # Simulate using all pieces for a player
        for i in range(1, 22):
            game.board.player_pieces_used[Player.RED].add(i)
        
        # Check that player has no legal moves
        moves = game.get_legal_moves(Player.RED)
        assert len(moves) == 0
        
        # Check that player cannot move
        assert not game.can_player_move(Player.RED)
    
    def test_scoring_consistency(self):
        """Test that scoring is consistent."""
        game = BlokusGame()
        
        # Make some moves
        for _ in range(5):
            moves = game.get_legal_moves()
            if moves:
                game.make_move(moves[0])
        
        # Check that scores are consistent
        scores = {}
        for player in Player:
            scores[player] = game.get_score(player)
        
        # All scores should be non-negative
        for score in scores.values():
            assert score >= 0


if __name__ == "__main__":
    pytest.main([__file__])
