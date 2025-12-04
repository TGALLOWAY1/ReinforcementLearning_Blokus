"""
Basic tests for frontier tracking functionality.
"""

import random
import unittest
from engine.board import Board, Player, Position
from engine.move_generator import LegalMoveGenerator


def assert_frontier_invariants(board: Board, player: Player, test_case: unittest.TestCase):
    """
    Assert that frontier invariants hold for a given board and player.
    
    Invariants:
    1. Every frontier cell is empty
    2. Every frontier cell is diagonally adjacent to at least one of the player's cells
    3. No frontier cell is orthogonally adjacent to any of the player's cells
    
    Args:
        board: Board to check
        player: Player to check frontier for
        test_case: TestCase instance for assertions
    """
    frontier = board.get_frontier(player)
    player_value = player.value
    grid = board.grid
    is_first_move = board.player_first_move[player]
    start_corner = board.player_start_corners[player]
    start_corner_tuple = (start_corner.row, start_corner.col)
    
    for r, c in frontier:
        # Invariant 1: Every frontier cell is empty
        test_case.assertEqual(grid[r, c], 0, 
                             f"Frontier cell ({r}, {c}) should be empty")
        
        # Special case: starting corner on first move
        if is_first_move and (r, c) == start_corner_tuple:
            continue
        
        # Invariant 2: Every frontier cell is diagonally adjacent to player's cells
        has_diagonal = False
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < board.SIZE and 0 <= nc < board.SIZE:
                if grid[nr, nc] == player_value:
                    has_diagonal = True
                    break
        
        test_case.assertTrue(has_diagonal,
                           f"Frontier cell ({r}, {c}) should be diagonally adjacent to {player.name}'s pieces")
        
        # Invariant 3: No frontier cell is orthogonally adjacent to player's cells
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < board.SIZE and 0 <= nc < board.SIZE:
                test_case.assertNotEqual(grid[nr, nc], player_value,
                                        f"Frontier cell ({r}, {c}) should not be orthogonally adjacent to {player.name}'s pieces")


class TestFrontierInitialization(unittest.TestCase):
    """Test frontier initialization."""
    
    def test_frontier_initialization(self):
        """Test that frontiers are properly initialized for all players."""
        board = Board()
        
        # Check that each player has their starting corner in their frontier
        for player in Player:
            frontier = board.get_frontier(player)
            start_corner = board.player_start_corners[player]
            start_corner_tuple = (start_corner.row, start_corner.col)
            
            # Starting corner should be in frontier
            self.assertIn(start_corner_tuple, frontier, 
                         f"{player.name} starting corner {start_corner_tuple} not in frontier")
            
            # Frontier should contain exactly one cell (the starting corner)
            self.assertEqual(len(frontier), 1, 
                           f"{player.name} should have exactly 1 frontier cell initially")
            
            # Starting corner should be empty
            self.assertTrue(board.is_empty(start_corner),
                           f"{player.name} starting corner should be empty")
    
    def test_frontier_initialization_after_reset(self):
        """Test that frontiers are reinitialized after a reset."""
        board = Board()
        
        # Place a piece to change frontier
        positions = [Position(0, 0), Position(0, 1)]
        board.place_piece(positions, Player.RED, 1)
        
        # Reset should reinitialize frontiers
        board = Board()  # Create new board (simulating reset)
        
        # Check that RED's frontier is back to just the starting corner
        red_frontier = board.get_frontier(Player.RED)
        self.assertEqual(len(red_frontier), 1)
        self.assertIn((0, 0), red_frontier)


class TestFrontierUpdateAfterSinglePiece(unittest.TestCase):
    """Test frontier updates after placing a single piece."""
    
    def test_frontier_update_after_single_piece(self):
        """Test that frontier is correctly updated after placing a single square piece."""
        board = Board()
        
        # Place a single square piece (monomino) at RED's starting corner
        positions = [Position(0, 0)]
        success = board.place_piece(positions, Player.RED, 1)
        self.assertTrue(success, "Should be able to place piece at starting corner")
        
        # Get updated frontier
        frontier = board.get_frontier(Player.RED)
        
        # The placed cell should NOT be in frontier
        self.assertNotIn((0, 0), frontier, "Placed cell should not be in frontier")
        
        # Diagonal neighbors should be in frontier (if empty and not orth adjacent)
        # Diagonal neighbors of (0, 0): (1, 1) only (others are out of bounds)
        # (1, 1) is diagonally adjacent to (0, 0) and not orthogonally adjacent
        self.assertIn((1, 1), frontier, "Diagonal neighbor (1, 1) should be in frontier")
        
        # Orthogonal neighbors should NOT be in frontier
        # Orthogonal neighbors of (0, 0): (0, 1), (1, 0)
        self.assertNotIn((0, 1), frontier, "Orthogonal neighbor (0, 1) should not be in frontier")
        self.assertNotIn((1, 0), frontier, "Orthogonal neighbor (1, 0) should not be in frontier")
        
        # Verify consistency
        self.assertTrue(board._verify_frontier_consistency(Player.RED),
                       "Frontier should be consistent after update")
    
    def test_frontier_update_after_two_square_piece(self):
        """Test frontier update after placing a two-square piece (domino)."""
        board = Board()
        
        # Place a domino at RED's starting corner: (0,0) and (0,1)
        positions = [Position(0, 0), Position(0, 1)]
        success = board.place_piece(positions, Player.RED, 2)
        self.assertTrue(success)
        
        frontier = board.get_frontier(Player.RED)
        
        # Placed cells should NOT be in frontier
        self.assertNotIn((0, 0), frontier)
        self.assertNotIn((0, 1), frontier)
        
        # Diagonal neighbors of (0,0): (1, 1) - but (1,1) is orthogonally adjacent to (0,1), so NOT in frontier
        # Diagonal neighbors of (0,1): (1, 0), (1, 2)
        # (1, 0) is orthogonally adjacent to (0, 0), so NOT in frontier
        # (1, 1) is diagonally adjacent to both (0,0) and (0,1), but orth adjacent to (0,1), so NOT in frontier
        # (1, 2) is diagonally adjacent to (0,1) and not orth adjacent, should be in frontier
        self.assertNotIn((1, 1), frontier, "(1,1) should not be in frontier (orthogonal to (0,1))")
        self.assertIn((1, 2), frontier, "(1,2) should be in frontier (diagonal to (0,1), not orth adjacent)")
        self.assertNotIn((1, 0), frontier, "(1,0) should not be in frontier (orthogonal to (0,0))")
        
        # Orthogonal neighbors should NOT be in frontier
        self.assertNotIn((0, 2), frontier, "(0,2) should not be in frontier (orthogonal to (0,1))")
        
        # Verify consistency
        self.assertTrue(board._verify_frontier_consistency(Player.RED))


class TestFrontierRebuildMatchesIncremental(unittest.TestCase):
    """Test that full recompute matches incremental updates."""
    
    def test_frontier_rebuild_matches_incremental(self):
        """Test that debug_rebuild_frontier correctly identifies and fixes mismatches."""
        board = Board()
        
        # Place a piece
        positions = [Position(0, 0), Position(0, 1)]
        board.place_piece(positions, Player.RED, 2)
        
        # Rebuild should match (no mismatch)
        match = board.debug_rebuild_frontier(Player.RED)
        self.assertTrue(match, "Frontier should match after first move")
        
        # Place another piece
        positions2 = [Position(1, 1), Position(1, 2)]
        board.place_piece(positions2, Player.RED, 3)
        
        # Rebuild should still match
        match = board.debug_rebuild_frontier(Player.RED)
        self.assertTrue(match, "Frontier should match after second move")
        
        # Verify consistency
        self.assertTrue(board._verify_frontier_consistency(Player.RED))
    
    def test_frontier_rebuild_after_multiple_moves(self):
        """Test frontier rebuild after multiple moves for different players."""
        board = Board()
        
        # Make several moves
        from engine.move_generator import LegalMoveGenerator
        generator = LegalMoveGenerator()
        
        for _ in range(4):
            player = board.current_player
            moves = generator.get_legal_moves(board, player)
            if not moves:
                break
            
            move = moves[0]
            orientations = generator.piece_orientations_cache[move.piece_id]
            positions = move.get_positions(orientations)
            board.place_piece(positions, player, move.piece_id)
            
            # Rebuild should match for current player
            match = board.debug_rebuild_frontier(player)
            self.assertTrue(match, f"Frontier should match for {player.name} after move")
            
            # Verify consistency
            self.assertTrue(board._verify_frontier_consistency(player),
                          f"Frontier should be consistent for {player.name}")


class TestFrontierEdgeCases(unittest.TestCase):
    """Test edge cases for frontier behavior."""
    
    def test_frontier_at_board_edges(self):
        """Test frontier behavior at board edges."""
        board = Board()
        
        # Place piece at corner (0, 0)
        positions = [Position(0, 0)]
        board.place_piece(positions, Player.RED, 1)
        
        frontier = board.get_frontier(Player.RED)
        
        # Only (1, 1) should be in frontier (other diagonals are out of bounds)
        self.assertIn((1, 1), frontier)
        # Should not have any out-of-bounds cells
        for r, c in frontier:
            self.assertGreaterEqual(r, 0)
            self.assertLess(r, board.SIZE)
            self.assertGreaterEqual(c, 0)
            self.assertLess(c, board.SIZE)
    
    def test_frontier_with_adjacent_pieces(self):
        """Test frontier when pieces are placed adjacent to each other."""
        board = Board()
        
        # Place first piece
        positions1 = [Position(0, 0), Position(0, 1)]
        board.place_piece(positions1, Player.RED, 2)
        
        # Find a legal second move that's diagonally adjacent
        from engine.move_generator import LegalMoveGenerator
        generator = LegalMoveGenerator()
        moves = generator.get_legal_moves(board, Player.RED)
        self.assertGreater(len(moves), 0, "Should have legal moves")
        
        # Use the first legal move (should be diagonally adjacent)
        move = moves[0]
        orientations = generator.piece_orientations_cache[move.piece_id]
        positions2 = move.get_positions(orientations)
        success = board.place_piece(positions2, Player.RED, move.piece_id)
        self.assertTrue(success, "Should be able to place second piece")
        
        frontier = board.get_frontier(Player.RED)
        
        # Placed cells should NOT be in frontier
        for pos in positions2:
            self.assertNotIn((pos.row, pos.col), frontier, 
                           f"Placed cell ({pos.row}, {pos.col}) should not be in frontier")
        
        # Verify consistency
        self.assertTrue(board._verify_frontier_consistency(Player.RED))
        
        # Verify that frontier matches full recompute
        match = board.debug_rebuild_frontier(Player.RED)
        self.assertTrue(match, "Frontier should match full recompute")


class TestFrontierInvariants(unittest.TestCase):
    """Test that frontier invariants hold across various game states."""
    
    def test_frontier_invariants_after_random_moves(self):
        """Test frontier invariants after several random moves."""
        board = Board()
        generator = LegalMoveGenerator()
        random.seed(42)  # For reproducibility
        
        # Make several random moves
        for move_num in range(8):
            player = board.current_player
            moves = generator.get_legal_moves(board, player)
            
            if not moves:
                # No legal moves, try next player
                board._update_current_player()
                continue
            
            # Choose a random move
            move = random.choice(moves)
            orientations = generator.piece_orientations_cache[move.piece_id]
            positions = move.get_positions(orientations)
            board.place_piece(positions, player, move.piece_id)
            
            # Check invariants for all players after each move
            for p in Player:
                assert_frontier_invariants(board, p, self)
                
                # Verify that incremental and full recompute match
                # Skip rebuild check for first move (starting corner is special case)
                if not board.player_first_move[p]:
                    match = board.debug_rebuild_frontier(p)
                    self.assertTrue(match, 
                                   f"Frontier should match full recompute for {p.name} after move {move_num}")
    
    def test_frontier_invariants_self_play(self):
        """Test frontier invariants in a self-play scenario."""
        board = Board()
        generator = LegalMoveGenerator()
        
        # Make moves for all players in rotation
        for move_num in range(12):
            player = board.current_player
            moves = generator.get_legal_moves(board, player)
            
            if not moves:
                # No legal moves, skip this player
                board._update_current_player()
                continue
            
            # Use first move (deterministic)
            move = moves[0]
            orientations = generator.piece_orientations_cache[move.piece_id]
            positions = move.get_positions(orientations)
            board.place_piece(positions, player, move.piece_id)
            
            # Check invariants for the player who just moved
            assert_frontier_invariants(board, player, self)
            
            # Verify consistency check
            self.assertTrue(board._verify_frontier_consistency(player),
                           f"Frontier consistency check should pass for {player.name} after move {move_num}")
            
            # Verify that incremental and full recompute match
            match = board.debug_rebuild_frontier(player)
            self.assertTrue(match,
                           f"Frontier should match full recompute for {player.name} after move {move_num}")


if __name__ == "__main__":
    unittest.main()

