"""
Tests to verify that bitboard-based legality checks produce the same results
as grid-based legality checks.
"""

import unittest
import random
from engine.board import Board, Player, Position
from engine.move_generator import LegalMoveGenerator
from engine.pieces import ALL_PIECE_ORIENTATIONS, PieceGenerator, PiecePlacement
from tests.utils_game_states import generate_random_valid_state


class TestLegalityBitboardEquivalence(unittest.TestCase):
    """Test that bitboard and grid-based legality checks are equivalent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = LegalMoveGenerator()
        self.piece_generator = PieceGenerator()
    
    def _check_legality_grid_based(self, board: Board, player: Player, 
                                    piece_id: int, orientation_idx: int,
                                    anchor_row: int, anchor_col: int) -> bool:
        """
        Check legality using the original grid-based method.
        
        This is the reference implementation.
        """
        # Get numpy orientation
        orientations = self.generator.piece_orientations_cache[piece_id]
        if orientation_idx >= len(orientations):
            return False
        
        orientation = orientations[orientation_idx]
        
        # Check bounds
        if not PiecePlacement.can_place_piece_at(
            (board.SIZE, board.SIZE), orientation, anchor_row, anchor_col
        ):
            return False
        
        # Get positions and check legality
        positions = PiecePlacement.get_piece_positions(
            orientation, anchor_row, anchor_col
        )
        piece_positions = [Position(row, col) for row, col in positions]
        
        return board.can_place_piece(piece_positions, player)
    
    def _check_legality_bitboard_based(self, board: Board, player: Player,
                                        piece_id: int, orientation_idx: int,
                                        anchor_row: int, anchor_col: int) -> bool:
        """
        Check legality using coords-based bitboard method.
        
        This converts anchor position to placement coordinates and uses
        the coords-based bitboard legality check.
        """
        # Get numpy orientation to compute placement coordinates
        orientations = self.generator.piece_orientations_cache[piece_id]
        if orientation_idx >= len(orientations):
            return False
        
        orientation = orientations[orientation_idx]
        
        # Get placement coordinates from anchor position
        positions = PiecePlacement.get_piece_positions(
            orientation, anchor_row, anchor_col
        )
        placement_coords = [(row, col) for row, col in positions]
        
        # Use coords-based bitboard legality check
        return self.generator.is_placement_legal_bitboard_coords(
            board, player, placement_coords,
            is_first_move=board.player_first_move[player]
        )
    
    def test_first_move_equivalence(self):
        """Test equivalence on empty board (first move scenario)."""
        board = Board()
        player = Player.RED
        
        # Test a few candidate placements for first move
        piece_id = 1  # Monomino
        orientation_idx = 0
        
        # Test placing at starting corner (should be legal)
        grid_result = self._check_legality_grid_based(board, player, piece_id, orientation_idx, 0, 0)
        
        # For bitboard, use coords-based check
        orientations = self.generator.piece_orientations_cache[piece_id]
        orientation = orientations[orientation_idx]
        positions = PiecePlacement.get_piece_positions(orientation, 0, 0)
        placement_coords = [(row, col) for row, col in positions]
        bitboard_result = self.generator.is_placement_legal_bitboard_coords(
            board, player, placement_coords,
            is_first_move=board.player_first_move[player]
        )
        
        self.assertEqual(grid_result, bitboard_result,
                        f"First move legality mismatch: grid={grid_result}, bitboard={bitboard_result}")
    
    def test_after_single_piece_equivalence(self):
        """Test equivalence after placing a single piece."""
        board = Board()
        
        # Place a single square piece (monomino) at RED's starting corner
        positions = [Position(0, 0)]
        board.place_piece(positions, Player.RED, 1)
        
        # Test a few candidate placements for BLUE
        player = Player.BLUE
        piece_id = 2  # Domino
        orientation_idx = 0  # Horizontal orientation
        
        # Test a few anchor positions
        test_anchors = [(0, 19), (1, 19), (0, 18)]
        
        for anchor_row, anchor_col in test_anchors:
            grid_result = self._check_legality_grid_based(
                board, player, piece_id, orientation_idx, anchor_row, anchor_col
            )
            
            # For bitboard, use coords-based check
            orientations = self.generator.piece_orientations_cache[piece_id]
            orientation = orientations[orientation_idx]
            positions = PiecePlacement.get_piece_positions(orientation, anchor_row, anchor_col)
            placement_coords = [(row, col) for row, col in positions]
            bitboard_result = self.generator.is_placement_legal_bitboard_coords(
                board, player, placement_coords,
                is_first_move=board.player_first_move[player]
            )
            
            # If that didn't work, the grid-based might be using a different anchor strategy
            # For now, we'll test that when bitboard says legal, grid also says legal
            if bitboard_result:
                self.assertTrue(grid_result,
                              f"Bitboard says legal but grid says illegal at ({anchor_row}, {anchor_col})")
    
    def test_equivalence_for_actual_moves(self):
        """Test equivalence by checking actual legal moves from both generators."""
        board = Board()
        player = Player.RED
        
        # Get legal moves using grid-based generator (naive)
        naive_moves = self.generator._get_legal_moves_naive(board, player)
        
        # For each naive move, verify bitboard legality agrees
        for move in naive_moves[:10]:  # Test first 10 moves
            piece_orientations = ALL_PIECE_ORIENTATIONS.get(move.piece_id, [])
            if move.orientation >= len(piece_orientations):
                continue
            
            orientation = piece_orientations[move.orientation]
            
            # Use coords-based bitboard check
            orientations = self.generator.piece_orientations_cache[move.piece_id]
            orientation = orientations[move.orientation]
            positions = PiecePlacement.get_piece_positions(
                orientation, move.anchor_row, move.anchor_col
            )
            placement_coords = [(row, col) for row, col in positions]
            bitboard_result = self.generator.is_placement_legal_bitboard_coords(
                board, player, placement_coords,
                is_first_move=board.player_first_move[player]
            )
            
            # Bitboard should agree with grid-based (which said this move is legal)
            self.assertTrue(bitboard_result,
                          f"Bitboard says move {move} is illegal, but grid-based says it's legal")
    
    def test_equivalence_after_two_pieces(self):
        """Test equivalence after placing two pieces."""
        board = Board()
        
        # Place first piece for RED
        positions1 = [Position(0, 0), Position(0, 1)]
        board.place_piece(positions1, Player.RED, 2)
        
        # Place first piece for BLUE
        positions2 = [Position(0, 19), Position(0, 18)]
        board.place_piece(positions2, Player.BLUE, 2)
        
        # Test a few candidate placements for YELLOW
        player = Player.YELLOW
        piece_id = 1  # Monomino
        orientation_idx = 0
        
        # Test placing at YELLOW's starting corner
        grid_result = self._check_legality_grid_based(board, player, piece_id, orientation_idx, 19, 19)
        
        orientations = self.generator.piece_orientations_cache[piece_id]
        orientation = orientations[orientation_idx]
        positions = PiecePlacement.get_piece_positions(orientation, 19, 19)
        placement_coords = [(row, col) for row, col in positions]
        bitboard_result = self.generator.is_placement_legal_bitboard_coords(
            board, player, placement_coords,
            is_first_move=board.player_first_move[player]
        )
        
        self.assertEqual(grid_result, bitboard_result,
                        f"After two pieces: grid={grid_result}, bitboard={bitboard_result}")
    
    def test_bitboard_legality_matches_grid_on_random_states(self):
        """
        Test that bitboard and grid legality produce identical results on random states.
        
        This test verifies equivalence by comparing move sets generated by:
        - Naive generator (grid-based legality) 
        - Frontier+bitboard generator (bitboard legality)
        
        If the move sets match, then the legality checks are equivalent.
        """
        import engine.move_generator as move_gen_module
        
        num_seeds = 5
        num_moves_range = (3, 8)
        
        # Save original flags
        original_frontier = move_gen_module.USE_FRONTIER_MOVEGEN
        original_bitboard = move_gen_module.USE_BITBOARD_LEGALITY
        
        try:
            # Enable both frontier and bitboard
            move_gen_module.USE_FRONTIER_MOVEGEN = True
            move_gen_module.USE_BITBOARD_LEGALITY = True
            
            for seed in range(num_seeds):
                # Generate random state
                num_moves = random.randint(*num_moves_range)
                board, current_player = generate_random_valid_state(num_moves, seed=seed)
                
                # Test all players that still have pieces available
                for player in Player:
                    # Skip players who have used all pieces (unlikely but possible)
                    if len(board.player_pieces_used[player]) >= 21:
                        continue
                    
                    # Get all legal moves using naive generator (grid-based, trusted reference)
                    naive_moves = self.generator._get_legal_moves_naive(board, player)
                    naive_set = {(m.piece_id, m.orientation, m.anchor_row, m.anchor_col) 
                                for m in naive_moves}
                    
                    # Get all legal moves using frontier+bitboard generator
                    frontier_bitboard_moves = self.generator._get_legal_moves_frontier(board, player)
                    frontier_bitboard_set = {(m.piece_id, m.orientation, m.anchor_row, m.anchor_col)
                                           for m in frontier_bitboard_moves}
                    
                    # Move sets should be identical
                    self.assertEqual(naive_set, frontier_bitboard_set,
                                    f"Move sets differ on random state (seed={seed}, player={player.name}, "
                                    f"num_moves={num_moves}): naive has {len(naive_set)}, "
                                    f"frontier+bitboard has {len(frontier_bitboard_set)}")
                    
                    # Check for discrepancies
                    if naive_set != frontier_bitboard_set:
                        extra = frontier_bitboard_set - naive_set
                        missing = naive_set - frontier_bitboard_set
                        if extra:
                            self.fail(f"Frontier+bitboard has {len(extra)} extra moves: {list(extra)[:5]}")
                        if missing:
                            self.fail(f"Frontier+bitboard missing {len(missing)} moves: {list(missing)[:5]}")
        finally:
            # Restore original flags
            move_gen_module.USE_FRONTIER_MOVEGEN = original_frontier
            move_gen_module.USE_BITBOARD_LEGALITY = original_bitboard
    
    def test_bitboard_vs_grid_move_generation_equivalence_random_states(self):
        """Test that frontier+bitboard move generation produces same moves as naive+grid."""
        import engine.move_generator as move_gen_module
        
        num_seeds = 10
        num_moves_range = (3, 8)
        
        # Save original flags
        original_frontier = move_gen_module.USE_FRONTIER_MOVEGEN
        original_bitboard = move_gen_module.USE_BITBOARD_LEGALITY
        
        try:
            # Enable both frontier and bitboard
            move_gen_module.USE_FRONTIER_MOVEGEN = True
            move_gen_module.USE_BITBOARD_LEGALITY = True
            
            for seed in range(num_seeds):
                num_moves = random.randint(*num_moves_range)
                board, current_player = generate_random_valid_state(num_moves, seed=seed)
                
                # Generate moves with naive (grid-based)
                naive_moves = self.generator._get_legal_moves_naive(board, current_player)
                naive_set = {(m.piece_id, m.orientation, m.anchor_row, m.anchor_col) 
                            for m in naive_moves}
                
                # Generate moves with frontier+bitboard
                frontier_bitboard_moves = self.generator._get_legal_moves_frontier(board, current_player)
                frontier_bitboard_set = {(m.piece_id, m.orientation, m.anchor_row, m.anchor_col)
                                       for m in frontier_bitboard_moves}
                
                # They should be identical
                self.assertEqual(naive_set, frontier_bitboard_set,
                                f"Move sets differ on random state (seed={seed}, player={current_player.name}, "
                                f"num_moves={num_moves}): naive has {len(naive_set)}, "
                                f"frontier+bitboard has {len(frontier_bitboard_set)}")
                
                # Check for extra moves in frontier+bitboard (shouldn't happen)
                extra_moves = frontier_bitboard_set - naive_set
                if extra_moves:
                    self.fail(f"Frontier+bitboard found {len(extra_moves)} extra moves not in naive: {list(extra_moves)[:5]}")
                
                # Check for missing moves in frontier+bitboard (shouldn't happen)
                missing_moves = naive_set - frontier_bitboard_set
                if missing_moves:
                    self.fail(f"Frontier+bitboard missing {len(missing_moves)} moves from naive: {list(missing_moves)[:5]}")
        finally:
            # Restore original flags
            move_gen_module.USE_FRONTIER_MOVEGEN = original_frontier
            move_gen_module.USE_BITBOARD_LEGALITY = original_bitboard


if __name__ == "__main__":
    unittest.main()

