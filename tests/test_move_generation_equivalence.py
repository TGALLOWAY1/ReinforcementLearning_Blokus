"""
Tests to verify that frontier-based move generation produces the same results
as the naive (full-board scan) move generation.
"""

import random
import unittest
from engine.board import Board, Player, Position
from engine.move_generator import LegalMoveGenerator, Move, debug_compare_bitboard_vs_grid
from engine.pieces import ALL_PIECE_ORIENTATIONS, PiecePlacement
from tests.utils_game_states import generate_random_valid_state


def moves_to_set(moves: list) -> set:
    """Convert a list of Move objects to a set of tuples for comparison."""
    return {(move.piece_id, move.orientation, move.anchor_row, move.anchor_col) 
            for move in moves}


def move_to_coord_key(generator: LegalMoveGenerator, move: Move) -> tuple:
    """
    Convert a Move to a coordinate-based key for comparison.
    
    This allows comparing moves regardless of orientation indices or internal IDs.
    The key is (piece_id, tuple(sorted_coords)) where coords are the board positions
    the piece would occupy.
    
    Args:
        generator: LegalMoveGenerator instance (for accessing orientation cache)
        move: Move object to convert
        
    Returns:
        Tuple of (piece_id, tuple(sorted_coords))
    """
    # Get the positions this move would occupy
    orientations = generator.piece_orientations_cache.get(move.piece_id, [])
    if move.orientation >= len(orientations):
        # Fallback: use anchor position only
        return (move.piece_id, ((move.anchor_row, move.anchor_col),))
    
    orientation = orientations[move.orientation]
    positions = PiecePlacement.get_piece_positions(
        orientation, move.anchor_row, move.anchor_col
    )
    
    # Sort coordinates for canonical ordering
    sorted_coords = tuple(sorted(positions))
    return (move.piece_id, sorted_coords)


def generate_random_board_state(num_moves: int, seed: int = None) -> tuple:
    """
    Generate a random but valid board state by playing random legal moves.
    
    Args:
        num_moves: Number of moves to make
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (board, current_player) representing the final state
    """
    if seed is not None:
        random.seed(seed)
    
    board = Board()
    generator = LegalMoveGenerator()
    
    moves_made = 0
    for _ in range(num_moves):
        player = board.current_player
        moves = generator._get_legal_moves_naive(board, player)
        
        if not moves:
            # No legal moves for current player, try next player
            board._update_current_player()
            continue
        
        # Choose a random move
        move = random.choice(moves)
        orientations = generator.piece_orientations_cache[move.piece_id]
        positions = move.get_positions(orientations)
        
        success = board.place_piece(positions, player, move.piece_id)
        if success:
            moves_made += 1
    
    return board, board.current_player


class TestMoveGenerationEquivalence(unittest.TestCase):
    """Test that naive and frontier-based generators produce equivalent results."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = LegalMoveGenerator()
    
    def test_empty_board_first_move(self):
        """Test equivalence on empty board (first move scenario)."""
        board = Board()
        
        # Generate moves with naive generator
        naive_moves = self.generator._get_legal_moves_naive(board, Player.RED)
        
        # Generate moves with frontier generator
        frontier_moves = self.generator._get_legal_moves_frontier(board, Player.RED)
        
        # Convert to sets for comparison
        naive_set = moves_to_set(naive_moves)
        frontier_set = moves_to_set(frontier_moves)
        
        # They should be equal
        self.assertEqual(naive_set, frontier_set,
                        f"Moves differ: naive has {len(naive_set)}, frontier has {len(frontier_set)}")
        
        # Both should have moves (first move must cover starting corner)
        self.assertGreater(len(naive_set), 0, "Should have legal moves on empty board")
        self.assertGreater(len(frontier_set), 0, "Should have legal moves on empty board")
    
    def test_after_single_piece(self):
        """Test equivalence after placing a single piece."""
        board = Board()
        
        # Place a single square piece (monomino) at RED's starting corner
        positions = [Position(0, 0)]
        board.place_piece(positions, Player.RED, 1)
        
        # Generate moves for next player (BLUE)
        naive_moves = self.generator._get_legal_moves_naive(board, Player.BLUE)
        frontier_moves = self.generator._get_legal_moves_frontier(board, Player.BLUE)
        
        naive_set = moves_to_set(naive_moves)
        frontier_set = moves_to_set(frontier_moves)
        
        self.assertEqual(naive_set, frontier_set,
                        f"Moves differ after single piece: naive has {len(naive_set)}, frontier has {len(frontier_set)}")
    
    def test_after_two_pieces(self):
        """Test equivalence after placing two pieces."""
        board = Board()
        
        # Place first piece for RED
        positions1 = [Position(0, 0), Position(0, 1)]
        board.place_piece(positions1, Player.RED, 2)
        
        # Place first piece for BLUE
        positions2 = [Position(0, 19), Position(0, 18)]
        board.place_piece(positions2, Player.BLUE, 2)
        
        # Generate moves for YELLOW (third player)
        naive_moves = self.generator._get_legal_moves_naive(board, Player.YELLOW)
        frontier_moves = self.generator._get_legal_moves_frontier(board, Player.YELLOW)
        
        naive_set = moves_to_set(naive_moves)
        frontier_set = moves_to_set(frontier_moves)
        
        self.assertEqual(naive_set, frontier_set,
                        f"Moves differ after two pieces: naive has {len(naive_set)}, frontier has {len(frontier_set)}")
    
    def test_after_multiple_moves(self):
        """Test equivalence after multiple moves from different players."""
        board = Board()
        generator = LegalMoveGenerator()
        
        # Make several moves
        for _ in range(4):
            player = board.current_player
            moves = generator._get_legal_moves_naive(board, player)
            if not moves:
                break
            
            move = moves[0]
            orientations = generator.piece_orientations_cache[move.piece_id]
            positions = move.get_positions(orientations)
            board.place_piece(positions, player, move.piece_id)
        
        # Test equivalence for current player
        current_player = board.current_player
        naive_moves = self.generator._get_legal_moves_naive(board, current_player)
        frontier_moves = self.generator._get_legal_moves_frontier(board, current_player)
        
        naive_set = moves_to_set(naive_moves)
        frontier_set = moves_to_set(frontier_moves)
        
        self.assertEqual(naive_set, frontier_set,
                        f"Moves differ after multiple moves for {current_player.name}: "
                        f"naive has {len(naive_set)}, frontier has {len(frontier_set)}")
    
    def test_all_players_after_first_move(self):
        """Test equivalence for all players after first move."""
        board = Board()
        
        # Place first piece for RED
        positions = [Position(0, 0), Position(0, 1)]
        board.place_piece(positions, Player.RED, 2)
        
        # Test all players
        for player in Player:
            naive_moves = self.generator._get_legal_moves_naive(board, player)
            frontier_moves = self.generator._get_legal_moves_frontier(board, player)
            
            naive_set = moves_to_set(naive_moves)
            frontier_set = moves_to_set(frontier_moves)
            
            self.assertEqual(naive_set, frontier_set,
                            f"Moves differ for {player.name}: naive has {len(naive_set)}, frontier has {len(frontier_set)}")
    
    def test_feature_flag_wrapper(self):
        """Test that the feature flag wrapper correctly delegates to both generators."""
        board = Board()
        
        # Test with flag disabled (should use naive)
        import engine.move_generator as move_gen_module
        original_flag = move_gen_module.USE_FRONTIER_MOVEGEN
        try:
            move_gen_module.USE_FRONTIER_MOVEGEN = False
            wrapper_moves = self.generator.get_legal_moves(board, Player.RED)
            naive_moves = self.generator._get_legal_moves_naive(board, Player.RED)
            self.assertEqual(moves_to_set(wrapper_moves), moves_to_set(naive_moves))
            
            # Test with flag enabled (should use frontier)
            move_gen_module.USE_FRONTIER_MOVEGEN = True
            wrapper_moves = self.generator.get_legal_moves(board, Player.RED)
            frontier_moves = self.generator._get_legal_moves_frontier(board, Player.RED)
            self.assertEqual(moves_to_set(wrapper_moves), moves_to_set(frontier_moves))
        finally:
            move_gen_module.USE_FRONTIER_MOVEGEN = original_flag
    
    def test_move_generation_equivalence_random_states_small(self):
        """Test equivalence on random board states with few moves (early game)."""
        num_states = 15
        num_moves_per_state = 5
        
        for seed in range(num_states):
            board, current_player = generate_random_board_state(num_moves_per_state, seed=seed)
            
            # Generate moves with both generators
            naive_moves = self.generator._get_legal_moves_naive(board, current_player)
            frontier_moves = self.generator._get_legal_moves_frontier(board, current_player)
            
            naive_set = moves_to_set(naive_moves)
            frontier_set = moves_to_set(frontier_moves)
            
            self.assertEqual(naive_set, frontier_set,
                            f"Moves differ for random state (seed={seed}, player={current_player.name}): "
                            f"naive has {len(naive_set)}, frontier has {len(frontier_set)}")
    
    def test_move_generation_equivalence_random_states_midgame(self):
        """Test equivalence on random board states with more moves (mid-game)."""
        num_states = 10
        num_moves_per_state = 10
        
        for seed in range(num_states):
            board, current_player = generate_random_board_state(num_moves_per_state, seed=seed)
            
            # Generate moves with both generators
            naive_moves = self.generator._get_legal_moves_naive(board, current_player)
            frontier_moves = self.generator._get_legal_moves_frontier(board, current_player)
            
            naive_set = moves_to_set(naive_moves)
            frontier_set = moves_to_set(frontier_moves)
            
            self.assertEqual(naive_set, frontier_set,
                            f"Moves differ for random mid-game state (seed={seed}, player={current_player.name}): "
                            f"naive has {len(naive_set)}, frontier has {len(frontier_set)}")
    
    def test_move_generation_equivalence_all_players_random_states(self):
        """Test equivalence for all players on random board states."""
        num_states = 5
        num_moves_per_state = 8
        
        for seed in range(num_states):
            board, _ = generate_random_board_state(num_moves_per_state, seed=seed)
            
            # Test all players
            for player in Player:
                naive_moves = self.generator._get_legal_moves_naive(board, player)
                frontier_moves = self.generator._get_legal_moves_frontier(board, player)
                
                naive_set = moves_to_set(naive_moves)
                frontier_set = moves_to_set(frontier_moves)
                
                self.assertEqual(naive_set, frontier_set,
                                f"Moves differ for {player.name} on random state (seed={seed}): "
                                f"naive has {len(naive_set)}, frontier has {len(frontier_set)}")
    
    def test_frontier_bitboard_vs_naive_random_states(self):
        """Test that frontier+bitboard move generation matches naive on random states."""
        import engine.move_generator as move_gen_module
        
        # Save original flags
        original_frontier = move_gen_module.USE_FRONTIER_MOVEGEN
        original_bitboard = move_gen_module.USE_BITBOARD_LEGALITY
        
        try:
            # Enable both frontier and bitboard
            move_gen_module.USE_FRONTIER_MOVEGEN = True
            move_gen_module.USE_BITBOARD_LEGALITY = True
            
            num_states = 15
            num_moves_range = (3, 10)
            
            for seed in range(num_states):
                num_moves = random.randint(*num_moves_range)
                board, current_player = generate_random_valid_state(num_moves, seed=seed)
                
                # Generate moves with naive (grid-based)
                naive_moves = self.generator._get_legal_moves_naive(board, current_player)
                naive_set = moves_to_set(naive_moves)
                
                # Generate moves with frontier+bitboard
                frontier_bitboard_moves = self.generator._get_legal_moves_frontier(board, current_player)
                frontier_bitboard_set = moves_to_set(frontier_bitboard_moves)
                
                # Check for discrepancies using coordinate-based keys
                naive_coord_map = {move_to_coord_key(self.generator, move): move 
                                  for move in naive_moves}
                frontier_coord_map = {move_to_coord_key(self.generator, move): move 
                                     for move in frontier_bitboard_moves}
                
                naive_coord_set = set(naive_coord_map.keys())
                frontier_coord_set = set(frontier_coord_map.keys())
                
                missing_coords = naive_coord_set - frontier_coord_set
                extra_coords = frontier_coord_set - naive_coord_set
                
                # If there's a mismatch, log detailed diagnostics for the first one
                if missing_coords or extra_coords:
                    print("\n" + "=" * 80)
                    print("DEBUG MISMATCH DETECTED")
                    print("=" * 80)
                    print(f"State: seed={seed}, player={current_player.name}, num_moves={num_moves}")
                    print(f"Naive moves: {len(naive_set)}, Frontier+bitboard moves: {len(frontier_bitboard_set)}")
                    print(f"Missing moves (in naive but not frontier+bitboard): {len(missing_coords)}")
                    print(f"Extra moves (in frontier+bitboard but not naive): {len(extra_coords)}")
                    print()
                    
                    # Analyze first missing move
                    if missing_coords:
                        missing_key = next(iter(missing_coords))
                        missing_move = naive_coord_map[missing_key]
                        piece_id = missing_move.piece_id
                        player_id = current_player
                        
                        print("DEBUG MISMATCH - Missing move (first):")
                        print(f"  Piece ID: {piece_id}")
                        print(f"  Player: {player_id.name} (value={player_id.value})")
                        print(f"  Move: {missing_move}")
                        print(f"  Anchor: ({missing_move.anchor_row}, {missing_move.anchor_col})")
                        print(f"  Orientation index: {missing_move.orientation}")
                        
                        # Get coordinates this move would occupy
                        coords = move_to_coord_key(self.generator, missing_move)[1]
                        print(f"  Board coordinates: {coords}")
                        print()
                        
                        # Compare legality directly
                        print("Legality comparison:")
                        # Grid-based legality
                        orientations = self.generator.piece_orientations_cache.get(piece_id, [])
                        if missing_move.orientation < len(orientations):
                            orientation = orientations[missing_move.orientation]
                            positions = PiecePlacement.get_piece_positions(
                                orientation, missing_move.anchor_row, missing_move.anchor_col
                            )
                            piece_positions = [Position(row, col) for row, col in positions]
                            grid_legal = board.can_place_piece(piece_positions, player_id)
                            print(f"  Grid-based legality: {grid_legal}")
                        else:
                            print(f"  Grid-based legality: ERROR (orientation index {missing_move.orientation} out of range)")
                            grid_legal = False
                        
                        # Bitboard legality (coords-based)
                        placement_coords_list = list(coords)
                        bitboard_legal = self.generator.is_placement_legal_bitboard_coords(
                            board, player_id, placement_coords_list,
                            is_first_move=board.player_first_move[player_id]
                        )
                        print(f"  Bitboard legality (coords-based): {bitboard_legal}")
                        
                        # Call deep debug helper for detailed comparison (if needed)
                        piece_orientations = ALL_PIECE_ORIENTATIONS.get(piece_id, [])
                        if missing_move.orientation < len(piece_orientations):
                            piece_orientation = piece_orientations[missing_move.orientation]
                            # Find matching anchor for debug helper
                            matching_anchor_idx = None
                            matching_anchor_coord = None
                            for anchor_idx in piece_orientation.anchor_indices:
                                if anchor_idx >= len(piece_orientation.offsets):
                                    continue
                                rel_r, rel_c = piece_orientation.offsets[anchor_idx]
                                anchor_row = missing_move.anchor_row - rel_r
                                anchor_col = missing_move.anchor_col - rel_c
                                test_coords = tuple(sorted(
                                    (anchor_row + offset[0], anchor_col + offset[1])
                                    for offset in piece_orientation.offsets
                                ))
                                if test_coords == coords:
                                    matching_anchor_idx = anchor_idx
                                    matching_anchor_coord = (anchor_row, anchor_col)
                                    break
                            
                            if matching_anchor_idx is not None and matching_anchor_coord is not None:
                                debug_compare_bitboard_vs_grid(
                                    board, player_id, piece_orientation,
                                    matching_anchor_coord, matching_anchor_idx,
                                    placement_coords_list
                                )
                            
                            # Orientation debug info
                            print()
                            print("Orientation debug info:")
                            print(f"  PieceOrientation offsets: {piece_orientation.offsets}")
                            print(f"  PieceOrientation anchor_indices: {piece_orientation.anchor_indices}")
                            if missing_move.orientation < len(orientations):
                                old_orientation = orientations[missing_move.orientation]
                                old_positions = PiecePlacement.get_piece_positions(
                                    old_orientation, missing_move.anchor_row, missing_move.anchor_col
                                )
                                print(f"  Old orientation shape positions: {old_positions}")
                        else:
                            print(f"  Bitboard legality: ERROR (orientation index {missing_move.orientation} out of range)")
                            bitboard_legal = False
                        
                        print()
                        print("=" * 80)
                        print()
                        
                        # Assertions for debugging
                        self.assertTrue(grid_legal, 
                                       f"Grid-based legality should be True for move {missing_move}")
                        self.assertFalse(bitboard_legal,
                                        f"Bitboard legality should be False for move {missing_move} (this is why it's missing)")
                    
                    # Also log first extra move if any
                    if extra_coords:
                        extra_key = next(iter(extra_coords))
                        extra_move = frontier_coord_map[extra_key]
                        print("DEBUG MISMATCH - Extra move (first):")
                        print(f"  Piece ID: {extra_move.piece_id}")
                        print(f"  Player: {current_player.name}")
                        print(f"  Move: {extra_move}")
                        print(f"  Board coordinates: {move_to_coord_key(self.generator, extra_move)[1]}")
                        print()
                
                # They should be identical
                self.assertEqual(naive_coord_set, frontier_coord_set,
                                f"Moves differ for random state (seed={seed}, player={current_player.name}, "
                                f"num_moves={num_moves}): naive has {len(naive_set)}, "
                                f"frontier+bitboard has {len(frontier_bitboard_set)}")
        finally:
            # Restore original flags
            move_gen_module.USE_FRONTIER_MOVEGEN = original_frontier
            move_gen_module.USE_BITBOARD_LEGALITY = original_bitboard


if __name__ == "__main__":
    unittest.main()

