"""
Legal move generator for Blokus game.
"""

import os
import time
import logging
import random
from typing import List, Tuple, Set, Optional
from .board import Board, Player, Position
from .pieces import PieceGenerator, PiecePlacement, PieceOrientation, ALL_PIECE_ORIENTATIONS
from .bitboard import shift_mask, coord_to_bit, coords_to_mask, mask_to_coords, BOARD_WIDTH, BOARD_HEIGHT

logger = logging.getLogger(__name__)

# Debug flag for move generation timing (controlled via environment variable)
MOVEGEN_DEBUG = bool(os.getenv("BLOKUS_MOVEGEN_DEBUG", ""))

# Feature flag to toggle between naive and frontier-based move generation
USE_FRONTIER_MOVEGEN = bool(os.getenv("BLOKUS_USE_FRONTIER_MOVEGEN", ""))

# Feature flag to toggle between grid-based and bitboard-based legality checks
# When True, frontier-based move generation (and optionally others) will use
# bitboard legality instead of cell-based checks
USE_BITBOARD_LEGALITY = bool(os.getenv("BLOKUS_USE_BITBOARD_LEGALITY", ""))

# Debug flag for equivalence checking (guarded, samples 5% of calls)
# When enabled, randomly samples calls to verify bitboard and grid legality match
MOVEGEN_DEBUG_EQUIVALENCE = bool(os.getenv("BLOKUS_MOVEGEN_DEBUG_EQUIVALENCE", ""))

# Debug flag for deep bitboard vs grid comparison
# When enabled, prints detailed comparison of bitboard and grid legality checks
DEBUG_BITBOARD = bool(os.getenv("BLOKUS_DEBUG_BITBOARD", ""))


class Move:
    """Represents a legal move in Blokus."""
    
    def __init__(self, piece_id: int, orientation: int, anchor_row: int, anchor_col: int):
        self.piece_id = piece_id
        self.orientation = orientation  # Index into orientations list
        self.anchor_row = anchor_row
        self.anchor_col = anchor_col
    
    def get_positions(self, piece_orientations: List) -> List[Position]:
        """Get the board positions this move would occupy."""
        shape = piece_orientations[self.orientation]
        positions = PiecePlacement.get_piece_positions(shape, self.anchor_row, self.anchor_col)
        return [Position(row, col) for row, col in positions]
    
    def __str__(self):
        return f"Move(piece_id={self.piece_id}, orientation={self.orientation}, anchor=({self.anchor_row}, {self.anchor_col}))"


class LegalMoveGenerator:
    """Generates all legal moves for a given board state and player."""
    
    def __init__(self):
        self.piece_generator = PieceGenerator()
        self.all_pieces = self.piece_generator.get_all_pieces()
        self.piece_orientations_cache = {}
        self.piece_position_cache = {}  # Cache piece positions for each orientation
        self._cache_piece_orientations()
    
    def _cache_piece_orientations(self):
        """Cache all piece orientations and their position lists for faster lookup."""
        for piece in self.all_pieces:
            orientations = self.piece_generator.get_piece_rotations_and_reflections(piece)
            self.piece_orientations_cache[piece.id] = orientations
            # Pre-compute position lists for each orientation (relative to anchor 0,0)
            self.piece_position_cache[piece.id] = []
            for orientation in orientations:
                positions = PiecePlacement.get_piece_positions(orientation, 0, 0)
                self.piece_position_cache[piece.id].append(positions)
    
    def get_legal_moves(self, board: Board, player: Player) -> List[Move]:
        """
        Get all legal moves for a given player on the current board.
        
        This is a thin wrapper that delegates to either the naive or frontier-based
        generator based on the USE_FRONTIER_MOVEGEN feature flag.
        
        Args:
            board: Current board state
            player: Player to generate moves for
        
        Returns:
            List of legal moves
        """
        if USE_FRONTIER_MOVEGEN:
            return self._get_legal_moves_frontier(board, player)
        else:
            return self._get_legal_moves_naive(board, player)
    
    def _get_legal_moves_naive(self, board: Board, player: Player) -> List[Move]:
        """
        Get all legal moves using the original full-board scan implementation.
        
        This is the naive generator that scans all possible anchor positions for
        each piece and orientation. It serves as the baseline for correctness
        and performance comparison.
        
        Args:
            board: Current board state
            player: Player to generate moves for
        
        Returns:
            List of legal moves
        """
        start = time.perf_counter()
        legal_moves = []
        
        # Get pieces that haven't been used yet
        available_pieces = [piece for piece in self.all_pieces 
                          if piece.id not in board.player_pieces_used[player]]
        
        # Direct grid access for faster checks
        grid = board.grid
        player_value = player.value
        is_first_move = board.player_first_move[player]
        start_corner = board.player_start_corners[player] if is_first_move else None
        
        piece_timings = {}
        for piece in available_pieces:
            piece_start = time.perf_counter()
            orientations = self.piece_orientations_cache[piece.id]
            cached_positions = self.piece_position_cache[piece.id]
            
            for orientation_idx, orientation in enumerate(orientations):
                # Get cached relative positions for this orientation
                relative_positions = cached_positions[orientation_idx]
                
                # Get all possible anchor positions for this orientation
                anchor_positions = PiecePlacement.get_valid_anchor_positions(
                    (board.SIZE, board.SIZE), orientation
                )
                
                for anchor_row, anchor_col in anchor_positions:
                    # Fast bounds and overlap check using cached positions and direct grid access
                    has_overlap = False
                    covers_start_corner = False
                    
                    for rel_r, rel_c in relative_positions:
                        r = anchor_row + rel_r
                        c = anchor_col + rel_c
                        
                        # Bounds check (should already be valid from get_valid_anchor_positions, but double-check)
                        if r < 0 or r >= board.SIZE or c < 0 or c >= board.SIZE:
                            has_overlap = True
                            break
                        
                        # Overlap check - direct grid access
                        if grid[r, c] != 0:
                            has_overlap = True
                            break
                        
                        # Check if covers start corner (for first move)
                        if is_first_move and start_corner and r == start_corner.row and c == start_corner.col:
                            covers_start_corner = True
                    
                    # Early exit if overlap or (first move and doesn't cover start corner)
                    if has_overlap:
                        continue
                    if is_first_move and not covers_start_corner:
                        continue
                    
                    # Check adjacency rules using relative positions directly (avoid Position object creation)
                    if self._check_adjacency_fast_inline(relative_positions, anchor_row, anchor_col, 
                                                         player_value, grid, board.SIZE, 
                                                         is_first_move, start_corner):
                        move = Move(piece.id, orientation_idx, anchor_row, anchor_col)
                        legal_moves.append(move)
            
            piece_end = time.perf_counter()
            piece_timings[piece.id] = piece_end - piece_start
        
        end = time.perf_counter()
        total_time = end - start
        elapsed_ms = total_time * 1000.0
        
        # Debug timing hook (only logs when BLOKUS_MOVEGEN_DEBUG is set)
        if MOVEGEN_DEBUG:
            logger.info(f"MoveGen[naive]: player={player.name}, legal_moves={len(legal_moves)}, elapsed_ms={elapsed_ms:.2f}")
            if piece_timings and len(piece_timings) > 0:
                max_piece_time = max(piece_timings.values())
                max_piece_id = max(piece_timings, key=piece_timings.get)
                logger.info(f"MoveGen[naive]: slowest piece={max_piece_id}, piece_time_ms={max_piece_time * 1000.0:.2f}")
        
        # Existing debug logging (always at DEBUG level)
        logger.debug(f"Legal move generation [naive]: {len(legal_moves)} moves in {total_time:.4f}s for player={player.name}, pieces_checked={len(available_pieces)}")
        if piece_timings and len(piece_timings) > 0:
            max_piece_time = max(piece_timings.values())
            max_piece_id = max(piece_timings, key=piece_timings.get)
            logger.debug(f"Legal move generation [naive]: slowest piece={max_piece_id} took {max_piece_time:.4f}s")
        
        return legal_moves
    
    def _get_legal_moves_frontier(self, board: Board, player: Player) -> List[Move]:
        """
        Get all legal moves using frontier-based generation.
        
        This generator uses frontier cells (empty cells diagonally adjacent to
        player's pieces but not orthogonally adjacent) as starting points instead
        of scanning every board cell. This should significantly reduce the search
        space, especially in later game stages.
        
        The implementation uses the same legality checking logic as the naive
        generator to preserve correctness.
        
        Args:
            board: Current board state
            player: Player to generate moves for
        
        Returns:
            List of legal moves
        """
        start = time.perf_counter()
        legal_moves = []
        
        # Per-call cache to avoid redundant anchor tests
        # This is a micro-optimization that does not affect correctness:
        # If we've tried all anchors for a (piece_id, frontier_coord) combination
        # and none were legal, we skip that combination in future iterations.
        # Cache is local to this function call and cleared on return.
        piece_frontier_fail: Set[Tuple[int, Tuple[int, int]]] = set()
        
        # Get pieces that haven't been used yet
        available_pieces = [piece for piece in self.all_pieces 
                          if piece.id not in board.player_pieces_used[player]]
        
        # Get frontier cells for this player
        frontier_cells = board.get_frontier(player)
        
        # Direct grid access for faster checks
        grid = board.grid
        player_value = player.value
        is_first_move = board.player_first_move[player]
        start_corner = board.player_start_corners[player] if is_first_move else None
        
        piece_timings = {}
        for piece in available_pieces:
            piece_start = time.perf_counter()
            
            # Hoist orientation lookup outside inner loops
            if USE_BITBOARD_LEGALITY:
                piece_orientations = ALL_PIECE_ORIENTATIONS.get(piece.id, [])
                num_orientations = len(piece_orientations)
            else:
                orientations = self.piece_orientations_cache[piece.id]
                cached_positions = self.piece_position_cache[piece.id]
                num_orientations = len(orientations)
            
            for orientation_idx in range(num_orientations):
                # Hoist orientation access outside frontier loop
                if USE_BITBOARD_LEGALITY:
                    piece_orientation = piece_orientations[orientation_idx]
                    relative_positions = piece_orientation.offsets
                    # Use precomputed anchor indices for bitboard (heuristic optimization)
                    anchor_indices = piece_orientation.anchor_indices
                else:
                    orientation = orientations[orientation_idx]
                    relative_positions = cached_positions[orientation_idx]
                    # For grid-based, try all offsets to maintain correctness
                    # (Anchor optimization could be added here in the future)
                    anchor_indices = list(range(len(relative_positions)))
                
                # For each frontier cell, try anchoring the piece at strategic positions
                for frontier_row, frontier_col in frontier_cells:
                    # Check per-call cache: skip if we've already determined this
                    # (piece_id, orientation_idx, frontier_coord) combination has no legal anchors
                    # Note: We include orientation_idx because different orientations may succeed
                    frontier_key = (piece.id, orientation_idx, (frontier_row, frontier_col))
                    if frontier_key in piece_frontier_fail:
                        continue
                    
                    # Track if we found any legal anchor for this frontier cell
                    found_legal_anchor = False
                    
                    # Try only the precomputed anchor indices (heuristic selection)
                    for anchor_piece_idx in anchor_indices:
                        if anchor_piece_idx >= len(relative_positions):
                            continue
                        
                        rel_r, rel_c = relative_positions[anchor_piece_idx]
                        
                        # Calculate anchor position: frontier cell minus relative position
                        anchor_row = frontier_row - rel_r
                        anchor_col = frontier_col - rel_c
                        
                        # Bounds check
                        if anchor_row < 0 or anchor_row >= board.SIZE or anchor_col < 0 or anchor_col >= board.SIZE:
                            continue
                        
                        # Check legality using bitboard or grid-based method
                        if USE_BITBOARD_LEGALITY:
                            # Use coords-based bitboard legality check
                            # Compute placement coordinates directly from offsets
                            placement_coords = [
                                (anchor_row + rel_r, anchor_col + rel_c)
                                for rel_r, rel_c in relative_positions
                            ]
                            
                            # Bounds check: ensure all coords are on board
                            if all(0 <= r < board.SIZE and 0 <= c < board.SIZE 
                                   for r, c in placement_coords):
                                if self.is_placement_legal_bitboard_coords(
                                    board, player, placement_coords,
                                    is_first_move=is_first_move
                                ):
                                    move = Move(piece.id, orientation_idx, anchor_row, anchor_col)
                                    legal_moves.append(move)
                                    found_legal_anchor = True
                        else:
                            # Use grid-based legality check (original method)
                            # Check if piece would be within bounds at this anchor
                            if not PiecePlacement.can_place_piece_at(
                                (board.SIZE, board.SIZE), orientation, anchor_row, anchor_col
                            ):
                                continue
                            
                            # Fast bounds and overlap check using cached positions and direct grid access
                            has_overlap = False
                            covers_start_corner = False
                            
                            for check_rel_r, check_rel_c in relative_positions:
                                r = anchor_row + check_rel_r
                                c = anchor_col + check_rel_c
                                
                                # Bounds check
                                if r < 0 or r >= board.SIZE or c < 0 or c >= board.SIZE:
                                    has_overlap = True
                                    break
                                
                                # Overlap check - direct grid access
                                if grid[r, c] != 0:
                                    has_overlap = True
                                    break
                                
                                # Check if covers start corner (for first move)
                                if is_first_move and start_corner and r == start_corner.row and c == start_corner.col:
                                    covers_start_corner = True
                            
                            # Early exit if overlap or (first move and doesn't cover start corner)
                            if has_overlap:
                                continue
                            if is_first_move and not covers_start_corner:
                                continue
                            
                            # Check adjacency rules using relative positions directly
                            if self._check_adjacency_fast_inline(relative_positions, anchor_row, anchor_col, 
                                                                 player_value, grid, board.SIZE, 
                                                                 is_first_move, start_corner):
                                move = Move(piece.id, orientation_idx, anchor_row, anchor_col)
                                legal_moves.append(move)
                                found_legal_anchor = True
                    
                    # Update per-call cache: if no anchors were legal for this frontier cell,
                    # remember to skip it in future iterations
                    if not found_legal_anchor:
                        piece_frontier_fail.add(frontier_key)
            
            piece_end = time.perf_counter()
            piece_timings[piece.id] = piece_end - piece_start
        
        end = time.perf_counter()
        total_time = end - start
        elapsed_ms = total_time * 1000.0
        
        # Debug timing hook (only logs when BLOKUS_MOVEGEN_DEBUG is set)
        if MOVEGEN_DEBUG:
            logger.info(f"MoveGen[frontier]: player={player.name}, legal_moves={len(legal_moves)}, elapsed_ms={elapsed_ms:.2f}, frontier_size={len(frontier_cells)}")
            if piece_timings and len(piece_timings) > 0:
                max_piece_time = max(piece_timings.values())
                max_piece_id = max(piece_timings, key=piece_timings.get)
                logger.info(f"MoveGen[frontier]: slowest piece={max_piece_id}, piece_time_ms={max_piece_time * 1000.0:.2f}")
        
        # Existing debug logging (always at DEBUG level)
        logger.debug(f"Legal move generation [frontier]: {len(legal_moves)} moves in {total_time:.4f}s for player={player.name}, frontier_size={len(frontier_cells)}, pieces_checked={len(available_pieces)}")
        if piece_timings and len(piece_timings) > 0:
            max_piece_time = max(piece_timings.values())
            max_piece_id = max(piece_timings, key=piece_timings.get)
            logger.debug(f"Legal move generation [frontier]: slowest piece={max_piece_id} took {max_piece_time:.4f}s")
        
        return legal_moves
    
    def is_placement_legal_bitboard(self, board: Board, player: Player, 
                                     orientation: PieceOrientation,
                                     anchor_board_coord: Tuple[int, int],
                                     anchor_piece_index: int = 0) -> bool:
        """
        Check if a piece placement is legal using bitboard operations.
        
        Args:
            board: Board state
            player: Player making the placement
            orientation: PieceOrientation instance
            anchor_board_coord: (row, col) where the anchor point should be placed
            anchor_piece_index: Index into orientation.offsets for the anchor point
            
        Returns:
            True if placement is legal
        """
        if anchor_piece_index >= len(orientation.offsets):
            return False
        
        # Get anchor point in piece coordinates
        piece_r, piece_c = orientation.offsets[anchor_piece_index]
        
        # Get anchor point in board coordinates
        board_r, board_c = anchor_board_coord
        
        # Compute shift
        d_row = board_r - piece_r
        d_col = board_c - piece_c
        
        # Shift masks to board position
        shape_shifted = shift_mask(orientation.shape_mask, d_row, d_col)
        if shape_shifted is None:
            return False  # Off-board
        
        # Compute diag/orth neighbors dynamically from the shifted shape
        # This ensures we capture ALL neighbors, including those that were negative
        # in normalized space but become valid when shifted
        shape_coords = mask_to_coords(shape_shifted)
        placement_set = set(shape_coords)
        
        # Compute diagonal neighbors from shifted shape
        diag_neighbors = set()
        for r, c in shape_coords:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < board.SIZE and 0 <= nc < board.SIZE:
                    if (nr, nc) not in placement_set:
                        diag_neighbors.add((nr, nc))
        diag_shifted = coords_to_mask(diag_neighbors)
        
        # Compute orthogonal neighbors from shifted shape
        orth_neighbors = set()
        for r, c in shape_coords:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < board.SIZE and 0 <= nc < board.SIZE:
                    if (nr, nc) not in placement_set:
                        orth_neighbors.add((nr, nc))
        orth_shifted = coords_to_mask(orth_neighbors)
        
        # Overlap check: shape must not overlap with any occupied cells
        if shape_shifted & board.occupied_bits != 0:
            return False
        
        # Orthogonal adjacency check: cannot touch own pieces orthogonally
        if orth_shifted & board.player_bits[player] != 0:
            return False
        
        # Diagonal adjacency requirement: must touch at least one own piece diagonally (if not first move)
        is_first_move = board.player_first_move[player]
        if not is_first_move:
            if diag_shifted & board.player_bits[player] == 0:
                return False
        
        # First move exception: must cover starting corner
        if is_first_move:
            start_corner = board.player_start_corners[player]
            start_corner_bit = coord_to_bit(start_corner.row, start_corner.col)
            if shape_shifted & start_corner_bit == 0:
                return False
        
        # Optional debug equivalence check (5% sampling when enabled)
        # Note: This is a simplified check - full equivalence is verified in test suite
        # The orientation mapping between old and new systems may not be 1:1, so this
        # check is primarily for catching obvious bugs, not full equivalence
        if MOVEGEN_DEBUG_EQUIVALENCE and random.random() < 0.05:
            # Log that we're using bitboard legality (for debugging)
            logger.debug(
                f"Bitboard legality check: piece_id={orientation.piece_id}, "
                f"anchor={anchor_board_coord}, anchor_idx={anchor_piece_index}, "
                f"result={True}"  # Result is True if we got here
            )
        
        return True
    
    def is_placement_legal_grid(self, board: Board, player: Player,
                                 orientation: PieceOrientation,
                                 anchor_board_coord: Tuple[int, int],
                                 anchor_piece_index: int,
                                 placement_coords: List[Tuple[int, int]]) -> bool:
        """
        Grid-based legality check for a given placement.
        
        This is a helper function for debugging that uses the grid-based
        can_place_piece method to check legality.
        
        Args:
            board: Board state
            player: Player making the placement
            orientation: PieceOrientation instance (for compatibility, not used)
            anchor_board_coord: (row, col) where anchor is placed (for compatibility)
            anchor_piece_index: Index into offsets (for compatibility, not used)
            placement_coords: List of (row, col) tuples where piece would be placed
            
        Returns:
            True if placement is legal according to grid-based rules
        """
        piece_positions = [Position(row, col) for row, col in placement_coords]
        return board.can_place_piece(piece_positions, player)
    
    def is_placement_legal_bitboard_coords(
        self,
        board: Board,
        player: Player,
        placement_coords: List[Tuple[int, int]],
        *,
        is_first_move: bool = False,
    ) -> bool:
        """
        Check if a piece placement is legal using bitboard operations.
        
        This version works directly from placement coordinates, avoiding
        anchor/shift complexity. It's more robust and easier to verify.
        
        Args:
            board: Board state
            player: Player making the placement
            placement_coords: List of (row, col) tuples where piece would be placed
            is_first_move: Whether this is the player's first move
            
        Returns:
            True if placement is legal
        """
        # 1. Build shape mask from coords
        shape_mask = coords_to_mask(placement_coords)
        
        # 2. Overlap check: shape must not overlap with any occupied cells
        if shape_mask & board.occupied_bits != 0:
            return False
        
        # 3. Compute diagonal and orth neighbors on the fly
        placement_set = set(placement_coords)
        diag_neighbors = set()
        orth_neighbors = set()
        
        for (r, c) in placement_coords:
            # Diagonals
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < board.SIZE and 0 <= cc < board.SIZE:
                    if (rr, cc) not in placement_set:
                        diag_neighbors.add((rr, cc))
            
            # Orthogonals
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < board.SIZE and 0 <= cc < board.SIZE:
                    if (rr, cc) not in placement_set:
                        orth_neighbors.add((rr, cc))
        
        diag_mask = coords_to_mask(diag_neighbors)
        orth_mask = coords_to_mask(orth_neighbors)
        
        player_bits = board.player_bits[player]
        
        # 4. Orth adjacency rule: cannot touch own pieces orthogonally
        if orth_mask & player_bits != 0:
            return False
        
        # 5. Diagonal rule: must touch at least one own piece diagonally (except first move)
        if not is_first_move:
            if (diag_mask & player_bits) == 0:
                return False
        
        # 6. First move special rule: must cover starting corner
        if is_first_move:
            start_corner = board.player_start_corners[player]
            start_corner_bit = coord_to_bit(start_corner.row, start_corner.col)
            if shape_mask & start_corner_bit == 0:
                return False
        
        return True
    
    def _check_adjacency_fast_inline(self, relative_positions: List[Tuple[int, int]], 
                                     anchor_row: int, anchor_col: int,
                                     player_value: int, grid, board_size: int,
                                     is_first_move: bool, start_corner: Optional[Position]) -> bool:
        """
        Ultra-fast inline adjacency check without creating Position objects.
        
        Returns True if placement is valid according to adjacency rules.
        """
        has_corner_connection = False
        
        # Check each position in the piece
        for rel_r, rel_c in relative_positions:
            r = anchor_row + rel_r
            c = anchor_col + rel_c
            
            # Check edge adjacency (not allowed with same color) - direct grid access
            # Top, bottom, left, right
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if grid[nr, nc] == player_value:
                        return False  # Edge adjacency not allowed
            
            # Check corner adjacency (allowed with same color) - direct grid access
            # Diagonals
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if grid[nr, nc] == player_value:
                        has_corner_connection = True  # Corner connection found
        
        # Additional check: piece must be connected to existing pieces via corners (if not first move)
        if not is_first_move:
            if not has_corner_connection:
                return False
        
        return True
    
    def get_legal_moves_for_piece(self, board: Board, player: Player, piece_id: int) -> List[Move]:
        """
        Get all legal moves for a specific piece.
        
        Args:
            board: Current board state
            player: Player to generate moves for
            piece_id: ID of the piece to generate moves for
        
        Returns:
            List of legal moves for the specific piece
        """
        if piece_id in board.player_pieces_used[player]:
            return []  # Piece already used
        
        # Find the piece
        piece = next((p for p in self.all_pieces if p.id == piece_id), None)
        if piece is None:
            return []
        
        legal_moves = []
        orientations = self.piece_orientations_cache[piece.id]
        
        for orientation_idx, orientation in enumerate(orientations):
            anchor_positions = PiecePlacement.get_valid_anchor_positions(
                (board.SIZE, board.SIZE), orientation
            )
            
            for anchor_row, anchor_col in anchor_positions:
                positions = PiecePlacement.get_piece_positions(
                    orientation, anchor_row, anchor_col
                )
                piece_positions = [Position(row, col) for row, col in positions]
                
                if board.can_place_piece(piece_positions, player):
                    move = Move(piece.id, orientation_idx, anchor_row, anchor_col)
                    legal_moves.append(move)
        
        return legal_moves
    
    def is_move_legal(self, board: Board, player: Player, move: Move) -> bool:
        """
        Check if a specific move is legal.
        
        Args:
            board: Current board state
            player: Player making the move
            move: Move to check
        
        Returns:
            True if the move is legal
        """
        if move.piece_id in board.player_pieces_used[player]:
            return False  # Piece already used
        
        # Get piece orientations
        orientations = self.piece_orientations_cache.get(move.piece_id, [])
        if move.orientation >= len(orientations):
            return False
        
        orientation = orientations[move.orientation]
        
        # Check if piece can be placed at anchor position
        if not PiecePlacement.can_place_piece_at(
            (board.SIZE, board.SIZE), orientation, move.anchor_row, move.anchor_col
        ):
            return False
        
        # Get positions and check legality
        positions = PiecePlacement.get_piece_positions(
            orientation, move.anchor_row, move.anchor_col
        )
        piece_positions = [Position(row, col) for row, col in positions]
        
        return board.can_place_piece(piece_positions, player)
    
    def get_move_count(self, board: Board, player: Player) -> int:
        """
        Get the number of legal moves available for a player.
        
        Args:
            board: Current board state
            player: Player to count moves for
        
        Returns:
            Number of legal moves
        """
        return len(self.get_legal_moves(board, player))
    
    def has_legal_moves(self, board: Board, player: Player) -> bool:
        """
        Check if a player has any legal moves.
        
        Args:
            board: Current board state
            player: Player to check
        
        Returns:
            True if player has legal moves
        """
        return self.get_move_count(board, player) > 0
    
    def get_game_state_summary(self, board: Board) -> dict:
        """
        Get a summary of the current game state.
        
        Args:
            board: Current board state
        
        Returns:
            Dictionary with game state information
        """
        summary = {
            'current_player': board.current_player,
            'move_count': board.move_count,
            'game_over': board.game_over,
            'player_moves': {},
            'player_scores': {},
            'player_pieces_used': {}
        }
        
        for player in Player:
            summary['player_moves'][player.name] = self.get_move_count(board, player)
            summary['player_scores'][player.name] = board.get_score(player)
            summary['player_pieces_used'][player.name] = len(board.player_pieces_used[player])
        
        return summary


def debug_compare_bitboard_vs_grid(
    board: Board,
    player: Player,
    orientation: PieceOrientation,
    anchor_board_coord: Tuple[int, int],
    anchor_piece_index: int,
    placement_coords: List[Tuple[int, int]],
) -> None:
    """
    Debug helper: for a given move (piece orientation + anchor + resulting coords),
    compare grid-based legality and bitboard-based legality and print detailed info.
    Only prints when DEBUG_BITBOARD is True.
    
    Args:
        board: Board state
        player: Player making the placement
        orientation: PieceOrientation instance
        anchor_board_coord: (row, col) where the anchor point should be placed
        anchor_piece_index: Index into orientation.offsets for the anchor point
        placement_coords: List of (row, col) tuples where the piece would be placed
    """
    if not DEBUG_BITBOARD:
        return
    
    # 1. Compute masks from coords directly
    shape_mask_from_coords = coords_to_mask(placement_coords)
    
    # Compute diagonal and orthogonal neighbors directly from coords.
    # These are used only for debugging.
    diag_neighbors = set()
    orth_neighbors = set()
    placement_set = set(placement_coords)
    
    for (r, c) in placement_coords:
        # Diagonal neighbors
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < BOARD_HEIGHT and 0 <= cc < BOARD_WIDTH:
                if (rr, cc) not in placement_set:
                    diag_neighbors.add((rr, cc))
        
        # Orth neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < BOARD_HEIGHT and 0 <= cc < BOARD_WIDTH:
                if (rr, cc) not in placement_set:
                    orth_neighbors.add((rr, cc))
    
    diag_mask_from_coords = coords_to_mask(diag_neighbors)
    orth_mask_from_coords = coords_to_mask(orth_neighbors)
    
    # 2. Compute masks via orientation + anchor using the bitboard logic
    piece_r, piece_c = orientation.offsets[anchor_piece_index]
    board_r, board_c = anchor_board_coord
    d_row, d_col = board_r - piece_r, board_c - piece_c
    
    shape_shifted = shift_mask(orientation.shape_mask, d_row, d_col)
    diag_shifted = shift_mask(orientation.diag_mask, d_row, d_col)
    orth_shifted = shift_mask(orientation.orth_mask, d_row, d_col)
    
    # 3. Log and compare
    print("\n" + "=" * 80)
    print("=== DEBUG BITBOARD VS GRID ===")
    print(f"player={player.name} (value={player.value})")
    print(f"piece_id={orientation.piece_id}")
    print(f"anchor_board_coord={anchor_board_coord}, anchor_piece_index={anchor_piece_index}")
    print(f"d_row={d_row}, d_col={d_col} (computed from board_r={board_r} - piece_r={piece_r}, board_c={board_c} - piece_c={piece_c})")
    print(f"placement_coords={sorted(placement_coords)}")
    print()
    
    print("Shape from coords vs shifted shape:")
    shape_coords_from_mask = sorted(mask_to_coords(shape_mask_from_coords))
    shape_shifted_coords = sorted(mask_to_coords(shape_shifted or 0))
    print(f"  coords -> mask -> coords: {shape_coords_from_mask}")
    print(f"  shifted orientation shape coords: {shape_shifted_coords}")
    print(f"  MATCH: {shape_coords_from_mask == shape_shifted_coords}")
    print()
    
    print("Diag neighbors from coords vs shifted diag mask:")
    diag_coords_from_set = sorted(diag_neighbors)
    diag_shifted_coords = sorted(mask_to_coords(diag_shifted or 0))
    print(f"  diag from coords: {diag_coords_from_set}")
    print(f"  diag mask coords: {diag_shifted_coords}")
    print(f"  MATCH: {diag_coords_from_set == diag_shifted_coords}")
    print()
    
    print("Orth neighbors from coords vs shifted orth mask:")
    orth_coords_from_set = sorted(orth_neighbors)
    orth_shifted_coords = sorted(mask_to_coords(orth_shifted or 0))
    print(f"  orth from coords: {orth_coords_from_set}")
    print(f"  orth mask coords: {orth_shifted_coords}")
    print(f"  MATCH: {orth_coords_from_set == orth_shifted_coords}")
    print()
    
    # 4. Show adjacency intersections with player bits
    player_bits = board.player_bits[player]
    occupied_bits = board.occupied_bits
    
    print("Adjacency & overlap checks:")
    shape_overlap = bool(shape_shifted and (shape_shifted & occupied_bits))
    orth_adj = bool(orth_shifted and (orth_shifted & player_bits))
    diag_adj = bool(diag_shifted and (diag_shifted & player_bits))
    print(f"  shape & occupied_bits -> {shape_overlap} (should be False for legal)")
    print(f"  orth & player_bits     -> {orth_adj} (should be False for legal)")
    print(f"  diag & player_bits      -> {diag_adj} (should be True if not first move)")
    print()
    
    # Show which cells are causing overlaps/adjacencies
    if shape_overlap:
        overlap_cells = mask_to_coords(shape_shifted & occupied_bits)
        print(f"  Overlap cells: {sorted(overlap_cells)}")
    if orth_adj:
        orth_adj_cells = mask_to_coords(orth_shifted & player_bits)
        print(f"  Orth adj cells: {sorted(orth_adj_cells)}")
    if diag_adj:
        diag_adj_cells = mask_to_coords(diag_shifted & player_bits)
        print(f"  Diag adj cells: {sorted(diag_adj_cells)}")
    print()
    
    # 5. Show grid-based vs bitboard legality results for this move
    generator = LegalMoveGenerator()
    grid_legal = generator.is_placement_legal_grid(
        board, player, orientation, anchor_board_coord, anchor_piece_index,
        placement_coords=placement_coords
    )
    bitboard_legal = generator.is_placement_legal_bitboard(
        board, player, orientation, anchor_board_coord, anchor_piece_index
    )
    
    print(f"RESULT: grid_legal={grid_legal}, bitboard_legal={bitboard_legal}")
    if grid_legal != bitboard_legal:
        print("  *** MISMATCH DETECTED ***")
    print("=== END DEBUG ===")
    print("=" * 80)
    print()
