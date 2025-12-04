"""
Legal move generator for Blokus game.
"""

import os
import time
import logging
from typing import List, Tuple, Set, Optional
from .board import Board, Player, Position
from .pieces import PieceGenerator, PiecePlacement

logger = logging.getLogger(__name__)

# Debug flag for move generation timing (controlled via environment variable)
MOVEGEN_DEBUG = bool(os.getenv("BLOKUS_MOVEGEN_DEBUG", ""))


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
        
        OPTIMIZED: Uses fast bounds/overlap checks before expensive adjacency validation.
        
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
            logger.info(f"MoveGen: player={player.name}, legal_moves={len(legal_moves)}, elapsed_ms={elapsed_ms:.2f}")
            if piece_timings and len(piece_timings) > 0:
                max_piece_time = max(piece_timings.values())
                max_piece_id = max(piece_timings, key=piece_timings.get)
                logger.info(f"MoveGen: slowest piece={max_piece_id}, piece_time_ms={max_piece_time * 1000.0:.2f}")
        
        # Existing debug logging (always at DEBUG level)
        logger.debug(f"Legal move generation: {len(legal_moves)} moves in {total_time:.4f}s for player={player.name}, pieces_checked={len(available_pieces)}")
        if piece_timings and len(piece_timings) > 0:
            max_piece_time = max(piece_timings.values())
            max_piece_id = max(piece_timings, key=piece_timings.get)
            logger.debug(f"Legal move generation: slowest piece={max_piece_id} took {max_piece_time:.4f}s")
        
        return legal_moves
    
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
