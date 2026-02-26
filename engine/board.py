

"""
Blokus Board implementation with 20x20 grid and game state management.
"""

import os
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .bitboard import coord_to_bit, coords_to_mask


class Player(Enum):
    """Player enumeration."""
    RED = 1
    BLUE = 2
    YELLOW = 3
    GREEN = 4


@dataclass
class Position:
    """Represents a position on the board."""
    row: int
    col: int
    
    def __hash__(self):
        return hash((self.row, self.col))
    
    def __eq__(self, other):
        return self.row == other.row and self.col == other.col


class Board:
    """
    Blokus game board implementation.
    
    The board is a 20x20 grid where:
    - 0 represents empty space
    - 1-4 represent players (RED, BLUE, YELLOW, GREEN)
    """
    
    SIZE = 20
    
    def __init__(self):
        self.grid = np.zeros((self.SIZE, self.SIZE), dtype=int)
        self.player_start_corners = {
            Player.RED: Position(0, 0),
            Player.BLUE: Position(0, self.SIZE - 1),
            Player.YELLOW: Position(self.SIZE - 1, self.SIZE - 1),
            Player.GREEN: Position(self.SIZE - 1, 0),
        }
        self.player_pieces_used = {player: set() for player in Player}
        self.player_first_move = {player: True for player in Player}
        self.game_over = False
        self.current_player = Player.RED
        self.move_count = 0
        # Frontier tracking: per-player set of (row, col) tuples
        # Frontier = empty cells diagonally adjacent to player's pieces, 
        # but NOT orthogonally adjacent to player's pieces
        self.player_frontiers: Dict[Player, Set[Tuple[int, int]]] = {
            player: set() for player in Player
        }
        # Initialize frontiers for all players
        self.init_frontiers()
        
        # Bitboard state: maintain bit-level occupancy for efficient operations
        self.occupied_bits: int = 0  # Bitmask of all occupied cells
        self.player_bits: Dict[Player, int] = defaultdict(int)  # Bitmask per player
    
    def is_valid_position(self, pos: Position) -> bool:
        """Check if position is within board bounds."""
        return 0 <= pos.row < self.SIZE and 0 <= pos.col < self.SIZE
    
    def get_cell(self, pos: Position) -> int:
        """Get the value at a position."""
        if not self.is_valid_position(pos):
            return -1  # Invalid position
        return self.grid[pos.row, pos.col]
    
    def set_cell(self, pos: Position, value: int) -> None:
        """Set the value at a position."""
        if self.is_valid_position(pos):
            self.grid[pos.row, pos.col] = value
    
    def is_empty(self, pos: Position) -> bool:
        """Check if a position is empty."""
        return self.get_cell(pos) == 0
    
    def get_player_at(self, pos: Position) -> Optional[Player]:
        """Get the player at a position, or None if empty."""
        value = self.get_cell(pos)
        if value == 0:
            return None
        return Player(value)
    
    def get_adjacent_positions(self, pos: Position) -> List[Position]:
        """Get all adjacent positions (including diagonals)."""
        positions = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_pos = Position(pos.row + dr, pos.col + dc)
                if self.is_valid_position(new_pos):
                    positions.append(new_pos)
        return positions
    
    def get_edge_adjacent_positions(self, pos: Position) -> List[Position]:
        """Get positions that share an edge (not diagonal)."""
        positions = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = Position(pos.row + dr, pos.col + dc)
            if self.is_valid_position(new_pos):
                positions.append(new_pos)
        return positions
    
    def get_corner_adjacent_positions(self, pos: Position) -> List[Position]:
        """Get positions that are diagonally adjacent (corner touching)."""
        positions = []
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            new_pos = Position(pos.row + dr, pos.col + dc)
            if self.is_valid_position(new_pos):
                positions.append(new_pos)
        return positions
    
    def can_place_piece(self, piece_positions: List[Position], player: Player) -> bool:
        """
        Check if a piece can be placed at the given positions.
        
        Rules:
        1. First move must cover player's start corner
        2. Pieces of same color must touch only at corners
        3. No edge-to-edge adjacency with same color
        
        OPTIMIZED: Uses direct grid access to avoid Position object overhead.
        """
        if not piece_positions:
            return False
        
        player_value = player.value
        grid = self.grid  # Direct reference for faster access
        
        # Fast bounds and overlap check using direct grid access
        for pos in piece_positions:
            r, c = pos.row, pos.col
            # Bounds check
            if r < 0 or r >= self.SIZE or c < 0 or c >= self.SIZE:
                return False
            # Overlap check - direct grid access
            if grid[r, c] != 0:
                return False
        
        # Rule 1: First move must cover player's start corner
        if self.player_first_move[player]:
            start_corner = self.player_start_corners[player]
            if start_corner not in piece_positions:
                return False
        
        # Rule 2 & 3: Adjacency rules (optimized)
        if not self._check_adjacency_rules_fast(piece_positions, player_value, grid):
            return False
        
        return True
    
    def _check_adjacency_rules(self, piece_positions: List[Position], player: Player) -> bool:
        """
        Check adjacency rules for piece placement.
        
        - Pieces of same color must touch only at corners
        - No edge-to-edge adjacency with same color
        
        DEPRECATED: Use _check_adjacency_rules_fast for better performance.
        """
        return self._check_adjacency_rules_fast(piece_positions, player.value, self.grid)
    
    def _check_adjacency_rules_fast(self, piece_positions: List[Position], player_value: int, grid: np.ndarray) -> bool:
        """
        Optimized adjacency rules check using direct grid access.
        
        - Pieces of same color must touch only at corners
        - No edge-to-edge adjacency with same color
        """
        has_corner_connection = False
        
        # Check each position in the piece
        for pos in piece_positions:
            r, c = pos.row, pos.col
            
            # Check edge adjacency (not allowed with same color) - direct grid access
            # Top, bottom, left, right
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                    if grid[nr, nc] == player_value:
                        return False  # Edge adjacency not allowed
            
            # Check corner adjacency (allowed with same color) - direct grid access
            # Diagonals
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                    if grid[nr, nc] == player_value:
                        has_corner_connection = True  # Corner connection found
        
        # Additional check: piece must be connected to existing pieces via corners (if not first move)
        if not self.player_first_move[Player(player_value)]:
            if not has_corner_connection:
                return False
        
        return True
    
    def _is_connected_via_corners(self, piece_positions: List[Position], player: Player) -> bool:
        """
        Check if the piece is connected to existing pieces of the same color via corners.
        
        DEPRECATED: This is now handled in _check_adjacency_rules_fast for better performance.
        """
        # For first move, this is handled by the start corner rule
        if self.player_first_move[player]:
            return True
        
        player_value = player.value
        grid = self.grid
        
        # Check if any position in the piece touches an existing piece at a corner
        for pos in piece_positions:
            r, c = pos.row, pos.col
            # Check diagonals - direct grid access
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                    if grid[nr, nc] == player_value:
                        return True
        
        return False
    
    def get_frontier(self, player: Player) -> Set[Tuple[int, int]]:
        """
        Get the current frontier cells for a player.
        
        Frontier = empty cells that are:
        - Diagonally adjacent to at least one cell occupied by the player
        - NOT orthogonally adjacent to any cell occupied by the player
        
        Returns:
            Set of (row, col) tuples representing frontier cells
        """
        return self.player_frontiers[player].copy()
    
    def _compute_full_frontier(self, player: Player) -> Set[Tuple[int, int]]:
        """
        Recompute the frontier from scratch based on the current board state.
        
        This is the source of truth for frontier computation. It scans the entire
        board to find all cells that meet the frontier criteria:
        - On the board
        - Empty (no piece placed)
        - Diagonally adjacent to at least one cell occupied by player
        - NOT orthogonally adjacent to any cell occupied by player
        
        Args:
            player: Player to compute frontier for
            
        Returns:
            Set of (row, col) tuples representing frontier cells
        """
        frontier = set()
        player_value = player.value
        grid = self.grid
        
        # Scan all board positions
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                # Must be empty
                if grid[r, c] != 0:
                    continue
                
                # Check if diagonally adjacent to player's pieces
                has_diagonal_adjacency = False
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                        if grid[nr, nc] == player_value:
                            has_diagonal_adjacency = True
                            break
                
                if not has_diagonal_adjacency:
                    continue
                
                # Check if orthogonally adjacent to player's pieces (must NOT be)
                has_orthogonal_adjacency = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                        if grid[nr, nc] == player_value:
                            has_orthogonal_adjacency = True
                            break
                
                if not has_orthogonal_adjacency:
                    frontier.add((r, c))
        
        return frontier
    
    def update_frontier_after_move(self, player: Player, placed_cells: List[Tuple[int, int]]) -> None:
        """
        Incrementally update the frontier after a piece is placed.
        
        This method updates the frontier by:
        1. Removing placed cells from the frontier (they're now occupied)
        2. Adding new frontier candidates from diagonal neighbors of placed cells
        3. Removing cells that become orthogonally adjacent to player's pieces
        
        Note: This method does NOT rely on previous frontier correctness beyond
        removing/adding around placed_cells. The full recompute method remains
        the source of truth for testing.
        
        Args:
            player: Player who placed the piece
            placed_cells: List of (row, col) tuples for the newly placed piece
        """
        player_value = player.value
        grid = self.grid
        frontier = self.player_frontiers[player]
        
        # For each placed cell
        for r, c in placed_cells:
            # Remove this cell from frontier (it's now occupied)
            frontier.discard((r, c))
            
            # Check diagonal neighbors (potential new frontier cells)
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                    # Must be empty
                    if grid[nr, nc] != 0:
                        continue
                    
                    # Check if orthogonally adjacent to any of player's pieces
                    is_orth_adjacent = False
                    for odr, odc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        onr, onc = nr + odr, nc + odc
                        if 0 <= onr < self.SIZE and 0 <= onc < self.SIZE:
                            if grid[onr, onc] == player_value:
                                is_orth_adjacent = True
                                break
                    
                    # Add to frontier if not orthogonally adjacent
                    if not is_orth_adjacent:
                        frontier.add((nr, nc))
            
            # Check orthogonal neighbors (remove from frontier if present)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                    # Remove from frontier (orthogonal adjacency not allowed)
                    frontier.discard((nr, nc))
    
    def init_frontiers(self) -> None:
        """
        Initialize frontiers for all players.
        
        This method clears all frontier sets and initializes them based on
        the starting corners. For a fresh board, each player's initial frontier
        contains only their starting corner (which must be covered by the first move).
        
        This should be called:
        - From Board.__init__ when creating a new board
        - When resetting the game to initial state
        """
        # Clear all frontiers
        for player in Player:
            self.player_frontiers[player].clear()
        
        # Initialize each player's frontier with their starting corner
        for player in Player:
            self.init_frontier_for_player(player)
    
    def init_frontier_for_player(self, player: Player) -> None:
        """
        Initialize the frontier for a single player based on their starting corner.
        
        For a fresh board, the initial frontier is the player's starting corner
        (since it's empty and will be the first placement location).
        
        Args:
            player: Player to initialize frontier for
        """
        start_corner = self.player_start_corners[player]
        # Initially, the starting corner is the only frontier cell
        # (it's empty and will be covered by the first move)
        # Note: This is a special case - normally frontier requires diagonal adjacency,
        # but on first move there are no pieces yet, so we allow the starting corner
        if self.is_empty(start_corner):
            self.player_frontiers[player].add((start_corner.row, start_corner.col))
    
    def debug_rebuild_frontier(self, player: Player) -> bool:
        """
        Debug helper: recompute frontier from scratch and verify consistency.
        
        This method recomputes the frontier using _compute_full_frontier and
        compares it with the current incremental frontier. 
        
        In debug mode (BLOKUS_FRONTIER_DEBUG env var set), this will assert
        that the frontiers match and log helpful error messages if they don't.
        
        Args:
            player: Player to rebuild frontier for
            
        Returns:
            True if incremental frontier matches full recompute, False otherwise
        """
        computed_frontier = self._compute_full_frontier(player)
        current_frontier = self.player_frontiers[player]
        
        if computed_frontier != current_frontier:
            # Find differences
            extra_cells = current_frontier - computed_frontier
            missing_cells = computed_frontier - current_frontier
            
            # Log helpful error message
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"Frontier mismatch for {player.name}: "
                f"incremental has {len(current_frontier)} cells, "
                f"computed has {len(computed_frontier)} cells"
            )
            if extra_cells:
                logger.error(f"  Extra cells in incremental frontier: {sorted(extra_cells)}")
            if missing_cells:
                logger.error(f"  Missing cells in incremental frontier: {sorted(missing_cells)}")
            
            # In debug mode, assert equality
            FRONTIER_DEBUG = bool(os.getenv("BLOKUS_FRONTIER_DEBUG", ""))
            if FRONTIER_DEBUG:
                raise AssertionError(
                    f"Frontier mismatch for {player.name}: "
                    f"incremental={sorted(current_frontier)}, "
                    f"computed={sorted(computed_frontier)}"
                )
            
            # Update to match computed frontier
            self.player_frontiers[player] = computed_frontier
            return False
        
        return True
    
    def _verify_frontier_consistency(self, player: Player) -> bool:
        """
        Verify that the frontier for a player is consistent with board state.
        
        Checks:
        - Every frontier cell is empty
        - Every frontier cell is diagonally adjacent to player's pieces (or is the starting corner on first move)
        - No frontier cell is orthogonally adjacent to player's pieces
        
        Special case: On first move, the starting corner may be in the frontier
        even though it's not yet diagonally adjacent (no pieces placed yet).
        
        Args:
            player: Player to verify frontier for
            
        Returns:
            True if frontier is consistent, False otherwise
        """
        player_value = player.value
        grid = self.grid
        frontier = self.player_frontiers[player]
        is_first_move = self.player_first_move[player]
        start_corner = self.player_start_corners[player]
        start_corner_tuple = (start_corner.row, start_corner.col)
        
        for r, c in frontier:
            # Check: must be empty
            if grid[r, c] != 0:
                return False
            
            # Special case: starting corner on first move
            if is_first_move and (r, c) == start_corner_tuple:
                # Starting corner is valid for first move even without diagonal adjacency
                continue
            
            # Check: must be diagonally adjacent to player's pieces
            has_diagonal = False
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                    if grid[nr, nc] == player_value:
                        has_diagonal = True
                        break
            
            if not has_diagonal:
                return False
            
            # Check: must NOT be orthogonally adjacent to player's pieces
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                    if grid[nr, nc] == player_value:
                        return False
        
        return True
    
    def place_piece(self, piece_positions: List[Position], player: Player, piece_id: int) -> bool:
        """
        Place a piece on the board.
        
        OPTIMIZED: Uses direct grid access for faster placement.
        
        Returns True if placement was successful, False otherwise.
        """
        if not self.can_place_piece(piece_positions, player):
            return False
        
        # Place the piece using direct grid access (faster than set_cell)
        player_value = player.value
        placed_cells = [(pos.row, pos.col) for pos in piece_positions]
        for pos in piece_positions:
            self.grid[pos.row, pos.col] = player_value
        
        # Update bitboard state
        placed_mask = coords_to_mask(placed_cells)
        self.occupied_bits |= placed_mask
        self.player_bits[player] |= placed_mask
        
        # Record the piece as used
        self.player_pieces_used[player].add(piece_id)
        
        # Mark first move as completed
        self.player_first_move[player] = False
        
        # Update frontier after placing piece
        self.update_frontier_after_move(player, placed_cells)
        
        # Update game state
        self.move_count += 1
        self._update_current_player()
        
        return True
    
    def _update_current_player(self) -> None:
        """Update the current player for the next turn."""
        players = list(Player)
        current_index = players.index(self.current_player)
        self.current_player = players[(current_index + 1) % len(players)]
    
    def get_score(self, player: Player) -> int:
        """
        Calculate score for a player.
        
        Score is based on:
        - Number of squares covered by pieces
        - Bonus for using all pieces
        """
        # Count squares covered by player
        squares_covered = np.sum(self.grid == player.value)
        
        # Bonus for using all pieces (21 pieces total)
        if len(self.player_pieces_used[player]) == 21:
            squares_covered += 15  # Bonus for using all pieces
        
        return squares_covered
    
    def get_winner(self) -> Optional[Player]:
        """Determine the winner based on scores."""
        if not self.game_over:
            return None
        
        scores = {player: self.get_score(player) for player in Player}
        return max(scores, key=scores.get)
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        # Game is over when no player can make a legal move
        # This is a simplified check - in practice, you'd need to check all possible moves
        return self.game_over
    
    def assert_bitboard_consistent(self) -> None:
        """
        Assert that bitboard state is consistent with grid state.
        
        Checks:
        - Every occupied cell in grid has corresponding bit set in occupied_bits
        - Every player-owned cell has corresponding bit set in player_bits[player]
        - No bit is set in occupied_bits for an empty cell
        
        Raises:
            AssertionError if any inconsistency is found
        """
        # Check that occupied_bits matches grid
        for row in range(self.SIZE):
            for col in range(self.SIZE):
                cell_value = self.grid[row, col]
                bit = coord_to_bit(row, col)
                is_occupied_in_grid = (cell_value != 0)
                is_occupied_in_bits = ((self.occupied_bits & bit) != 0)
                
                if is_occupied_in_grid != is_occupied_in_bits:
                    raise AssertionError(
                        f"Bitboard inconsistency at ({row}, {col}): "
                        f"grid={cell_value}, occupied_bits has bit={is_occupied_in_bits}"
                    )
                
                # Check player_bits
                if cell_value != 0:
                    player = Player(cell_value)
                    has_player_bit = ((self.player_bits[player] & bit) != 0)
                    if not has_player_bit:
                        raise AssertionError(
                            f"Bitboard inconsistency at ({row}, {col}): "
                            f"grid has {player.name}, but player_bits[{player.name}] missing bit"
                        )
                
                # Check that no player_bits have bits set for empty cells
                for p in Player:
                    has_player_bit = ((self.player_bits[p] & bit) != 0)
                    if has_player_bit and cell_value == 0:
                        raise AssertionError(
                            f"Bitboard inconsistency at ({row}, {col}): "
                            f"cell is empty but player_bits[{p.name}] has bit set"
                        )
                    if has_player_bit and cell_value != p.value:
                        raise AssertionError(
                            f"Bitboard inconsistency at ({row}, {col}): "
                            f"grid has {Player(cell_value).name} but player_bits[{p.name}] has bit set"
                        )
    
    def copy(self) -> 'Board':
        """Create a deep copy of the board."""
        new_board = Board()
        new_board.grid = self.grid.copy()
        new_board.player_pieces_used = {k: v.copy() for k, v in self.player_pieces_used.items()}
        new_board.player_first_move = self.player_first_move.copy()
        new_board.game_over = self.game_over
        new_board.current_player = self.current_player
        new_board.move_count = self.move_count
        # Copy frontiers
        new_board.player_frontiers = {k: v.copy() for k, v in self.player_frontiers.items()}
        # Copy bitboard state
        new_board.occupied_bits = self.occupied_bits
        new_board.player_bits = self.player_bits.copy()
        return new_board
    
    def __str__(self) -> str:
        """String representation of the board."""
        result = []
        for row in range(self.SIZE):
            row_str = ""
            for col in range(self.SIZE):
                value = self.grid[row, col]
                if value == 0:
                    row_str += "."
                else:
                    row_str += str(value)
            result.append(row_str)
        return "\n".join(result)
