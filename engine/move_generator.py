"""
Legal move generator for Blokus game.
"""

from typing import List, Tuple, Set, Optional
from .board import Board, Player, Position
from .pieces import PieceGenerator, PiecePlacement


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
        self._cache_piece_orientations()
    
    def _cache_piece_orientations(self):
        """Cache all piece orientations for faster lookup."""
        for piece in self.all_pieces:
            orientations = self.piece_generator.get_piece_rotations_and_reflections(piece)
            self.piece_orientations_cache[piece.id] = orientations
    
    def get_legal_moves(self, board: Board, player: Player) -> List[Move]:
        """
        Get all legal moves for a given player on the current board.
        
        Args:
            board: Current board state
            player: Player to generate moves for
        
        Returns:
            List of legal moves
        """
        legal_moves = []
        
        # Get pieces that haven't been used yet
        available_pieces = [piece for piece in self.all_pieces 
                          if piece.id not in board.player_pieces_used[player]]
        
        for piece in available_pieces:
            orientations = self.piece_orientations_cache[piece.id]
            
            for orientation_idx, orientation in enumerate(orientations):
                # Get all possible anchor positions for this orientation
                anchor_positions = PiecePlacement.get_valid_anchor_positions(
                    (board.SIZE, board.SIZE), orientation
                )
                
                for anchor_row, anchor_col in anchor_positions:
                    # Get positions this piece would occupy
                    positions = PiecePlacement.get_piece_positions(
                        orientation, anchor_row, anchor_col
                    )
                    piece_positions = [Position(row, col) for row, col in positions]
                    
                    # Check if this placement is legal
                    if board.can_place_piece(piece_positions, player):
                        move = Move(piece.id, orientation_idx, anchor_row, anchor_col)
                        legal_moves.append(move)
        
        return legal_moves
    
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
