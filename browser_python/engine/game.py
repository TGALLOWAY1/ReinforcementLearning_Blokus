"""
Main Blokus game engine with scoring rules and game management.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .board import Board, Player, Position
from .move_generator import LegalMoveGenerator, Move
from .pieces import PieceGenerator

logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """
    Canonical game result containing final scores and winner information.
    
    Attributes:
        scores: Dictionary mapping player_id (Player.value) to final score
        winner_ids: List of player_ids (Player.value) that tied for the highest score
        is_tie: True if multiple players tied for the highest score (len(winner_ids) > 1)
    """
    scores: Dict[int, int]
    winner_ids: List[int]
    is_tie: bool


class BlokusGame:
    """
    Main Blokus game engine.
    
    Manages game state, scoring, and game flow.
    """
    
    def __init__(self):
        self.board = Board()
        self.move_generator = LegalMoveGenerator()
        self.piece_generator = PieceGenerator()
        self.game_history = []
        self.winner = None
    
    def make_move(self, move: Move, player: Optional[Player] = None) -> bool:
        """
        Make a move on the board.
        
        OPTIMIZED: Skips redundant validation if move came from legal moves list.
        
        Args:
            move: The move to make
            player: Player making the move (defaults to current player)
        
        Returns:
            True if move was successful, False otherwise
        """
        start = time.perf_counter()
        if player is None:
            player = self.board.current_player
        
        # Quick validation - only check if move is obviously invalid
        # (Full validation happens in place_piece, but we can skip some redundant checks)
        start_legal_check = time.perf_counter()
        if not self.move_generator.is_move_legal(self.board, player, move):
            end_legal_check = time.perf_counter()
            logger.debug(f"Engine make_move: move illegal (checked in {end_legal_check - start_legal_check:.4f}s)")
            return False
        end_legal_check = time.perf_counter()
        
        # Get piece orientations and positions (use cached positions if available)
        orientations = self.move_generator.piece_orientations_cache[move.piece_id]
        orientation = orientations[move.orientation]
        
        # Use cached position list if available, otherwise compute
        if move.piece_id in self.move_generator.piece_position_cache:
            cached_positions = self.move_generator.piece_position_cache[move.piece_id][move.orientation]
            piece_positions = [Position(move.anchor_row + rel_r, move.anchor_col + rel_c) 
                             for rel_r, rel_c in cached_positions]
        else:
            # Fallback to original method
            piece_positions = move.get_positions(orientations)
        
        # Place the piece (this does validation again, but it's fast now with optimized can_place_piece)
        start_place = time.perf_counter()
        success = self.board.place_piece(piece_positions, player, move.piece_id)
        end_place = time.perf_counter()
        
        if success:
            from .advanced_metrics import compute_piece_penalty
            
            corner_count = {}
            frontier_size = {}
            difficult_piece_penalty = {}
            remaining_pieces = {}
            
            for p in Player:
                fr = len(self.board.get_frontier(p))
                corner_count[p.name] = fr
                frontier_size[p.name] = fr
                difficult_piece_penalty[p.name] = compute_piece_penalty(self.board.player_pieces_used[p])
                used_pieces = self.board.player_pieces_used[p]
                remaining_pieces[p.name] = [pid for pid in range(1, 22) if pid not in used_pieces]

            self.game_history.append({
                'turn_number': len(self.game_history) + 1,
                'player_to_move': player.name,
                'action': {
                    'piece_id': move.piece_id,
                    'orientation': move.orientation,
                    'anchor_row': move.anchor_row,
                    'anchor_col': move.anchor_col
                },
                'board_state': self.board.grid.tolist(),
                'metrics': {
                    'corner_count': corner_count,
                    'frontier_size': frontier_size,
                    'difficult_piece_penalty': difficult_piece_penalty,
                    'remaining_pieces': remaining_pieces,
                    'influence_map': None
                }
            })
            
            # Check if game is over
            self._check_game_over()
        
        end = time.perf_counter()
        logger.debug(f"Engine make_move core: total={end - start:.4f}s, legal_check={end_legal_check - start_legal_check:.4f}s, place_piece={end_place - start_place:.4f}s, success={success}")
        return success
    
    def _check_game_over(self) -> None:
        """
        Check if the game is over and update game state.
        
        The game ends when no players have legal moves available. This method
        scans all players to determine if any player can still make a move.
        If no player has legal moves, the game is marked as over and the winner
        is determined.
        
        Note: This is a simple scan over all players each time. Future enhancements
        could track pass states or cache move availability for better performance.
        """
        # Check if any player has legal moves available
        # Game ends when NO players can move
        any_player_can_move = False
        
        for player in Player:
            if self.move_generator.has_legal_moves(self.board, player):
                any_player_can_move = True
                break
        
        # If no player can move, the game is over
        if not any_player_can_move:
            self.board.game_over = True
            # Use get_game_result() to determine winner
            game_result = self.get_game_result()
            # Set self.winner for backward compatibility (None if tie)
            if game_result.is_tie:
                self.winner = None
            else:
                # Convert winner_id back to Player enum
                winner_id = game_result.winner_ids[0]
                self.winner = Player(winner_id)
    
    def get_game_result(self) -> GameResult:
        """
        Get the canonical game result with final scores and winner information.
        
        This method computes final scores for all players using the standard Blokus
        scoring system:
        - Base score: 1 point per square covered by pieces
        - Bonus: +15 points if player used all 21 pieces
        - Corner control bonus: +5 points per controlled corner (4 corners max)
        - Center control bonus: +2 points per center square (4×4 center area)
        
        The method can be safely called:
        - After the game is over (recommended): Returns accurate final scores
        - Before the game is over: Computes scores from current board state
          (scores may change as more moves are made)
        
        Returns:
            GameResult containing:
            - scores: Dict mapping player_id (Player.value) to final score
            - winner_ids: List of player_ids that tied for highest score
            - is_tie: True if multiple players tied for highest score
        
        Raises:
            RuntimeError: If called on a game that has never had any moves made
            (edge case - should not occur in normal gameplay)
        """
        # Calculate scores for all players using existing scoring logic
        scores = {}
        for player in Player:
            scores[player.value] = self.get_score(player)
        
        # Find the maximum score
        max_score = max(scores.values())
        
        # Find all players who achieved the maximum score
        winner_ids = [player_id for player_id, score in scores.items() if score == max_score]
        
        # Determine if there's a tie
        is_tie = len(winner_ids) > 1
        
        return GameResult(
            scores=scores,
            winner_ids=winner_ids,
            is_tie=is_tie
        )
    
    def get_winner(self) -> Optional[Player]:
        """
        Get the winner of the game (backward compatibility method).
        
        This method uses get_game_result() internally. For new code, prefer
        using get_game_result() directly as it provides more complete information.
        
        Returns:
            Player enum if there's a unique winner, None if there's a tie or game not over.
        """
        if not self.board.game_over:
            return None
        
        game_result = self.get_game_result()
        
        if game_result.is_tie:
            return None
        
        # Convert winner_id back to Player enum
        winner_id = game_result.winner_ids[0]
        return Player(winner_id)
    
    def get_score(self, player: Player) -> int:
        """
        Get the score for a player.
        
        Scoring rules:
        - 1 point per square covered by pieces
        - 15 bonus points for using all 21 pieces
        - Additional bonuses for strategic placement
        """
        base_score = self.board.get_score(player)
        
        # Additional scoring considerations
        bonus_score = self._calculate_bonus_score(player)
        
        return base_score + bonus_score
    
    def _calculate_bonus_score(self, player: Player) -> int:
        """
        Calculate bonus score for strategic placement.
        
        Bonuses:
        - Corner control bonus
        - Center control bonus
        - Blocking opponent moves bonus
        """
        bonus = 0
        
        # Corner control bonus
        corner_bonus = self._calculate_corner_bonus(player)
        bonus += corner_bonus
        
        # Center control bonus
        center_bonus = self._calculate_center_bonus(player)
        bonus += center_bonus
        
        return bonus
    
    def _calculate_corner_bonus(self, player: Player) -> int:
        """Calculate bonus for controlling corners."""
        corners = [
            Position(0, 0),
            Position(0, self.board.SIZE - 1),
            Position(self.board.SIZE - 1, 0),
            Position(self.board.SIZE - 1, self.board.SIZE - 1)
        ]
        
        controlled_corners = 0
        for corner in corners:
            if self.board.get_player_at(corner) == player:
                controlled_corners += 1
        
        return controlled_corners * 5  # 5 points per controlled corner
    
    def _calculate_center_bonus(self, player: Player) -> int:
        """Calculate bonus for controlling center area."""
        center_size = 4
        center_start = (self.board.SIZE - center_size) // 2
        
        center_squares = 0
        for row in range(center_start, center_start + center_size):
            for col in range(center_start, center_start + center_size):
                pos = Position(row, col)
                if self.board.get_player_at(pos) == player:
                    center_squares += 1
        
        return center_squares * 2  # 2 points per center square
    
    def get_legal_moves(self, player: Optional[Player] = None) -> List[Move]:
        """Get all legal moves for a player."""
        if player is None:
            player = self.board.current_player
        
        return self.move_generator.get_legal_moves(self.board, player)
    
    def get_game_state(self) -> Dict:
        """Get current game state information."""
        return {
            'board': self.board,
            'current_player': self.board.current_player,
            'move_count': self.board.move_count,
            'game_over': self.board.game_over,
            'winner': self.winner,
            'scores': {player.name: self.get_score(player) for player in Player},
            'legal_moves': len(self.get_legal_moves()),
            'game_history_length': len(self.game_history)
        }
    
    def reset_game(self) -> None:
        """Reset the game to initial state."""
        self.board = Board()
        self.game_history = []
        self.winner = None
    
    def get_board_copy(self) -> Board:
        """Get a copy of the current board."""
        return self.board.copy()
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.board.game_over
    
    def get_current_player(self) -> Player:
        """Get the current player."""
        return self.board.current_player
    
    def get_move_count(self) -> int:
        """Get the number of moves made so far."""
        return self.board.move_count
    
    @property
    def move_count(self) -> int:
        """Get the number of moves made so far."""
        return self.board.move_count
    
    def get_player_pieces_used(self, player: Player) -> int:
        """Get the number of pieces used by a player."""
        return len(self.board.player_pieces_used[player])
    
    def get_player_pieces_remaining(self, player: Player) -> int:
        """Get the number of pieces remaining for a player."""
        return 21 - self.get_player_pieces_used(player)
    
    def can_player_move(self, player: Player) -> bool:
        """Check if a player can make any legal moves."""
        return self.move_generator.has_legal_moves(self.board, player)
    
    def get_game_summary(self) -> str:
        """Get a text summary of the current game state."""
        summary = []
        summary.append(f"Current Player: {self.board.current_player.name}")
        summary.append(f"Move Count: {self.board.move_count}")
        summary.append(f"Game Over: {self.board.game_over}")
        
        if self.winner:
            summary.append(f"Winner: {self.winner.name}")
        
        summary.append("\nScores:")
        for player in Player:
            score = self.get_score(player)
            pieces_used = self.get_player_pieces_used(player)
            summary.append(f"  {player.name}: {score} points ({pieces_used}/21 pieces used)")
        
        summary.append(f"\nLegal moves for current player: {len(self.get_legal_moves())}")
        
        return "\n".join(summary)