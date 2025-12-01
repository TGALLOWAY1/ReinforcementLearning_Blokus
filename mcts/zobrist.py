"""
Zobrist hashing for efficient state representation in MCTS.
"""

import numpy as np
from typing import Dict, List, Optional
from engine.board import Board, Player, Position


class ZobristHash:
    """
    Zobrist hashing system for efficient state representation.
    
    Zobrist hashing allows for efficient state comparison and transposition
    table lookups in game tree search algorithms like MCTS.
    """
    
    def __init__(self, board_size: int = 20, num_players: int = 4, seed: Optional[int] = None):
        """
        Initialize Zobrist hash system.
        
        Args:
            board_size: Size of the game board
            num_players: Number of players in the game
            seed: Random seed for reproducible hash values
        """
        self.board_size = board_size
        self.num_players = num_players
        
        # Initialize random number generator
        self.rng = np.random.RandomState(seed)
        
        # Generate random hash values
        self._generate_hash_values()
        
    def _generate_hash_values(self):
        """Generate random hash values for each position and player."""
        # Hash values for each (row, col, player) combination
        self.position_player_hashes = np.zeros(
            (self.board_size, self.board_size, self.num_players + 1), 
            dtype=np.uint64
        )
        
        # Generate random 64-bit values
        for row in range(self.board_size):
            for col in range(self.board_size):
                for player in range(self.num_players + 1):  # +1 for empty
                    self.position_player_hashes[row, col, player] = self.rng.randint(
                        0, 2**64, dtype=np.uint64
                    )
                    
        # Hash values for player turn
        self.player_turn_hashes = np.zeros(self.num_players, dtype=np.uint64)
        for player in range(self.num_players):
            self.player_turn_hashes[player] = self.rng.randint(0, 2**64, dtype=np.uint64)
            
        # Hash values for pieces used by each player
        self.piece_used_hashes = np.zeros(
            (self.num_players, 21),  # 21 pieces per player
            dtype=np.uint64
        )
        for player in range(self.num_players):
            for piece_id in range(21):
                self.piece_used_hashes[player, piece_id] = self.rng.randint(
                    0, 2**64, dtype=np.uint64
                )
                
    def hash_board(self, board: Board) -> int:
        """
        Compute Zobrist hash for a board state.
        
        Args:
            board: Board state to hash
            
        Returns:
            Hash value for the board state
        """
        hash_value = 0
        
        # Hash board positions
        for row in range(self.board_size):
            for col in range(self.board_size):
                cell_value = board.get_cell(Position(row, col))
                hash_value ^= self.position_player_hashes[row, col, cell_value]
                
        # Hash current player
        current_player_idx = board.current_player.value - 1
        hash_value ^= self.player_turn_hashes[current_player_idx]
        
        # Hash pieces used by each player
        for player in Player:
            player_idx = player.value - 1
            used_pieces = board.player_pieces_used[player]
            for piece_id in used_pieces:
                hash_value ^= self.piece_used_hashes[player_idx, piece_id - 1]
                
        return hash_value
        
    def hash_move(self, board: Board, move_hash: int, player: Player, piece_id: int) -> int:
        """
        Update hash value after making a move.
        
        Args:
            board: Board state before the move
            move_hash: Current hash value
            player: Player making the move
            piece_id: Piece being placed
            
        Returns:
            Updated hash value
        """
        new_hash = move_hash
        
        # XOR out the current player
        current_player_idx = board.current_player.value - 1
        new_hash ^= self.player_turn_hashes[current_player_idx]
        
        # XOR in the new player (next player)
        next_player_idx = (current_player_idx + 1) % self.num_players
        new_hash ^= self.player_turn_hashes[next_player_idx]
        
        # XOR in the piece being used
        player_idx = player.value - 1
        new_hash ^= self.piece_used_hashes[player_idx, piece_id - 1]
        
        return new_hash
        
    def hash_position_placement(self, row: int, col: int, player: Player) -> int:
        """
        Get hash contribution for placing a piece at a position.
        
        Args:
            row: Row position
            col: Column position
            player: Player placing the piece
            
        Returns:
            Hash contribution for this placement
        """
        return self.position_player_hashes[row, col, player.value]
        
    def get_hash_info(self) -> Dict[str, any]:
        """Get information about the hash system."""
        return {
            "board_size": self.board_size,
            "num_players": self.num_players,
            "hash_bits": 64,
            "total_positions": self.board_size * self.board_size,
            "total_pieces": 21
        }


class TranspositionTable:
    """
    Transposition table for caching MCTS node information.
    """
    
    def __init__(self, max_size: int = 1000000):
        """
        Initialize transposition table.
        
        Args:
            max_size: Maximum number of entries to store
        """
        self.max_size = max_size
        self.table: Dict[int, Dict] = {}
        self.access_count = 0
        self.hit_count = 0
        
    def get(self, hash_value: int) -> Optional[Dict]:
        """
        Get entry from transposition table.
        
        Args:
            hash_value: Hash value to look up
            
        Returns:
            Cached entry or None if not found
        """
        self.access_count += 1
        if hash_value in self.table:
            self.hit_count += 1
            return self.table[hash_value]
        return None
        
    def put(self, hash_value: int, entry: Dict):
        """
        Store entry in transposition table.
        
        Args:
            hash_value: Hash value to store
            entry: Entry to store
        """
        # If table is full, remove oldest entries
        if len(self.table) >= self.max_size:
            # Simple LRU: remove 10% of entries
            keys_to_remove = list(self.table.keys())[:self.max_size // 10]
            for key in keys_to_remove:
                del self.table[key]
                
        self.table[hash_value] = entry
        
    def clear(self):
        """Clear the transposition table."""
        self.table.clear()
        self.access_count = 0
        self.hit_count = 0
        
    def get_stats(self) -> Dict[str, float]:
        """Get transposition table statistics."""
        hit_rate = self.hit_count / max(self.access_count, 1)
        return {
            "size": len(self.table),
            "max_size": self.max_size,
            "access_count": self.access_count,
            "hit_count": self.hit_count,
            "hit_rate": hit_rate
        }
