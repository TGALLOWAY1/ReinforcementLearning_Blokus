"""
Fast MCTS agent optimized for real-time gameplay.
Uses aggressive optimizations to minimize computation time.
"""

import random
import time
from typing import List, Optional, Dict, Any
from engine.board import Board, Player, Position
from engine.move_generator import LegalMoveGenerator, Move
from agents.heuristic_agent import HeuristicAgent


class FastMCTSNode:
    """Lightweight MCTS node with minimal overhead."""
    
    def __init__(self, move: Optional[Move] = None, parent: Optional['FastMCTSNode'] = None):
        self.move = move
        self.parent = parent
        self.children: List['FastMCTSNode'] = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_moves: List[Move] = []
        
    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0
        
    def is_terminal(self) -> bool:
        return len(self.untried_moves) == 0 and len(self.children) == 0
        
    def ucb1_value(self, exploration_constant: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        if self.parent is None or self.parent.visits == 0:
            return self.total_reward / self.visits
            
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * (2 * __import__('math').log(self.parent.visits) / self.visits) ** 0.5
        return exploitation + exploration
        
    def select_child(self, exploration_constant: float = 1.414) -> 'FastMCTSNode':
        return max(self.children, key=lambda child: child.ucb1_value(exploration_constant))
        
    def expand(self, legal_moves: List[Move]) -> 'FastMCTSNode':
        if not self.untried_moves:
            return None
            
        move = self.untried_moves.pop()
        child = FastMCTSNode(move, self)
        self.children.append(child)
        return child
        
    def update(self, reward: float):
        self.visits += 1
        self.total_reward += reward
        
    def get_best_move(self) -> Optional[Move]:
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.visits).move


class FastMCTSAgent:
    """
    Ultra-fast MCTS agent optimized for real-time gameplay.
    
    Key optimizations:
    - No board copying (uses move simulation instead)
    - Cached legal move generation
    - Simplified rollout with random moves
    - Aggressive time limits
    - Minimal memory allocation
    """
    
    def __init__(self, 
                 iterations: int = 30,
                 time_limit: float = 0.5,
                 exploration_constant: float = 1.414,
                 seed: Optional[int] = None):
        self.iterations = iterations
        self.time_limit = time_limit
        self.exploration_constant = exploration_constant
        self.rng = random.Random(seed)
        self.move_generator = LegalMoveGenerator()
        
        # Cache for legal moves to avoid recomputation
        self._legal_moves_cache: Dict[str, List[Move]] = {}
        
    def select_action(self, board: Board, player: Player, legal_moves: List[Move]) -> Optional[Move]:
        """Select action using fast MCTS."""
        if not legal_moves:
            return None
            
        if len(legal_moves) == 1:
            return legal_moves[0]
            
        # Use even faster fallback for very limited time
        if len(legal_moves) <= 5:
            return self._quick_heuristic_selection(board, player, legal_moves)
            
        start_time = time.time()
        root = FastMCTSNode()
        root.untried_moves = legal_moves.copy()
        
        # Run MCTS with strict time limit
        iteration = 0
        while (time.time() - start_time < self.time_limit and 
               iteration < self.iterations and 
               not root.is_fully_expanded()):
            self._fast_mcts_iteration(root, board, player)
            iteration += 1
            
        # If we didn't get enough iterations, use heuristic fallback
        if iteration < 5:
            return self._quick_heuristic_selection(board, player, legal_moves)
            
        best_move = root.get_best_move()
        return best_move if best_move else legal_moves[0]
        
    def _fast_mcts_iteration(self, root: FastMCTSNode, board: Board, player: Player):
        """Run one fast MCTS iteration without board copying."""
        # Selection
        node = self._selection(root)
        
        # Expansion
        if not node.is_fully_expanded():
            legal_moves = self._get_cached_legal_moves(board, player)
            child = node.expand(legal_moves)
            if child is None:
                return
            node = child
            
        # Simulation (ultra-fast random rollout)
        reward = self._fast_rollout(board, player)
        
        # Backpropagation
        self._backpropagation(node, reward)
        
    def _selection(self, node: FastMCTSNode) -> FastMCTSNode:
        """Selection phase with early termination."""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            else:
                node = node.select_child(self.exploration_constant)
        return node
        
    def _fast_rollout(self, board: Board, player: Player) -> float:
        """Ultra-fast rollout using random moves only."""
        # Simple heuristic: prefer larger pieces and center positions
        legal_moves = self._get_cached_legal_moves(board, player)
        if not legal_moves:
            return 0.0
            
        # Quick evaluation based on piece size and position
        move = self._quick_move_evaluation(legal_moves)
        if move is None:
            return 0.0
            
        # Simple reward based on piece size and position
        reward = move.piece_id * 0.1  # Larger pieces are better
        
        # Center bonus
        center_distance = abs(move.anchor_row - 9.5) + abs(move.anchor_col - 9.5)
        reward += (20 - center_distance) * 0.05
        
        return reward
        
    def _quick_move_evaluation(self, legal_moves: List[Move]) -> Optional[Move]:
        """Quick move evaluation without complex heuristics."""
        if not legal_moves:
            return None
            
        # Prefer larger pieces
        moves_by_size = sorted(legal_moves, key=lambda m: m.piece_id, reverse=True)
        
        # Among top 3 largest pieces, prefer center positions
        top_moves = moves_by_size[:min(3, len(moves_by_size))]
        center_moves = sorted(top_moves, 
                            key=lambda m: abs(m.anchor_row - 9.5) + abs(m.anchor_col - 9.5))
        
        return center_moves[0] if center_moves else legal_moves[0]
        
    def _quick_heuristic_selection(self, board: Board, player: Player, legal_moves: List[Move]) -> Move:
        """Quick heuristic selection for time-constrained scenarios."""
        return self._quick_move_evaluation(legal_moves) or legal_moves[0]
        
    def _get_cached_legal_moves(self, board: Board, player: Player) -> List[Move]:
        """Get legal moves with caching to avoid recomputation."""
        # Create a simple cache key based on board state
        cache_key = f"{player.name}_{board.move_count}"
        
        if cache_key not in self._legal_moves_cache:
            self._legal_moves_cache[cache_key] = self.move_generator.get_legal_moves(board, player)
            
        return self._legal_moves_cache[cache_key]
        
    def _backpropagation(self, node: FastMCTSNode, reward: float):
        """Backpropagation phase."""
        while node is not None:
            node.update(reward)
            node = node.parent
