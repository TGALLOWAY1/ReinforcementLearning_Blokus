"""
Monte Carlo Tree Search agent for Blokus.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from agents.heuristic_agent import HeuristicAgent
from engine.board import Board, Player, Position
from engine.move_generator import LegalMoveGenerator, Move
from engine.pieces import PieceGenerator

from .zobrist import TranspositionTable, ZobristHash


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search tree.
    """
    
    def __init__(self, board: Board, player: Player, move: Optional[Move] = None, parent: Optional['MCTSNode'] = None):
        """
        Initialize MCTS node.
        
        Args:
            board: Board state at this node
            player: Player whose turn it is
            move: Move that led to this node
            parent: Parent node
        """
        self.board = board.copy()
        self.player = player
        self.move = move
        self.parent = parent
        self.children: List['MCTSNode'] = []
        
        # MCTS statistics
        self.visits = 0
        self.total_reward = 0.0
        self.untried_moves: List[Move] = []
        
        # Initialize untried moves
        self._initialize_untried_moves()
        
    def _initialize_untried_moves(self):
        """Initialize list of untried moves."""
        move_generator = LegalMoveGenerator()
        self.untried_moves = move_generator.get_legal_moves(self.board, self.player)
        
    def is_fully_expanded(self) -> bool:
        """Check if node is fully expanded."""
        return len(self.untried_moves) == 0
        
    def is_terminal(self) -> bool:
        """Check if node is terminal."""
        return len(self.untried_moves) == 0 and len(self.children) == 0
        
    def ucb1_value(self, exploration_constant: float = 1.414) -> float:
        """
        Calculate UCB1 value for node selection.
        
        Args:
            exploration_constant: Exploration parameter (sqrt(2) is common)
            
        Returns:
            UCB1 value
        """
        if self.visits == 0:
            return float('inf')
            
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * np.sqrt(np.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
        
    def select_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """
        Select child node using UCB1.
        
        Args:
            exploration_constant: Exploration parameter
            
        Returns:
            Selected child node
        """
        return max(self.children, key=lambda child: child.ucb1_value(exploration_constant))
        
    def expand(self) -> 'MCTSNode':
        """
        Expand node by adding a new child.
        
        Returns:
            New child node
        """
        if not self.untried_moves:
            return None
            
        # Select random untried move
        move = self.untried_moves.pop()
        
        # Create new board state
        new_board = self.board.copy()
        success = new_board.place_piece(
            self._get_move_positions(move), 
            self.player, 
            move.piece_id
        )
        
        if not success:
            return None
            
        # Determine next player
        next_player = self._get_next_player()
        
        # Create child node
        child = MCTSNode(new_board, next_player, move, self)
        self.children.append(child)
        
        return child
        
    def _get_move_positions(self, move: Move) -> List[Position]:
        """Get positions that a move would occupy."""
        move_generator = LegalMoveGenerator()
        orientations = move_generator.piece_orientations_cache[move.piece_id]
        orientation = orientations[move.orientation]
        
        positions = []
        rows, cols = orientation.shape
        
        for i in range(rows):
            for j in range(cols):
                if orientation[i, j] == 1:
                    pos = Position(move.anchor_row + i, move.anchor_col + j)
                    positions.append(pos)
                    
        return positions
        
    def _get_next_player(self) -> Player:
        """Get next player in turn order."""
        players = list(Player)
        current_idx = players.index(self.player)
        next_idx = (current_idx + 1) % len(players)
        return players[next_idx]
        
    def update(self, reward: float):
        """
        Update node statistics.
        
        Args:
            reward: Reward from simulation
        """
        self.visits += 1
        self.total_reward += reward
        
    def get_best_move(self) -> Optional[Move]:
        """
        Get the best move based on visit counts.
        
        Returns:
            Best move or None if no children
        """
        if not self.children:
            return None
            
        best_child = max(self.children, key=lambda child: child.visits)
        return best_child.move


class MCTSAgent:
    """
    Monte Carlo Tree Search agent for Blokus.
    
    Uses UCT (Upper Confidence Bound applied to Trees) algorithm with
    heuristic rollouts and Zobrist hashing for efficient state representation.
    """
    
    def __init__(self, 
                 iterations: int = 1000,
                 time_limit: Optional[float] = None,
                 exploration_constant: float = 1.414,
                 rollout_agent: Optional[HeuristicAgent] = None,
                 use_transposition_table: bool = True,
                 seed: Optional[int] = None):
        """
        Initialize MCTS agent.
        
        Args:
            iterations: Maximum number of MCTS iterations
            time_limit: Maximum time in seconds (overrides iterations)
            exploration_constant: UCB1 exploration parameter
            rollout_agent: Agent to use for rollouts (default: HeuristicAgent)
            use_transposition_table: Whether to use transposition table
            seed: Random seed for reproducible behavior
        """
        self.iterations = iterations
        self.time_limit = time_limit
        self.exploration_constant = exploration_constant
        self.use_transposition_table = use_transposition_table
        
        # Initialize components
        self.move_generator = LegalMoveGenerator()
        self.piece_generator = PieceGenerator()
        self.zobrist_hash = ZobristHash(seed=seed)
        
        if rollout_agent is None:
            self.rollout_agent = HeuristicAgent(seed=seed)
        else:
            self.rollout_agent = rollout_agent
            
        if use_transposition_table:
            self.transposition_table = TranspositionTable()
        else:
            self.transposition_table = None
            
        # Statistics
        self.stats = {
            "iterations_run": 0,
            "time_elapsed": 0.0,
            "transposition_hits": 0,
            "rollout_rewards": []
        }
        
    def select_action(self, board: Board, player: Player, legal_moves: List[Move]) -> Optional[Move]:
        """
        Select action using MCTS.
        
        Args:
            board: Current board state
            player: Player making the move
            legal_moves: List of legal moves available
            
        Returns:
            Selected move, or None if no legal moves available
        """
        if not legal_moves:
            return None
            
        # If only one move, return it
        if len(legal_moves) == 1:
            return legal_moves[0]
            
        # Run MCTS
        start_time = time.time()
        root = MCTSNode(board, player)
        
        if self.time_limit:
            self._run_mcts_with_time_limit(root)
        else:
            self._run_mcts_with_iterations(root)
            
        self.stats["time_elapsed"] = time.time() - start_time
        
        # Get best move
        best_move = root.get_best_move()
        
        # Clean up transposition table if needed
        if self.transposition_table and len(self.transposition_table.table) > 500000:
            self.transposition_table.clear()
            
        return best_move
        
    def _run_mcts_with_iterations(self, root: MCTSNode):
        """Run MCTS for specified number of iterations."""
        for i in range(self.iterations):
            self._mcts_iteration(root)
            self.stats["iterations_run"] = i + 1
            
    def _run_mcts_with_time_limit(self, root: MCTSNode):
        """Run MCTS for specified time limit."""
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < self.time_limit:
            self._mcts_iteration(root)
            iteration += 1
            
        self.stats["iterations_run"] = iteration
        
    def _mcts_iteration(self, root: MCTSNode):
        """
        Run one MCTS iteration.
        
        Args:
            root: Root node of the search tree
        """
        # Selection: traverse tree using UCB1
        node = self._selection(root)
        
        # Expansion: expand node if not fully expanded
        if not node.is_fully_expanded() and not node.is_terminal():
            node = node.expand()
            if node is None:
                return
                
        # Simulation: run rollout
        reward = self._simulation(node)
        
        # Backpropagation: update statistics up the tree
        self._backpropagation(node, reward)
        
    def _selection(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: traverse tree using UCB1.
        
        Args:
            node: Starting node
            
        Returns:
            Selected node
        """
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            else:
                node = node.select_child(self.exploration_constant)
                
        return node
        
    def _simulation(self, node: MCTSNode) -> float:
        """
        Simulation phase: run rollout from node.
        
        Args:
            node: Node to simulate from
            
        Returns:
            Reward from simulation
        """
        # Check transposition table
        if self.transposition_table:
            board_hash = self.zobrist_hash.hash_board(node.board)
            cached_result = self.transposition_table.get(board_hash)
            if cached_result:
                self.stats["transposition_hits"] += 1
                return cached_result["reward"]
                
        # Run rollout
        reward = self._rollout(node.board, node.player)
        
        # Cache result
        if self.transposition_table:
            self.transposition_table.put(board_hash, {"reward": reward})
            
        self.stats["rollout_rewards"].append(reward)
        return reward
        
    def _rollout(self, board: Board, player: Player) -> float:
        """
        Run rollout simulation.
        
        Args:
            board: Board state to simulate from
            player: Player whose turn it is
            
        Returns:
            Reward from rollout
        """
        # Create copy for simulation
        sim_board = board.copy()
        current_player = player
        
        # Get initial score
        initial_score = sim_board.get_score(player)
        
        # Simulate until game ends or max moves
        max_rollout_moves = 50
        moves_made = 0
        
        while moves_made < max_rollout_moves:
            # Get legal moves
            legal_moves = self.move_generator.get_legal_moves(sim_board, current_player)
            
            if not legal_moves:
                break
                
            # Select move using rollout agent
            move = self.rollout_agent.select_action(sim_board, current_player, legal_moves)
            
            if move is None:
                break
                
            # Make move
            move_positions = self._get_move_positions(move)
            success = sim_board.place_piece(move_positions, current_player, move.piece_id)
            
            if not success:
                break
                
            # Move to next player
            players = list(Player)
            current_idx = players.index(current_player)
            current_player = players[(current_idx + 1) % len(players)]
            moves_made += 1
            
        # Calculate reward
        final_score = sim_board.get_score(player)
        reward = final_score - initial_score
        
        # Add bonus for winning
        if sim_board.is_game_over():
            winner = sim_board.get_winner()
            if winner == player:
                reward += 100
            elif winner is None:
                reward += 10
                
        return reward
        
    def _get_move_positions(self, move: Move) -> List[Position]:
        """Get positions that a move would occupy."""
        orientations = self.move_generator.piece_orientations_cache[move.piece_id]
        orientation = orientations[move.orientation]
        
        positions = []
        rows, cols = orientation.shape
        
        for i in range(rows):
            for j in range(cols):
                if orientation[i, j] == 1:
                    pos = Position(move.anchor_row + i, move.anchor_col + j)
                    positions.append(pos)
                    
        return positions
        
    def _backpropagation(self, node: MCTSNode, reward: float):
        """
        Backpropagation phase: update statistics up the tree.
        
        Args:
            node: Node to start backpropagation from
            reward: Reward to propagate
        """
        while node is not None:
            node.update(reward)
            node = node.parent
            
    def get_action_info(self) -> Dict[str, Any]:
        """Get information about the agent."""
        info = {
            "name": "MCTSAgent",
            "type": "mcts",
            "description": "Monte Carlo Tree Search with UCT and heuristic rollouts",
            "parameters": {
                "iterations": self.iterations,
                "time_limit": self.time_limit,
                "exploration_constant": self.exploration_constant,
                "use_transposition_table": self.use_transposition_table
            },
            "stats": self.stats.copy()
        }
        
        if self.transposition_table:
            info["transposition_stats"] = self.transposition_table.get_stats()
            
        return info
        
    def reset(self):
        """Reset agent state."""
        self.stats = {
            "iterations_run": 0,
            "time_elapsed": 0.0,
            "transposition_hits": 0,
            "rollout_rewards": []
        }
        
        if self.transposition_table:
            self.transposition_table.clear()
            
    def set_seed(self, seed: int):
        """Set random seed for reproducible behavior."""
        self.zobrist_hash = ZobristHash(seed=seed)
        self.rollout_agent.set_seed(seed)
