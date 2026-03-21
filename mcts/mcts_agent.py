"""
Monte Carlo Tree Search agent for Blokus.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from agents.heuristic_agent import HeuristicAgent
from engine.board import Board, Player, Position, _PLAYERS
from engine.move_generator import LegalMoveGenerator, Move
from engine.pieces import PieceGenerator

from .learned_evaluator import LearnedWinProbabilityEvaluator
from .zobrist import TranspositionTable, ZobristHash


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search tree.
    """

    def __init__(
        self,
        board: Board,
        player: Player,
        move: Optional[Move] = None,
        parent: Optional['MCTSNode'] = None,
    ):
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
        self.children: List[MCTSNode] = []

        # MCTS statistics
        self.visits = 0
        self.total_reward = 0.0
        self.prior_bias = 0.0
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

    def ucb1_value(
        self,
        exploration_constant: float = 1.414,
        progressive_bias_weight: float = 0.0,
    ) -> float:
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

        bias_term = progressive_bias_weight * (self.prior_bias / (1.0 + self.visits))
        return exploitation + exploration + bias_term

    def select_child(
        self,
        exploration_constant: float = 1.414,
        progressive_bias_weight: float = 0.0,
    ) -> 'MCTSNode':
        """
        Select child node using UCB1.
        
        Args:
            exploration_constant: Exploration parameter
            
        Returns:
            Selected child node
        """
        return max(
            self.children,
            key=lambda child: child.ucb1_value(
                exploration_constant=exploration_constant,
                progressive_bias_weight=progressive_bias_weight,
            ),
        )

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
            move.piece_id,
            validate=False
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
        current_idx = _PLAYERS.index(self.player)
        next_idx = (current_idx + 1) % len(_PLAYERS)
        return _PLAYERS[next_idx]

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

    def __init__(
        self,
        iterations: int = 1000,
        time_limit: Optional[float] = None,
        exploration_constant: float = 1.414,
        rollout_agent: Optional[HeuristicAgent] = None,
        use_transposition_table: bool = True,
        seed: Optional[int] = None,
        learned_model_path: Optional[str] = None,
        leaf_evaluation_enabled: bool = False,
        progressive_bias_enabled: bool = False,
        progressive_bias_weight: float = 0.25,
        potential_shaping_enabled: bool = False,
        potential_shaping_gamma: float = 1.0,
        potential_shaping_weight: float = 1.0,
        potential_mode: str = "prob",
        max_rollout_moves: int = 50,
    ):
        """
        Initialize MCTS agent.
        
        Args:
            iterations: Maximum number of MCTS iterations
            time_limit: Maximum time in seconds (overrides iterations)
            exploration_constant: UCB1 exploration parameter
            rollout_agent: Agent to use for rollouts (default: HeuristicAgent)
            use_transposition_table: Whether to use transposition table
            seed: Random seed for reproducible behavior
            learned_model_path: Optional model artifact path (`.pkl`) for learned evaluation
            leaf_evaluation_enabled: Enable learned leaf evaluation in place of rollout
            progressive_bias_enabled: Enable progressive bias term from learned value deltas
            progressive_bias_weight: Selection bias weight (decays with child visits)
            potential_shaping_enabled: Enable potential-based shaping in truncated rollouts
            potential_shaping_gamma: Gamma term in shaping: gamma * Phi(s') - Phi(s)
            potential_shaping_weight: Scale factor for shaping contribution
            potential_mode: Potential representation ('prob' or 'logit')
            max_rollout_moves: Maximum rollout length when rollouts are used
        """
        self.iterations = iterations
        self.time_limit = time_limit
        self.exploration_constant = exploration_constant
        self.use_transposition_table = use_transposition_table
        self.leaf_evaluation_enabled = bool(leaf_evaluation_enabled)
        self.progressive_bias_enabled = bool(progressive_bias_enabled)
        self.progressive_bias_weight = float(progressive_bias_weight)
        self.potential_shaping_enabled = bool(potential_shaping_enabled)
        self.potential_shaping_gamma = float(potential_shaping_gamma)
        self.potential_shaping_weight = float(potential_shaping_weight)
        self.potential_mode = potential_mode
        self.learned_model_path = learned_model_path
        self.max_rollout_moves = int(max_rollout_moves)

        if self.potential_mode not in {"prob", "logit"}:
            raise ValueError("potential_mode must be either 'prob' or 'logit'.")
        if self.max_rollout_moves <= 0:
            raise ValueError("max_rollout_moves must be > 0.")

        self._requires_learned_evaluator = (
            self.leaf_evaluation_enabled
            or self.progressive_bias_enabled
            or self.potential_shaping_enabled
        )
        if self._requires_learned_evaluator and not self.learned_model_path:
            raise ValueError(
                "learned_model_path is required when learned evaluation, progressive bias, "
                "or potential shaping is enabled."
            )

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

        self.learned_evaluator: Optional[LearnedWinProbabilityEvaluator] = None
        if self.learned_model_path:
            self.learned_evaluator = LearnedWinProbabilityEvaluator(
                self.learned_model_path,
                potential_mode=potential_mode,
            )

        # Statistics
        self.stats = {
            "iterations_run": 0,
            "time_elapsed": 0.0,
            "transposition_hits": 0,
            "rollout_rewards": [],
            "leaf_eval_calls": 0,
            "progressive_bias_updates": 0,
            "potential_shaping_terms": [],
            "evaluator_errors": 0,
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
            parent = node
            node = node.expand()
            if node is None:
                return
            self._update_progressive_bias(parent, node)

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
                progressive_bias_weight = (
                    self.progressive_bias_weight if self.progressive_bias_enabled else 0.0
                )
                node = node.select_child(
                    exploration_constant=self.exploration_constant,
                    progressive_bias_weight=progressive_bias_weight,
                )

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

        # Run learned leaf evaluation or rollout.
        if self.leaf_evaluation_enabled and self.learned_evaluator is not None:
            reward = self._evaluate_leaf(node.board, node.player)
        else:
            reward = self._rollout(node.board, node.player)

        # Cache result
        if self.transposition_table:
            self.transposition_table.put(board_hash, {"reward": reward})

        self.stats["rollout_rewards"].append(reward)
        return reward

    def _evaluate_leaf(self, board: Board, player: Player) -> float:
        """Evaluate leaf with learned model-backed win probability."""
        if self.learned_evaluator is None:
            return self._rollout(board, player)
        try:
            probability = self.learned_evaluator.predict_player_win_probability(
                board, player
            )
            self.stats["leaf_eval_calls"] += 1
            return float(probability)
        except Exception:
            self.stats["evaluator_errors"] += 1
            return self._rollout(board, player)

    def _update_progressive_bias(self, parent: MCTSNode, child: MCTSNode) -> None:
        """Set child prior bias from learned value delta."""
        if not self.progressive_bias_enabled or self.learned_evaluator is None:
            return
        try:
            parent_value = self.learned_evaluator.predict_player_win_probability(
                parent.board, parent.player
            )
            child_value = self.learned_evaluator.predict_player_win_probability(
                child.board, parent.player
            )
            child.prior_bias = float(child_value - parent_value)
            self.stats["progressive_bias_updates"] += 1
        except Exception:
            child.prior_bias = 0.0
            self.stats["evaluator_errors"] += 1

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
        initial_potential = None
        if self.potential_shaping_enabled and self.learned_evaluator is not None:
            try:
                initial_potential = self.learned_evaluator.potential(sim_board, player)
            except Exception:
                self.stats["evaluator_errors"] += 1
                initial_potential = None

        # Simulate until game ends or max moves
        moves_made = 0

        while moves_made < self.max_rollout_moves:
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
            success = sim_board.place_piece(move_positions, current_player, move.piece_id, validate=False)

            if not success:
                break

            # Move to next player
            current_idx = _PLAYERS.index(current_player)
            current_player = _PLAYERS[(current_idx + 1) % len(_PLAYERS)]
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

        # Apply potential-based shaping only on truncated rollouts.
        if (
            self.potential_shaping_enabled
            and self.learned_evaluator is not None
            and initial_potential is not None
            and moves_made >= self.max_rollout_moves
            and not sim_board.is_game_over()
        ):
            try:
                final_potential = self.learned_evaluator.potential(sim_board, player)
                shaping_term = self.potential_shaping_weight * (
                    (self.potential_shaping_gamma * final_potential) - initial_potential
                )
                reward += shaping_term
                if len(self.stats["potential_shaping_terms"]) < 2048:
                    self.stats["potential_shaping_terms"].append(float(shaping_term))
            except Exception:
                self.stats["evaluator_errors"] += 1

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
                "use_transposition_table": self.use_transposition_table,
                "max_rollout_moves": self.max_rollout_moves,
                "learned_model_path": self.learned_model_path,
                "leaf_evaluation_enabled": self.leaf_evaluation_enabled,
                "progressive_bias_enabled": self.progressive_bias_enabled,
                "progressive_bias_weight": self.progressive_bias_weight,
                "potential_shaping_enabled": self.potential_shaping_enabled,
                "potential_shaping_gamma": self.potential_shaping_gamma,
                "potential_shaping_weight": self.potential_shaping_weight,
                "potential_mode": self.potential_mode,
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
            "rollout_rewards": [],
            "leaf_eval_calls": 0,
            "progressive_bias_updates": 0,
            "potential_shaping_terms": [],
            "evaluator_errors": 0,
        }

        if self.transposition_table:
            self.transposition_table.clear()

    def set_seed(self, seed: int):
        """Set random seed for reproducible behavior."""
        self.zobrist_hash = ZobristHash(seed=seed)
        self.rollout_agent.set_seed(seed)
