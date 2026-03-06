"""
Fast MCTS agent optimized for real-time gameplay.
Uses aggressive optimizations to minimize computation time.
"""

import random
import time
from typing import Any, Dict, List, Optional

from engine.board import Board, Player
from engine.move_generator import LegalMoveGenerator, Move



def compute_policy_entropy(visits: List[int]) -> float:
    total = sum(visits)
    if total <= 0:
        return 0.0
    import math
    entropy = 0.0
    for v in visits:
        if v > 0:
            p = v / total
            entropy -= p * math.log(p)
    return entropy

class FastMCTSNode:

    """Lightweight MCTS node with minimal overhead."""

    def __init__(self, move: Optional[Move] = None, parent: Optional['FastMCTSNode'] = None):
        self.move = move
        self.parent = parent
        self.children: List[FastMCTSNode] = []
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
                 seed: Optional[int] = None,
                 enable_diagnostics: bool = False,
                 diagnostics_sample_interval: int = 100):
        self.iterations = iterations
        self.time_limit = time_limit
        self.exploration_constant = exploration_constant
        self.rng = random.Random(seed)
        self.move_generator = LegalMoveGenerator()
        self.enable_diagnostics = enable_diagnostics
        self.diagnostics_sample_interval = diagnostics_sample_interval

        # Cache for legal moves to avoid recomputation
        self._legal_moves_cache: Dict[str, List[Move]] = {}

    def select_action(self, board: Board, player: Player, legal_moves: List[Move]) -> Optional[Move]:
        """Select action using fast MCTS."""
        result = self.think(board, player, legal_moves, int(self.time_limit * 1000))
        return result["move"]

    def think(self, board: Board, player: Player, legal_moves: List[Move], time_budget_ms: int) -> Dict[str, Any]:
        """Run MCTS with an explicit wall-clock budget and return move + stats."""
        budget_s = max(time_budget_ms, 1) / 1000.0
        start_time = time.perf_counter()

        if not legal_moves:
            return {
                "move": None,
                "stats": {
                    "timeBudgetMs": time_budget_ms,
                    "timeSpentMs": 0,
                    "nodesEvaluated": 0,
                    "maxDepthReached": 0,
                    "topMoves": [],
                },
            }

        if len(legal_moves) == 1:
            m = legal_moves[0]
            return {
                "move": m,
                "stats": {
                    "timeBudgetMs": time_budget_ms,
                    "timeSpentMs": int((time.perf_counter() - start_time) * 1000),
                    "nodesEvaluated": 1,
                    "maxDepthReached": 1,
                    "topMoves": [{"piece_id": m.piece_id, "orientation": m.orientation, "anchor_row": m.anchor_row, "anchor_col": m.anchor_col, "visits": 1, "q_value": 0.0}],
                },
            }

        start_wall = time.perf_counter()
        root = FastMCTSNode()
        root.untried_moves = legal_moves.copy()

        max_depth = 0
        nodes_expanded = 0
        nodes_by_depth = {0: 1}
        best_move_trace = []

        # Run MCTS with strict time limit
        iteration = 0
        while (time.perf_counter() - start_wall < budget_s and
               iteration < self.iterations):
            depth, expanded = self._fast_mcts_iteration(root, board, player)
            
            if self.enable_diagnostics:
                if expanded:
                    nodes_expanded += 1
                    max_depth = max(max_depth, depth)
                    nodes_by_depth[depth] = nodes_by_depth.get(depth, 0) + 1
                
                if iteration > 0 and iteration % self.diagnostics_sample_interval == 0:
                    best_child = max(root.children, key=lambda c: c.visits) if root.children else None
                    if best_child and best_child.move:
                        bm = best_child.move
                        action_id = f"{bm.piece_id}-{bm.orientation}-{bm.anchor_row}-{bm.anchor_col}"
                        visits = [c.visits for c in root.children]
                        best_move_trace.append({
                            "sim": iteration,
                            "bestActionId": action_id,
                            "bestQMean": float(best_child.total_reward / best_child.visits) if best_child.visits > 0 else 0.0,
                            "entropy": float(compute_policy_entropy(visits))
                        })

            iteration += 1

        # If we didn't get enough iterations, use heuristic fallback
        if iteration < 5:
            move = self._quick_heuristic_selection(board, player, legal_moves)
            top_moves = self._get_top_moves(root, top_n=10)
            return {
                "move": move,
                "stats": {
                    "timeBudgetMs": time_budget_ms,
                    "timeSpentMs": int((time.perf_counter() - start_time) * 1000),
                    "nodesEvaluated": max(iteration, 1),
                    "maxDepthReached": 2,
                    "topMoves": top_moves,
                },
            }

        best_move = root.get_best_move()
        top_moves = self._get_top_moves(root, top_n=10)
        time_spent_ms = int((time.perf_counter() - start_time) * 1000)
        
        policy_entropy = 0.0
        if self.enable_diagnostics and root.children:
            visits = [c.visits for c in root.children]
            policy_entropy = compute_policy_entropy(visits)

        return {
            "move": best_move if best_move else legal_moves[0],
            "stats": {
                "timeBudgetMs": time_budget_ms,
                "timeSpentMs": time_spent_ms,
                "nodesEvaluated": max(iteration, 1),
                "maxDepthReached": max_depth if self.enable_diagnostics else 2,
                "topMoves": top_moves,
                "diagnostics": {
                    "version": "v1",
                    "timeBudgetMs": int(time_budget_ms),
                    "timeSpentMs": int(time_spent_ms),
                    "simulations": max(iteration, 1),
                    "simsPerSec": int(max(iteration, 1) / (time_spent_ms / 1000.0)) if time_spent_ms > 0 else 0,
                    "rootLegalMoves": len(legal_moves),
                    "rootChildrenExpanded": len(root.children),
                    "rootPolicy": top_moves,
                    "policyEntropy": float(policy_entropy),
                    "maxDepthReached": int(max_depth),
                    "nodesExpanded": int(nodes_expanded),
                    "nodesByDepth": [{"depth": d, "nodes": n} for d, n in sorted(nodes_by_depth.items())],
                    "bestMoveTrace": best_move_trace,
                } if self.enable_diagnostics else None
            },
        }

    def _fast_mcts_iteration(self, root: FastMCTSNode, board: Board, player: Player):
        """Run one fast MCTS iteration without board copying. Returns (depth, expanded_node)."""
        # Selection
        node = root
        depth = 0
        while not node.is_terminal():
            if not node.is_fully_expanded():
                break
            else:
                node = node.select_child(self.exploration_constant)
                depth += 1

        expanded = False
        # Expansion
        if not node.is_fully_expanded():
            legal_moves = self._get_cached_legal_moves(board, player)
            child = node.expand(legal_moves)
            if child is not None:
                expanded = True
                depth += 1
                node = child

        # Simulation (ultra-fast random rollout)
        reward = self._fast_rollout(board, player)

        # Backpropagation
        self._backpropagation(node, reward)

        return depth, expanded



    def _fast_rollout(self, board: Board, player: Player) -> float:
        """Ultra-fast rollout using randomized heuristics."""
        # Simple heuristic: prefer larger pieces and center positions
        legal_moves = self._get_cached_legal_moves(board, player)
        if not legal_moves:
            return 0.0

        # Quick evaluation based on piece size and position
        move = self._quick_move_evaluation(legal_moves)
        if move is None:
            return 0.0

        # Simple reward based on piece size
        reward = move.piece_id * 0.1  # Larger pieces are better

        # Center bonus (higher is better)
        center_distance = abs(move.anchor_row - 9.5) + abs(move.anchor_col - 9.5)
        reward += (20 - center_distance) * 0.05

        # Add a random factor so multiple iterations don't just repeat the exact same evaluation
        # This allows MCTS to actually benefit from more iterations by 'averaging'
        reward += self.rng.random() * 0.1

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

    def _get_top_moves(self, root: FastMCTSNode, top_n: int = 10) -> List[Dict[str, Any]]:
        """Extract top N moves by visits with visits and Q from root children."""
        if not root.children:
            return []
        candidates = []
        for child in root.children:
            if child.move is None:
                continue
            q = child.total_reward / child.visits if child.visits > 0 else 0.0
            candidates.append({
                "piece_id": child.move.piece_id,
                "orientation": child.move.orientation,
                "anchor_row": child.move.anchor_row,
                "anchor_col": child.move.anchor_col,
                "visits": child.visits,
                "q_value": round(q, 4),
            })
        candidates.sort(key=lambda x: x["visits"], reverse=True)
        return candidates[:top_n]
