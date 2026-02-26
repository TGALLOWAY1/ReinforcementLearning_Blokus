import os
import sys
from typing import Any, Dict, List

# Add current directory to path so engine, mcts, agents modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.fast_mcts_agent import FastMCTSAgent
from engine.advanced_metrics import (
    compute_center_proximity,
    compute_corner_differential,
    compute_dead_zones,
    compute_opponent_adjacency,
    compute_piece_penalty,
    compute_territory_control,
)
from engine.board import Player as EnginePlayer
from engine.game import BlokusGame
from engine.mobility_metrics import compute_player_mobility_metrics
from engine.move_generator import Move as EngineMove


class WebWorkerGameBridge:
    def __init__(self):
        self.game = BlokusGame()
        self.agents = {}
        self.players_config = []
        self.mcts_top_moves = []
        self.game_id = "local-webworker"

    def init_game(self, config_dict: Dict[str, Any]):
        self.game = BlokusGame()
        self.players_config = config_dict.get("players", [])
        self.agents = {}
        self.mcts_top_moves = []
        self.game_id = config_dict.get("game_id", "local-webworker")
        
        for pc in self.players_config:
            player_enum = EnginePlayer[pc["player"]]
            agent_type = pc.get("agent_type", "human")
            agent_config = pc.get("agent_config", {})
            if agent_type == "mcts":
                budget_ms = int(agent_config.get("time_budget_ms", 1000))
                self.agents[player_enum] = FastMCTSAgent(
                    iterations=5000,
                    time_limit=max(budget_ms, 1) / 1000.0,
                    exploration_constant=1.414
                )
            elif agent_type == "human":
                self.agents[player_enum] = None
        
        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        game = self.game
        
        # Convert board
        board_list = game.board.grid.tolist()
        
        scores = {p.name: game.get_score(p) for p in EnginePlayer}
        pieces_used = {p.name: list(game.board.player_pieces_used[p]) for p in EnginePlayer}
        
        legal_moves_out = []
        heatmap = [[0.0 for _ in range(20)] for _ in range(20)]
        
        current_player = game.get_current_player()
        engine_moves = []
        
        if not game.is_game_over():
            engine_moves = game.get_legal_moves(current_player)
            for m in engine_moves:
                positions = []
                cached_ops = game.move_generator.piece_orientations_cache.get(m.piece_id)
                if cached_ops:
                    pts = m.get_positions(cached_ops)
                    for pt in pts:
                        positions.append({"row": pt.row, "col": pt.col})
                        if 0 <= pt.row < 20 and 0 <= pt.col < 20:
                            heatmap[pt.row][pt.col] = 1.0
                
                legal_moves_out.append({
                    "piece_id": m.piece_id,
                    "orientation": m.orientation,
                    "anchor_row": m.anchor_row,
                    "anchor_col": m.anchor_col,
                    "positions": positions
                })
        
        pieces_used_current = list(game.board.player_pieces_used[current_player])
        mobility = compute_player_mobility_metrics(engine_moves, pieces_used_current)
        center_control = game._calculate_center_bonus(current_player) // 2
        frontier_size = len(game.board.get_frontier(current_player))
        
        mobility_metrics = {
            "totalPlacements": mobility.totalPlacements,
            "totalOrientationNormalized": mobility.totalOrientationNormalized,
            "totalCellWeighted": mobility.totalCellWeighted,
            "buckets": mobility.buckets,
            "centerControl": center_control,
            "frontierSize": frontier_size,
        }
        
        winner_name = None
        if game.is_game_over():
            w = game.board.get_winner()
            if w:
                winner_name = w.name
                
        status = "finished" if game.is_game_over() else "in_progress"
        
        influence_map, territory_ratios = compute_territory_control(game.board)
        dead_zones = compute_dead_zones(game.board)
        
        advanced_metrics_out = {}
        for p in EnginePlayer:
            advanced_metrics_out[p.name] = {
                "corner_differential": float(compute_corner_differential(game.board, p)),
                "territory_ratio": float(territory_ratios.get(p.name, 0.0)),
                "piece_penalty": float(compute_piece_penalty(game.board.player_pieces_used[p])),
                "center_proximity": float(compute_center_proximity(game.board, p)),
                "opponent_adjacency": float(compute_opponent_adjacency(game.board, p))
            }
        
        return {
            "game_id": self.game_id,
            "status": status,
            "current_player": current_player.name,
            "board": board_list,
            "scores": scores,
            "pieces_used": pieces_used,
            "move_count": game.get_move_count(),
            "game_over": game.is_game_over(),
            "winner": winner_name,
            "legal_moves": legal_moves_out,
            "created_at": "",
            "updated_at": "",
            "players": self.players_config,
            "heatmap": heatmap,
            "mobility_metrics": mobility_metrics,
            "mcts_top_moves": self.mcts_top_moves,
            "influence_map": influence_map,
            "dead_zones": dead_zones,
            "advanced_metrics": advanced_metrics_out,
            "game_history": game.game_history
        }

    def load_game(self, history: List[Dict[str, Any]]):
        self.game = BlokusGame()
        self.mcts_top_moves = []
        
        # Replay history
        for entry in history:
            action = entry.get("action")
            if action:
                mv = EngineMove(
                    action["piece_id"], 
                    action["orientation"], 
                    action["anchor_row"], 
                    action["anchor_col"]
                )
                player = EnginePlayer[entry["player_to_move"]]
                self.game.make_move(mv, player)
        
        return self.get_state()

    def make_move(self, piece_id: int, orientation: int, anchor_row: int, anchor_col: int):
        engine_move = EngineMove(piece_id, orientation, anchor_row, anchor_col)
        player = self.game.get_current_player()
        
        success = self.game.make_move(engine_move, player)
        if not success:
            return {"success": False, "message": "Invalid move", "game_state": self.get_state()}
            
        return {"success": True, "message": "Move made", "game_state": self.get_state()}
        
    def pass_turn(self):
        player = self.game.get_current_player()
        # To skip, we just update the player turn and check over
        self.game.board._update_current_player()
        self.game._check_game_over()
        return {"success": True, "message": "Turn passed", "game_state": self.get_state()}

    def advance_turn(self) -> Dict[str, Any]:
        if self.game.is_game_over():
            return {"success": False, "message": "Game over", "game_state": self.get_state()}
            
        current_player = self.game.get_current_player()
        agent = self.agents.get(current_player)
        
        if agent is None:
            return {"success": False, "message": "No agent for current player", "game_state": self.get_state()}
            
        legal_moves = self.game.get_legal_moves(current_player)
        if not legal_moves:
            self.game.board._update_current_player()
            self.game._check_game_over()
            return {"success": True, "message": "Agent passed", "game_state": self.get_state()}
            
        # Get agent's time budget from config
        budget_ms = 1000
        for pc in self.players_config:
            if pc.get("player") == current_player.name:
                ac = pc.get("agent_config", {})
                budget_ms = int(ac.get("time_budget_ms", 1000))
                break
                
        # Handle agent thinking
        result = agent.think(self.game.board, current_player, legal_moves, budget_ms)
        move = result.get("move")
        if move is None:
            move = agent.select_action(self.game.board, current_player, legal_moves)
            
        if "stats" in result and "topMoves" in result["stats"]:
            self.mcts_top_moves = result["stats"]["topMoves"]
            
        if move:
            success = self.game.make_move(move, current_player)
            return {"success": success, "message": "Agent moved", "game_state": self.get_state()}
        else:
            self.game.board._update_current_player()
            self.game._check_game_over()
            return {"success": True, "message": "Agent failed to move and passed", "game_state": self.get_state()}

# Expose a global instance for Pyodide to interact with
bridge = WebWorkerGameBridge()
