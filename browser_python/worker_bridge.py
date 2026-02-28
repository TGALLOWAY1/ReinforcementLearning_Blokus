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
        
        self.frontend_to_backend = {}
        self.backend_to_frontend = {}
        self._init_orientation_mappings()

    def _init_orientation_mappings(self):
        import numpy as np
        from engine.pieces import PieceGenerator, normalize_offsets, shape_to_offsets, ALL_PIECE_ORIENTATIONS
        
        for piece in PieceGenerator.get_all_pieces():
            self.frontend_to_backend[piece.id] = {}
            self.backend_to_frontend[piece.id] = {}
            
            base_shape = piece.shape.copy()
            for frontend_idx in range(8):
                shape = base_shape.copy()
                if frontend_idx >= 4:
                    shape = np.fliplr(shape)
                for _ in range(frontend_idx % 4):
                    shape = np.rot90(shape)
                    
                normalized = normalize_offsets(shape_to_offsets(shape))
                
                backend_ori_id = 0
                for o in ALL_PIECE_ORIENTATIONS.get(piece.id, []):
                    if tuple(o.offsets) == tuple(normalized):
                        backend_ori_id = o.orientation_id
                        break
                        
                self.frontend_to_backend[piece.id][frontend_idx] = backend_ori_id
                
                if backend_ori_id not in self.backend_to_frontend[piece.id]:
                    self.backend_to_frontend[piece.id][backend_ori_id] = frontend_idx

    def _get_backend_ori(self, piece_id, frontend_ori):
        return self.frontend_to_backend.get(int(piece_id), {}).get(int(frontend_ori), 0)
        
    def _get_frontend_ori(self, piece_id, backend_ori):
        return self.backend_to_frontend.get(int(piece_id), {}).get(int(backend_ori), 0)

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
                
                frontend_ori = self._get_frontend_ori(m.piece_id, m.orientation)
                legal_moves_out.append({
                    "piece_id": m.piece_id,
                    "orientation": frontend_ori,
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
        
        # Calculate utility + block_pressure + urgency
        frontier_metrics = {
            "utility": {},
            "block_pressure": {},
            "urgency": {}
        }
        
        frontier_cells = game.board.get_frontier(current_player)
        
        # Initialize utilities based on legal moves already fetched
        for fr, fc in frontier_cells:
            frontier_metrics["utility"][f"{fr},{fc}"] = 0
            frontier_metrics["block_pressure"][f"{fr},{fc}"] = 0
            
        for m in engine_moves:
            cached_ops = game.move_generator.piece_orientations_cache.get(m.piece_id)
            if cached_ops:
                pts = m.get_positions(cached_ops)
                # Naive inference of which frontier point was used: check adjacency overlap
                # Just find the first frontier point that this move connects to diagonally
                used_f = None
                pts_set = set((pt.row, pt.col) for pt in pts)
                for fr, fc in frontier_cells:
                    for r, c in pts_set:
                        if abs(fr - r) == 1 and abs(fc - c) == 1:
                            used_f = (fr, fc)
                            break
                    if used_f: break
                
                if used_f:
                    frontier_metrics["utility"][f"{used_f[0]},{used_f[1]}"] += 1
                    
        # 1-ply BlockPressure: opponent's legal moves next turn
        # Naive generator fetch to get coverage map
        block_pressure_map = [[False]*20 for _ in range(20)]
        for op in EnginePlayer:
            if op != current_player and not game.is_game_over():
                op_moves = game.get_legal_moves(op)
                for m in op_moves:
                    cached_ops = game.move_generator.piece_orientations_cache.get(m.piece_id)
                    if cached_ops:
                        pts = m.get_positions(cached_ops)
                        for pt in pts:
                            if 0 <= pt.row < 20 and 0 <= pt.col < 20:
                                block_pressure_map[pt.row][pt.col] = True

        for fr, fc in frontier_cells:
            if block_pressure_map[fr][fc]:
                frontier_metrics["block_pressure"][f"{fr},{fc}"] += 1
            
            u = frontier_metrics["utility"][f"{fr},{fc}"]
            bp = frontier_metrics["block_pressure"][f"{fr},{fc}"]
            frontier_metrics["urgency"][f"{fr},{fc}"] = u * (1 + bp)
        
        # --- Milestone 2: Frontier Redundancy Clusters ---
        # 1. Build support sets: mapping each frontier point to a set of move IDs
        support_sets = {f"{fr},{fc}": set() for fr, fc in frontier_cells}
        for move_idx, m in enumerate(engine_moves):
            cached_ops = game.move_generator.piece_orientations_cache.get(m.piece_id)
            if cached_ops:
                pts = m.get_positions(cached_ops)
                pts_set = set((pt.row, pt.col) for pt in pts)
                
                # Assign this move to any frontier point it connects to diagonally
                # We could just assign it to the *first* one like in Utility, but
                # assigning to all it touches is technically more accurate for overlap clustering
                for fr, fc in frontier_cells:
                    for r, c in pts_set:
                        if abs(fr - r) == 1 and abs(fc - c) == 1:
                            support_sets[f"{fr},{fc}"].add(move_idx)
                            # Once we found a connection to *this* frontier point, move to the next frontier point
                            break

        # Filter out low utility frontiers as a guardrail - evaluate only the top 60
        sorted_frontiers = sorted(frontier_cells, key=lambda f: len(support_sets[f"{f[0]},{f[1]}"]), reverse=True)
        top_frontiers = sorted_frontiers[:60]
        
        # 2. Compute overlaps and build adjacency list
        overlap_threshold = 0.35
        adjacency = {f"{fr},{fc}": [] for fr, fc in top_frontiers}
        
        for i in range(len(top_frontiers)):
            f1 = top_frontiers[i]
            k1 = f"{f1[0]},{f1[1]}"
            s1 = support_sets[k1]
            if not s1: continue
            
            for j in range(i + 1, len(top_frontiers)):
                f2 = top_frontiers[j]
                k2 = f"{f2[0]},{f2[1]}"
                s2 = support_sets[k2]
                if not s2: continue
                
                # Overlap = |S1 ∩ S2| / min(|S1|, |S2|)
                intersection = s1.intersection(s2)
                if len(intersection) == 0: continue
                
                overlap = len(intersection) / min(len(s1), len(s2))
                if overlap >= overlap_threshold:
                    adjacency[k1].append(k2)
                    adjacency[k2].append(k1)
                    
        # 3. Cluster using DFS
        visited = set()
        frontier_clusters = {
            "cluster_id": {},
            "cluster_sizes": [],
            "num_clusters": 0
        }
        
        cluster_id = 0
        for f in top_frontiers:
            k = f"{f[0]},{f[1]}"
            if k not in visited and support_sets[k]: # only cluster frontiers with >0 moves
                # DFS
                stack = [k]
                size = 0
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        frontier_clusters["cluster_id"][curr] = cluster_id
                        size += 1
                        for neighbor in adjacency[curr]:
                            if neighbor not in visited:
                                stack.append(neighbor)
                
                if size > 0:
                    frontier_clusters["cluster_sizes"].append(size)
                    cluster_id += 1
                    
        frontier_clusters["num_clusters"] = cluster_id
        
        # Fill in remaining isolated nodes or non-top-60 nodes (0 support, or low utility)
        for fr, fc in frontier_cells:
            k = f"{fr},{fc}"
            if k not in frontier_clusters["cluster_id"]:
                if support_sets[k]:
                    frontier_clusters["cluster_id"][k] = cluster_id
                    frontier_clusters["cluster_sizes"].append(1)
                    cluster_id += 1
                    frontier_clusters["num_clusters"] = cluster_id
                else:
                    # Point has 0 support, it's not even a cluster
                    frontier_clusters["cluster_id"][k] = -1

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
        
        history_out = []
        for entry in game.game_history:
            entry_copy = dict(entry)
            action = entry.get("action")
            if action:
                action_copy = dict(action)
                action_copy["orientation"] = self._get_frontend_ori(action_copy["piece_id"], action_copy["orientation"])
                entry_copy["action"] = action_copy
            history_out.append(entry_copy)

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
            "frontier_metrics": frontier_metrics,
            "frontier_clusters": frontier_clusters,
            "game_history": history_out
        }

    def load_game(self, history: List[Dict[str, Any]]):
        self.game = BlokusGame()
        self.mcts_top_moves = []
        
        # Replay history
        for entry in history:
            action = entry.get("action")
            if action:
                backend_ori = self._get_backend_ori(action["piece_id"], action["orientation"])
                mv = EngineMove(
                    action["piece_id"], 
                    backend_ori, 
                    action["anchor_row"], 
                    action["anchor_col"]
                )
                player = EnginePlayer[entry["player_to_move"]]
                self.game.make_move(mv, player)
        
        return self.get_state()

    def make_move(self, piece_id: int, orientation: int, anchor_row: int, anchor_col: int):
        backend_ori = self._get_backend_ori(piece_id, orientation)
        engine_move = EngineMove(piece_id, backend_ori, anchor_row, anchor_col)
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
            translated_top_moves = []
            for tm in result["stats"]["topMoves"]:
                tm_copy = dict(tm)
                tm_copy["orientation"] = self._get_frontend_ori(tm_copy["piece_id"], tm_copy["orientation"])
                translated_top_moves.append(tm_copy)
            self.mcts_top_moves = translated_top_moves
            
        if move:
            success = self.game.make_move(move, current_player)
            return {"success": success, "message": "Agent moved", "game_state": self.get_state()}
        else:
            self.game.board._update_current_player()
            self.game._check_game_over()
            return {"success": True, "message": "Agent failed to move and passed", "game_state": self.get_state()}

# Expose a global instance for Pyodide to interact with
bridge = WebWorkerGameBridge()
