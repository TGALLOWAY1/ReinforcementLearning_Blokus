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
        self.mcts_stats = {}
        self.game_id = "local-webworker"

        self.frontend_to_backend = {}
        self.backend_to_frontend = {}
        self._init_orientation_mappings()

    def _init_orientation_mappings(self):
        import numpy as np

        from engine.pieces import (
            ALL_PIECE_ORIENTATIONS,
            PieceGenerator,
            normalize_offsets,
            shape_to_offsets,
        )

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
        # Convert config_dict to native Python objects to avoid Pyodide proxy destruction issues
        self.players_config = []
        for p in config_dict.get("players", []):
            p_dict = {
                "player": str(p["player"]),
                "agent_type": str(p["agent_type"]),
                "agent_config": dict(p["agent_config"]) if p.get("agent_config") else {}
            }
            self.players_config.append(p_dict)

        self.agents = {}
        self.mcts_top_moves = []
        self.mcts_stats = {}
        self.game_id = str(config_dict.get("game_id", "local-webworker"))

        for pc in self.players_config:
            player_enum = EnginePlayer[pc["player"]]
            agent_type = pc.get("agent_type", "human")
            agent_config = pc.get("agent_config", {})
            if agent_type == "mcts":
                budget_ms = int(agent_config.get("time_budget_ms", 1000))
                self.agents[player_enum] = FastMCTSAgent(
                    iterations=500000,
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
        all_players_legal_moves = {}
        for p in EnginePlayer:
            if not game.is_game_over():
                all_players_legal_moves[p] = game.get_legal_moves(p)
            else:
                all_players_legal_moves[p] = []

        current_player_moves = all_players_legal_moves[current_player]

        for m in current_player_moves:
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
        mobility = compute_player_mobility_metrics(current_player_moves, pieces_used_current)
        center_control = game._calculate_center_bonus(current_player) // 2
        frontier_size = len(game.board.get_frontier(current_player))

        piece_lock_risk = {}

        mobility_metrics = {
            "totalPlacements": mobility.totalPlacements,
            "totalOrientationNormalized": mobility.totalOrientationNormalized,
            "totalCellWeighted": mobility.totalCellWeighted,
            "buckets": mobility.buckets,
            "centerControl": center_control,
            "frontierSize": frontier_size,
        }

        all_frontier_metrics = {}
        all_frontier_clusters = {}

        for p in EnginePlayer:
            p_frontier_cells = game.board.get_frontier(p)
            p_moves = all_players_legal_moves[p]
            p_pieces_used = list(game.board.player_pieces_used[p])
            p_has_move = {pid: False for pid in range(1, 22) if pid not in p_pieces_used}
            for m in p_moves:
                if m.piece_id in p_has_move:
                    p_has_move[m.piece_id] = True
            piece_lock_risk[p.name] = sum(1 for can_place in p_has_move.values() if not can_place)

            block_pressure_map = [[False]*20 for _ in range(20)]
            for opp in EnginePlayer:
                if opp == p: continue
                for m in all_players_legal_moves[opp]:
                    cached_ops = game.move_generator.piece_orientations_cache.get(m.piece_id)
                    if cached_ops:
                        pts = m.get_positions(cached_ops)
                        for pt in pts:
                            if 0 <= pt.row < 20 and 0 <= pt.col < 20:
                                block_pressure_map[pt.row][pt.col] = True

            p_metrics = {
                "utility": {f"{fr},{fc}": 0 for fr, fc in p_frontier_cells},
                "block_pressure": {f"{fr},{fc}": 0 for fr, fc in p_frontier_cells},
                "urgency": {f"{fr},{fc}": 0 for fr, fc in p_frontier_cells}
            }
            p_support_sets = {f"{fr},{fc}": set() for fr, fc in p_frontier_cells}

            for move_idx, m in enumerate(p_moves):
                cached_ops = game.move_generator.piece_orientations_cache.get(m.piece_id)
                if cached_ops:
                    pts = m.get_positions(cached_ops)
                    pts_set = set((pt.row, pt.col) for pt in pts)
                    for fr, fc in p_frontier_cells:
                        if (fr, fc) in pts_set:
                            p_metrics["utility"][f"{fr},{fc}"] += 1
                            p_support_sets[f"{fr},{fc}"].add(move_idx)

            for fr, fc in p_frontier_cells:
                key = f"{fr},{fc}"
                if block_pressure_map[fr][fc]: p_metrics["block_pressure"][key] = 1
                u, bp = p_metrics["utility"][key], p_metrics["block_pressure"][key]
                p_metrics["urgency"][key] = u * bp

            all_frontier_metrics[p.name] = p_metrics

            p_top_frontiers = sorted(p_frontier_cells, key=lambda f: len(p_support_sets[f"{f[0]},{f[1]}"]), reverse=True)[:60]
            p_adjacency = {f"{fr},{fc}": [] for fr, fc in p_top_frontiers}
            overlap_threshold = 0.35

            for i in range(len(p_top_frontiers)):
                f1 = p_top_frontiers[i]
                k1, s1 = f"{f1[0]},{f1[1]}", p_support_sets[f"{f1[0]},{f1[1]}"]
                if not s1: continue
                for j in range(i + 1, len(p_top_frontiers)):
                    f2 = p_top_frontiers[j]
                    k2, s2 = f"{f2[0]},{f2[1]}", p_support_sets[f"{f2[0]},{f2[1]}"]
                    if not s2: continue
                    intersection = s1.intersection(s2)
                    if len(intersection) > 0:
                        overlap = len(intersection) / min(len(s1), len(s2))
                        if overlap >= overlap_threshold:
                            p_adjacency[k1].append(k2)
                            p_adjacency[k2].append(k1)

            p_visited = set()
            p_clusters = {"cluster_id": {}, "cluster_sizes": [], "num_clusters": 0}
            p_cluster_id = 0
            for f in p_top_frontiers:
                k = f"{f[0]},{f[1]}"
                if k not in p_visited and p_support_sets[k]:
                    stack, size = [k], 0
                    while stack:
                        curr = stack.pop()
                        if curr not in p_visited:
                            p_visited.add(curr)
                            p_clusters["cluster_id"][curr] = p_cluster_id
                            size += 1
                            for n in p_adjacency[curr]:
                                if n not in p_visited: stack.append(n)
                    if size > 0:
                        p_clusters["cluster_sizes"].append(size)
                        p_cluster_id += 1

            p_clusters["num_clusters"] = p_cluster_id
            for fr, fc in p_frontier_cells:
                k = f"{fr},{fc}"
                if k not in p_clusters["cluster_id"]:
                    if p_support_sets[k]:
                        p_clusters["cluster_id"][k] = p_cluster_id
                        p_clusters["cluster_sizes"].append(1)
                        p_cluster_id += 1
                        p_clusters["num_clusters"] = p_cluster_id
                    else:
                        p_clusters["cluster_id"][k] = -1

            all_frontier_clusters[p.name] = p_clusters

        self_block_risk_moves = []
        curr_support = {f"{fr},{fc}": set() for fr, fc in game.board.get_frontier(current_player)}
        for move_idx, m in enumerate(current_player_moves):
            cached_ops = game.move_generator.piece_orientations_cache.get(m.piece_id)
            if cached_ops:
                pts = m.get_positions(cached_ops)
                pts_set = set((pt.row, pt.col) for pt in pts)
                for fr, fc in game.board.get_frontier(current_player):
                    if (fr, fc) in pts_set: curr_support[f"{fr},{fc}"].add(move_idx)

        curr_clusters = all_frontier_clusters[current_player.name]
        for move_idx, m in enumerate(current_player_moves):
            used_f = [k for k, s in curr_support.items() if move_idx in s]
            c_touched = set(curr_clusters["cluster_id"].get(k, -1) for k in used_f if curr_clusters["cluster_id"].get(k, -1) != -1)
            n_clusters, n_frontiers = len(c_touched), len(used_f)
            if n_clusters > 0 or n_frontiers > 0:
                risk = 2 * n_clusters + 1 * n_frontiers
                self_block_risk_moves.append({
                    "piece_id": m.piece_id, "orientation": self._get_frontend_ori(m.piece_id, m.orientation),
                    "anchor_row": m.anchor_row, "anchor_col": m.anchor_col, "risk": risk,
                    "clusters_touched": n_clusters, "frontier_points_used": n_frontiers
                })
        self_block_risk_moves.sort(key=lambda x: x["risk"], reverse=True)
        self_block_risk = {"top_moves": self_block_risk_moves[:10]}

        winner_name = None
        if game.is_game_over():
            w = game.board.get_winner()
            if w: winner_name = w.name

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

        seat_index_map = {"RED": 0, "BLUE": 1, "YELLOW": 2, "GREEN": 3}
        for idx, hist_entry in enumerate(game.game_history):
            if "move_index" not in hist_entry:
                hist_entry["move_index"], hist_entry["round_index"], hist_entry["position_in_round"] = idx, idx // 4, idx % 4
                player_name = hist_entry.get("player_to_move")
                if player_name in seat_index_map: hist_entry["seat_index"] = seat_index_map[player_name]
            m = hist_entry.get("metrics", {})
            if "frontier_metrics" not in m:
                m["frontier_metrics"], m["frontier_clusters"], m["piece_lock_risk"], m["self_block_risk"] = all_frontier_metrics, all_frontier_clusters, piece_lock_risk, self_block_risk

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
            "game_id": self.game_id, "status": status, "current_player": current_player.name,
            "board": board_list, "scores": scores, "pieces_used": pieces_used,
            "move_count": game.get_move_count(), "game_over": game.is_game_over(), "winner": winner_name,
            "legal_moves": legal_moves_out, "created_at": "", "updated_at": "", "players": self.players_config,
            "heatmap": heatmap, "mobility_metrics": mobility_metrics,
            "mcts_top_moves": self.mcts_top_moves, "mcts_stats": self.mcts_stats,
            "influence_map": influence_map, "dead_zones": dead_zones, "advanced_metrics": advanced_metrics_out,
            "frontier_metrics": all_frontier_metrics, "frontier_clusters": all_frontier_clusters,
            "piece_lock_risk": piece_lock_risk, "self_block_risk": self_block_risk,
            "game_history": history_out
        }

    def load_game(self, history: List[Dict[str, Any]]):
        self.game = BlokusGame()
        self.mcts_top_moves = []
        self.game_id = "loaded-game"
        for entry in history:
            action = entry.get("action")
            if action:
                backend_ori = self._get_backend_ori(action["piece_id"], action["orientation"])
                mv = EngineMove(action["piece_id"], backend_ori, action["anchor_row"], action["anchor_col"])
                player = EnginePlayer[entry["player_to_move"]]
                self.game.make_move(mv, player)
        return self.get_state()

    def make_move(self, piece_id: int, orientation: int, anchor_row: int, anchor_col: int):
        backend_ori = self._get_backend_ori(piece_id, orientation)
        engine_move = EngineMove(piece_id, backend_ori, anchor_row, anchor_col)
        player = self.game.get_current_player()
        success = self.game.make_move(engine_move, player)
        if not success: return {"success": False, "message": "Invalid move", "game_state": self.get_state()}
        return {"success": True, "message": "Move made", "game_state": self.get_state()}

    def pass_turn(self):
        self.game.board._update_current_player()
        self.game._check_game_over()
        return {"success": True, "message": "Turn passed", "game_state": self.get_state()}

    def advance_turn(self, enable_diagnostics: bool = False) -> Dict[str, Any]:
        if self.game.is_game_over(): return {"success": False, "message": "Game over", "game_state": self.get_state()}
        current_player = self.game.get_current_player()
        agent = self.agents.get(current_player)
        if agent is None: return {"success": False, "message": "No agent for current player", "game_state": self.get_state()}
        legal_moves = self.game.get_legal_moves(current_player)
        if not legal_moves:
            self.game.board._update_current_player()
            self.game._check_game_over()
            return {"success": True, "message": "Agent passed", "game_state": self.get_state()}
        budget_ms = 1000
        for pc in self.players_config:
            if pc.get("player") == current_player.name:
                ac = pc.get("agent_config", {})
                budget_ms = int(ac.get("time_budget_ms", 1000))
                break
                
        # Toggle diagnostics based on request
        if hasattr(agent, 'enable_diagnostics'):
            agent.enable_diagnostics = enable_diagnostics
            
        result = agent.think(self.game.board, current_player, legal_moves, budget_ms)
        move = result.get("move")
        if move is None: move = agent.select_action(self.game.board, current_player, legal_moves)
        if "stats" in result:
            self.mcts_stats = result["stats"]
            if "topMoves" in result["stats"]:
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
