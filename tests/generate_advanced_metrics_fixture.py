import json
import os
import random
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from engine.game import BlokusGame
from engine.board import Player
from engine.advanced_metrics import (
    compute_corner_differential,
    compute_territory_control,
    compute_piece_penalty,
    compute_center_proximity,
    compute_opponent_adjacency,
    compute_dead_zones
)

FIXTURE_PATH = os.path.join(project_root, "tests", "fixtures", "advanced_metrics_snapshot.json")

def generate_fixture():
    random.seed(42)
    game = BlokusGame()
    
    snapshot_data = {
        "history": [],
        "snapshots": {}
    }
    
    move_count = 0
    while not game.is_game_over() and move_count < 20:
        player = game.get_current_player()
        legal_moves = game.get_legal_moves(player)
        if not legal_moves:
            game.board._update_current_player()
            continue
            
        move = random.choice(legal_moves)
        success = game.make_move(move, player)
        if success:
            move_count += 1
            
            # Save snapshots at move 10 and 20
            if move_count in [10, 20]:
                influence_map, territory_ratios = compute_territory_control(game.board)
                dead_zones = compute_dead_zones(game.board)
                
                advanced_metrics_out = {}
                for p in Player:
                    advanced_metrics_out[p.name] = {
                        "corner_differential": float(compute_corner_differential(game.board, p)),
                        "territory_ratio": float(territory_ratios.get(p.name, 0.0)),
                        "piece_penalty": float(compute_piece_penalty(game.board.player_pieces_used[p])),
                        "center_proximity": float(compute_center_proximity(game.board, p)),
                        "opponent_adjacency": float(compute_opponent_adjacency(game.board, p))
                    }
                
                snapshot_data["snapshots"][str(move_count)] = {
                    "advanced_metrics": advanced_metrics_out,
                    "dead_zones": dead_zones
                }
                
    # Extract history in a serializable format
    history_serializable = []
    for entry in game.game_history:
        h = {
            "player": entry["player"].name,
            "action": None
        }
        if entry["action"]:
            h["action"] = {
                "piece_id": entry["action"].piece_id,
                "orientation": entry["action"].orientation,
                "anchor_row": entry["action"].anchor_row,
                "anchor_col": entry["action"].anchor_col
            }
        history_serializable.append(h)
        
    snapshot_data["history"] = history_serializable
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    with open(FIXTURE_PATH, "w") as f:
        json.dump(snapshot_data, f, indent=2)
        
    print(f"Generated fixture at {FIXTURE_PATH} with {len(history_serializable)} moves.")

if __name__ == "__main__":
    generate_fixture()
