import json
import os
import pytest

from engine.game import BlokusGame
from engine.board import Player
from engine.move_generator import Move
from engine.advanced_metrics import (
    compute_corner_differential,
    compute_territory_control,
    compute_piece_penalty,
    compute_center_proximity,
    compute_opponent_adjacency,
    compute_dead_zones
)

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "advanced_metrics_snapshot.json")

def generate_snapshot():
    import random
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
            "player": entry["player_to_move"],
            "action": None
        }
        if entry["action"]:
            h["action"] = {
                "piece_id": entry["action"]["piece_id"],
                "orientation": entry["action"]["orientation"],
                "anchor_row": entry["action"]["anchor_row"],
                "anchor_col": entry["action"]["anchor_col"]
            }
        history_serializable.append(h)
        
    snapshot_data["history"] = history_serializable
    
    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    with open(FIXTURE_PATH, "w") as f:
        json.dump(snapshot_data, f, indent=2)

def test_advanced_metrics_snapshot():
    """
    Loads a known game replay sequence (20 moves) and verifies that the 
    advanced_metrics outputs at turn 10 and turn 20 match the expected snapshots.
    This acts as a guardrail against regressions when refactoring metrics.
    """
    if not os.path.exists(FIXTURE_PATH):
        generate_snapshot()
    
    with open(FIXTURE_PATH, "r") as f:
        data = json.load(f)
        
    history = data["history"]
    expected_snapshots = data["snapshots"]
    
    game = BlokusGame()
    move_count = 0
    
    for entry in history:
        player = Player[entry["player"]]
        action = entry["action"]
        
        if action:
            move = Move(
                piece_id=action["piece_id"],
                orientation=action["orientation"],
                anchor_row=action["anchor_row"],
                anchor_col=action["anchor_col"]
            )
            success = game.make_move(move, player)
            assert success, f"Failed to replay move {move} for {player}"
            move_count += 1
        else:
            game.board._update_current_player()
            
        # Check against snapshots if we hit a milestone
        if str(move_count) in expected_snapshots:
            snapshot = expected_snapshots[str(move_count)]
            expected_adv_metrics = snapshot["advanced_metrics"]
            expected_dead_zones = snapshot["dead_zones"]
            
            # Recompute current actual metrics
            influence_map, territory_ratios = compute_territory_control(game.board)
            actual_dead_zones = compute_dead_zones(game.board)
            
            actual_adv_metrics = {}
            for p in Player:
                actual_adv_metrics[p.name] = {
                    "corner_differential": float(compute_corner_differential(game.board, p)),
                    "territory_ratio": float(territory_ratios.get(p.name, 0.0)),
                    "piece_penalty": float(compute_piece_penalty(game.board.player_pieces_used[p])),
                    "center_proximity": float(compute_center_proximity(game.board, p)),
                    "opponent_adjacency": float(compute_opponent_adjacency(game.board, p))
                }
                
            # Assertions
            assert actual_dead_zones == expected_dead_zones, f"Dead zones mismatch at move {move_count}"
            
            for p_name in expected_adv_metrics:
                for metric_name, expected_val in expected_adv_metrics[p_name].items():
                    actual_val = actual_adv_metrics[p_name][metric_name]
                    # Use pytest.approx for floating point ratios
                    if isinstance(expected_val, float) and metric_name == "territory_ratio":
                        assert actual_val == pytest.approx(expected_val), f"Metric {metric_name} for {p_name} mismatch at move {move_count}: {actual_val} != {expected_val}"
                    else:
                        assert actual_val == expected_val, f"Metric {metric_name} for {p_name} mismatch at move {move_count}: {actual_val} != {expected_val}"
