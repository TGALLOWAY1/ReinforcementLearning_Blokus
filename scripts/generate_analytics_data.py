
import sys
import os
import time
import shutil

# Ensure root is in path
sys.path.insert(0, os.getcwd())

from engine.game import BlokusGame
from engine.board import Player
from agents.random_agent import RandomAgent
from analytics.logging.logger import StrategyLogger
from analytics.aggregate.aggregate_games import aggregate_games
from analytics.aggregate.aggregate_agents import aggregate_agents

def run_game_with_logging(game_id, agents, logger):
    game = BlokusGame()
    
    # Agent map
    agent_ids = {p.value: f"RandomAgent_{p.name}" for p in Player}
    
    # Reset logger
    logger.on_reset(game_id, 42, agent_ids, {})
    
    print(f"Starting game {game_id}...")
    turn_index = 0
    
    while not game.is_game_over():
        current_player = game.get_current_player()
        agent = agents[current_player]
        
        # Snapshot state before move
        state_before = game.get_board_copy()
        
        legal_moves = game.get_legal_moves(current_player)
        
        if not legal_moves:
            # Pass
            game.board._update_current_player()
            # We might want to check gameover immediately after pass in case all passed?
            game._check_game_over()
            continue
            
        move = agent.select_action(game.board, current_player, legal_moves)
        
        if move:
            success = game.make_move(move, current_player)
            if success:
                # Log step
                # game.board is now the next state
                logger.on_step(game_id, turn_index, current_player.value, 
                               state_before, move, game.board)
                turn_index += 1
        else:
            # Agent returned None despite legal moves? Should not happen for RandomAgent
            game.board._update_current_player()
            game._check_game_over()
            
    # Game End
    res = game.get_game_result()
    
    winner_id = res.winner_ids[0] if not res.is_tie else None
    
    logger.on_game_end(game_id, res.scores, winner_id, turn_index)
    print(f"Game {game_id} finished. Winner values: {res.winner_ids}")

def main():
    log_dir = "logs/verification_run"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    
    logger = StrategyLogger(log_dir=log_dir)
    
    # Create 4 random agents
    agents = {p: RandomAgent(seed=p.value) for p in Player}
    
    # Run 1 short game
    run_game_with_logging("game_001", agents, logger)
    
    # Run aggregation
    print("\nRunning aggregation...")
    try:
        aggregate_games(log_dir, os.path.join(log_dir, "game_summary.parquet"))
        aggregate_agents(os.path.join(log_dir, "game_summary.parquet"), 
                        os.path.join(log_dir, "agent_summary.parquet"))
        
        print("\nSuccess! Analytics pipeline verified.")
        
        # Print profiling table if pandas exists
        import pandas as pd
        if os.path.exists(os.path.join(log_dir, "agent_summary.parquet")):
            df = pd.read_parquet(os.path.join(log_dir, "agent_summary.parquet"))
            print("\nAgent Strategy Profile:")
            print(df[['games_played', 'AggressionIndex', 'CenterFocus', 'MobilityCare', 'ExpansionIndex']].to_string())
            
    except ImportError:
        print("Pandas/PyArrow not installed, skipping aggregation step in verification.")
    except Exception as e:
        print(f"Aggregation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
