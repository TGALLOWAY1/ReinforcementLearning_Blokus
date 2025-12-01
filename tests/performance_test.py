"""
Performance test script to collect timing data for optimization analysis.

Runs games with agents and collects timing metrics from logs.
"""

import sys
import os
import time
import logging
from io import StringIO

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from engine.game import BlokusGame
from engine.board import Player
from agents.random_agent import RandomAgent

# Configure logging to capture timing data
log_capture = StringIO()
handler = logging.StreamHandler(log_capture)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
handler.setFormatter(formatter)

# Get the logger and add handler
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def run_agent_game(num_players=4, max_moves=20):
    """Run a game with all random agents and collect timing data."""
    print(f"\n{'='*60}")
    print(f"Running {num_players}-player agent game (max {max_moves} moves)")
    print(f"{'='*60}\n")
    
    game = BlokusGame()
    agents = {
        Player.RED: RandomAgent(),
        Player.BLUE: RandomAgent(),
        Player.YELLOW: RandomAgent(),
        Player.GREEN: RandomAgent()
    }
    
    move_count = 0
    timing_data = {
        'legal_times': [],
        'agent_times': [],
        'apply_times': [],
        'total_times': []
    }
    
    while not game.is_game_over() and move_count < max_moves:
        current_player = game.get_current_player()
        agent = agents.get(current_player)
        
        if not agent:
            break
        
        # Time legal move generation
        start_legal = time.perf_counter()
        legal_moves = game.get_legal_moves(current_player)
        end_legal = time.perf_counter()
        legal_time = end_legal - start_legal
        timing_data['legal_times'].append(legal_time)
        
        if not legal_moves:
            print(f"  Move {move_count + 1}: {current_player.name} - No legal moves, skipping")
            game.board._update_current_player()
            continue
        
        # Time agent selection
        start_agent = time.perf_counter()
        move = agent.select_action(game.board, current_player, legal_moves)
        end_agent = time.perf_counter()
        agent_time = end_agent - start_agent
        timing_data['agent_times'].append(agent_time)
        
        # Time move application
        start_apply = time.perf_counter()
        success = game.make_move(move, current_player)
        end_apply = time.perf_counter()
        apply_time = end_apply - start_apply
        timing_data['apply_times'].append(apply_time)
        
        total_time = end_apply - start_legal
        timing_data['total_times'].append(total_time)
        
        move_count += 1
        print(f"  Move {move_count}: {current_player.name} - legal={legal_time*1000:.2f}ms, agent={agent_time*1000:.2f}ms, apply={apply_time*1000:.2f}ms, total={total_time*1000:.2f}ms")
    
    # Print summary
    print(f"\n{'='*60}")
    print("AGENT GAME TIMING SUMMARY")
    print(f"{'='*60}")
    if timing_data['legal_times']:
        avg_legal = sum(timing_data['legal_times']) / len(timing_data['legal_times'])
        avg_agent = sum(timing_data['agent_times']) / len(timing_data['agent_times'])
        avg_apply = sum(timing_data['apply_times']) / len(timing_data['apply_times'])
        avg_total = sum(timing_data['total_times']) / len(timing_data['total_times'])
        
        print(f"Average legal move generation: {avg_legal*1000:.2f}ms")
        print(f"Average agent selection: {avg_agent*1000:.2f}ms")
        print(f"Average move application: {avg_apply*1000:.2f}ms")
        print(f"Average total per move: {avg_total*1000:.2f}ms")
        print(f"\nLegal moves: {len(timing_data['legal_times'])} samples")
        print(f"  Min: {min(timing_data['legal_times'])*1000:.2f}ms")
        print(f"  Max: {max(timing_data['legal_times'])*1000:.2f}ms")
    
    return timing_data


def run_human_simulation_game(num_agents=3, max_moves=15):
    """Simulate a game with 1 human + agents (human moves simulated)."""
    print(f"\n{'='*60}")
    print(f"Running 1 human + {num_agents} agents game (max {max_moves} moves)")
    print(f"{'='*60}\n")
    
    game = BlokusGame()
    agents = {
        Player.BLUE: RandomAgent(),
        Player.YELLOW: RandomAgent(),
        Player.GREEN: RandomAgent()
    }
    
    move_count = 0
    timing_data = {
        'human_make_move_times': [],
        'human_total_times': []
    }
    
    while not game.is_game_over() and move_count < max_moves:
        current_player = game.get_current_player()
        
        # Get legal moves
        legal_moves = game.get_legal_moves(current_player)
        
        if not legal_moves:
            print(f"  Move {move_count + 1}: {current_player.name} - No legal moves, skipping")
            game.board._update_current_player()
            continue
        
        if current_player == Player.RED:
            # Simulate human move - just pick first legal move
            move = legal_moves[0]
            
            start_total = time.perf_counter()
            start_make_move = time.perf_counter()
            success = game.make_move(move, current_player)
            end_make_move = time.perf_counter()
            end_total = time.perf_counter()
            
            make_move_time = end_make_move - start_make_move
            total_time = end_total - start_total
            
            timing_data['human_make_move_times'].append(make_move_time)
            timing_data['human_total_times'].append(total_time)
            
            move_count += 1
            print(f"  Move {move_count} (HUMAN): {current_player.name} - make_move={make_move_time*1000:.2f}ms, total={total_time*1000:.2f}ms")
        else:
            # Agent move
            agent = agents.get(current_player)
            if agent:
                move = agent.select_action(game.board, current_player, legal_moves)
                game.make_move(move, current_player)
                move_count += 1
                print(f"  Move {move_count} (AGENT): {current_player.name}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("HUMAN SIMULATION GAME TIMING SUMMARY")
    print(f"{'='*60}")
    if timing_data['human_make_move_times']:
        avg_make_move = sum(timing_data['human_make_move_times']) / len(timing_data['human_make_move_times'])
        avg_total = sum(timing_data['human_total_times']) / len(timing_data['human_total_times'])
        
        print(f"Average human make_move: {avg_make_move*1000:.2f}ms")
        print(f"Average human total: {avg_total*1000:.2f}ms")
        print(f"\nHuman moves: {len(timing_data['human_make_move_times'])} samples")
        print(f"  Min make_move: {min(timing_data['human_make_move_times'])*1000:.2f}ms")
        print(f"  Max make_move: {max(timing_data['human_make_move_times'])*1000:.2f}ms")
    
    return timing_data


if __name__ == "__main__":
    print("="*60)
    print("BLOKUS RL PERFORMANCE TEST - BASELINE MEASUREMENTS")
    print("="*60)
    
    # Test 1: 4-agent game
    agent_timings = run_agent_game(num_players=4, max_moves=20)
    
    # Test 2: 1 human + 3 agents
    human_timings = run_human_simulation_game(num_agents=3, max_moves=15)
    
    print(f"\n{'='*60}")
    print("BASELINE MEASUREMENTS COMPLETE")
    print(f"{'='*60}")
    print("\nKey findings:")
    print("- Review timing data above to identify bottlenecks")
    print("- Focus optimizations on stages with highest average times")
    print("- Target: <100-150ms total per move")

