"""
Arena script for running round-robin matches between Blokus agents.
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.board import Board, Player
from engine.move_generator import LegalMoveGenerator
from engine.game import BlokusGame
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from mcts.mcts_agent import MCTSAgent


class ArenaMatch:
    """
    Represents a single match between two agents.
    """
    
    def __init__(self, agent1_name: str, agent2_name: str, agent1, agent2):
        """
        Initialize match.
        
        Args:
            agent1_name: Name of first agent
            agent2_name: Name of second agent
            agent1: First agent instance
            agent2: Second agent instance
        """
        self.agent1_name = agent1_name
        self.agent2_name = agent2_name
        self.agent1 = agent1
        self.agent2 = agent2
        
        # Match results
        self.winner = None
        self.scores = {agent1_name: 0, agent2_name: 0}
        self.moves_made = 0
        self.game_duration = 0.0
        self.error = None
        
    def play_match(self, max_moves: int = 1000, verbose: bool = False) -> Dict[str, Any]:
        """
        Play a complete match.
        
        Args:
            max_moves: Maximum moves per match
            verbose: Whether to print game progress
            
        Returns:
            Match results dictionary
        """
        try:
            start_time = time.time()
            
            # Initialize game
            game = BlokusGame()
            move_generator = LegalMoveGenerator()
            
            if verbose:
                print(f"Starting match: {self.agent1_name} vs {self.agent2_name}")
                
            # Play game
            while not game.is_game_over() and self.moves_made < max_moves:
                current_player = game.get_current_player()
                
                # Determine which agent to use
                if current_player == Player.RED:
                    agent = self.agent1
                    agent_name = self.agent1_name
                elif current_player == Player.BLUE:
                    agent = self.agent2
                    agent_name = self.agent2_name
                else:
                    # For 4-player games, we'll use agents in rotation
                    # This is a simplified version for 2-agent matches
                    agent = self.agent1 if self.moves_made % 2 == 0 else self.agent2
                    agent_name = self.agent1_name if self.moves_made % 2 == 0 else self.agent2_name
                
                # Get legal moves
                legal_moves = move_generator.get_legal_moves(game.board, current_player)
                
                if not legal_moves:
                    if verbose:
                        print(f"No legal moves for {agent_name}, skipping turn")
                    game.board._update_current_player()
                    continue
                    
                # Agent selects move
                move = agent.select_action(game.board, current_player, legal_moves)
                
                if move is None:
                    if verbose:
                        print(f"Agent {agent_name} returned None move, skipping turn")
                    game.board._update_current_player()
                    continue
                    
                # Make move
                success = game.make_move(move, current_player)
                
                if not success:
                    if verbose:
                        print(f"Invalid move from {agent_name}, skipping turn")
                    game.board._update_current_player()
                    continue
                    
                self.moves_made += 1
                
                if verbose and self.moves_made % 10 == 0:
                    print(f"Move {self.moves_made}: {agent_name} placed piece {move.piece_id}")
                    
            # Game finished
            self.game_duration = time.time() - start_time
            
            # Get final scores
            self.scores[self.agent1_name] = game.get_score(Player.RED)
            self.scores[self.agent2_name] = game.get_score(Player.BLUE)
            
            # Determine winner
            if self.scores[self.agent1_name] > self.scores[self.agent2_name]:
                self.winner = self.agent1_name
            elif self.scores[self.agent2_name] > self.scores[self.agent1_name]:
                self.winner = self.agent2_name
            else:
                self.winner = "tie"
                
            if verbose:
                print(f"Match finished: {self.winner}")
                print(f"Scores: {self.scores}")
                print(f"Moves: {self.moves_made}, Duration: {self.game_duration:.2f}s")
                
        except Exception as e:
            self.error = str(e)
            if verbose:
                print(f"Error in match: {e}")
                
        return self.get_results()
        
    def get_results(self) -> Dict[str, Any]:
        """Get match results."""
        return {
            "agent1": self.agent1_name,
            "agent2": self.agent2_name,
            "winner": self.winner,
            "scores": self.scores,
            "moves_made": self.moves_made,
            "game_duration": self.game_duration,
            "error": self.error
        }


class Arena:
    """
    Arena for running round-robin tournaments between agents.
    """
    
    def __init__(self, agents: Dict[str, Any], output_dir: str = "arena_results"):
        """
        Initialize arena.
        
        Args:
            agents: Dictionary of agent names to agent instances
            output_dir: Directory to save results
        """
        self.agents = agents
        self.output_dir = output_dir
        self.results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def run_round_robin(self, 
                       rounds: int = 1, 
                       max_moves: int = 1000,
                       verbose: bool = False) -> Dict[str, Any]:
        """
        Run round-robin tournament.
        
        Args:
            rounds: Number of rounds to play
            max_moves: Maximum moves per match
            verbose: Whether to print progress
            
        Returns:
            Tournament results
        """
        agent_names = list(self.agents.keys())
        total_matches = len(agent_names) * (len(agent_names) - 1) * rounds
        
        if verbose:
            print(f"Starting round-robin tournament with {len(agent_names)} agents")
            print(f"Total matches: {total_matches}")
            print(f"Agents: {agent_names}")
            
        match_count = 0
        start_time = time.time()
        
        # Run rounds
        for round_num in range(rounds):
            if verbose:
                print(f"\n--- Round {round_num + 1} ---")
                
            # Play all pairs
            for i, agent1_name in enumerate(agent_names):
                for j, agent2_name in enumerate(agent_names):
                    if i != j:  # Don't play against self
                        match_count += 1
                        
                        if verbose:
                            print(f"\nMatch {match_count}/{total_matches}: {agent1_name} vs {agent2_name}")
                            
                        # Create and play match
                        match = ArenaMatch(
                            agent1_name, agent2_name,
                            self.agents[agent1_name], self.agents[agent2_name]
                        )
                        
                        result = match.play_match(max_moves=max_moves, verbose=verbose)
                        self.results.append(result)
                        
                        if verbose:
                            print(f"Result: {result['winner']} wins")
                            
        total_time = time.time() - start_time
        
        # Calculate statistics
        stats = self._calculate_statistics()
        
        # Save results
        self._save_results(stats, total_time)
        
        if verbose:
            self._print_summary(stats, total_time)
            
        return stats
        
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate tournament statistics."""
        agent_names = list(self.agents.keys())
        stats = {
            "agents": agent_names,
            "total_matches": len(self.results),
            "agent_stats": {},
            "match_results": self.results
        }
        
        # Initialize agent statistics
        for agent_name in agent_names:
            stats["agent_stats"][agent_name] = {
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "total_score": 0,
                "avg_score": 0,
                "win_rate": 0
            }
            
        # Calculate statistics
        for result in self.results:
            if result["error"]:
                continue
                
            agent1 = result["agent1"]
            agent2 = result["agent2"]
            winner = result["winner"]
            
            # Update scores
            stats["agent_stats"][agent1]["total_score"] += result["scores"][agent1]
            stats["agent_stats"][agent2]["total_score"] += result["scores"][agent2]
            
            # Update win/loss/ties
            if winner == agent1:
                stats["agent_stats"][agent1]["wins"] += 1
                stats["agent_stats"][agent2]["losses"] += 1
            elif winner == agent2:
                stats["agent_stats"][agent2]["wins"] += 1
                stats["agent_stats"][agent1]["losses"] += 1
            else:  # tie
                stats["agent_stats"][agent1]["ties"] += 1
                stats["agent_stats"][agent2]["ties"] += 1
                
        # Calculate averages and win rates
        for agent_name in agent_names:
            agent_stat = stats["agent_stats"][agent_name]
            total_games = agent_stat["wins"] + agent_stat["losses"] + agent_stat["ties"]
            
            if total_games > 0:
                agent_stat["avg_score"] = agent_stat["total_score"] / total_games
                agent_stat["win_rate"] = agent_stat["wins"] / total_games
                
        return stats
        
    def _save_results(self, stats: Dict[str, Any], total_time: float):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(self.output_dir, f"arena_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "total_time": total_time,
                "stats": stats
            }, f, indent=2)
            
        # Save summary
        summary_file = os.path.join(self.output_dir, f"arena_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Arena Tournament Results - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total time: {total_time:.2f} seconds\n")
            f.write(f"Total matches: {stats['total_matches']}\n\n")
            
            f.write("Agent Rankings:\n")
            f.write("-" * 20 + "\n")
            
            # Sort agents by win rate
            sorted_agents = sorted(
                stats["agent_stats"].items(),
                key=lambda x: x[1]["win_rate"],
                reverse=True
            )
            
            for i, (agent_name, agent_stat) in enumerate(sorted_agents, 1):
                f.write(f"{i}. {agent_name}\n")
                f.write(f"   Win Rate: {agent_stat['win_rate']:.3f}\n")
                f.write(f"   Wins: {agent_stat['wins']}, Losses: {agent_stat['losses']}, Ties: {agent_stat['ties']}\n")
                f.write(f"   Avg Score: {agent_stat['avg_score']:.2f}\n\n")
                
    def _print_summary(self, stats: Dict[str, Any], total_time: float):
        """Print tournament summary."""
        print("\n" + "=" * 50)
        print("TOURNAMENT SUMMARY")
        print("=" * 50)
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Total matches: {stats['total_matches']}")
        print()
        
        print("Agent Rankings:")
        print("-" * 20)
        
        # Sort agents by win rate
        sorted_agents = sorted(
            stats["agent_stats"].items(),
            key=lambda x: x[1]["win_rate"],
            reverse=True
        )
        
        for i, (agent_name, agent_stat) in enumerate(sorted_agents, 1):
            print(f"{i}. {agent_name}")
            print(f"   Win Rate: {agent_stat['win_rate']:.3f}")
            print(f"   Wins: {agent_stat['wins']}, Losses: {agent_stat['losses']}, Ties: {agent_stat['ties']}")
            print(f"   Avg Score: {agent_stat['avg_score']:.2f}")
            print()


def create_agents(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create agents based on configuration.
    
    Args:
        config: Agent configuration
        
    Returns:
        Dictionary of agent names to agent instances
    """
    agents = {}
    
    for agent_name, agent_config in config.items():
        agent_type = agent_config["type"]
        
        if agent_type == "random":
            agents[agent_name] = RandomAgent(seed=agent_config.get("seed"))
        elif agent_type == "heuristic":
            agent = HeuristicAgent(seed=agent_config.get("seed"))
            if "weights" in agent_config:
                agent.set_weights(agent_config["weights"])
            agents[agent_name] = agent
        elif agent_type == "mcts":
            agent = MCTSAgent(
                iterations=agent_config.get("iterations", 1000),
                time_limit=agent_config.get("time_limit"),
                exploration_constant=agent_config.get("exploration_constant", 1.414),
                use_transposition_table=agent_config.get("use_transposition_table", True),
                seed=agent_config.get("seed")
            )
            agents[agent_name] = agent
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
    return agents


def main():
    """Main function for arena script."""
    parser = argparse.ArgumentParser(description="Blokus Arena - Round-robin tournament")
    parser.add_argument("--config", type=str, default="arena_config.json",
                       help="Configuration file for agents")
    parser.add_argument("--rounds", type=int, default=1,
                       help="Number of rounds to play")
    parser.add_argument("--max-moves", type=int, default=1000,
                       help="Maximum moves per match")
    parser.add_argument("--output-dir", type=str, default="arena_results",
                       help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Default configuration if no config file
    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found, using default configuration")
        config = {
            "RandomAgent": {
                "type": "random",
                "seed": 42
            },
            "HeuristicAgent": {
                "type": "heuristic",
                "seed": 43,
                "weights": {
                    "piece_size": 1.0,
                    "corner_creation": 2.0,
                    "edge_avoidance": -1.5,
                    "center_preference": 0.5
                }
            },
            "MCTSAgent": {
                "type": "mcts",
                "iterations": 500,
                "exploration_constant": 1.414,
                "use_transposition_table": True,
                "seed": 44
            }
        }
    else:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Create agents
    agents = create_agents(config)
    
    # Create arena
    arena = Arena(agents, args.output_dir)
    
    # Run tournament
    stats = arena.run_round_robin(
        rounds=args.rounds,
        max_moves=args.max_moves,
        verbose=args.verbose
    )
    
    print(f"\nTournament completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
