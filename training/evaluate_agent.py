"""
Agent evaluation protocol.

This module provides functionality to evaluate trained RL agents against
baseline agents (Random, Heuristic) and in self-play scenarios.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from engine.board import Player
from engine.game import BlokusGame
from engine.move_generator import LegalMoveGenerator, Move
from envs.blokus_v0 import make_gymnasium_env
from training.checkpoints import load_checkpoint
from training.eval_logger import log_evaluation_result

# Import mask function from trainer
from training.trainer import mask_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class RLAgentWrapper:
    """
    Wrapper to use a trained RL agent (from checkpoint) in game evaluation.
    
    This wraps the Stable-Baselines3 model to match the agent interface:
    select_action(board, player, legal_moves) -> Move
    
    Note: This is a simplified wrapper. For full evaluation, consider using
    the environment directly or implementing proper observation conversion.
    """
    
    def __init__(self, model: MaskablePPO, env_wrapper, action_to_move: Dict[int, Tuple], move_to_action: Dict[Tuple, int]):
        """
        Initialize RL agent wrapper.
        
        Args:
            model: Loaded SB3 model
            env_wrapper: Environment wrapper (GymnasiumBlokusWrapper)
            action_to_move: Mapping from action IDs to (piece_id, orientation, row, col)
            move_to_action: Reverse mapping from (piece_id, orientation, row, col) to action IDs
        """
        self.model = model
        self.env_wrapper = env_wrapper
        self.action_to_move = action_to_move
        self.move_to_action = move_to_action
    
    def select_action(self, board, player: Player, legal_moves: List[Move]) -> Optional[Move]:
        """
        Select action using the RL model.
        
        Args:
            board: Current board state
            player: Current player
            legal_moves: List of legal moves
            
        Returns:
            Selected move or None
        """
        if not legal_moves:
            return None
        
        try:
            # Get action mask
            legal_action_mask = np.zeros(len(self.action_to_move), dtype=bool)
            for move in legal_moves:
                action_id = self.move_to_action.get(
                    (move.piece_id, move.orientation, move.anchor_row, move.anchor_col)
                )
                if action_id is not None:
                    legal_action_mask[action_id] = True
            
            # Ensure at least one action is legal
            if not np.any(legal_action_mask):
                return legal_moves[0] if legal_moves else None
            
            # Get observation from environment
            # Note: This requires the environment to be in sync with the board
            # For now, we'll use a simplified approach
            blokus_env = self.env_wrapper.env
            
            # Try to get observation (may not be perfectly synced)
            try:
                obs = blokus_env._get_observation("player_0")
            except Exception:
                # If observation fails, use fallback
                logger.debug("Could not get observation, using fallback")
                return legal_moves[0] if legal_moves else None
            
            # Predict action with masking
            action, _ = self.model.predict(obs, action_masks=legal_action_mask, deterministic=True)
            
            # Convert action to move
            move_tuple = self.action_to_move.get(action)
            if move_tuple:
                piece_id, orientation, anchor_row, anchor_col = move_tuple
                return Move(piece_id, orientation, anchor_row, anchor_col)
            
            # Fallback: return first legal move
            return legal_moves[0] if legal_moves else None
            
        except Exception as e:
            logger.warning(f"RL agent prediction failed: {e}, using first legal move")
            return legal_moves[0] if legal_moves else None


def play_game(
    agent1,
    agent2,
    agent1_name: str = "Agent1",
    agent2_name: str = "Agent2",
    max_moves: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Play a single game between two agents.
    
    Args:
        agent1: First agent
        agent2: Second agent
        agent1_name: Name of first agent
        agent2_name: Name of second agent
        max_moves: Maximum moves before truncation
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with game results
    """
    game = BlokusGame()
    move_generator = LegalMoveGenerator()
    
    if seed is not None:
        np.random.seed(seed)
    
    moves_made = 0
    agent1_score = 0
    agent2_score = 0
    
    # Map players to agents (simplified 2-player game)
    player_to_agent = {
        Player.RED: (agent1, agent1_name),
        Player.BLUE: (agent2, agent2_name),
    }
    
    while not game.is_game_over() and moves_made < max_moves:
        current_player = game.get_current_player()
        
        # Skip if player not in our mapping (for 4-player games, we'd handle all players)
        if current_player not in player_to_agent:
            game.board._update_current_player()
            continue
        
        agent, agent_name = player_to_agent[current_player]
        
        # Get legal moves
        legal_moves = move_generator.get_legal_moves(game.board, current_player)
        
        if not legal_moves:
            game.board._update_current_player()
            continue
        
        # Agent selects move
        move = agent.select_action(game.board, current_player, legal_moves)
        
        if move is None:
            game.board._update_current_player()
            continue
        
        # Make move
        success = game.make_move(move, current_player)
        if success:
            moves_made += 1
        else:
            logger.warning(f"Invalid move from {agent_name}")
            game.board._update_current_player()
    
    # Get final scores
    agent1_score = game.get_score(Player.RED)
    agent2_score = game.get_score(Player.BLUE)
    
    # Determine winner (highest score wins)
    if agent1_score > agent2_score:
        agent1_won = True
        agent2_won = False
    elif agent2_score > agent1_score:
        agent1_won = False
        agent2_won = True
    else:
        agent1_won = False
        agent2_won = False  # Tie
    
    return {
        "moves": moves_made,
        "agent1_score": agent1_score,
        "agent2_score": agent2_score,
        "agent1_won": agent1_won,
        "agent2_won": agent2_won,
        "tie": winner is None,
        "game_over": game.is_game_over()
    }


def evaluate_against_opponent(
    rl_agent,
    opponent,
    opponent_type: str,
    num_games: int = 50,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate RL agent against a specific opponent.
    
    Args:
        rl_agent: RL agent to evaluate
        opponent: Opponent agent
        opponent_type: Type of opponent ("random", "heuristic", "self_play")
        num_games: Number of games to play
        seed: Random seed for reproducibility
        
    Returns:
        Evaluation results dictionary
    """
    logger.info(f"Evaluating against {opponent_type} ({num_games} games)...")
    
    wins = 0
    losses = 0
    ties = 0
    total_reward = 0.0
    total_moves = 0
    scores = []
    
    for game_num in range(num_games):
        game_seed = (seed + game_num) if seed is not None else None
        
        # Play game with RL agent as player 1
        result = play_game(
            agent1=rl_agent,
            agent2=opponent,
            agent1_name="RL Agent",
            agent2_name=opponent_type,
            seed=game_seed
        )
        
        # Track results
        if result["agent1_won"]:
            wins += 1
        elif result["agent2_won"]:
            losses += 1
        else:
            ties += 1
        
        total_reward += result["agent1_score"]
        total_moves += result["moves"]
        scores.append(result["agent1_score"])
        
        if (game_num + 1) % 10 == 0:
            logger.info(f"  Completed {game_num + 1}/{num_games} games")
    
    win_rate = wins / num_games if num_games > 0 else 0.0
    avg_reward = total_reward / num_games if num_games > 0 else 0.0
    avg_game_length = total_moves / num_games if num_games > 0 else 0.0
    
    return {
        "opponent_type": opponent_type,
        "games_played": num_games,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": win_rate,
        "avg_reward": avg_reward,
        "avg_game_length": avg_game_length,
        "scores": scores
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    num_games: int = 50,
    opponents: Optional[List[str]] = None,
    seed: Optional[int] = None,
    training_run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate a trained checkpoint against baseline agents.
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_games: Number of games per opponent
        opponents: List of opponent types to test (default: ["random", "heuristic"])
        seed: Random seed for reproducibility
        training_run_id: Optional training run ID for logging
        
    Returns:
        Dictionary with evaluation results
    """
    if opponents is None:
        opponents = ["random", "heuristic"]
    
    logger.info("=" * 80)
    logger.info("Agent Evaluation")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Games per opponent: {num_games}")
    logger.info(f"Opponents: {opponents}")
    logger.info("=" * 80)
    
    # Load checkpoint
    logger.info("Loading checkpoint...")
    env = make_gymnasium_env()
    env = ActionMasker(env, mask_fn)
    
    try:
        model, config, extra_state = load_checkpoint(checkpoint_path, env=env)
        logger.info("Checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise
    
    # Get action mapping from underlying environment
    blokus_env = env.env.env  # Unwrap: ActionMasker -> GymnasiumBlokusWrapper -> BlokusEnv
    action_to_move = blokus_env.action_to_move
    move_to_action = blokus_env.move_to_action
    
    # Create RL agent wrapper
    rl_agent = RLAgentWrapper(model, env.env, action_to_move, move_to_action)
    
    # Evaluate against each opponent
    results = {}
    
    for opponent_type in opponents:
        if opponent_type == "random":
            opponent = RandomAgent(seed=seed)
        elif opponent_type == "heuristic":
            opponent = HeuristicAgent(seed=seed)
        elif opponent_type == "self_play":
            opponent = rl_agent  # Self-play
        else:
            logger.warning(f"Unknown opponent type: {opponent_type}, skipping")
            continue
        
        result = evaluate_against_opponent(
            rl_agent=rl_agent,
            opponent=opponent,
            opponent_type=opponent_type,
            num_games=num_games,
            seed=seed
        )
        
        results[opponent_type] = result
        
        logger.info(f"{opponent_type.upper()}: Win rate: {result['win_rate']:.2%}, "
                   f"Avg reward: {result['avg_reward']:.2f}, "
                   f"Avg length: {result['avg_game_length']:.1f}")
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Evaluation Summary")
    logger.info("=" * 80)
    for opponent_type, result in results.items():
        logger.info(f"{opponent_type}: {result['win_rate']:.2%} win rate "
                   f"({result['wins']}W/{result['losses']}L/{result['ties']}T)")
    logger.info("=" * 80)
    
    return {
        "checkpoint_path": checkpoint_path,
        "training_run_id": training_run_id,
        "num_games": num_games,
        "seed": seed,
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL agent against baseline agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint file to evaluate"
    )
    
    parser.add_argument(
        "--num-games",
        type=int,
        default=50,
        help="Number of games to play per opponent (default: 50)"
    )
    
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=["random", "heuristic"],
        choices=["random", "heuristic", "self_play"],
        help="Opponent types to evaluate against"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--training-run-id",
        type=str,
        default=None,
        help="Training run ID (for logging to MongoDB)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results JSON file"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    try:
        results = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            num_games=args.num_games,
            opponents=args.opponents,
            seed=args.seed,
            training_run_id=args.training_run_id
        )
        
        # Save results if output path provided
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        
        # Log to MongoDB EvaluationRun records
        if training_run_id:
            for opponent_type, result in results.items():
                log_evaluation_result(
                    training_run_id=training_run_id,
                    checkpoint_path=checkpoint_path,
                    opponent_type=opponent_type,
                    games_played=result["games_played"],
                    win_rate=result["win_rate"],
                    avg_reward=result["avg_reward"],
                    avg_game_length=result["avg_game_length"]
                )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

