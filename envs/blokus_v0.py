"""
PettingZoo AEC environment for Blokus game.

ACTION INDEXING SCHEME:
- Action space: Discrete with size ~36,400 (calculated as: 21 pieces × orientations × 20×20 board positions)
- Action encoding: Flattened mapping from (piece_id, orientation, anchor_row, anchor_col) to a single discrete action index
  - piece_id: 1-21 (Blokus pieces)
  - orientation: 0 to len(orientations)-1 for each piece (varies by piece, up to 8)
  - anchor_row, anchor_col: 0-19 (20×20 board positions)
- Action mapping: Created in _setup_action_space() as action_to_move and move_to_action dictionaries

MASK CONSTRUCTION:
- Legal action mask is constructed in _get_info() method
- Mask is a boolean numpy array of shape (action_space_size,) where:
  - True = legal action (action_id corresponds to a valid move)
  - False = illegal action
- Mask is stored in env.infos[agent]["legal_action_mask"]
- When no legal moves exist, mask will be all False (this is a problem for MaskablePPO)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import logging
import os
import time

from engine.board import Board, Player, Position
from engine.pieces import PieceGenerator
from engine.move_generator import LegalMoveGenerator, Move
from engine.game import BlokusGame

# Diagnostic logging for action masking (can be disabled)
MASK_DEBUG_LOGGING = True  # Set to False to disable diagnostic logging
_mask_logger = logging.getLogger(__name__ + ".mask_diagnostics")

# Move generation profiling (enabled via BLOKUS_PROFILE_MOVEGEN env var)
PROFILE_MOVEGEN = os.getenv("BLOKUS_PROFILE_MOVEGEN", "0") == "1"
_movegen_profiler_logger = logging.getLogger(__name__ + ".movegen_profiler")


class BlokusEnv(AECEnv):
    """
    PettingZoo AEC environment for Blokus.
    
    Features:
    - Discrete action space mapping to (piece_id, orientation, anchor_row, anchor_col)
    - Multi-channel observations (board, remaining pieces, last move)
    - Action masking for legal moves only
    - Dense and sparse rewards
    - Gymnasium compatibility
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "blokus_v0",
    }
    
    def __init__(self, render_mode: Optional[str] = None, max_episode_steps: int = 1000):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        
        # Game components
        self.game = BlokusGame()
        self.move_generator = LegalMoveGenerator()
        self.piece_generator = PieceGenerator()
        
        # Action space setup
        self._setup_action_space()
        
        # Observation space setup
        self._setup_observation_space()
        
        # Agent setup
        self.possible_agents = [f"player_{i}" for i in range(4)]
        self.agents = self.possible_agents[:]
        
        # AEC required attributes
        self.observation_spaces = {agent: self.observation_space for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space for agent in self.possible_agents}
        
        # State tracking
        self.step_count = 0
        self.last_scores = {agent: 0 for agent in self.possible_agents}
        self.last_move_info = {agent: None for agent in self.possible_agents}
        self.game_result = None  # Stores GameResult when game is over
        
        # Move generation profiling (if enabled)
        self._movegen_profiling_enabled = PROFILE_MOVEGEN
        self._movegen_total_time = 0.0  # Total time in seconds
        self._movegen_call_count = 0
        self._movegen_max_time = 0.0  # Maximum single call time
        self._episode_count = 0
        self._profile_log_interval = 10  # Log every N episodes
        
        # Agent selector for turn management
        self._agent_selector = agent_selector(self.agents)
        
    def _setup_action_space(self):
        """Setup the discrete action space."""
        # Calculate maximum possible actions
        # Each action is: piece_id (1-21) + orientation (0-7) + anchor_row (0-19) + anchor_col (0-19)
        # We'll use a flattened action space for simplicity
        self.max_piece_id = 21
        self.max_orientation = 8  # Up to 8 orientations per piece
        self.board_size = 20
        
        # Create action mapping
        self.action_to_move = {}
        self.move_to_action = {}
        action_id = 0
        
        for piece_id in range(1, self.max_piece_id + 1):
            orientations = self.move_generator.piece_orientations_cache.get(piece_id, [])
            for orientation_idx in range(len(orientations)):
                for row in range(self.board_size):
                    for col in range(self.board_size):
                        self.action_to_move[action_id] = (piece_id, orientation_idx, row, col)
                        self.move_to_action[(piece_id, orientation_idx, row, col)] = action_id
                        action_id += 1
        
        self.action_space_size = action_id
        self.action_space = spaces.Discrete(self.action_space_size)
        
    def _setup_observation_space(self):
        """Setup the observation space."""
        # Board channels: 4 channels for each player + 1 empty channel
        board_channels = 5
        
        # Remaining pieces: 21 pieces per player
        remaining_pieces_channels = 21
        
        # Last move: 4 channels (piece_id, orientation, row, col)
        last_move_channels = 4
        
        # Total observation shape
        obs_height = self.board_size
        obs_width = self.board_size
        obs_channels = board_channels + remaining_pieces_channels + last_move_channels
        
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(obs_channels, obs_height, obs_width), 
            dtype=np.float32
        )
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None, **kwargs) -> None:
        """Reset the environment."""
        # Accept seed and options as both positional and keyword arguments for compatibility
        # Also accept **kwargs for compatibility with wrappers that pass additional arguments
        if seed is not None:
            np.random.seed(seed)
        
        # Log move generation profiling summary before resetting (if enabled)
        if self._movegen_profiling_enabled and self._movegen_call_count > 0:
            # Log summary before resetting
            avg_time_ms = (self._movegen_total_time / self._movegen_call_count) * 1000.0
            total_time_ms = self._movegen_total_time * 1000.0
            max_time_ms = self._movegen_max_time * 1000.0
            
            # Log every N episodes (or on first episode with data)
            if self._episode_count % self._profile_log_interval == 0 or self._episode_count == 0:
                _movegen_profiler_logger.info(
                    f"MoveGen profiling (episode {self._episode_count}): "
                    f"total_ms={total_time_ms:.2f}, calls={self._movegen_call_count}, "
                    f"avg_ms={avg_time_ms:.2f}, max_ms={max_time_ms:.2f}"
                )
            
            # Reset stats for next episode
            self._movegen_total_time = 0.0
            self._movegen_call_count = 0
            self._movegen_max_time = 0.0
        
        # Increment episode count
        self._episode_count += 1
            
        # Reset game state
        self.game.reset_game()
        self.step_count = 0
        self.last_scores = {agent: 0 for agent in self.possible_agents}
        self.last_move_info = {agent: None for agent in self.possible_agents}
        self.game_result = None  # Clear game result on reset
        
        # Reset agents
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        
        # Set initial agent
        self.agent_selection = self._agent_selector.next()
        
        # Initialize observations and infos
        self.observations = {}
        self.infos = {}
        self.rewards = {}
        self.terminations = {}
        self.truncations = {}
        
        for agent in self.agents:
            self.observations[agent] = self._get_observation(agent)
            self.infos[agent] = self._get_info(agent)
            self.rewards[agent] = 0
            self.terminations[agent] = False
            self.truncations[agent] = False
            
    def _get_observation(self, agent: str) -> np.ndarray:
        """Get observation for an agent."""
        player = self._agent_to_player(agent)
        
        # Initialize observation array
        obs_channels, height, width = self.observation_space.shape
        obs = np.zeros((obs_channels, height, width), dtype=np.float32)
        
        # Board channels (0-4)
        board_channels = 5
        for row in range(height):
            for col in range(width):
                cell_value = self.game.board.get_cell(Position(row, col))
                if cell_value == 0:
                    obs[0, row, col] = 1  # Empty channel
                else:
                    obs[cell_value, row, col] = 1  # Player channel
                    
        # Remaining pieces channels (5-25)
        remaining_start = board_channels
        used_pieces = self.game.board.player_pieces_used[player]
        for i, piece_id in enumerate(range(1, 22)):
            if piece_id not in used_pieces:
                obs[remaining_start + i, :, :] = 1
                
        # Last move channels (26-29)
        last_move_start = remaining_start + 21
        if self.last_move_info[agent] is not None:
            piece_id, orientation, row, col = self.last_move_info[agent]
            obs[last_move_start, :, :] = piece_id / 21.0  # Normalize
            obs[last_move_start + 1, :, :] = orientation / 8.0  # Normalize
            obs[last_move_start + 2, :, :] = row / 20.0  # Normalize
            obs[last_move_start + 3, :, :] = col / 20.0  # Normalize
            
        return obs
        
    def _get_info(self, agent: str) -> Dict[str, Any]:
        """Get info for an agent."""
        player = self._agent_to_player(agent)
        
        # Get legal moves (with optional profiling)
        if self._movegen_profiling_enabled:
            start_time = time.perf_counter()
            legal_moves = self.move_generator.get_legal_moves(self.game.board, player)
            elapsed_time = time.perf_counter() - start_time
            
            # Accumulate profiling stats
            self._movegen_total_time += elapsed_time
            self._movegen_call_count += 1
            self._movegen_max_time = max(self._movegen_max_time, elapsed_time)
        else:
            legal_moves = self.move_generator.get_legal_moves(self.game.board, player)
        legal_action_mask = np.zeros(self.action_space_size, dtype=bool)
        
        # DIAGNOSTIC: Log when no legal moves are found
        if len(legal_moves) == 0:
            if MASK_DEBUG_LOGGING:
                _mask_logger.warning(
                    f"BlokusEnv: NO LEGAL MOVES for {agent} (player {player.name}) "
                    f"at step {self.step_count}. "
                    f"Mask will be all False - this will break MaskablePPO!"
                )
        
        # Build the mask by marking legal actions as True
        mapped_count = 0
        for move in legal_moves:
            action_id = self.move_to_action.get((move.piece_id, move.orientation, move.anchor_row, move.anchor_col))
            if action_id is not None:
                legal_action_mask[action_id] = True
                mapped_count += 1
        
        # DIAGNOSTIC: Log mask properties (only for first few calls or when issues detected)
        if MASK_DEBUG_LOGGING:
            mask_sum = legal_action_mask.sum()
            if mask_sum == 0 or (self.step_count < 10):  # Always log first 10 steps, or when mask is empty
                _mask_logger.info(
                    f"BlokusEnv._get_info({agent}): "
                    f"legal_moves={len(legal_moves)}, "
                    f"mapped_to_mask={mapped_count}, "
                    f"mask.sum()={mask_sum}, "
                    f"mask.shape={legal_action_mask.shape}, "
                    f"mask.dtype={legal_action_mask.dtype}, "
                    f"action_space_size={self.action_space_size}"
                )
                if mask_sum > 0:
                    # Log sample of legal action indices
                    legal_indices = np.where(legal_action_mask)[0]
                    sample_size = min(10, len(legal_indices))
                    _mask_logger.debug(
                        f"Sample legal action indices: {legal_indices[:sample_size].tolist()}"
                    )
                if len(legal_moves) != mapped_count:
                    _mask_logger.warning(
                        f"Mismatch: {len(legal_moves)} legal moves but only {mapped_count} mapped to action space"
                    )
                
        info = {
            "legal_action_mask": legal_action_mask,
            "legal_moves_count": len(legal_moves),
            "score": self.game.get_score(player),
            "pieces_used": len(self.game.board.player_pieces_used[player]),
            "pieces_remaining": 21 - len(self.game.board.player_pieces_used[player]),
            "can_move": len(legal_moves) > 0,
        }
        
        # Add game result information on terminal steps
        # These fields are only guaranteed to be present when the game is over
        if self.game_result is not None:
            info["final_scores"] = self.game_result.scores
            info["winner_ids"] = self.game_result.winner_ids
            info["is_tie"] = self.game_result.is_tie
            
            # Convenience flag: player_0 (RED, value=1) won
            # player_0 corresponds to Player.RED which has value=1
            player_0_id = Player.RED.value  # player_0 = RED = 1
            info["player0_won"] = (player_0_id in self.game_result.winner_ids) and not self.game_result.is_tie
        
        return info
        
    def _agent_to_player(self, agent: str) -> Player:
        """Convert agent string to Player enum."""
        agent_idx = int(agent.split("_")[1])
        return list(Player)[agent_idx]
        
    def _player_to_agent(self, player: Player) -> str:
        """Convert Player enum to agent string."""
        player_idx = player.value - 1
        return f"player_{player_idx}"
        
    def step(self, action: int) -> None:
        """Execute one step in the environment."""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
            
        # Convert action to move
        if action not in self.action_to_move:
            # Invalid action - skip turn
            self._skip_turn()
            return
            
        piece_id, orientation_idx, anchor_row, anchor_col = self.action_to_move[action]
        
        # Create move object
        move = Move(piece_id, orientation_idx, anchor_row, anchor_col)
        
        # Execute move
        player = self._agent_to_player(self.agent_selection)
        success = self.game.make_move(move, player)
        
        if success:
            # Update last move info
            self.last_move_info[self.agent_selection] = (piece_id, orientation_idx, anchor_row, anchor_col)
            
            # Calculate reward
            current_score = self.game.get_score(player)
            reward = current_score - self.last_scores[self.agent_selection]
            self.last_scores[self.agent_selection] = current_score
            
            self.rewards[self.agent_selection] = reward
        else:
            # Invalid move - no reward
            self.rewards[self.agent_selection] = 0
            
        # Update observations and infos for all agents
        for agent in self.agents:
            self.observations[agent] = self._get_observation(agent)
            self.infos[agent] = self._get_info(agent)
            
        # Check for termination/truncation
        self._check_termination_truncation()
        
        # Move to next agent
        self.agent_selection = self._agent_selector.next()
        self.step_count += 1
        
    def _skip_turn(self):
        """Skip turn for current agent."""
        self.rewards[self.agent_selection] = 0
        self.terminations[self.agent_selection] = True
        
        # Update observations and infos
        for agent in self.agents:
            self.observations[agent] = self._get_observation(agent)
            self.infos[agent] = self._get_info(agent)
            
        # Move to next agent
        self.agent_selection = self._agent_selector.next()
        self.step_count += 1
        
    def _check_termination_truncation(self):
        """Check for episode termination or truncation."""
        # Check if game is over
        if self.game.is_game_over():
            # Get canonical game result (includes scores, winner_ids, is_tie)
            self.game_result = self.game.get_game_result()
            
            # Calculate final rewards
            winner = self.game.get_winner()
            for agent in self.agents:
                player = self._agent_to_player(agent)
                if winner == player:
                    self.rewards[agent] += 100  # Winner bonus
                elif winner is None:
                    self.rewards[agent] += 10   # Tie bonus
                    
                self.terminations[agent] = True
                
        # Check for truncation
        if self.step_count >= self.max_episode_steps:
            for agent in self.agents:
                self.truncations[agent] = True
                
        # Check if any agent can't move
        agents_can_move = []
        for agent in self.agents:
            if not self.terminations[agent] and not self.truncations[agent]:
                can_move = self.infos[agent]["can_move"]
                if can_move:
                    agents_can_move.append(agent)
                else:
                    self.terminations[agent] = True
                    
        # If no agents can move, terminate all
        if not agents_can_move:
            for agent in self.agents:
                self.terminations[agent] = True
                
    def observe(self, agent: str) -> np.ndarray:
        """Get observation for a specific agent."""
        return self.observations[agent]
        
    def close(self):
        """Close the environment."""
        pass
        
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(f"Step: {self.step_count}")
            print(f"Current agent: {self.agent_selection}")
            print("Board:")
            print(self.game.board)
            print("Scores:")
            for agent in self.agents:
                player = self._agent_to_player(agent)
                score = self.game.get_score(player)
                print(f"  {agent}: {score}")
            print()
        elif self.render_mode == "rgb_array":
            # Return RGB array representation
            return self._get_rgb_array()
            
    def _get_rgb_array(self) -> np.ndarray:
        """Get RGB array representation of the board."""
        # Create RGB array
        rgb_array = np.zeros((self.board_size * 20, self.board_size * 20, 3), dtype=np.uint8)
        
        # Color mapping
        colors = {
            0: (255, 255, 255),  # Empty - white
            1: (255, 0, 0),      # RED - red
            2: (0, 0, 255),      # BLUE - blue
            3: (255, 255, 0),    # YELLOW - yellow
            4: (0, 255, 0),      # GREEN - green
        }
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                cell_value = self.game.board.get_cell(Position(row, col))
                color = colors.get(cell_value, (128, 128, 128))  # Default gray
                
                # Fill 20x20 pixel area for each cell
                start_row = row * 20
                start_col = col * 20
                rgb_array[start_row:start_row+20, start_col:start_col+20] = color
                
        return rgb_array


def env(render_mode: Optional[str] = None, max_episode_steps: int = 1000):
    """
    Create Blokus environment.
    
    Args:
        render_mode: Rendering mode ("human" or "rgb_array")
        max_episode_steps: Maximum steps per episode
        
    Returns:
        BlokusEnv instance
    """
    return BlokusEnv(render_mode=render_mode, max_episode_steps=max_episode_steps)


# Gymnasium compatibility wrapper
class GymnasiumBlokusWrapper(gym.Env):
    """
    Wrapper to make Blokus environment compatible with Gymnasium/Stable-Baselines3.
    
    This wrapper is fully Gymnasium-compatible:
    - Inherits from gymnasium.Env (required for SB3's _patch_env to recognize it)
    - reset() signature matches gymnasium.Env.reset() with keyword-only arguments
    - step() returns correct 5-tuple (obs, reward, terminated, truncated, info)
    - Compatible with SB3's automatic Monitor wrapping and VecEnv reset behavior
    """
    
    def __init__(self, env: BlokusEnv):
        super().__init__()
        self.env = env
        
        # Single agent wrapper - focuses on one agent
        self.agent_name = "player_0"  # Default to first agent
        
        # Action space
        self.action_space = env.action_space
        
        # Observation space
        self.observation_space = env.observation_space
        
        # Gymnasium required attributes
        self.metadata = env.metadata
        self.render_mode = env.render_mode
        self.spec = None  # Optional, but some wrappers expect it
    
    @property
    def unwrapped(self):
        """Return the unwrapped environment (required by Stable-Baselines3)."""
        return self.env
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None, **kwargs):
        """
        Reset environment.
        
        This method is fully Gymnasium-compatible and works with SB3's Monitor wrapper.
        The signature matches gymnasium.Env.reset() with keyword-only arguments (*).
        
        Args:
            seed: Optional seed for environment reset
            options: Optional dictionary of reset options
            **kwargs: Additional keyword arguments (for compatibility with wrappers)
        
        Returns:
            Tuple of (observation, info) as required by Gymnasium API
        """
        # Accept seed and options as keyword arguments (matching gymnasium.Wrapper.reset() behavior)
        # Call underlying env's reset with seed and options
        # BlokusEnv.reset() now accepts **kwargs, so any remaining kwargs will be ignored
        self.env.reset(seed=seed, options=options, **kwargs)
        
        obs = self.env.observe(self.agent_name)
        info = self.env.infos[self.agent_name]
        
        return obs, info
        
    def step(self, action: int):
        """
        Execute step.
        
        Returns the correct 5-tuple as required by Gymnasium API:
        (observation, reward, terminated, truncated, info)
        
        Args:
            action: Action to execute
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Execute action for current agent
        self.env.step(action)
        
        # Get observation and info for our agent
        obs = self.env.observe(self.agent_name)
        reward = self.env.rewards[self.agent_name]
        terminated = self.env.terminations[self.agent_name]
        truncated = self.env.truncations[self.agent_name]
        info = self.env.infos[self.agent_name]
        
        return obs, reward, terminated, truncated, info
        
    def render(self):
        """Render environment."""
        return self.env.render()


def make_gymnasium_env(render_mode: Optional[str] = None, max_episode_steps: int = 1000):
    """
    Create Gymnasium-compatible Blokus environment.
    
    Args:
        render_mode: Rendering mode
        max_episode_steps: Maximum steps per episode
        
    Returns:
        GymnasiumBlokusWrapper instance
    """
    blokus_env = env(render_mode=render_mode, max_episode_steps=max_episode_steps)
    return GymnasiumBlokusWrapper(blokus_env)


# Diagnostic helper for testing Monitor compatibility
# This can be called manually for debugging, but is not used in production code
def _test_monitor_reset_compat():
    """
    Diagnostic test to verify Monitor + GymnasiumBlokusWrapper reset compatibility.
    
    This function tests that SB3's Monitor wrapper can successfully call reset()
    on GymnasiumBlokusWrapper with keyword arguments, as it does in VecEnv mode.
    
    Usage:
        from envs.blokus_v0 import _test_monitor_reset_compat
        _test_monitor_reset_compat()
    """
    try:
        from stable_baselines3.common.monitor import Monitor
        
        # Create a base environment
        base_env = make_gymnasium_env(render_mode=None, max_episode_steps=50)
        print(f"✓ Created GymnasiumBlokusWrapper: {type(base_env)}")
        
        # Wrap with Monitor (as SB3 does automatically)
        monitored_env = Monitor(base_env)
        print(f"✓ Wrapped with Monitor: {type(monitored_env)}")
        
        # Test reset with seed (as DummyVecEnv does)
        obs, info = monitored_env.reset(seed=42)
        print(f"✓ Monitor.reset(seed=42) succeeded")
        print(f"  - obs type: {type(obs)}, shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
        print(f"  - info type: {type(info)}, keys: {list(info.keys()) if isinstance(info, dict) else 'N/A'}")
        
        # Test reset with seed and options
        obs2, info2 = monitored_env.reset(seed=123, options={})
        print(f"✓ Monitor.reset(seed=123, options={{}}) succeeded")
        
        # Test step
        obs3, reward, terminated, truncated, info3 = monitored_env.step(0)
        print(f"✓ Monitor.step(0) succeeded")
        print(f"  - reward: {reward}, terminated: {terminated}, truncated: {truncated}")
        
        print("\n✅ All Monitor compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Monitor compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Allow running diagnostic test directly
    print("Running Monitor compatibility diagnostic test...")
    _test_monitor_reset_compat()
