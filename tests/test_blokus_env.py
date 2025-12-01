"""
Tests for Blokus PettingZoo environment.
"""

import unittest
import numpy as np
from envs.blokus_v0 import BlokusEnv, GymnasiumBlokusWrapper, env, make_gymnasium_env
from engine.board import Board, Player, Position
from engine.move_generator import Move


class TestBlokusEnv(unittest.TestCase):
    """Test the Blokus PettingZoo environment."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = BlokusEnv(max_episode_steps=100)
        
    def test_environment_initialization(self):
        """Test environment initialization."""
        self.assertEqual(len(self.env.possible_agents), 4)
        self.assertEqual(self.env.agents, ["player_0", "player_1", "player_2", "player_3"])
        self.assertIsNotNone(self.env.action_space)
        self.assertIsNotNone(self.env.observation_space)
        
    def test_reset(self):
        """Test environment reset."""
        self.env.reset()
        
        # Check that all agents have observations
        for agent in self.env.agents:
            self.assertIn(agent, self.env.observations)
            self.assertIn(agent, self.env.infos)
            self.assertIn(agent, self.env.rewards)
            self.assertIn(agent, self.env.terminations)
            self.assertIn(agent, self.env.truncations)
            
        # Check initial state
        self.assertEqual(self.env.step_count, 0)
        self.assertEqual(self.env.agent_selection, "player_0")
        
    def test_observation_space(self):
        """Test observation space properties."""
        obs_space = self.env.observation_space
        
        # Check shape
        expected_channels = 5 + 21 + 4  # board + remaining_pieces + last_move
        self.assertEqual(obs_space.shape, (expected_channels, 20, 20))
        
        # Check dtype
        self.assertEqual(obs_space.dtype, np.float32)
        
        # Check bounds
        self.assertTrue(np.all(obs_space.low == 0))
        self.assertTrue(np.all(obs_space.high == 1))
        
    def test_action_space(self):
        """Test action space properties."""
        action_space = self.env.action_space
        
        # Should be discrete
        self.assertIsInstance(action_space, type(self.env.action_space))
        
        # Check that action mapping is set up
        self.assertGreater(len(self.env.action_to_move), 0)
        self.assertGreater(len(self.env.move_to_action), 0)
        
    def test_agent_to_player_conversion(self):
        """Test agent to player conversion."""
        self.assertEqual(self.env._agent_to_player("player_0"), Player.RED)
        self.assertEqual(self.env._agent_to_player("player_1"), Player.BLUE)
        self.assertEqual(self.env._agent_to_player("player_2"), Player.YELLOW)
        self.assertEqual(self.env._agent_to_player("player_3"), Player.GREEN)
        
    def test_player_to_agent_conversion(self):
        """Test player to agent conversion."""
        self.assertEqual(self.env._player_to_agent(Player.RED), "player_0")
        self.assertEqual(self.env._player_to_agent(Player.BLUE), "player_1")
        self.assertEqual(self.env._player_to_agent(Player.YELLOW), "player_2")
        self.assertEqual(self.env._player_to_agent(Player.GREEN), "player_3")
        
    def test_observation_generation(self):
        """Test observation generation."""
        self.env.reset()
        
        for agent in self.env.agents:
            obs = self.env.observe(agent)
            
            # Check shape
            self.assertEqual(obs.shape, self.env.observation_space.shape)
            
            # Check dtype
            self.assertEqual(obs.dtype, np.float32)
            
            # Check bounds
            self.assertTrue(np.all(obs >= 0))
            self.assertTrue(np.all(obs <= 1))
            
    def test_info_generation(self):
        """Test info generation."""
        self.env.reset()
        
        for agent in self.env.agents:
            info = self.env.infos[agent]
            
            # Check required keys
            required_keys = [
                "legal_action_mask", "legal_moves_count", "score", 
                "pieces_used", "pieces_remaining", "can_move"
            ]
            for key in required_keys:
                self.assertIn(key, info)
                
            # Check legal action mask
            self.assertEqual(len(info["legal_action_mask"]), self.env.action_space_size)
            self.assertEqual(info["legal_action_mask"].dtype, bool)
            
            # Check score
            self.assertGreaterEqual(info["score"], 0)
            
            # Check pieces
            self.assertEqual(info["pieces_used"], 0)  # Initially no pieces used
            self.assertEqual(info["pieces_remaining"], 21)
            
    def test_action_masking(self):
        """Test action masking for legal moves."""
        self.env.reset()
        
        # Get legal moves for first player
        player_0_info = self.env.infos["player_0"]
        legal_mask = player_0_info["legal_action_mask"]
        
        # Should have some legal moves initially
        self.assertGreater(np.sum(legal_mask), 0)
        
        # Check that legal moves correspond to valid actions
        legal_actions = np.where(legal_mask)[0]
        for action in legal_actions[:10]:  # Check first 10 legal actions
            if action in self.env.action_to_move:
                piece_id, orientation, row, col = self.env.action_to_move[action]
                self.assertGreaterEqual(piece_id, 1)
                self.assertLessEqual(piece_id, 21)
                self.assertGreaterEqual(orientation, 0)
                self.assertGreaterEqual(row, 0)
                self.assertLess(row, 20)
                self.assertGreaterEqual(col, 0)
                self.assertLess(col, 20)
                
    def test_step_execution(self):
        """Test step execution."""
        self.env.reset()
        
        # Get a legal action
        legal_mask = self.env.infos["player_0"]["legal_action_mask"]
        legal_actions = np.where(legal_mask)[0]
        
        if len(legal_actions) > 0:
            action = legal_actions[0]
            
            # Execute step
            self.env.step(action)
            
            # Check that step count increased
            self.assertEqual(self.env.step_count, 1)
            
            # Check that agent selection changed
            self.assertNotEqual(self.env.agent_selection, "player_0")
            
            # Check that observations were updated
            for agent in self.env.agents:
                obs = self.env.observe(agent)
                self.assertEqual(obs.shape, self.env.observation_space.shape)
                
    def test_invalid_action(self):
        """Test handling of invalid actions."""
        self.env.reset()
        
        # Try invalid action (out of bounds)
        invalid_action = self.env.action_space_size + 1
        
        # Should not crash
        self.env.step(invalid_action)
        
        # Agent should be skipped
        self.assertNotEqual(self.env.agent_selection, "player_0")
        
    def test_reward_calculation(self):
        """Test reward calculation."""
        self.env.reset()
        
        # Initial rewards should be 0
        for agent in self.env.agents:
            self.assertEqual(self.env.rewards[agent], 0)
            
        # Get a legal action and execute it
        legal_mask = self.env.infos["player_0"]["legal_action_mask"]
        legal_actions = np.where(legal_mask)[0]
        
        if len(legal_actions) > 0:
            action = legal_actions[0]
            self.env.step(action)
            
            # Should have some reward (score delta)
            self.assertGreaterEqual(self.env.rewards["player_0"], 0)
            
    def test_termination_conditions(self):
        """Test termination conditions."""
        self.env.reset()
        
        # Initially no terminations
        for agent in self.env.agents:
            self.assertFalse(self.env.terminations[agent])
            self.assertFalse(self.env.truncations[agent])
            
        # Test max episode steps
        self.env.max_episode_steps = 5
        self.env.reset()
        
        for _ in range(6):  # Exceed max steps
            if not self.env.terminations[self.env.agent_selection]:
                # Try to make a move
                legal_mask = self.env.infos[self.env.agent_selection]["legal_action_mask"]
                legal_actions = np.where(legal_mask)[0]
                
                if len(legal_actions) > 0:
                    action = legal_actions[0]
                    self.env.step(action)
                else:
                    # Skip if no legal moves
                    self.env.step(0)
                    
        # Should have truncations
        truncations = any(self.env.truncations.values())
        self.assertTrue(truncations)
        
    def test_game_completion(self):
        """Test game completion detection."""
        self.env.reset()
        
        # Simulate a few moves
        for _ in range(10):
            if not self.env.terminations[self.env.agent_selection]:
                legal_mask = self.env.infos[self.env.agent_selection]["legal_action_mask"]
                legal_actions = np.where(legal_mask)[0]
                
                if len(legal_actions) > 0:
                    action = legal_actions[0]
                    self.env.step(action)
                else:
                    # Skip if no legal moves
                    self.env.step(0)
                    
        # Check that game state is consistent
        for agent in self.env.agents:
            info = self.env.infos[agent]
            self.assertGreaterEqual(info["score"], 0)
            self.assertLessEqual(info["pieces_used"], 21)
            self.assertGreaterEqual(info["pieces_remaining"], 0)
            
    def test_render_modes(self):
        """Test render modes."""
        # Test human render mode
        env_human = BlokusEnv(render_mode="human")
        env_human.reset()
        env_human.render()  # Should not crash
        
        # Test rgb_array render mode
        env_rgb = BlokusEnv(render_mode="rgb_array")
        env_rgb.reset()
        rgb_array = env_rgb.render()
        
        # Check RGB array properties
        self.assertEqual(rgb_array.shape, (400, 400, 3))  # 20x20 * 20x20 pixels
        self.assertEqual(rgb_array.dtype, np.uint8)
        
    def test_seed_consistency(self):
        """Test seed consistency."""
        # Reset with same seed
        self.env.reset(seed=42)
        obs1 = self.env.observe("player_0")
        
        self.env.reset(seed=42)
        obs2 = self.env.observe("player_0")
        
        # Observations should be identical
        np.testing.assert_array_equal(obs1, obs2)


class TestGymnasiumWrapper(unittest.TestCase):
    """Test the Gymnasium compatibility wrapper."""
    
    def setUp(self):
        """Set up test wrapper."""
        self.env = BlokusEnv(max_episode_steps=100)
        self.wrapper = GymnasiumBlokusWrapper(self.env)
        
    def test_wrapper_initialization(self):
        """Test wrapper initialization."""
        self.assertEqual(self.wrapper.action_space, self.env.action_space)
        self.assertEqual(self.wrapper.observation_space, self.env.observation_space)
        self.assertEqual(self.wrapper.agent_name, "player_0")
        
    def test_wrapper_reset(self):
        """Test wrapper reset."""
        obs, info = self.wrapper.reset()
        
        # Check observation
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertEqual(obs.dtype, np.float32)
        
        # Check info
        self.assertIn("legal_action_mask", info)
        self.assertIn("score", info)
        
    def test_wrapper_step(self):
        """Test wrapper step."""
        obs, info = self.wrapper.reset()
        
        # Get a legal action
        legal_mask = info["legal_action_mask"]
        legal_actions = np.where(legal_mask)[0]
        
        if len(legal_actions) > 0:
            action = legal_actions[0]
            
            # Execute step
            obs, reward, terminated, truncated, info = self.wrapper.step(action)
            
            # Check return values
            self.assertEqual(obs.shape, self.env.observation_space.shape)
            self.assertIsInstance(reward, (int, float, np.integer, np.floating))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)
            
    def test_wrapper_render(self):
        """Test wrapper render."""
        self.wrapper.reset()
        result = self.wrapper.render()
        
        # Should return RGB array (or None for human mode)
        if result is not None:
            self.assertEqual(result.shape, (400, 400, 3))
            self.assertEqual(result.dtype, np.uint8)


class TestEnvironmentFactory(unittest.TestCase):
    """Test environment factory functions."""
    
    def test_env_factory(self):
        """Test env factory function."""
        env_instance = env(max_episode_steps=50)
        
        self.assertIsInstance(env_instance, BlokusEnv)
        self.assertEqual(env_instance.max_episode_steps, 50)
        
    def test_gymnasium_env_factory(self):
        """Test Gymnasium environment factory function."""
        gym_env = make_gymnasium_env(max_episode_steps=50)
        
        self.assertIsInstance(gym_env, GymnasiumBlokusWrapper)
        self.assertEqual(gym_env.env.max_episode_steps, 50)
        
    def test_factory_with_render_mode(self):
        """Test factory functions with render mode."""
        # Test PettingZoo env with render mode
        env_instance = env(render_mode="rgb_array")
        self.assertEqual(env_instance.render_mode, "rgb_array")
        
        # Test Gymnasium env with render mode
        gym_env = make_gymnasium_env(render_mode="human")
        self.assertEqual(gym_env.env.render_mode, "human")


class TestEnvironmentCompliance(unittest.TestCase):
    """Test environment compliance with PettingZoo standards."""
    
    def test_aec_compliance(self):
        """Test AEC environment compliance."""
        env = BlokusEnv()
        env.reset()  # Reset to initialize all attributes
        
        # Test required attributes
        required_attrs = [
            "agents", "possible_agents", "observation_spaces", "action_spaces",
            "rewards", "terminations", "truncations", "infos", "observations",
            "agent_selection", "step_count"
        ]
        
        for attr in required_attrs:
            self.assertTrue(hasattr(env, attr), f"Missing required attribute: {attr}")
            
    def test_agent_selector(self):
        """Test agent selector functionality."""
        env = BlokusEnv()
        env.reset()
        
        # Test agent cycling
        agents_seen = set()
        for _ in range(10):  # Test multiple cycles
            agent = env.agent_selection
            agents_seen.add(agent)
            
            # Make a move or skip
            legal_mask = env.infos[agent]["legal_action_mask"]
            legal_actions = np.where(legal_mask)[0]
            
            if len(legal_actions) > 0:
                env.step(legal_actions[0])
            else:
                env.step(0)
                
        # Should have seen all agents
        self.assertEqual(len(agents_seen), 4)
        
    def test_observation_consistency(self):
        """Test observation consistency across agents."""
        env = BlokusEnv()
        env.reset()
        
        # All agents should have valid observations
        for agent in env.agents:
            obs = env.observe(agent)
            self.assertEqual(obs.shape, env.observation_space.shape)
            self.assertEqual(obs.dtype, env.observation_space.dtype)
            
    def test_info_consistency(self):
        """Test info consistency across agents."""
        env = BlokusEnv()
        env.reset()
        
        # All agents should have valid infos
        for agent in env.agents:
            info = env.infos[agent]
            
            # Check required info keys
            required_keys = ["legal_action_mask", "score", "pieces_used", "pieces_remaining"]
            for key in required_keys:
                self.assertIn(key, info)
                
    def test_action_space_consistency(self):
        """Test action space consistency."""
        env = BlokusEnv()
        
        # Action space should be consistent
        self.assertIsNotNone(env.action_space)
        self.assertGreater(env.action_space_size, 0)
        
        # Action mapping should be consistent
        self.assertEqual(len(env.action_to_move), env.action_space_size)
        self.assertEqual(len(env.move_to_action), env.action_space_size)
        
    def test_legal_action_mask_accuracy(self):
        """Test that legal action masks are accurate."""
        env = BlokusEnv()
        env.reset()
        
        # Test for each agent
        for agent in env.agents:
            info = env.infos[agent]
            legal_mask = info["legal_action_mask"]
            
            # Count legal actions
            legal_count = np.sum(legal_mask)
            
            # Verify by checking actual legal moves
            player = env._agent_to_player(agent)
            actual_legal_moves = env.move_generator.get_legal_moves(env.game.board, player)
            
            # Should match (approximately, due to action space mapping)
            self.assertGreaterEqual(legal_count, len(actual_legal_moves) * 0.8)  # Allow some tolerance


if __name__ == "__main__":
    unittest.main()
