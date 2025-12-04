"""
Tests for win detection and game result information in BlokusEnv.

These tests verify that when the environment reaches a terminal state,
the info dict contains final_scores, winner_ids, is_tie, and player0_won.
"""

import unittest
from unittest.mock import Mock, patch
from engine.game import BlokusGame, GameResult
from engine.board import Player
from envs.blokus_v0 import BlokusEnv


class TestEnvWinDetection(unittest.TestCase):
    """Test win detection information in environment info dict."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = BlokusEnv(max_episode_steps=1000)
    
    def test_info_contains_game_result_on_terminal_step(self):
        """
        Test that info dict contains game result fields when game is over.
        
        This test simulates a finished game by:
        1. Marking the game as over
        2. Setting up a game result
        3. Checking that info dict contains the expected fields
        """
        # Reset environment
        self.env.reset()
        
        # Mark game as over and set up a game result
        self.env.game.board.game_over = True
        
        # Create a mock game result where player_0 (RED) wins
        game_result = GameResult(
            scores={1: 25, 2: 20, 3: 15, 4: 10},  # RED wins
            winner_ids=[1],  # RED (player_0)
            is_tie=False
        )
        self.env.game_result = game_result
        
        # Also set the game's winner for consistency
        self.env.game.winner = Player.RED
        
        # Get info for player_0
        info = self.env._get_info("player_0")
        
        # Verify game result fields are present
        self.assertIn("final_scores", info)
        self.assertIn("winner_ids", info)
        self.assertIn("is_tie", info)
        self.assertIn("player0_won", info)
        
        # Verify values
        self.assertEqual(info["final_scores"], {1: 25, 2: 20, 3: 15, 4: 10})
        self.assertEqual(info["winner_ids"], [1])
        self.assertFalse(info["is_tie"])
        self.assertTrue(info["player0_won"])
    
    def test_info_contains_tie_information(self):
        """Test that info dict correctly identifies ties."""
        # Reset environment
        self.env.reset()
        
        # Mark game as over and set up a tie scenario
        self.env.game.board.game_over = True
        
        # Create a game result with a tie
        game_result = GameResult(
            scores={1: 20, 2: 20, 3: 15, 4: 10},  # RED and BLUE tie
            winner_ids=[1, 2],  # RED and BLUE tie
            is_tie=True
        )
        self.env.game_result = game_result
        self.env.game.winner = None  # Tie
        
        # Get info for player_0
        info = self.env._get_info("player_0")
        
        # Verify tie information
        self.assertTrue(info["is_tie"])
        self.assertEqual(len(info["winner_ids"]), 2)
        self.assertIn(1, info["winner_ids"])  # RED
        self.assertIn(2, info["winner_ids"])  # BLUE
        
        # player0_won should be False for ties (even though player_0 is in winner_ids)
        self.assertFalse(info["player0_won"])
    
    def test_info_no_game_result_before_terminal(self):
        """
        Test that info dict does not contain game result fields before game is over.
        
        This ensures we only add these fields on terminal steps.
        """
        # Reset environment (game not over)
        self.env.reset()
        
        # Ensure game_result is None
        self.env.game_result = None
        
        # Get info for player_0
        info = self.env._get_info("player_0")
        
        # Verify game result fields are NOT present
        self.assertNotIn("final_scores", info)
        self.assertNotIn("winner_ids", info)
        self.assertNotIn("is_tie", info)
        self.assertNotIn("player0_won", info)
        
        # But other fields should still be present
        self.assertIn("score", info)
        self.assertIn("legal_action_mask", info)
        self.assertIn("can_move", info)
    
    def test_info_consistent_across_agents(self):
        """
        Test that all agents get the same game result information on terminal steps.
        
        In a multi-agent environment, all agents should see the same final scores
        and winner information.
        """
        # Reset environment
        self.env.reset()
        
        # Mark game as over
        self.env.game.board.game_over = True
        
        # Set up game result
        game_result = GameResult(
            scores={1: 30, 2: 25, 3: 20, 4: 15},
            winner_ids=[1],  # RED wins
            is_tie=False
        )
        self.env.game_result = game_result
        
        # Get info for all agents
        info_player_0 = self.env._get_info("player_0")
        info_player_1 = self.env._get_info("player_1")
        info_player_2 = self.env._get_info("player_2")
        info_player_3 = self.env._get_info("player_3")
        
        # All agents should see the same final_scores and winner_ids
        self.assertEqual(info_player_0["final_scores"], info_player_1["final_scores"])
        self.assertEqual(info_player_0["final_scores"], info_player_2["final_scores"])
        self.assertEqual(info_player_0["final_scores"], info_player_3["final_scores"])
        
        self.assertEqual(info_player_0["winner_ids"], info_player_1["winner_ids"])
        self.assertEqual(info_player_0["winner_ids"], info_player_2["winner_ids"])
        self.assertEqual(info_player_0["winner_ids"], info_player_3["winner_ids"])
        
        self.assertEqual(info_player_0["is_tie"], info_player_1["is_tie"])
        self.assertEqual(info_player_0["is_tie"], info_player_2["is_tie"])
        self.assertEqual(info_player_0["is_tie"], info_player_3["is_tie"])
    
    def test_player0_won_false_when_loses(self):
        """Test that player0_won is False when player_0 loses."""
        # Reset environment
        self.env.reset()
        
        # Mark game as over
        self.env.game.board.game_over = True
        
        # Create game result where BLUE wins (player_1)
        game_result = GameResult(
            scores={1: 15, 2: 25, 3: 20, 4: 10},  # BLUE wins
            winner_ids=[2],  # BLUE (player_1)
            is_tie=False
        )
        self.env.game_result = game_result
        self.env.game.winner = Player.BLUE
        
        # Get info for player_0
        info = self.env._get_info("player_0")
        
        # player0_won should be False
        self.assertFalse(info["player0_won"])
        self.assertEqual(info["winner_ids"], [2])  # BLUE won
    
    def test_check_termination_sets_game_result(self):
        """
        Test that _check_termination_truncation() sets game_result when game is over.
        
        This verifies the integration - when the game ends, game_result should
        be populated automatically.
        """
        # Reset environment
        self.env.reset()
        
        # Mock game.is_game_over() to return True
        with patch.object(self.env.game, 'is_game_over', return_value=True):
            # Mock get_game_result() to return a known result
            mock_result = GameResult(
                scores={1: 20, 2: 18, 3: 16, 4: 14},
                winner_ids=[1],
                is_tie=False
            )
            with patch.object(self.env.game, 'get_game_result', return_value=mock_result):
                # Call _check_termination_truncation()
                self.env._check_termination_truncation()
                
                # Verify game_result was set
                self.assertIsNotNone(self.env.game_result)
                self.assertEqual(self.env.game_result, mock_result)
                
                # Verify all agents are terminated
                for agent in self.env.agents:
                    self.assertTrue(self.env.terminations[agent])


if __name__ == '__main__':
    unittest.main()

