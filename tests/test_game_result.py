"""
Tests for GameResult and get_game_result() API in BlokusGame.

These tests verify that the canonical game result API correctly computes
final scores and winner information using the standard Blokus scoring system.
"""

import unittest
from unittest.mock import Mock, patch
from engine.game import BlokusGame, GameResult
from engine.board import Board, Player
from engine.move_generator import LegalMoveGenerator, Move


class TestGameResult(unittest.TestCase):
    """Test the GameResult dataclass and get_game_result() method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.game = BlokusGame()
    
    def test_game_result_dataclass(self):
        """Test that GameResult dataclass can be instantiated."""
        scores = {1: 10, 2: 15, 3: 8, 4: 12}
        winner_ids = [2]
        is_tie = False
        
        result = GameResult(scores=scores, winner_ids=winner_ids, is_tie=is_tie)
        
        self.assertEqual(result.scores, scores)
        self.assertEqual(result.winner_ids, winner_ids)
        self.assertFalse(result.is_tie)
    
    def test_game_result_tie(self):
        """Test GameResult with a tie scenario."""
        scores = {1: 15, 2: 15, 3: 10, 4: 8}
        winner_ids = [1, 2]
        is_tie = True
        
        result = GameResult(scores=scores, winner_ids=winner_ids, is_tie=is_tie)
        
        self.assertEqual(result.scores, scores)
        self.assertEqual(result.winner_ids, winner_ids)
        self.assertTrue(result.is_tie)
        self.assertEqual(len(result.winner_ids), 2)
    
    def test_get_game_result_before_game_over(self):
        """
        Test that get_game_result() can be called before game is over.
        
        It should compute scores from current board state (even if game isn't over).
        """
        # Game should not be over initially
        self.assertFalse(self.game.is_game_over())
        
        # Should be able to call get_game_result() even before game over
        result = self.game.get_game_result()
        
        # Should return a valid GameResult
        self.assertIsInstance(result, GameResult)
        self.assertIn(Player.RED.value, result.scores)
        self.assertIn(Player.BLUE.value, result.scores)
        self.assertIn(Player.YELLOW.value, result.scores)
        self.assertIn(Player.GREEN.value, result.scores)
        
        # All scores should be non-negative
        for score in result.scores.values():
            self.assertGreaterEqual(score, 0)
    
    def test_get_game_result_after_game_over(self):
        """
        Test that get_game_result() returns correct result after game is over.
        
        This test simulates a finished game by:
        1. Marking game as over
        2. Setting up a scenario where one player has a higher score
        """
        # Mark game as over
        self.game.board.game_over = True
        
        # Mock get_score to return different scores for each player
        def mock_get_score(player):
            # RED wins with highest score
            scores = {
                Player.RED: 25,
                Player.BLUE: 20,
                Player.YELLOW: 15,
                Player.GREEN: 10
            }
            return scores.get(player, 0)
        
        with patch.object(self.game, 'get_score', side_effect=mock_get_score):
            result = self.game.get_game_result()
            
            # Verify result structure
            self.assertIsInstance(result, GameResult)
            self.assertEqual(result.scores[Player.RED.value], 25)
            self.assertEqual(result.scores[Player.BLUE.value], 20)
            self.assertEqual(result.scores[Player.YELLOW.value], 15)
            self.assertEqual(result.scores[Player.GREEN.value], 10)
            
            # RED should be the winner
            self.assertEqual(result.winner_ids, [Player.RED.value])
            self.assertFalse(result.is_tie)
    
    def test_get_game_result_with_tie(self):
        """Test get_game_result() correctly identifies ties."""
        # Mark game as over
        self.game.board.game_over = True
        
        # Mock get_score to return a tie scenario
        def mock_get_score(player):
            # RED and BLUE tie for highest score
            scores = {
                Player.RED: 20,
                Player.BLUE: 20,
                Player.YELLOW: 15,
                Player.GREEN: 10
            }
            return scores.get(player, 0)
        
        with patch.object(self.game, 'get_score', side_effect=mock_get_score):
            result = self.game.get_game_result()
            
            # Verify tie detection
            self.assertTrue(result.is_tie)
            self.assertEqual(len(result.winner_ids), 2)
            self.assertIn(Player.RED.value, result.winner_ids)
            self.assertIn(Player.BLUE.value, result.winner_ids)
            self.assertNotIn(Player.YELLOW.value, result.winner_ids)
            self.assertNotIn(Player.GREEN.value, result.winner_ids)
    
    def test_get_game_result_three_way_tie(self):
        """Test get_game_result() with a three-way tie."""
        # Mark game as over
        self.game.board.game_over = True
        
        # Mock get_score to return a three-way tie
        def mock_get_score(player):
            scores = {
                Player.RED: 18,
                Player.BLUE: 18,
                Player.YELLOW: 18,
                Player.GREEN: 10
            }
            return scores.get(player, 0)
        
        with patch.object(self.game, 'get_score', side_effect=mock_get_score):
            result = self.game.get_game_result()
            
            # Verify three-way tie
            self.assertTrue(result.is_tie)
            self.assertEqual(len(result.winner_ids), 3)
            self.assertIn(Player.RED.value, result.winner_ids)
            self.assertIn(Player.BLUE.value, result.winner_ids)
            self.assertIn(Player.YELLOW.value, result.winner_ids)
            self.assertNotIn(Player.GREEN.value, result.winner_ids)
    
    def test_get_game_result_uses_existing_scoring_logic(self):
        """
        Test that get_game_result() uses the same scoring logic as get_score().
        
        This ensures consistency - the scores in GameResult should match
        what get_score() returns for each player.
        """
        # Mark game as over
        self.game.board.game_over = True
        
        # Get result
        result = self.game.get_game_result()
        
        # Verify that scores match what get_score() returns
        for player in Player:
            expected_score = self.game.get_score(player)
            actual_score = result.scores[player.value]
            self.assertEqual(actual_score, expected_score,
                           f"Score mismatch for {player.name}: "
                           f"get_score()={expected_score}, GameResult={actual_score}")
    
    def test_get_winner_uses_get_game_result(self):
        """
        Test that get_winner() correctly uses get_game_result() internally.
        
        This verifies the refactoring - get_winner() should return the same
        result as extracting the winner from get_game_result().
        """
        # Mark game as over
        self.game.board.game_over = True
        
        # Mock get_score to return known scores
        def mock_get_score(player):
            scores = {
                Player.RED: 30,
                Player.BLUE: 25,
                Player.YELLOW: 20,
                Player.GREEN: 15
            }
            return scores.get(player, 0)
        
        with patch.object(self.game, 'get_score', side_effect=mock_get_score):
            # Get winner using get_winner()
            winner = self.game.get_winner()
            
            # Get result using get_game_result()
            result = self.game.get_game_result()
            
            # Verify consistency
            if result.is_tie:
                self.assertIsNone(winner, "get_winner() should return None for ties")
            else:
                winner_id = result.winner_ids[0]
                expected_winner = Player(winner_id)
                self.assertEqual(winner, expected_winner,
                               f"get_winner()={winner}, GameResult winner={expected_winner}")
    
    def test_get_winner_returns_none_for_tie(self):
        """Test that get_winner() returns None when there's a tie."""
        # Mark game as over
        self.game.board.game_over = True
        
        # Mock get_score to return a tie
        def mock_get_score(player):
            scores = {
                Player.RED: 20,
                Player.BLUE: 20,
                Player.YELLOW: 15,
                Player.GREEN: 10
            }
            return scores.get(player, 0)
        
        with patch.object(self.game, 'get_score', side_effect=mock_get_score):
            winner = self.game.get_winner()
            self.assertIsNone(winner, "get_winner() should return None for ties")
    
    def test_check_game_over_uses_get_game_result(self):
        """
        Test that _check_game_over() correctly uses get_game_result().
        
        This verifies that when the game ends, _check_game_over() uses
        get_game_result() to determine the winner.
        """
        # Mock has_legal_moves to return False (game over condition)
        with patch.object(self.game.move_generator, 'has_legal_moves', return_value=False):
            # Mock get_score to return known scores
            def mock_get_score(player):
                scores = {
                    Player.RED: 28,
                    Player.BLUE: 22,
                    Player.YELLOW: 18,
                    Player.GREEN: 12
                }
                return scores.get(player, 0)
            
            with patch.object(self.game, 'get_score', side_effect=mock_get_score):
                # Call _check_game_over()
                self.game._check_game_over()
                
                # Verify game is marked as over
                self.assertTrue(self.game.board.game_over)
                
                # Verify winner is set correctly (RED should win)
                self.assertEqual(self.game.winner, Player.RED)
                
                # Verify get_game_result() returns consistent result
                result = self.game.get_game_result()
                self.assertEqual(result.winner_ids, [Player.RED.value])
                self.assertFalse(result.is_tie)
    
    def test_check_game_over_sets_winner_none_for_tie(self):
        """Test that _check_game_over() sets winner to None for ties."""
        # Mock has_legal_moves to return False (game over condition)
        with patch.object(self.game.move_generator, 'has_legal_moves', return_value=False):
            # Mock get_score to return a tie
            def mock_get_score(player):
                scores = {
                    Player.RED: 20,
                    Player.BLUE: 20,
                    Player.YELLOW: 15,
                    Player.GREEN: 10
                }
                return scores.get(player, 0)
            
            with patch.object(self.game, 'get_score', side_effect=mock_get_score):
                # Call _check_game_over()
                self.game._check_game_over()
                
                # Verify game is marked as over
                self.assertTrue(self.game.board.game_over)
                
                # Verify winner is None for tie
                self.assertIsNone(self.game.winner)
                
                # Verify get_game_result() confirms tie
                result = self.game.get_game_result()
                self.assertTrue(result.is_tie)
                self.assertEqual(len(result.winner_ids), 2)


if __name__ == '__main__':
    unittest.main()

