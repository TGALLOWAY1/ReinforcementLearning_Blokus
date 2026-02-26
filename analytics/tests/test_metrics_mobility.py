
import unittest
from unittest.mock import MagicMock, patch

from analytics.metrics import MetricInput, compute_mobility_metrics


class TestMobilityMetrics(unittest.TestCase):
    def test_precomputed(self):
        mock_inp = MagicMock(spec=MetricInput)
        mock_inp.player_id = 1
        mock_inp.opponents = [2]
        mock_inp.precomputed_values = {
            'mobility_counts_before': {1: 10, 2: 20},
            'mobility_counts_after': {1: 12, 2: 15}
        }
        # Ensure state attributes exist even if ignored
        mock_inp.state = MagicMock()
        mock_inp.next_state = MagicMock()
        
        res = compute_mobility_metrics(mock_inp)
        
        self.assertEqual(res['mobility_me_before'], 10)
        self.assertEqual(res['mobility_me_after'], 12)
        self.assertEqual(res['mobility_me_delta'], 2)
        self.assertAlmostEqual(res['mobility_me_ratio'], 1.2)
        
        self.assertEqual(res['mobility_opp_before_sum'], 20)
        self.assertEqual(res['mobility_opp_after_sum'], 15)
        self.assertEqual(res['mobility_opp_delta_sum'], -5)

    @patch('analytics.metrics.mobility.get_move_generator')
    def test_computed_fallback(self, mock_get_gen):
        # Setup mock generator
        mock_gen = MagicMock()
        mock_get_gen.return_value = mock_gen
        
        # Mock legal moves return (lists of dummy items)
        # get_legal_moves(board, player) -> List[Move]
        def fake_moves(board, player):
            if board == 'state_before':
                return [1]*10 if player.value == 1 else [1]*20
            if board == 'state_after':
                return [1]*12 if player.value == 1 else [1]*15
            return []
            
        mock_gen.get_legal_moves.side_effect = fake_moves
        
        mock_inp = MagicMock(spec=MetricInput)
        mock_inp.player_id = 1
        mock_inp.opponents = [2]
        mock_inp.precomputed_values = None # Force computation
        mock_inp.state = 'state_before'
        mock_inp.next_state = 'state_after'
        
        res = compute_mobility_metrics(mock_inp)
        
        self.assertEqual(res['mobility_me_before'], 10)
        self.assertEqual(res['mobility_me_after'], 12)
        self.assertEqual(res['mobility_opp_delta_sum'], -5)

if __name__ == '__main__':
    unittest.main()
