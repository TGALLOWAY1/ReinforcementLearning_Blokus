
import unittest
from unittest.mock import MagicMock, patch

from analytics.metrics import MetricInput, compute_blocking_metrics


class TestBlockingMetrics(unittest.TestCase):
    @patch('analytics.metrics.blocking.get_mobility_counts')
    def test_blocking(self, mock_get_counts):
        # Before: P2 has 20 moves
        # After: P2 has 15 moves (Loss 5)
        # Area gain = 5
        
        mock_get_counts.return_value = (
            {1: 10, 2: 20}, # Before
            {1: 12, 2: 15}  # After
        )
        
        mock_inp = MagicMock(spec=MetricInput)
        mock_inp.player_id = 1
        mock_inp.opponents = [2]
        # placed_squares list of length 5
        mock_inp.get_placed_squares.return_value = [(0,0), (0,1), (0,2), (0,3), (0,4)] 
        
        res = compute_blocking_metrics(mock_inp)
        
        self.assertEqual(res['blocking'], 5)
        self.assertEqual(res['block_eff'], 1.0) # 5 / 5 = 1.0
        self.assertEqual(res['blocking_target_id'], 2)
        self.assertEqual(res['blocking_target_loss'], 5)
        
    @patch('analytics.metrics.blocking.get_mobility_counts')
    def test_no_blocking(self, mock_get_counts):
        # Before: P2 has 20
        # After: P2 has 20 (Loss 0)
        
        mock_get_counts.return_value = (
            {1: 10, 2: 20}, 
            {1: 12, 2: 20}
        )
        
        mock_inp = MagicMock(spec=MetricInput)
        mock_inp.player_id = 1
        mock_inp.opponents = [2]
        mock_inp.get_placed_squares.return_value = [(0,0)] # len 1
        
        res = compute_blocking_metrics(mock_inp)
        
        self.assertEqual(res['blocking'], 0)
        self.assertEqual(res['blocking_target_loss'], 0)

if __name__ == '__main__':
    unittest.main()
