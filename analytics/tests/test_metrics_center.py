
import unittest
from unittest.mock import MagicMock

from analytics.metrics import (
    MetricInput,
    compute_center_metrics,
)


class TestCenterMetrics(unittest.TestCase):
    def test_center_gain(self):
        # Create a MetricInput with known placed_squares
        # Center is (9.5, 9.5) for 20x20
        # Radius 4
        
        # Square (9,9) is dist 0.5+0.5=1.0 <= 4 -> Inside
        # Square (0,0) is dist 9.5+9.5=19 > 4 -> Outside
        
        mock_inp = MagicMock(spec=MetricInput)
        mock_inp.get_placed_squares.return_value = [(9, 9), (0, 0)]
        
        res = compute_center_metrics(mock_inp)
        
        self.assertEqual(res['center_gain'], 1)
        # Dist: (1.0 + 19.0) / 2 = 10.0
        self.assertAlmostEqual(res['center_distance'], 10.0)
        
    def test_empty_move(self):
        mock_inp = MagicMock(spec=MetricInput)
        mock_inp.get_placed_squares.return_value = []
        
        res = compute_center_metrics(mock_inp)
        self.assertEqual(res['center_gain'], 0)
        self.assertEqual(res['center_distance'], 0.0)

if __name__ == '__main__':
    unittest.main()
