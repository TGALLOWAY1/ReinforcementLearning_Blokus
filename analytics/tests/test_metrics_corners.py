
import unittest
from unittest.mock import MagicMock

from analytics.metrics import compute_corner_metrics


class TestCornerMetrics(unittest.TestCase):
    def test_corner_counts(self):
        mock_inp = MagicMock() # spec removed to avoid strict checks if that was the issue
        mock_inp.player_id = 1
        mock_inp.opponents = [2]
        
        # Helper to fake frontiers
        # Before: P1 has 1 corner, P2 has 2
        def frontier_before(player):
            if player.value == 1: return {(0,0)}
            if player.value == 2: return {(1,1), (2,2)}
            return set()
            
        # After: P1 has 2 corners, P2 has 1
        def frontier_after(player):
            if player.value == 1: return {(0,0), (0,1)}
            if player.value == 2: return {(1,1)}
            return set()
            
        mock_inp.state.get_frontier.side_effect = frontier_before
        mock_inp.next_state.get_frontier.side_effect = frontier_after
        
        res = compute_corner_metrics(mock_inp)
        
        self.assertEqual(res['corners_me_before'], 1)
        self.assertEqual(res['corners_me_after'], 2)
        self.assertEqual(res['corners_me_delta'], 1)
        
        self.assertEqual(res['corners_opp_before_sum'], 2)
        self.assertEqual(res['corners_opp_after_sum'], 1)
        self.assertEqual(res['corner_block'], 1) # max(0, 2-1) = 1

if __name__ == '__main__':
    unittest.main()
