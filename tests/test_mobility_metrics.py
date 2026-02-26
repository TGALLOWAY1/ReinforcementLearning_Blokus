"""
Unit tests for mobility metrics per docs/metrics/mobility.md.
Ensures backend and frontend produce identical results for the same inputs.
"""

import json
import os
import unittest

from engine.mobility_metrics import (
    compute_player_mobility_metrics,
    compute_player_mobility_metrics_from_dicts,
)
from engine.move_generator import Move


def _fixture_path():
    return os.path.join(os.path.dirname(__file__), "fixtures", "mobility_metrics_fixture.json")


class TestMobilityMetrics(unittest.TestCase):
    """Test mobility metrics against spec and shared fixture."""

    def test_fixture_matches_expected(self):
        """Backend produces expected values for shared fixture."""
        with open(_fixture_path()) as f:
            data = json.load(f)
        legal_moves = data["legal_moves"]
        pieces_used = data["pieces_used"]
        expected = data["expected"]

        m = compute_player_mobility_metrics_from_dicts(legal_moves, pieces_used)

        self.assertEqual(m.totalPlacements, expected["totalPlacements"])
        self.assertAlmostEqual(m.totalOrientationNormalized, expected["totalOrientationNormalized"])
        self.assertAlmostEqual(m.totalCellWeighted, expected["totalCellWeighted"])
        for k, v in expected["buckets"].items():
            self.assertAlmostEqual(m.buckets[int(k)], v, msg=f"bucket[{k}]")

    def test_engine_moves_match_dicts(self):
        """compute_player_mobility_metrics and _from_dicts produce same result."""
        with open(_fixture_path()) as f:
            data = json.load(f)
        legal_moves_dicts = data["legal_moves"]
        pieces_used = data["pieces_used"]

        moves = [
            Move(m["piece_id"], m["orientation"], m["anchor_row"], m["anchor_col"])
            for m in legal_moves_dicts
        ]
        m1 = compute_player_mobility_metrics(moves, pieces_used)
        m2 = compute_player_mobility_metrics_from_dicts(legal_moves_dicts, pieces_used)

        self.assertEqual(m1.totalPlacements, m2.totalPlacements)
        self.assertAlmostEqual(m1.totalOrientationNormalized, m2.totalOrientationNormalized)
        self.assertAlmostEqual(m1.totalCellWeighted, m2.totalCellWeighted)
        for k in range(1, 6):
            self.assertAlmostEqual(m1.buckets.get(k, 0), m2.buckets.get(k, 0))

    def test_pieces_used_excluded(self):
        """Pieces in pieces_used are excluded from metrics."""
        moves = [
            Move(1, 0, 0, 0),
            Move(1, 0, 1, 0),
        ]
        m = compute_player_mobility_metrics(moves, [1])
        self.assertEqual(m.totalPlacements, 0)
        self.assertAlmostEqual(m.totalOrientationNormalized, 0.0)
        self.assertAlmostEqual(m.totalCellWeighted, 0.0)
        self.assertAlmostEqual(m.buckets[1], 0.0)

    def test_empty_moves(self):
        """Empty legal moves yields zeros."""
        m = compute_player_mobility_metrics([], [])
        self.assertEqual(m.totalPlacements, 0)
        self.assertAlmostEqual(m.totalOrientationNormalized, 0.0)
        self.assertAlmostEqual(m.totalCellWeighted, 0.0)
        for k in range(1, 6):
            self.assertAlmostEqual(m.buckets.get(k, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
