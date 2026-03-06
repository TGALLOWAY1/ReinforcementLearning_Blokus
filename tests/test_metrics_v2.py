
import pytest

from engine.advanced_metrics import compute_dead_space_split, compute_effective_frontier
from engine.board import Board, Player
from engine.metrics_config import TelemetryConfig
from engine.move_generator import LegalMoveGenerator
from engine.telemetry import (
    compute_move_telemetry_delta,
    simulate_mobility_stability,
)


@pytest.fixture
def empty_board():
    return Board()

@pytest.fixture
def move_generator():
    return LegalMoveGenerator()

def test_effective_frontier_boxed_in(empty_board):
    # Setup a mock board where player RED is boxed in a corner
    empty_board.grid[0][0] = Player.RED.value
    # Blocked by blue and board edge
    empty_board.grid[1][0] = Player.BLUE.value
    empty_board.grid[0][1] = Player.BLUE.value
    empty_board.grid[1][1] = Player.BLUE.value

    # Calculate effective frontier for RED
    empty_board.init_frontier_for_player(Player.RED)
    eff_f = compute_effective_frontier(empty_board, Player.RED)

    # Because space is highly restricted and vulnerability is capped at maximum
    assert 0.0 < eff_f < 4.0

def test_dead_space_causality_split(empty_board):
    # Red creates a cavity that is walled off by Red itself
    # 0 0 0
    # R R R
    # R 0 R
    # R R R
    for c in range(3):
        empty_board.grid[1][c] = Player.RED.value
        empty_board.grid[3][c] = Player.RED.value
    empty_board.grid[2][0] = Player.RED.value
    empty_board.grid[2][2] = Player.RED.value

    empty_board.init_frontier_for_player(Player.RED)

    ds_self, ds_opp = compute_dead_space_split(empty_board, Player.RED)

    # Grid[2][1] is a dead space bordered only by RED
    # So ds_self should perfectly capture it
    assert ds_self >= 1.0
    assert ds_opp == 0.0

def test_stability_determinism(empty_board, move_generator):
    """Same board + seed => identical stability percentiles."""
    # Place a piece for red and blue so they have moves

    # Initialize some minimal board state to avoid 0 mobility
    empty_board.grid[4][4] = Player.RED.value
    empty_board.grid[4][10] = Player.BLUE.value
    empty_board.init_frontier_for_player(Player.RED)
    empty_board.init_frontier_for_player(Player.BLUE)

    # Override seed
    TelemetryConfig.DETERMINISM_SEED = 123

    res1 = simulate_mobility_stability(empty_board, Player.RED, move_generator, fast_mode=True)

    TelemetryConfig.DETERMINISM_SEED = 123
    res2 = simulate_mobility_stability(empty_board, Player.RED, move_generator, fast_mode=True)

    assert res1["mobilityNextP10"] == res2["mobilityNextP10"]
    assert res1["mobilityNextMean"] == res2["mobilityNextMean"]

def test_telemetry_schema_compatibility(empty_board, move_generator):
    """Ensure old fields are present and new fields calculate without crashing"""
    empty_board.grid[0][0] = Player.RED.value
    empty_board.init_frontier_for_player(Player.RED)

    from engine.telemetry import collect_all_player_metrics
    metrics = collect_all_player_metrics(empty_board, move_generator, fast_mode=True)[Player.RED.name]

    # Check old schema keys
    assert "frontierSize" in metrics
    assert "mobility" in metrics
    assert "deadSpace" in metrics
    assert "centerControl" in metrics
    assert "pieceLockRisk" in metrics

    # Check new schema keys
    assert "effectiveFrontier" in metrics
    assert "mobilityEntropy" in metrics
    assert "lockedArea" in metrics
    assert "bottleneckScore" in metrics

def test_advantage_delta_polarity():
    """Ensure that advantage delta correctly flips sign for lower-is-better metrics."""
    # lockedArea is polarity -1.
    # If I have 0 locked area and opponents have 10, my advantage should be Positive.

    before = {
        "RED": {"lockedArea": 5.0},
        "BLUE": {"lockedArea": 5.0}
    }
    after = {
        "RED": {"lockedArea": 0.0},
        "BLUE": {"lockedArea": 10.0}
    }

    # Mover reduces locked area, opponent increases. So RED's locked area advantage goes up.
    # Before Adv = -1 * (5 - 5) = 0
    # After Adv = -1 * (0 - 10) = 10
    # Delta Adv should be +10

    delta = compute_move_telemetry_delta(
        ply=1,
        mover_id="RED",
        move_id="",
        before=before,
        after=after
    )

    assert "lockedAreaAdv" in delta["deltaAdvantage"]
    assert delta["deltaAdvantage"]["lockedAreaAdv"] == 10.0
