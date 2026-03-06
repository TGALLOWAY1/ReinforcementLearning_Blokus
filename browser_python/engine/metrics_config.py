"""
Central configuration for advanced metrics, fast-mode tuning, and telemetry logic.
"""

from typing import Dict


class TelemetryConfig:
    # Sampling & Performance Knobs
    MOBILITY_SAMPLES_K = 3         # How many opponent moves to sample for stability in fast-mode
    PIECE_SUBSET_N = 5             # How many top remaining pieces to check for viability/utility in fast-mode
    ANCHOR_RADIUS = 3              # Radius for space/vulnerability approximations
    ENABLE_FULL_ENUM = False      # If true, overrides fast-mode to compute exhaustively

    # Random Seed
    DETERMINISM_SEED = 42

    # Polarity Map for Advantage Calculations (1 = higher is better, -1 = lower is better)
    METRIC_POLARITY: Dict[str, int] = {
        # Expansion
        "frontierSize": 1,
        "effectiveFrontier": 1,
        "frontierComponentCount": 1,
        "frontierQuadrantCoverage": 1,

        # Flexibility
        "mobility": 1,
        "mobilityWeighted": 1,
        "mobilityEntropy": 1,
        "pieceTop1Share": -1,       # High reliance on 1 piece is bad
        "anchorTop1Share": -1,      # High reliance on 1 anchor is bad

        # Stability
        "mobilityNextMean": 1,
        "mobilityNextP10": 1,
        "mobilityNextMin": 1,
        "mobilityDropRisk": -1,     # Large drop is bad

        # Position
        "centerControl": 1,
        "centerControlWeighted": 1,

        # Lockdown (Dead Space)
        "deadSpaceNearOpponents": 1, # Implies sealing off others (generally good)
        "deadSpaceNearSelf": -1,     # Implies sealing yourself off (generally bad)

        # Pieces & Material
        "lockedArea": -1,            # High trapped material is bad
        "criticalPiecesCount": -1,   # Many pieces near death is bad
        "bottleneckScore": 1,        # High min-placements is good
        "remainingArea": -1,         # Less remaining area is good
        "largestRemainingPiece": -1,
        "unloadPotential": 1,

        # Old fallbacks
        "deadSpace": 1,
        "pieceLockRisk": -1,
    }

    # Coefficients for Win Proxy Score
    # These base weights are balanced roughly around identical normalization scales.
    WIN_PROXY_WEIGHTS: Dict[str, float] = {
        "remainingAreaAdv": 2.0,       # Heaviest weight: reducing hand size relative to opps
        "effectiveFrontierAdv": 1.5,   # Core expansion
        "mobilityNextP10Adv": 1.5,     # Robust flexibility
        "lockedAreaAdv": 1.0,          # Penalize trapped chunks
        "centerControlWeightedAdv": 0.5, # Small positional bump
    }
