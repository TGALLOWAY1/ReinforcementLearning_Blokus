#!/usr/bin/env python3
"""
Verification Script for Metrics V2
Checks that metrics follow analytical invariants (e.g. non-negativity, probability scores in [0,1])
in simulated games.
"""

import logging
import random

from engine.game import BlokusGame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_invariants():
    """Run a randomized game and check metric invariants at every step."""
    logger.info("Starting Metrics V2 Invariant Verification...")

    # We turn on full simulation and telemetry
    game = BlokusGame(enable_telemetry=True, telemetry_fast_mode=False)

    # Force seed
    random.seed(42)

    err_count = 0
    checks = 0

    while True:
        if game.is_game_over():
            break

        current_player = game.get_current_player()
        legal_moves = game.get_legal_moves(current_player)
        if not legal_moves:
            # Pass turn manually for basic simulation
            game.board._update_current_player()
            game._check_game_over()
            continue

        move = random.choice(legal_moves)
        game.make_move(move, current_player)

        # Check telemetry History
        if not game.game_history:
            continue

        latest = game.game_history[-1]
        telemetry = latest.get("telemetry")
        if not telemetry:
            continue

        before = telemetry["before"]

        # Check invariants
        for player_metrics in before:
            pid = player_metrics["playerId"]
            m = player_metrics["metrics"]
            checks += 1

            # Non-negativity
            for k, v in m.items():
                if v < 0.0:
                    logger.error(f"Invariant failure: {k} is negative ({v}) for {pid} at ply {telemetry['ply']}")
                    err_count += 1
                    break

            # Entropy bounds
            entropy = m.get("mobilityEntropy", 0.0)
            if not (0.0 <= entropy <= 1.05): # Slight leeway for float
                logger.error(f"Invariant failure: Entropy {entropy} out of bounds")
                err_count += 1

            # Piece Share Bounds
            p_share = m.get("pieceTop1Share", 0.0)
            if not (0.0 <= p_share <= 1.05):
                logger.error(f"Invariant failure: pieceTop1Share {p_share} out of bounds")
                err_count += 1

            # Anchor Share Bounds
            a_share = m.get("anchorTop1Share", 0.0)
            if not (0.0 <= a_share <= 1.05):
                logger.error(f"Invariant failure: anchorTop1Share {a_share} out of bounds")
                err_count += 1

    logger.info(f"Verification complete. Performed {checks} metric checks.")
    if err_count == 0:
        logger.info("✅ All invariant checks passed!")
    else:
        logger.error(f"❌ Failed {err_count} invariant checks.")

if __name__ == "__main__":
    verify_invariants()
