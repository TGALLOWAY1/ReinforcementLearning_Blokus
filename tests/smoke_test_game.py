"""
Smoke test for Blokus game engine.
Verifies basic game functionality.
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Check for required dependencies
try:
    import numpy  # noqa: F401
except ImportError:
    print("ERROR: numpy is not installed in this Python environment.")
    print(f"Python path: {sys.executable}")
    print("\nPlease install dependencies:")
    print("  pip install -r requirements.txt")
    print("\nOr use the correct Python environment (e.g., conda base environment)")
    sys.exit(1)

from engine.game import BlokusGame


def test_engine():
    game = BlokusGame()
    print(f"Initial Player: {game.board.current_player}") # Should be RED
    
    # Get valid moves
    moves = game.get_legal_moves()
    print(f"Legal moves for RED: {len(moves)}")
    
    if not moves:
        print("CRITICAL: No legal moves generated for start of game!")
        return

    # Execute a move
    move = moves[0]
    print(f"Attempting move: {move}")
    success = game.make_move(move)
    print(f"Move success: {success}")
    
    # Verify turn change
    print(f"Next Player: {game.board.current_player}") # Should be BLUE

if __name__ == "__main__":
    test_engine()