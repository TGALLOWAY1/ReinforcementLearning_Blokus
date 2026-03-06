import os
import pprint
import sys

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.board import Player
from engine.game import BlokusGame


def main():
    game = BlokusGame(enable_telemetry=True, telemetry_fast_mode=True)

    print("Getting legal moves for RED (turn 1)...")
    legal_moves = game.get_legal_moves(Player.RED)
    first_move = legal_moves[10] # Pick some arbitrary move

    print(f"Making move: {first_move}...")
    game.make_move(first_move, Player.RED)

    print("\nTelemetry for Turn 1:")
    history_entry = game.game_history[-1]
    telemetry = history_entry.get('telemetry')
    pprint.pprint(telemetry)

    if telemetry:
        print("\nDelta Self for RED:")
        pprint.pprint(telemetry['deltaSelf'])

if __name__ == "__main__":
    main()
