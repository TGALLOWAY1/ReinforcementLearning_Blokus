"""
Smoke test to verify the core Blokus game engine.

This script performs basic verification of the game engine:
1. Initializes BlokusGame
2. Checks initial current player (should be RED)
3. Gets legal moves for RED
4. Attempts to make the first legal move
5. Verifies invalid moves are rejected
"""

from engine.board import Player
from engine.game import BlokusGame
from engine.move_generator import Move


def main():
    """Run the smoke test."""
    print("=" * 60)
    print("Blokus Engine Smoke Test")
    print("=" * 60)
    
    # Step 1: Initialize BlokusGame
    print("\n1. Initializing BlokusGame...")
    game = BlokusGame()
    print("   ✓ Game initialized successfully")
    
    # Step 2: Print the initial current player (should be RED)
    print("\n2. Checking initial current player...")
    current_player = game.get_current_player()
    print(f"   Current player: {current_player}")
    print(f"   Expected: {Player.RED}")
    if current_player == Player.RED:
        print("   ✓ Initial player is RED (correct)")
    else:
        print(f"   ✗ ERROR: Expected RED, got {current_player}")
        return False
    
    # Step 3: Get all legal moves for RED using game.get_legal_moves()
    print("\n3. Getting legal moves for RED...")
    legal_moves = game.get_legal_moves(Player.RED)
    print(f"   ✓ Retrieved legal moves")
    
    # Step 4: Print the number of legal moves
    print(f"\n4. Number of legal moves for RED: {len(legal_moves)}")
    if len(legal_moves) > 0:
        print("   ✓ Legal moves found")
    else:
        print("   ✗ ERROR: No legal moves found for RED")
        return False
    
    # Step 5: Attempt to make the first legal move found
    print("\n5. Attempting to make the first legal move...")
    if len(legal_moves) > 0:
        first_move = legal_moves[0]
        print(f"   Move: piece_id={first_move.piece_id}, "
              f"orientation={first_move.orientation}, "
              f"anchor=({first_move.anchor_row}, {first_move.anchor_col})")
        
        # Step 6: Print the result of the move (True/False) and the new current player
        move_result = game.make_move(first_move, Player.RED)
        print(f"   Move result: {move_result}")
        
        if move_result:
            print("   ✓ Move was successful")
            new_current_player = game.get_current_player()
            print(f"   New current player: {new_current_player}")
            if new_current_player != Player.RED:
                print(f"   ✓ Current player changed (expected)")
            else:
                print(f"   ✗ WARNING: Current player did not change")
        else:
            print("   ✗ ERROR: Move failed unexpectedly")
            return False
    else:
        print("   ✗ ERROR: No legal moves available to test")
        return False
    
    # Step 7: Attempt to place a piece in an invalid location (e.g., center of board)
    print("\n7. Attempting invalid move (center of board)...")
    # Create an invalid move - placing piece 1 at center (10, 10) with orientation 0
    # This should fail because:
    # - For RED's first move, it must cover corner (0, 0)
    # - Center placement is not connected to existing pieces
    invalid_move = Move(piece_id=1, orientation=0, anchor_row=10, anchor_col=10)
    print(f"   Invalid move: piece_id={invalid_move.piece_id}, "
          f"orientation={invalid_move.orientation}, "
          f"anchor=({invalid_move.anchor_row}, {invalid_move.anchor_col})")
    
    # Reset game to initial state for clean test
    game.reset_game()
    invalid_result = game.make_move(invalid_move, Player.RED)
    print(f"   Invalid move result: {invalid_result}")
    
    if not invalid_result:
        print("   ✓ Invalid move correctly rejected")
    else:
        print("   ✗ ERROR: Invalid move was accepted (should be rejected)")
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("Smoke Test Summary")
    print("=" * 60)
    print("✓ All tests passed!")
    print("\nThe core game engine is working correctly.")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

