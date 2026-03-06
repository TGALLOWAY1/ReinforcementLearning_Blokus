import random

from engine.game import BlokusGame


def test_telemetry_determinism_and_sanity():
    """Test that playing the same sequence of moves yields the identical telemetry deterministically."""

    def play_random_game(seed: int, fast_mode: bool = True):
        random.seed(seed)
        game = BlokusGame(enable_telemetry=True, telemetry_fast_mode=fast_mode)

        moves_made = 0
        while not game.is_game_over() and moves_made < 10:
            player = game.get_current_player()
            legal_moves = game.get_legal_moves(player)
            if not legal_moves:
                game._check_game_over()
                break

            move = random.choice(legal_moves)
            game.make_move(move, player)
            moves_made += 1

        return game.game_history

    history1 = play_random_game(42, fast_mode=True)
    history2 = play_random_game(42, fast_mode=True)

    assert len(history1) == len(history2)
    assert len(history1) > 0

    for h1, h2 in zip(history1, history2):
        assert 'telemetry' in h1
        assert h1['telemetry'] == h2['telemetry']

        # Sanity check values
        t = h1['telemetry']
        assert 'deltaSelf' in t
        assert 'before' in t
        assert 'after' in t

        # Metrics before/after should be non-negative
        for b_item in t['before']:
            for k, v in b_item['metrics'].items():
                assert v >= 0

        # Deltas can be anything, but let's just make sure they exist
        assert 'frontierSize' in t['deltaSelf']
        assert 'mobility' in t['deltaSelf']
        assert 'deadSpace' in t['deltaSelf']

def test_telemetry_exact_mobility():
    """Test exact mobility computation without fast mode."""
    game = BlokusGame(enable_telemetry=True, telemetry_fast_mode=False)
    player = game.get_current_player()
    legal_moves = game.get_legal_moves(player)

    # Just do 1 move to verify it doesn't crash and computes correctly
    game.make_move(legal_moves[0], player)

    assert len(game.game_history) == 1
    t = game.game_history[0]['telemetry']

    for b_item in t['before']:
        if b_item['playerId'] == player.name:
            assert b_item['metrics']['mobility'] > 0

    # Also verify deadSpace is populated
    assert 'deadSpace' in t['deltaSelf']
