"""
Versioned board/move serialization round-trip tests
(agent-strength rescue, Phase 2 task 3).

The correctness bar is the Board.copy() field set — the Zobrist hash omits
player_last_piece/player_first_move/move_count/game_over, so hash equality is
checked as a supplement, never as the sole criterion.
"""

import json
import random

import numpy as np
import pytest

from engine.board import STATE_SCHEMA_VERSION, Board, Player, Position
from engine.move_generator import ACTION_SCHEMA_VERSION, LegalMoveGenerator, Move
from mcts.zobrist import ZobristHash
from tests.utils_game_states import generate_random_valid_state

SEEDS = [20260620, 20260621, 20260622]


def _assert_boards_equal(a: Board, restored: Board):
    """Assert `restored` (a from_dict board) faithfully reproduces `a`.

    Frontiers are compared in CANONICAL form: a live board's incremental
    frontier may hold stale entries for non-placing players (only the mover's
    set is maintained by update_frontier_after_move), while from_dict rebuilds
    the clean recomputed set. Behavioral equivalence of the two forms is
    asserted separately via legal-move-set equality.
    """
    assert np.array_equal(a.grid, restored.grid)
    for p in Player:
        assert a.player_pieces_used[p] == restored.player_pieces_used[p]
        assert a.player_last_piece[p] == restored.player_last_piece[p]
        assert a.player_first_move[p] == restored.player_first_move[p]
        assert a.player_bits[p] == restored.player_bits[p]
        if a.player_first_move[p]:
            assert restored.player_frontiers[p] == a.player_frontiers[p]
        else:
            # Canonical frontier; also a subset of the (possibly stale) live one
            # restricted to still-empty cells.
            assert restored.player_frontiers[p] == a._compute_full_frontier(p)
    assert a.current_player == restored.current_player
    assert a.move_count == restored.move_count
    assert a.game_over == restored.game_over
    assert a.occupied_bits == restored.occupied_bits


def _round_trip(board: Board) -> Board:
    # Force an actual JSON round-trip so numpy scalars / non-string keys /
    # sets would be caught, not just dict identity.
    payload = json.loads(json.dumps(board.to_dict()))
    return Board.from_dict(payload)


class TestBoardRoundTrip:
    def test_fresh_board(self):
        _assert_boards_equal(Board(), _round_trip(Board()))

    def test_midgame_boards(self):
        for seed in SEEDS:
            board, _ = generate_random_valid_state(num_moves=16, seed=seed)
            restored = _round_trip(board)
            _assert_boards_equal(board, restored)
            restored.assert_bitboard_consistent()

    def test_terminal_board(self):
        # Play a full game (production path), then round-trip the terminal state.
        rng = random.Random(SEEDS[0])
        board = Board()
        generator = LegalMoveGenerator()
        passes = 0
        while passes < len(Player):
            player = board.current_player
            moves = generator.get_legal_moves(board, player)
            if not moves:
                board._update_current_player()
                passes += 1
                continue
            passes = 0
            move = rng.choice(moves)
            orientations = generator.piece_orientations_cache[move.piece_id]
            board.place_piece(move.get_positions(orientations), player, move.piece_id)

        restored = _round_trip(board)
        _assert_boards_equal(board, restored)
        for p in Player:
            assert restored.get_score(p) == board.get_score(p)

    def test_zobrist_hash_matches_after_round_trip(self):
        hasher = ZobristHash(seed=42)
        for seed in SEEDS:
            board, _ = generate_random_valid_state(num_moves=12, seed=seed)
            assert hasher.hash_board(board) == hasher.hash_board(_round_trip(board))

    def test_behavioral_equivalence_after_round_trip(self):
        generator = LegalMoveGenerator()
        for seed in SEEDS:
            board, _ = generate_random_valid_state(num_moves=14, seed=seed)
            restored = _round_trip(board)
            for p in Player:
                original_moves = {
                    (m.piece_id, m.orientation, m.anchor_row, m.anchor_col)
                    for m in generator.get_legal_moves(board, p)
                }
                restored_moves = {
                    (m.piece_id, m.orientation, m.anchor_row, m.anchor_col)
                    for m in generator.get_legal_moves(restored, p)
                }
                assert original_moves == restored_moves
                assert board.get_score(p) == restored.get_score(p)

    def test_restored_board_is_independent(self):
        board, _ = generate_random_valid_state(num_moves=10, seed=SEEDS[0])
        restored = _round_trip(board)
        generator = LegalMoveGenerator()
        player = restored.current_player
        moves = generator.get_legal_moves(restored, player)
        assert moves
        move = moves[0]
        orientations = generator.piece_orientations_cache[move.piece_id]
        snapshot_count = board.move_count
        assert restored.place_piece(move.get_positions(orientations), player, move.piece_id)
        assert board.move_count == snapshot_count  # original untouched

    def test_schema_version_stamped_and_enforced(self):
        payload = Board().to_dict()
        assert payload["schema_version"] == STATE_SCHEMA_VERSION
        payload["schema_version"] = "board_state_v0"
        with pytest.raises(ValueError, match="schema"):
            Board.from_dict(payload)

    def test_malformed_grid_rejected(self):
        payload = Board().to_dict()
        payload["grid"] = [[0] * 5] * 5
        with pytest.raises(ValueError, match="grid"):
            Board.from_dict(payload)


class TestMoveRoundTrip:
    def test_move_dict_round_trip(self):
        move = Move(piece_id=17, orientation=3, anchor_row=4, anchor_col=9)
        payload = json.loads(json.dumps(move.to_dict()))
        assert payload == {
            "piece_id": 17,
            "orientation": 3,
            "anchor_row": 4,
            "anchor_col": 9,
        }
        restored = Move.from_dict(payload)
        assert (restored.piece_id, restored.orientation,
                restored.anchor_row, restored.anchor_col) == (17, 3, 4, 9)

    def test_action_schema_version_exists(self):
        assert ACTION_SCHEMA_VERSION == "move_v2"


def test_from_dict_rejects_pre_standard_piece_set_payloads():
    """Payloads stamped with the pre-2026-09 schema (piece id 10 = S/Z tetromino)
    must not deserialize under the standard catalogue."""
    board = Board()
    payload = board.to_dict()
    assert payload["schema_version"] != "board_state_v1"
    payload["schema_version"] = "board_state_v1"
    with pytest.raises(ValueError):
        Board.from_dict(payload)
