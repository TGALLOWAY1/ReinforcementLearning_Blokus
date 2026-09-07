"""Pentobi (GTP) adapter: an external Blokus engine as anchor opponent and
rules cross-check. Pure mapping helpers are tested without the binary; the
integration tests are skipped when pentobi-gtp is not installed."""

import os

import pytest

from agents import pentobi_agent as pa
from agents.random_agent import RandomAgent
from engine.board import Board, Player
from engine.move_generator import LegalMoveGenerator

GEN = LegalMoveGenerator()
HAVE_BINARY = pa.find_binary() is not None
needs_binary = pytest.mark.skipif(not HAVE_BINARY, reason="pentobi-gtp not installed")


def test_point_conversion_round_trips_every_cell():
    for r in range(20):
        for c in range(20):
            assert pa.gtp_point_to_repo(pa.repo_point_to_gtp(r, c)) == (r, c)
    assert pa.repo_point_to_gtp(0, 0) == "a20"
    assert pa.repo_point_to_gtp(19, 19) == "t1"


def test_colour_mapping_matches_start_corners():
    assert pa.player_to_colour(Player.RED) == "1"      # Blue in Pentobi, a20 top-left
    assert pa.player_to_colour(Player.BLUE) == "2"     # Yellow, t20
    assert pa.player_to_colour(Player.YELLOW) == "3"   # Red, t1
    assert pa.player_to_colour(Player.GREEN) == "4"    # Green, a1
    board = Board()
    for player in Player:
        corner = board.player_start_corners[player]
        assert pa.repo_point_to_gtp(corner.row, corner.col) == pa.START_CORNERS[pa.player_to_colour(player)]


def test_move_string_round_trip_and_matching():
    cells = frozenset({(2, 2), (1, 1), (1, 2), (0, 0), (0, 1)})
    text = pa.cells_to_gtp(cells)
    assert pa.gtp_move_to_cells(text) == cells
    assert pa.gtp_move_to_cells("pass") == frozenset()
    board = Board()
    moves = GEN.get_legal_moves(board, Player.RED)
    target = moves[7]
    target_cells = pa.move_cells(target)
    found = pa.match_move(moves, target_cells)
    assert found is target
    assert pa.match_move(moves, frozenset({(5, 5)})) is None


def test_placed_cells_are_grouped_into_pieces():
    board = Board()
    mv = GEN.get_legal_moves(board, Player.RED)[0]
    board.place_piece(mv.get_positions(GEN.piece_orientations_cache[mv.piece_id]), Player.RED, mv.piece_id)
    pieces = pa.pieces_on_grid(board.grid, Player.RED)
    assert pieces == [pa.move_cells(mv)]


@needs_binary
def test_pentobi_plays_a_full_game_through_the_arena_adapter():
    from analytics.tournament.arena_runner import AgentConfig, build_agent

    adapter = build_agent(AgentConfig.from_dict({"name": "pentobi_l1", "type": "pentobi",
                                                 "params": {"level": 1}}), seed=11)
    board = Board()
    others = {p: RandomAgent(seed=p.value) for p in Player if p != Player.RED}
    order = list(Player)
    idle = 0
    ply = 0
    pentobi_moves = 0
    while idle < 4 and ply < 400:
        player = order[ply % 4]
        moves = GEN.get_legal_moves(board, player)
        if not moves:
            idle += 1
        else:
            idle = 0
            if player == Player.RED:
                mv, _stats = adapter.choose_move(board, player, moves, None)
                pentobi_moves += 1
            else:
                mv = others[player].select_action(board, player, moves)
            assert mv is not None and mv in moves
            board.place_piece(mv.get_positions(GEN.piece_orientations_cache[mv.piece_id]), player, mv.piece_id)
        ply += 1
    info = adapter.agent.get_action_info()
    assert pentobi_moves >= 10
    assert info["stats"]["mismatches"] == 0
    adapter.agent.close()


@needs_binary
def test_engine_legal_moves_match_pentobi_all_legal():
    """The outstanding M0 check: engine legal-move sets == Pentobi's, on
    every ply of seeded games (>= 200 positions by default)."""
    n_games = int(os.environ.get("BLOKUS_PENTOBI_GAMES", "3"))
    positions = disagreements = zero = 0
    examples = []
    for seed in range(1, n_games + 1):
        board = Board()
        agents = {p: RandomAgent(seed=seed * 10 + p.value) for p in Player}
        with pa.PentobiGtp(level=1, seed=seed) as eng:
            eng.set_game_classic()
            order = list(Player)
            idle = ply = 0
            while idle < 4 and ply < 400:
                player = order[ply % 4]
                moves = GEN.get_legal_moves(board, player)
                engine_set = {pa.move_cells(m) for m in moves}
                pentobi_set = eng.all_legal_cells(pa.player_to_colour(player))
                positions += 1
                zero += not engine_set
                if engine_set != pentobi_set:
                    disagreements += 1
                    if len(examples) < 3:
                        examples.append((seed, ply, player.name,
                                         sorted(map(sorted, engine_set - pentobi_set))[:2],
                                         sorted(map(sorted, pentobi_set - engine_set))[:2]))
                if moves:
                    idle = 0
                    mv = agents[player].select_action(board, player, moves)
                    board.place_piece(mv.get_positions(GEN.piece_orientations_cache[mv.piece_id]), player, mv.piece_id)
                    eng.play(pa.player_to_colour(player), pa.cells_to_gtp(pa.move_cells(mv)))
                else:
                    idle += 1
                ply += 1
    assert positions >= 200, positions
    assert zero >= 10, zero
    assert disagreements == 0, examples
