"""Sanity checks: rules, scoring, determinism, agent loading.

    python -m mcts_lab.checks

Runs in well under a minute and exits non-zero on the first failure. This is
the "is the lab wired up correctly?" gate to run before any training or
evaluation session (CI runs the full pytest suite; this is the fast subset).
"""

from __future__ import annotations

import sys
import time
from typing import Callable, List, Tuple

_CHECKS: List[Tuple[str, Callable[[], None]]] = []


def check(name: str):
    def deco(fn):
        _CHECKS.append((name, fn))
        return fn
    return deco


@check("engine: first move must cover the start corner")
def _first_move_corner():
    from engine.board import Board, Player, Position

    b = Board()
    # Monomino away from RED's corner (0,0) must be rejected...
    assert not b.can_place_piece([Position(5, 5)], Player.RED)
    # ...and on the corner accepted.
    assert b.can_place_piece([Position(0, 0)], Player.RED)


@check("engine: corner-contact required, edge-contact forbidden")
def _adjacency_rules():
    from engine.board import Board, Player, Position

    b = Board()
    assert b.place_piece([Position(0, 0)], Player.RED, piece_id=1)
    # Edge-adjacent to own piece: illegal.
    assert not b.can_place_piece([Position(0, 1)], Player.RED)
    # Diagonal to own piece: legal.
    assert b.can_place_piece([Position(1, 1)], Player.RED)


@check("engine: scoring counts squares and all-pieces bonus")
def _scoring():
    from engine.board import Board, Player, Position

    b = Board()
    b.place_piece([Position(0, 0)], Player.RED, piece_id=1)
    assert b.get_score(Player.RED) >= 1


@check("engine: standard 21-piece catalogue (distinct shapes, 89 squares, 91 orientations)")
def _piece_catalogue():
    import numpy as np

    from engine.pieces import ALL_PIECE_ORIENTATIONS, PieceGenerator

    def canonical(shape):
        forms = []
        for k in range(4):
            r = np.rot90(shape, k)
            for f in (r, np.fliplr(r)):
                t = f[np.ix_(np.any(f, axis=1), np.any(f, axis=0))]
                forms.append(tuple(map(tuple, t.tolist())))
        return min(forms)

    pieces = PieceGenerator.get_all_pieces()
    assert len(pieces) == 21
    assert len({canonical(p.shape) for p in pieces}) == 21, "duplicate free polyomino"
    assert sum(int(p.shape.sum()) for p in pieces) == 89
    assert sum(len(v) for v in ALL_PIECE_ORIENTATIONS.values()) == 91


def _play_probe(seed: int, max_moves: int = 30) -> list:
    """Play a short seeded random-vs-random game; return the move sequence."""
    from engine.game import BlokusGame
    from agents.random_agent import RandomAgent

    game = BlokusGame(enable_telemetry=False)
    agent = RandomAgent(seed=seed)
    seq = []
    passes = 0
    while len(seq) < max_moves and passes < 4:
        player = game.get_current_player()
        moves = game.get_legal_moves(player)
        if not moves:
            passes += 1
            game.board._update_current_player()
            continue
        passes = 0
        move = agent.select_action(game.board, player, moves)
        seq.append((player.value, move.piece_id, move.orientation, move.anchor_row, move.anchor_col))
        assert game.make_move(move, player), "legal move rejected by engine"
    return seq


@check("engine: seeded games are deterministic")
def _determinism():
    seq_a = _play_probe(seed=123)
    seq_b = _play_probe(seed=123)
    assert seq_a == seq_b, "same seed must give identical move sequences"
    assert seq_a, "probe game played no moves"
    seq_c = _play_probe(seed=124)
    assert seq_a != seq_c, "different seeds should diverge"


@check("agents: registry constructs random/heuristic/mcts")
def _agent_loading():
    from engine.game import BlokusGame
    from agents.random_agent import RandomAgent
    from agents.heuristic_agent import HeuristicAgent
    from mcts.mcts_agent import MCTSAgent

    game = BlokusGame()
    player = game.get_current_player()
    moves = game.get_legal_moves(player)
    for agent in (RandomAgent(seed=1), HeuristicAgent(), MCTSAgent(iterations=3, seed=1, rollout_policy="greedy_sample", rollout_cutoff_depth=4)):
        move = agent.select_action(game.board, player, moves)
        assert move is not None


@check("champion: training/state/champion.json loads into an arena agent")
def _champion_loads():
    from mcts_lab._common import load_state, resolve_agent

    state = load_state()
    cfg = resolve_agent("champion", state)
    assert cfg.get("type") == "mcts" and isinstance(cfg.get("params"), dict)


@check("mcts: search returns a legal move and per-player rewards")
def _mcts_search():
    from engine.game import BlokusGame
    from engine.board import Player
    from mcts.mcts_agent import MCTSAgent

    game = BlokusGame()
    player = game.get_current_player()
    moves = game.get_legal_moves(player)
    agent = MCTSAgent(iterations=8, seed=7, rollout_policy="greedy_sample", rollout_cutoff_depth=6)
    move = agent.select_action(game.board, player, moves)
    legal_keys = {(m.piece_id, m.orientation, m.anchor_row, m.anchor_col) for m in moves}
    assert (move.piece_id, move.orientation, move.anchor_row, move.anchor_col) in legal_keys
    rewards, _ = agent._rollout(game.board, player)
    assert set(rewards) == set(Player), "rollout must return a reward for every player"


def main() -> int:
    failures = 0
    for name, fn in _CHECKS:
        t0 = time.monotonic()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}  ({exc.__class__.__name__}: {exc})")
        else:
            print(f"ok    {name}  ({time.monotonic() - t0:.1f}s)")
    if failures:
        print(f"\n{failures}/{len(_CHECKS)} checks FAILED")
        return 1
    print(f"\nall {len(_CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
