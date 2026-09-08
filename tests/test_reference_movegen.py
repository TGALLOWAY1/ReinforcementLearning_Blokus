"""Cross-check the engine's legal-move generator against an independent
rules-based reference.

The reference below is written from the Blokus rules alone: its own 21 free
polyominoes, its own orientation generation, and a frontier-driven
placement enumeration that shares no code with engine/move_generator.py or
engine/pieces.py. Engine piece ids are mapped to reference pieces by shape,
never by name, so a catalogue defect shows up as a failed mapping.

First-move status and each colour's remaining inventory are recovered from
the grid (same-colour pieces never touch orthogonally, so each connected
component is one placed piece) and cross-checked against the engine's
bookkeeping.

Set BLOKUS_REF_GAMES (default 7) to scale the sampled positions.
"""

import os
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np

from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from engine.board import Board, Player
from engine.move_generator import LegalMoveGenerator

Cell = Tuple[int, int]
Shape = FrozenSet[Cell]

# --- Reference piece set (standard Blokus, free polyominoes) -----------------
REF_PIECES: Dict[str, Shape] = {
    "I1": frozenset({(0, 0)}),
    "I2": frozenset({(0, 0), (0, 1)}),
    "I3": frozenset({(0, 0), (0, 1), (0, 2)}),
    "V3": frozenset({(0, 0), (1, 0), (1, 1)}),
    "I4": frozenset({(0, 0), (0, 1), (0, 2), (0, 3)}),
    "O4": frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
    "T4": frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}),
    "L4": frozenset({(0, 0), (1, 0), (2, 0), (2, 1)}),
    "S4": frozenset({(0, 1), (0, 2), (1, 0), (1, 1)}),
    "F5": frozenset({(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)}),
    "I5": frozenset({(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)}),
    "L5": frozenset({(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)}),
    "N5": frozenset({(0, 0), (1, 0), (1, 1), (2, 1), (3, 1)}),
    "P5": frozenset({(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)}),
    "T5": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    "U5": frozenset({(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)}),
    "V5": frozenset({(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}),
    "W5": frozenset({(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)}),
    "X5": frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}),
    "Y5": frozenset({(0, 0), (1, 0), (1, 1), (2, 0), (3, 0)}),
    "Z5": frozenset({(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)}),
}


def _normalize(cells) -> Shape:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def orientations(shape: Shape) -> List[Shape]:
    """All distinct rotations/reflections, normalized to the origin."""
    out: List[Shape] = []
    cur = set(shape)
    for _ in range(4):
        cur = {(c, -r) for r, c in cur}  # rotate 90 degrees
        for cand in (cur, {(r, -c) for r, c in cur}):  # and its reflection
            n = _normalize(cand)
            if n not in out:
                out.append(n)
    return out


def canonical(cells) -> Shape:
    return min(orientations(_normalize(cells)), key=lambda s: sorted(s))


REF_ORIENTS: Dict[str, List[Shape]] = {n: orientations(s) for n, s in REF_PIECES.items()}
REF_CANON: Dict[Shape, str] = {canonical(s): n for n, s in REF_PIECES.items()}

SIZE = 20
ORTH = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def ref_legal_placements(grid: List[List[int]], colour: int, inventory: List[str],
                         first_move: bool, start: Cell) -> Set[Tuple[str, Shape]]:
    """Every legal placement for `colour` as (piece name, absolute cells)."""
    own = {(r, c) for r in range(SIZE) for c in range(SIZE) if grid[r][c] == colour}
    if first_move:
        seeds = {start} if grid[start[0]][start[1]] == 0 else set()
    else:
        seeds = set()
        for r, c in own:
            for dr, dc in DIAG:
                rr, cc = r + dr, c + dc
                if 0 <= rr < SIZE and 0 <= cc < SIZE and grid[rr][cc] == 0:
                    seeds.add((rr, cc))
    out: Set[Tuple[str, Shape]] = set()
    for name in inventory:
        for orient in REF_ORIENTS[name]:
            for (sr, sc) in seeds:
                for (pr, pc) in orient:  # put piece cell (pr,pc) onto the seed
                    off_r, off_c = sr - pr, sc - pc
                    cells = frozenset((r + off_r, c + off_c) for r, c in orient)
                    if not all(0 <= r < SIZE and 0 <= c < SIZE and grid[r][c] == 0
                               for r, c in cells):
                        continue
                    if first_move:
                        if start not in cells:
                            continue
                    else:
                        if any((r + dr, c + dc) in own for r, c in cells for dr, dc in ORTH):
                            continue
                        if not any((r + dr, c + dc) in own for r, c in cells for dr, dc in DIAG):
                            continue
                    out.add((name, cells))
    return out


# --- Engine adapters ----------------------------------------------------------
GEN = LegalMoveGenerator()


def engine_id_to_ref_name() -> Dict[int, str]:
    mapping = {}
    for piece in GEN.all_pieces:
        cells = {(int(r), int(c)) for r, c in zip(*np.nonzero(piece.shape))}
        mapping[piece.id] = REF_CANON.get(canonical(cells))
    return mapping


def engine_placements(board: Board, player: Player, mapping) -> Set[Tuple[str, Shape]]:
    moves = GEN.get_legal_moves(board, player)
    out = set()
    for m in moves:
        pos = m.get_positions(GEN.piece_orientations_cache[m.piece_id])
        out.add((mapping[m.piece_id], frozenset((p.row, p.col) for p in pos)))
    assert len(out) == len(moves), "engine emitted the same placement more than once"
    return out


START_CORNERS = {1: (0, 0), 2: (0, SIZE - 1), 3: (SIZE - 1, SIZE - 1), 4: (SIZE - 1, 0)}


def _pieces_on_grid(grid: List[List[int]], colour: int) -> List[str]:
    """Pieces a colour has already played, recovered from the grid alone.

    Same-colour pieces never share an edge in Blokus, so each orthogonally
    connected component of a colour is exactly one placed piece.
    """
    cells = {(r, c) for r in range(SIZE) for c in range(SIZE) if grid[r][c] == colour}
    played = []
    while cells:
        stack = [cells.pop()]
        comp = set(stack)
        while stack:
            r, c = stack.pop()
            for dr, dc in ORTH:
                n = (r + dr, c + dc)
                if n in cells:
                    cells.remove(n)
                    comp.add(n)
                    stack.append(n)
        played.append(REF_CANON[canonical(comp)])
    return played


def reference_placements(board: Board, player: Player, mapping) -> Set[Tuple[str, Shape]]:
    grid = board.grid.tolist()
    played = _pieces_on_grid(grid, player.value)
    inventory = [name for name in REF_PIECES if name not in played]
    first_move = not played
    # The reference derives these from the grid; the engine's bookkeeping must agree.
    assert first_move == bool(board.player_first_move[player])
    assert sorted(played) == sorted(mapping[pid] for pid in board.player_pieces_used[player])
    start = board.player_start_corners[player]
    assert (start.row, start.col) == START_CORNERS[player.value]
    return ref_legal_placements(grid, player.value, inventory, first_move,
                                START_CORNERS[player.value])


def _play_game(agent, seed: int):
    """Yield (board, player) for every ply, then all four players at the end."""
    board = Board()
    order = list(Player)
    idle = 0
    ply = 0
    while idle < 4 and ply < 400:
        player = order[ply % 4]
        moves = GEN.get_legal_moves(board, player)
        yield board, player
        if moves:
            idle = 0
            mv = agent.select_action(board, player, moves)
            pos = mv.get_positions(GEN.piece_orientations_cache[mv.piece_id])
            assert board.place_piece(pos, player, mv.piece_id), "engine rejected its own move"
        else:
            idle += 1
        ply += 1
    for player in order:
        yield board, player


# --- Tests --------------------------------------------------------------------
def test_reference_set_is_sane():
    assert len(REF_CANON) == 21
    assert sum(len(s) for s in REF_PIECES.values()) == 89
    assert sum(len(o) for o in REF_ORIENTS.values()) == 91


def test_engine_pieces_map_one_to_one_onto_the_standard_set():
    mapping = engine_id_to_ref_name()
    unmapped = sorted(pid for pid, name in mapping.items() if name is None)
    assert not unmapped, f"engine pieces with no standard shape: {unmapped}"
    names = sorted(mapping.values())
    assert names == sorted(REF_PIECES), (
        f"duplicates: {sorted({n for n in names if names.count(n) > 1})}, "
        f"missing: {sorted(set(REF_PIECES) - set(names))}"
    )


def test_opening_placements_match_reference_in_every_corner():
    mapping = engine_id_to_ref_name()
    for player in Player:
        board = Board()
        eng = engine_placements(board, player, mapping)
        ref = reference_placements(board, player, mapping)
        assert eng == ref
        assert len(eng) == 58
        assert sum(1 for name, _ in eng if name == "Z5") == 2


def test_engine_and_reference_agree_on_sampled_positions():
    mapping = engine_id_to_ref_name()
    n_games = int(os.environ.get("BLOKUS_REF_GAMES", "7"))
    seeds = [20260906 + i for i in range(n_games)]
    positions = zero_move = 0
    for i, seed in enumerate(seeds):
        agent = HeuristicAgent(seed=seed) if i % 2 else RandomAgent(seed=seed)
        for board, player in _play_game(agent, seed):
            eng = engine_placements(board, player, mapping)
            ref = reference_placements(board, player, mapping)
            assert eng == ref, (
                f"seed {seed} player {player.name} ply {board.move_count}: "
                f"engine-only {sorted(eng - ref)[:3]} reference-only {sorted(ref - eng)[:3]}"
            )
            assert GEN.has_legal_moves(board, player) == bool(ref)
            positions += 1
            zero_move += not ref
    assert positions >= 300, positions
    assert zero_move >= 30, zero_move
