"""Pin the engine's piece catalogue to the standard Blokus set.

Blokus pieces are free polyominoes: any rotation or reflection of a piece is
the same piece. The standard set is 1 monomino, 1 domino, 2 trominoes,
5 tetrominoes and 12 pentominoes (21 pieces, 89 squares, 91 fixed
orientations). These checks deliberately avoid the engine's own orientation
machinery so a catalogue error cannot hide behind it.
"""

from collections import Counter

import numpy as np

from engine.pieces import ALL_PIECE_ORIENTATIONS, PieceGenerator


def canonical_form(shape) -> tuple:
    """Smallest (row-tuple) representation over all 8 rotations/reflections."""
    a = np.asarray(shape, dtype=int)
    forms = []
    for k in range(4):
        r = np.rot90(a, k)
        for f in (r, np.fliplr(r)):
            rows = np.any(f, axis=1)
            cols = np.any(f, axis=0)
            t = f[np.ix_(rows, cols)]
            forms.append(tuple(map(tuple, t.tolist())))
    return min(forms)


Z_PENTOMINO = [[1, 1, 0], [0, 1, 0], [0, 1, 1]]
S_TETROMINO = [[0, 1, 1], [1, 1, 0]]


def _pieces():
    return PieceGenerator.get_all_pieces()


def test_twenty_one_pairwise_distinct_free_polyominoes():
    pieces = _pieces()
    assert len(pieces) == 21
    seen = {}
    for p in pieces:
        key = canonical_form(p.shape)
        assert key not in seen, (
            f"piece {p.id} ({p.name}) is the same free polyomino as piece {seen[key]}"
        )
        seen[key] = p.id


def test_total_area_is_89_squares():
    assert sum(int(np.sum(p.shape)) for p in _pieces()) == 89


def test_size_histogram_matches_standard_set():
    assert Counter(int(p.size) for p in _pieces()) == {1: 1, 2: 1, 3: 2, 4: 5, 5: 12}


def test_z_pentomino_present_and_s_tetromino_present_once():
    forms = [canonical_form(p.shape) for p in _pieces()]
    assert canonical_form(Z_PENTOMINO) in forms, "Z-pentomino missing"
    assert forms.count(canonical_form(S_TETROMINO)) == 1, "S/Z tetromino must appear once"


def test_ninety_one_fixed_orientations_in_total():
    assert sum(len(v) for v in ALL_PIECE_ORIENTATIONS.values()) == 91


def test_ids_are_1_to_21_and_monomino_is_id_1():
    pieces = _pieces()
    assert sorted(p.id for p in pieces) == list(range(1, 22))
    assert next(p for p in pieces if p.id == 1).size == 1


def test_piece_penalty_is_derived_from_piece_size_not_id_range():
    """engine.advanced_metrics.compute_piece_penalty charges unused pieces by
    size tier; the Z-pentomino (id 10) must be charged as a pentomino."""
    from engine.advanced_metrics import compute_piece_penalty

    all_ids = set(range(1, 22))
    only_z5_unused = compute_piece_penalty(all_ids - {10})
    only_f5_unused = compute_piece_penalty(all_ids - {11})   # an ordinary pentomino
    only_s4_unused = compute_piece_penalty(all_ids - {9})    # a tetromino
    assert only_z5_unused == only_f5_unused
    assert only_z5_unused > only_s4_unused > 0
