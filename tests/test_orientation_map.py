"""The web frontend indexes piece orientations 0-7 (rotation k, +4 = flipped);
the engine numbers orientations by its own canonical enumeration. The Pyodide
bridge translates between the two; the natively served backend must translate
identically so a move sent from the UI lands on the intended cells."""

import numpy as np

from engine import orientation_map as om
from engine.pieces import ALL_PIECE_ORIENTATIONS, PieceGenerator, normalize_offsets, shape_to_offsets

PIECES = PieceGenerator.get_all_pieces()


def _frontend_shape(base: np.ndarray, frontend_idx: int) -> np.ndarray:
    shape = base.copy()
    if frontend_idx >= 4:
        shape = np.fliplr(shape)
    for _ in range(frontend_idx % 4):
        shape = np.rot90(shape)
    return shape


def test_every_frontend_index_maps_to_a_valid_engine_orientation():
    for piece in PIECES:
        n_engine = len(ALL_PIECE_ORIENTATIONS[piece.id])
        for idx in range(8):
            engine_idx = om.frontend_to_engine(piece.id, idx)
            assert 0 <= engine_idx < n_engine, (piece.id, idx, engine_idx)


def test_engine_orientation_round_trips_through_frontend_index():
    for piece in PIECES:
        for orient in ALL_PIECE_ORIENTATIONS[piece.id]:
            f = om.engine_to_frontend(piece.id, orient.orientation_id)
            assert 0 <= f < 8
            assert om.frontend_to_engine(piece.id, f) == orient.orientation_id


def test_mapped_orientation_has_the_same_cells_as_the_frontend_shape():
    for piece in PIECES:
        offsets_by_id = {o.orientation_id: tuple(o.offsets) for o in ALL_PIECE_ORIENTATIONS[piece.id]}
        for idx in range(8):
            expected = tuple(normalize_offsets(shape_to_offsets(_frontend_shape(piece.shape, idx))))
            assert offsets_by_id[om.frontend_to_engine(piece.id, idx)] == expected


_BRIDGE_DUMP = """
import json, sys
from browser_python.worker_bridge import WebWorkerGameBridge
b = WebWorkerGameBridge()
json.dump({"f2b": {str(k): {str(i): v for i, v in m.items()} for k, m in b.frontend_to_backend.items()},
           "b2f": {str(k): {str(i): v for i, v in m.items()} for k, m in b.backend_to_frontend.items()}},
          sys.stdout)
"""


def test_matches_the_pyodide_bridge_translation():
    """Compared in a subprocess: importing the bridge appends browser_python/ to
    sys.path, which would make later spawn-based tests resolve the stale bundle
    copy of ``agents``/``engine`` instead of the repo packages."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run([sys.executable, "-c", _BRIDGE_DUMP], cwd=repo, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    bridge = json.loads(proc.stdout)
    for piece in PIECES:
        for idx in range(8):
            assert om.frontend_to_engine(piece.id, idx) == bridge["f2b"][str(piece.id)][str(idx)]
        for orient in ALL_PIECE_ORIENTATIONS[piece.id]:
            assert om.engine_to_frontend(piece.id, orient.orientation_id) == \
                bridge["b2f"][str(piece.id)][str(orient.orientation_id)]


def test_unknown_inputs_fall_back_to_zero_like_the_bridge():
    assert om.frontend_to_engine(99, 3) == 0
    assert om.engine_to_frontend(99, 3) == 0
    assert om.frontend_to_engine(1, 42) == 0
