"""Translate piece orientations between the web frontend and the engine.

The frontend enumerates orientations as ``0-7``: index ``k % 4`` rotations of
the base shape by 90 degrees, with ``k >= 4`` meaning the shape is mirrored
(``np.fliplr``) first. The engine numbers orientations by its own canonical
enumeration (:data:`engine.pieces.ALL_PIECE_ORIENTATIONS`). The Pyodide bridge
(``browser_python/worker_bridge.py``) builds this translation at start-up; the
natively served backend uses this module so both paths agree cell for cell.

Unknown piece ids or indices map to orientation ``0``, matching the bridge.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from engine.pieces import ALL_PIECE_ORIENTATIONS, PieceGenerator, normalize_offsets, shape_to_offsets

FRONTEND_ORIENTATIONS = 8

_FRONTEND_TO_ENGINE: Dict[int, Dict[int, int]] = {}
_ENGINE_TO_FRONTEND: Dict[int, Dict[int, int]] = {}


def _build() -> None:
    for piece in PieceGenerator.get_all_pieces():
        forward: Dict[int, int] = {}
        backward: Dict[int, int] = {}
        engine_by_offsets = {tuple(o.offsets): o.orientation_id for o in ALL_PIECE_ORIENTATIONS.get(piece.id, [])}
        for frontend_idx in range(FRONTEND_ORIENTATIONS):
            shape = piece.shape.copy()
            if frontend_idx >= 4:
                shape = np.fliplr(shape)
            for _ in range(frontend_idx % 4):
                shape = np.rot90(shape)
            key = tuple(normalize_offsets(shape_to_offsets(shape)))
            engine_idx = engine_by_offsets.get(key, 0)
            forward[frontend_idx] = engine_idx
            backward.setdefault(engine_idx, frontend_idx)
        _FRONTEND_TO_ENGINE[piece.id] = forward
        _ENGINE_TO_FRONTEND[piece.id] = backward


_build()


def frontend_to_engine(piece_id: int, frontend_orientation: int) -> int:
    """Engine orientation id for a frontend orientation index (0 when unknown)."""
    return _FRONTEND_TO_ENGINE.get(int(piece_id), {}).get(int(frontend_orientation), 0)


def engine_to_frontend(piece_id: int, engine_orientation: int) -> int:
    """Frontend orientation index for an engine orientation id (0 when unknown)."""
    return _ENGINE_TO_FRONTEND.get(int(piece_id), {}).get(int(engine_orientation), 0)
