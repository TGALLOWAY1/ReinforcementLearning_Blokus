"""The browser UI ships its own copy of the piece catalogue
(frontend/src/constants/gameConstants.ts). The Pyodide bridge maps frontend
orientations onto engine orientations by assuming the base shapes are
identical, so the two catalogues must match exactly, id by id.
"""

import json
import re
from pathlib import Path

import numpy as np

from engine.pieces import PieceGenerator

TS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "constants" / "gameConstants.ts"


def _ts_block(name: str) -> str:
    src = TS.read_text()
    m = re.search(rf"export const {name}[^=]*=\s*\{{(.*?)\n\}};", src, re.S)
    assert m, f"{name} not found in {TS}"
    return m.group(1)


def frontend_shapes():
    body = re.sub(r"//[^\n]*", "", _ts_block("PIECE_SHAPES"))
    return {int(k): json.loads(v) for k, v in re.findall(r"(\d+):\s*(\[\[.*?\]\])", body, re.S)}


def frontend_names():
    body = _ts_block("PIECE_NAMES")
    return {int(k): v for k, v in re.findall(r"(\d+):\s*'([^']*)'", body)}


def test_frontend_shapes_match_engine_exactly():
    shapes = frontend_shapes()
    engine = {p.id: p.shape.tolist() for p in PieceGenerator.get_all_pieces()}
    assert set(shapes) == set(engine) == set(range(1, 22))
    for pid in engine:
        assert shapes[pid] == engine[pid], f"piece {pid}: frontend {shapes[pid]} vs engine {engine[pid]}"


def test_frontend_names_match_engine():
    names = frontend_names()
    engine = {p.id: p.name for p in PieceGenerator.get_all_pieces()}
    assert names == engine
