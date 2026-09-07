"""Pentobi (GTP) adapter — an external Blokus engine as anchor opponent and
rules cross-check (decision D-020; assessment path (d)).

Pentobi (Markus Enzenberger, GPL-3, https://github.com/enz/pentobi) ships a
GTP engine, ``pentobi-gtp``, built here GUI-free (tag v30.3). Protocol facts
verified against that build:

* Classic 4-colour 20x20 is ``set_game Blokus``; colours ``1..4`` are
  Blue, Yellow, Red, Green in play order, starting at a20 (top-left), t20,
  t1, a1. The repo's RED(0,0), BLUE(0,19), YELLOW(19,19), GREEN(19,0) play in
  the same order from the same corners, so repo player i maps to colour i.
* Points are ``<col letter a..t><row 1..20>`` with row 20 at the top:
  repo ``(row, col)`` -> ``chr(ord('a') + col) + str(20 - row)``.
* Moves are comma-separated points (``play 1 a20,b20,b19``). ``genmove``
  answers ``pass`` when the colour has no move; ``all_legal <colour>`` lists
  every legal move. Illegal plays answer ``? illegal move``.
* Strength is the command-line ``--level 1..9`` (max simulations per move
  roughly 3, 30, 90, 181, 667, 5028, 69809, 349044, 1745221, scaled by move
  number). ``--seed`` makes a fresh process deterministic with one thread.

The adapter keeps Pentobi's board in sync by diffing the arena board against
a mirror before every move and replaying newly placed pieces (same-colour
pieces never touch orthogonally, so each new connected component is one
piece). A fresh process is started per agent instance, i.e. per game in the
arena, so games are reproducible from their seed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from engine.board import Board, Player
from engine.move_generator import LegalMoveGenerator, Move, get_shared_generator

PENTOBI_VERSION = "pentobi_v1"
SIZE = 20
Cell = Tuple[int, int]
Cells = FrozenSet[Cell]

COLOUR_NAMES = {"1": "Blue", "2": "Yellow", "3": "Red", "4": "Green"}
START_CORNERS = {"1": "a20", "2": "t20", "3": "t1", "4": "a1"}
_ORTH = ((1, 0), (-1, 0), (0, 1), (0, -1))


# --- pure helpers -------------------------------------------------------------
def find_binary(explicit: Optional[str] = None) -> Optional[str]:
    """Locate pentobi-gtp: explicit path, $PENTOBI_GTP, PATH, ~/.local/bin."""
    candidates = [explicit, os.environ.get("PENTOBI_GTP"), shutil.which("pentobi-gtp"),
                  os.path.expanduser("~/.local/bin/pentobi-gtp"),
                  os.path.expanduser("~/.local/opt/pentobi/bin/pentobi-gtp")]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def player_to_colour(player: Player) -> str:
    return str(int(player.value))


def repo_point_to_gtp(row: int, col: int) -> str:
    return f"{chr(ord('a') + int(col))}{SIZE - int(row)}"


def gtp_point_to_repo(point: str) -> Cell:
    point = point.strip().lower()
    return SIZE - int(point[1:]), ord(point[0]) - ord("a")


def cells_to_gtp(cells: Iterable[Cell]) -> str:
    return ",".join(repo_point_to_gtp(r, c) for r, c in sorted(cells))


def gtp_move_to_cells(text: str) -> Cells:
    text = text.strip()
    if text in ("", "pass", "resign"):
        return frozenset()
    return frozenset(gtp_point_to_repo(p) for p in text.split(","))


def move_cells(move: Move, generator: Optional[LegalMoveGenerator] = None) -> Cells:
    gen = generator or get_shared_generator()
    return frozenset((p.row, p.col) for p in
                     move.get_positions(gen.piece_orientations_cache[move.piece_id]))


def match_move(legal_moves: Sequence[Move], cells: Cells,
               generator: Optional[LegalMoveGenerator] = None) -> Optional[Move]:
    gen = generator or get_shared_generator()
    for m in legal_moves:
        if move_cells(m, gen) == cells:
            return m
    return None


def pieces_on_grid(grid: np.ndarray, player: Player) -> List[Cells]:
    """Connected (orthogonal) components of a colour == its placed pieces."""
    value = int(player.value)
    cells = {(int(r), int(c)) for r, c in zip(*np.nonzero(np.asarray(grid) == value))}
    pieces: List[Cells] = []
    while cells:
        stack = [cells.pop()]
        comp = set(stack)
        while stack:
            r, c = stack.pop()
            for dr, dc in _ORTH:
                n = (r + dr, c + dc)
                if n in cells:
                    cells.remove(n)
                    comp.add(n)
                    stack.append(n)
        pieces.append(frozenset(comp))
    return pieces


# --- GTP client ---------------------------------------------------------------
class GtpError(RuntimeError):
    pass


class PentobiGtp:
    """Minimal stdlib GTP client around a pentobi-gtp subprocess."""

    def __init__(self, level: int = 5, seed: Optional[int] = None, binary: Optional[str] = None,
                 use_book: bool = False, threads: int = 1):
        path = find_binary(binary)
        if path is None:
            raise FileNotFoundError("pentobi-gtp not found (set $PENTOBI_GTP or install to ~/.local/bin)")
        args = [path, "--game", "classic", "--level", str(int(level)), "--threads", str(int(threads)),
                "--noresign", "--quiet"]
        if seed is not None:
            args += ["--seed", str(int(seed) % (2**31 - 1))]
        if not use_book:
            args.append("--nobook")
        self.args = args
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        self._lock = threading.Lock()
        self._stderr: List[str] = []
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self._stderr.append(line.rstrip("\n"))

    def send_raw(self, command: str) -> Tuple[bool, str]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        with self._lock:
            self.proc.stdin.write(command.strip() + "\n")
            self.proc.stdin.flush()
            lines: List[str] = []
            while True:
                line = self.proc.stdout.readline()
                if line == "":
                    raise GtpError(f"pentobi-gtp closed stdout while handling {command!r}: "
                                   f"{self._stderr[-3:]}")
                line = line.rstrip("\n")
                if line == "":
                    if lines:
                        break
                    continue
                lines.append(line)
        status, first = lines[0][:1], lines[0][1:].lstrip(" ")
        lines[0] = first
        body = "\n".join(lines).strip("\n")
        if status == "=":
            return True, body
        if status == "?":
            return False, body
        raise GtpError(f"malformed GTP response to {command!r}: {lines!r}")

    def send(self, command: str) -> str:
        ok, body = self.send_raw(command)
        if not ok:
            raise GtpError(f"{command!r} -> {body}")
        return body

    def set_game_classic(self) -> None:
        self.send("set_game Blokus")

    def play(self, colour: str, points: str) -> None:
        self.send(f"play {colour} {points}")

    def genmove(self, colour: str) -> str:
        return self.send(f"genmove {colour}")

    def all_legal(self, colour: str) -> List[str]:
        return [m for m in self.send(f"all_legal {colour}").split("\n") if m]

    def all_legal_cells(self, colour: str) -> Set[Cells]:
        return {gtp_move_to_cells(m) for m in self.all_legal(colour)}

    def showboard(self) -> str:
        return self.send("showboard")

    def quit(self) -> None:
        try:
            if self.proc.poll() is None:
                self.send_raw("quit")
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass

    def __enter__(self) -> "PentobiGtp":
        return self

    def __exit__(self, *exc) -> None:
        self.quit()


# --- arena-facing agent -------------------------------------------------------
class PentobiAgent:
    """select_action(board, player, legal_moves) backed by pentobi-gtp."""

    def __init__(self, level: int = 5, seed: Optional[int] = None, binary: Optional[str] = None,
                 use_book: bool = False):
        self.level = int(level)
        self.seed = seed
        self.binary = binary
        self.use_book = bool(use_book)
        self._engine: Optional[PentobiGtp] = None
        self._mirror: Dict[Player, Set[Cells]] = {p: set() for p in Player}
        self._generator = get_shared_generator()
        self.stats: Dict[str, Any] = {"moves": 0, "mismatches": 0, "passes": 0, "resets": 0,
                                      "total_time_ms": 0.0, "level": self.level}

    # lifecycle
    def _ensure_engine(self) -> PentobiGtp:
        if self._engine is None:
            self._engine = PentobiGtp(level=self.level, seed=self.seed, binary=self.binary,
                                      use_book=self.use_book)
            self._engine.set_game_classic()
            self._mirror = {p: set() for p in Player}
        return self._engine

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    # board sync
    def _sync(self, board: Board) -> None:
        eng = self._ensure_engine()
        on_board = {player: set(pieces_on_grid(board.grid, player)) for player in Player}
        if any(self._mirror[p] - on_board[p] for p in Player):
            # A mirrored piece is no longer on the board: this is a different
            # (or restarted) game. Start a fresh engine and replay from scratch.
            self.stats["resets"] += 1
            self.close()
            eng = self._ensure_engine()
        pending: List[Tuple[Player, Cells]] = []
        for player in Player:
            for piece in on_board[player]:
                if piece not in self._mirror[player]:
                    pending.append((player, piece))
        # Replay new pieces; a piece can only be illegal-at-replay if it rests
        # on another new piece of the same colour, so retry until no progress.
        while pending:
            progressed = False
            remaining: List[Tuple[Player, Cells]] = []
            for player, piece in pending:
                ok, body = eng.send_raw(f"play {player_to_colour(player)} {cells_to_gtp(piece)}")
                if ok:
                    self._mirror[player].add(piece)
                    progressed = True
                else:
                    remaining.append((player, piece))
            if not progressed:
                raise GtpError(f"pentobi rejected board sync: {remaining[:2]} -> {body}")
            pending = remaining

    def select_action(self, board: Board, player: Player, legal_moves: List[Move]) -> Optional[Move]:
        if not legal_moves:
            return None
        start = time.perf_counter()
        self._sync(board)
        eng = self._ensure_engine()
        answer = eng.genmove(player_to_colour(player))
        cells = gtp_move_to_cells(answer)
        if not cells:
            self.stats["passes"] += 1
            raise GtpError(f"pentobi passed for {player.name} while the engine lists "
                           f"{len(legal_moves)} legal moves (rules disagreement)")
        chosen = match_move(legal_moves, cells, self._generator)
        if chosen is None:
            self.stats["mismatches"] += 1
            raise GtpError(f"pentobi played {answer!r} for {player.name}, which is not in the "
                           f"engine's legal-move list (rules disagreement)")
        self._mirror[player].add(cells)
        self.stats["moves"] += 1
        self.stats["total_time_ms"] += (time.perf_counter() - start) * 1000.0
        return chosen

    def get_action_info(self) -> Dict[str, Any]:
        return {"name": f"Pentobi L{self.level}", "type": "pentobi", "version": PENTOBI_VERSION,
                "description": "Pentobi 30.3 GTP engine (GPL-3) as external anchor",
                "stats": dict(self.stats)}

    def reset(self) -> None:
        self.close()

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self.close()
