"""Per-game JSON log for the human protocol (M5).

One file per game under a log directory, rewritten atomically (temp file +
rename) after every event, so a crashed or abandoned game still leaves a
readable record. The record carries everything the protocol needs to attribute
a result: seat -> player type, each AI seat's settings (engine, level, seed,
cap), every move with its timing and any fallback the serving adapter took,
passes, final scores/ranks/winner, and the code and engine versions.

Enable with ``GAME_LOG_DIR`` (the research profile defaults to
``data/human_games``; the deploy profile is off unless the variable is set,
since hosted filesystems may be read-only).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.board import STATE_SCHEMA_VERSION
from engine.move_generator import ACTION_SCHEMA_VERSION

GAME_LOG_VERSION = "human_game_log_v1"
# Pass reasons the GameManager records when an AI seat forfeits its move.
ATTRIBUTABLE_PASS_REASONS = ("timeout", "error")
GAME_LOG_DIR_ENV = "GAME_LOG_DIR"
DEFAULT_RESEARCH_LOG_DIR = "data/human_games"

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def resolve_log_dir(app_profile: str) -> Optional[Path]:
    """Directory to log games into, or None when logging is off."""
    raw = os.getenv(GAME_LOG_DIR_ENV)
    if raw is not None:
        raw = raw.strip()
        return Path(raw) if raw else None
    if app_profile == "research":
        return _REPO_ROOT / DEFAULT_RESEARCH_LOG_DIR
    return None


def _ranks(scores: Dict[str, int]) -> Dict[str, int]:
    ordered = sorted(scores.values(), reverse=True)
    return {name: ordered.index(score) + 1 for name, score in scores.items()}


def load_game_log(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class GameLogWriter:
    def __init__(self, log_dir: Path | str) -> None:
        self.log_dir = Path(log_dir)
        self._records: Dict[str, Dict[str, Any]] = {}
        self._commit = repo_commit()

    def path_for(self, game_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(game_id))
        return self.log_dir / f"{safe}.json"

    def start(self, game_id: str, *, players: List[Dict[str, Any]], scoring_mode: str,
              meta: Dict[str, Any]) -> Path:
        players = [_plain(dict(p)) for p in players]
        seats = {str(p["player"]): str(p["agent_type"]) for p in players}
        human_seats = [seat for seat, kind in seats.items() if kind == "human"]
        record: Dict[str, Any] = {
            "log_version": GAME_LOG_VERSION,
            "game_id": str(game_id),
            "status": "in_progress",
            "started_at": _now(),
            "finished_at": None,
            "scoring_mode": str(scoring_mode),
            "players": players,
            "seats": seats,
            "has_human": bool(human_seats),
            "human_seats": human_seats,
            "meta": dict(meta),
            "versions": {
                "repo_commit": self._commit,
                "state_schema_version": STATE_SCHEMA_VERSION,
                "action_schema_version": ACTION_SCHEMA_VERSION,
            },
            "moves": [],
            "fallbacks": [],
            "result": None,
        }
        self._records[str(game_id)] = record
        return self._flush(str(game_id))

    def record_move(self, game_id: str, *, player: str, is_human: bool, move: Dict[str, Any],
                    stats: Dict[str, Any]) -> None:
        record = self._records.get(str(game_id))
        if record is None:
            return
        index = len(record["moves"])
        record["moves"].append({"index": index, "player": str(player), "is_human": bool(is_human),
                                "is_pass": False, "move": dict(move), "stats": _jsonable(stats),
                                "at": _now()})
        fallback = stats.get("fallback") if isinstance(stats, dict) else None
        if fallback:
            record["fallbacks"].append({"index": index, "player": str(player), "fallback": str(fallback)})
        self._flush(str(game_id))

    def record_pass(self, game_id: str, *, player: str, is_human: bool, reason: str) -> None:
        record = self._records.get(str(game_id))
        if record is None:
            return
        index = len(record["moves"])
        record["moves"].append({"index": index, "player": str(player),
                                "is_human": bool(is_human), "is_pass": True, "reason": str(reason),
                                "move": None, "stats": {}, "at": _now()})
        if not is_human and str(reason) in ATTRIBUTABLE_PASS_REASONS:
            # An AI turn lost to the server's outer timeout or an exception is a
            # forfeited move: attributable to the harness, like a fallback move.
            record["fallbacks"].append({"index": index, "player": str(player), "fallback": f"pass_{reason}"})
        self._flush(str(game_id))

    def finish(self, game_id: str, *, scores: Dict[str, int], winner: Optional[str], status: str) -> None:
        record = self._records.get(str(game_id))
        if record is None:
            return
        scores = {str(k): int(v) for k, v in scores.items()}
        ranks = _ranks(scores) if scores else {}
        record["status"] = str(status)
        record["finished_at"] = _now()
        record["result"] = {
            "scores": scores,
            "ranks": ranks,
            "winner": winner,
            "human_first_place": any(ranks.get(seat) == 1 for seat in record["human_seats"]) if ranks else None,
        }
        self._flush(str(game_id))
        self._records.pop(str(game_id), None)

    def _flush(self, game_id: str) -> Path:
        path = self.path_for(game_id)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=self.log_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._records[game_id], handle, indent=1, default=str)
            os.chmod(tmp, 0o644)  # mkstemp creates 0600; logs must be readable by the analysis CLI
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return path


def _plain(value: Any) -> Any:
    """Recursively convert enums to their values and mappings/lists to plain JSON types."""
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, dict):
        return {str(_plain(k)): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _jsonable(value: Any) -> Any:
    value = _plain(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))
