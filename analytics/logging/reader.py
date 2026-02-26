"""
Read StrategyLogger JSONL outputs for API consumption.

Supports:
- Per-game path: {base_dir}/{sanitized_game_id}/steps.jsonl (webapi layout)
- Flat path: {base_dir}/steps.jsonl filtered by game_id (offline layout)
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple


def _sanitize_game_id(game_id: str) -> str:
    """Sanitize game_id for filesystem path."""
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in game_id)


def load_jsonl(path: str, filter_game_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load JSONL file. Optionally filter by game_id.
    Returns empty list if path does not exist.
    """
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if filter_game_id is not None and obj.get("game_id") != filter_game_id:
                    continue
                data.append(obj)
            except json.JSONDecodeError:
                continue
    return data


def resolve_steps_path(base_dir: str, game_id: str) -> Tuple[str, Optional[str]]:
    """
    Resolve path(s) to steps for a game.
    Returns (primary_path, fallback_path).
    - primary: per-game {base}/{game_id}/steps.jsonl
    - fallback: flat {base}/steps.jsonl (filter by game_id)
    """
    safe_id = _sanitize_game_id(game_id)
    per_game = os.path.join(base_dir, safe_id, "steps.jsonl")
    flat = os.path.join(base_dir, "steps.jsonl")
    return per_game, flat if per_game != flat else None


def load_steps_for_game(base_dir: str, game_id: str) -> List[Dict[str, Any]]:
    """
    Load StepLog entries for game_id in chronological order.
    Tries per-game path first, then flat path with filter.
    """
    per_game, flat = resolve_steps_path(base_dir, game_id)
    data = load_jsonl(per_game)
    if not data and flat:
        data = load_jsonl(flat, filter_game_id=game_id)
    data.sort(key=lambda x: (x.get("turn_index", 0), x.get("timestamp", 0)))
    return data
