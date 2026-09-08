"""Summarise logged human games for the M5 protocol.

Reads the per-game JSON records written by ``webapi/game_log.py`` (default
``data/human_games/``), keeps completed games with exactly one human seat and
Pentobi opponents, and reports per Pentobi level, from the AGENT's side: games,
Pentobi first-place count and rate with a 90% Wilson interval (a human tie for
first counts against the agent), the human's first places / outright wins /
mean score and rank, seat distribution, engine fallbacks and forfeited AI
turns, and the pre-registered gate (assessment M5: the agent takes first place
in >= 70% of 20 seat-rotated games, and no agent loss is attributable to an
illegal move, timeout or crash — i.e. no game the human won in which an AI
seat played a fallback move or forfeited a turn).

    python -m mcts_lab.human_games [--dir data/human_games] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

GATE_GAMES = 20
GATE_FIRST_PLACE = 0.7
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "human_games"
SEATS = ("RED", "BLUE", "GREEN", "YELLOW")


def wilson_interval(successes: int, n: int, z: float = 1.6449) -> Tuple[float, float]:
    """Wilson score interval (default 90% two-sided)."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _level_of(record: Dict[str, Any]) -> Optional[int]:
    levels = {int(p.get("agent_config", {}).get("level", 0)) for p in record.get("players", [])
              if p.get("agent_type") == "pentobi"}
    return levels.pop() if len(levels) == 1 else None


def _eligible(record: Dict[str, Any]) -> bool:
    if record.get("status") != "finished" or not record.get("result"):
        return False
    if len(record.get("human_seats") or []) != 1:
        return False
    seats = record.get("seats") or {}
    return all(kind in ("human", "pentobi") for kind in seats.values()) and _level_of(record) is not None


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_level: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    eligible = [r for r in records if _eligible(r)]
    for record in eligible:
        by_level[_level_of(record)].append(record)  # type: ignore[index]

    rows = []
    for level in sorted(by_level):
        games = by_level[level]
        firsts = wins = agent_firsts = 0
        seats: Counter = Counter({seat: 0 for seat in SEATS})
        scores, ranks = [], []
        fallback_moves = attributable_losses = 0
        human_think_ms: List[int] = []
        ai_move_ms: List[int] = []
        for record in games:
            human = record["human_seats"][0]
            result = record["result"]
            rank = result["ranks"].get(human)
            first = rank == 1                       # human first, ties included
            agent_first = not first                 # a tie for first counts against the agent
            outright = first and list(result["ranks"].values()).count(1) == 1
            firsts += int(first)
            agent_firsts += int(agent_first)
            wins += int(outright)
            seats[human] += 1
            scores.append(int(result["scores"].get(human, 0)))
            ranks.append(int(rank or 0))
            n_fallback = len(record.get("fallbacks") or [])   # fallback moves + forfeited AI turns
            fallback_moves += n_fallback
            if n_fallback and not agent_first:
                attributable_losses += 1
            for move in record.get("moves", []):
                if move.get("is_pass"):
                    continue
                spent = (move.get("stats") or {}).get("timeSpentMs")
                if spent is None:
                    continue
                (human_think_ms if move.get("is_human") else ai_move_ms).append(int(spent))
        n = len(games)
        rate = agent_firsts / n if n else 0.0
        ci = wilson_interval(agent_firsts, n)
        rows.append({
            "level": level,
            "games": n,
            "agent_first_place": agent_firsts,
            "agent_first_place_rate": round(rate, 4),
            "agent_first_place_ci90": (round(ci[0], 4), round(ci[1], 4)),
            "human_first_place": firsts,
            "human_outright_wins": wins,
            "seats": dict(seats),
            "mean_human_score": round(sum(scores) / n, 1) if n else None,
            "mean_human_rank": round(sum(ranks) / n, 2) if n else None,
            "fallback_moves": fallback_moves,
            "attributable_losses": attributable_losses,
            "mean_human_think_s": round(sum(human_think_ms) / len(human_think_ms) / 1000, 1) if human_think_ms else None,
            "max_ai_move_s": round(max(ai_move_ms) / 1000, 2) if ai_move_ms else None,
            "gate": {
                "games_ok": n >= GATE_GAMES,
                "first_place_ok": rate >= GATE_FIRST_PLACE,
                "fallback_ok": attributable_losses == 0,
                "pass": n >= GATE_GAMES and rate >= GATE_FIRST_PLACE and attributable_losses == 0,
            },
        })
    unfinished = sorted(r.get("game_id", "?") for r in records
                        if r.get("status") != "finished" and len(r.get("human_seats") or []) == 1
                        and _level_of(r) is not None)
    return {"games_found": len(records), "completed_human_games": len(eligible),
            "unfinished_human_games": unfinished, "levels": rows,
            "gate": {"games": GATE_GAMES, "agent_first_place": GATE_FIRST_PLACE,
                     "tie_policy": "a human tie for first counts against the agent"}}


def summarize_dir(directory: Path | str) -> Dict[str, Any]:
    directory = Path(directory)
    records = [r for r in (_load(p) for p in sorted(directory.glob("*.json"))) if r]
    return summarize_records(records)


def format_table(summary: Dict[str, Any]) -> str:
    lines = [f"human games: {summary['completed_human_games']} completed of {summary['games_found']} logged"
             + (f"; unfinished: {', '.join(summary['unfinished_human_games'])}" if summary["unfinished_human_games"] else "")]
    if not summary["levels"]:
        lines.append("no completed human-vs-Pentobi games yet")
    for row in summary["levels"]:
        lo, hi = row["agent_first_place_ci90"]
        gate = row["gate"]
        status = "PASS" if gate["pass"] else ("pending" if not gate["games_ok"] else "FAIL")
        lines.append(
            f"level {row['level']}: Pentobi first place {row['agent_first_place']}/{row['games']} "
            f"({row['agent_first_place_rate'] * 100:.0f}%, 90% CI {lo * 100:.0f}-{hi * 100:.0f}%; gate >= 70%), "
            f"human first place {row['human_first_place']} (outright {row['human_outright_wins']}), "
            f"human mean score {row['mean_human_score']}, mean rank {row['mean_human_rank']}, seats {row['seats']}, "
            f"fallbacks/forfeits {row['fallback_moves']} (agent losses with one: {row['attributable_losses']}), "
            f"gate {status}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="directory of game logs")
    parser.add_argument("--json", default=None, help="write the summary to this JSON file")
    args = parser.parse_args(argv)
    summary = summarize_dir(args.dir)
    print(format_table(summary))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
