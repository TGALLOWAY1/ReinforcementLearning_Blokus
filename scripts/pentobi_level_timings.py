"""Per-move genmove wall time per Pentobi level (one 4-seat self-play game per
level, one thread). Produces the numbers recorded in DECISIONS.md D-024.

    python scripts/pentobi_level_timings.py --levels 4,5,6,8 --out timings.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.pentobi_agent import PentobiGtp  # noqa: E402


def time_level(level: int, seed: int, max_moves: int = 100) -> dict:
    times = []
    with PentobiGtp(level=level, seed=seed, threads=1) as eng:
        eng.set_game_classic()
        passes, colour, moves = 0, 1, 0
        while passes < 4 and moves < max_moves:
            t0 = time.perf_counter()
            answer = eng.genmove(str(colour))
            dt = (time.perf_counter() - t0) * 1000
            if answer.strip().lower() == "pass":
                passes += 1
            else:
                passes = 0
                moves += 1
                times.append(round(dt, 1))
            colour = colour % 4 + 1
    ordered = sorted(times)
    return {"level": level, "seed": seed, "moves": len(times), "mean_ms": round(sum(times) / len(times), 1),
            "p90_ms": ordered[int(0.9 * len(ordered)) - 1], "max_ms": ordered[-1], "first8_ms": times[:8]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results = {}
    for level in [int(x) for x in args.levels.split(",")]:
        results[level] = time_level(level, args.seed)
        print(json.dumps(results[level]), flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
