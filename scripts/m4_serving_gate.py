"""M4 serving-parity gate (DIAGNOSTIC_ASSESSMENT_2026-09-06.md, milestone M4).

Two checks for the natively served Pentobi agent (webapi.gameplay_agent_factory.
PentobiGameplayAdapter):

1. Parity: on N positions of a game, the served adapter and the arena's
   Pentobi agent (same level, same seed) choose the same move.
2. Latency: a full 4-seat game driven through the web GameManager (the exact
   code path the UI uses) records every move's wall time and any fallback; the
   gate is max per-move latency <= 8 s and zero fallbacks/timeouts.

Usage: python scripts/m4_serving_gate.py --levels 3,7 --positions 50 --out training/reports/m4_serving_gate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from engine.board import Board, Player  # noqa: E402
from engine.move_generator import LegalMoveGenerator  # noqa: E402

GATE_MAX_MOVE_MS = 8000
CAP_MS = 10000


def parity(level: int, positions: int, seed: int) -> dict:
    from analytics.tournament.arena_runner import AgentConfig, build_agent
    from schemas.game_state import AgentType
    from webapi.gameplay_agent_factory import build_deploy_gameplay_agent

    gen = LegalMoveGenerator()
    served = build_deploy_gameplay_agent(AgentType.PENTOBI, {"level": level, "seed": seed, "time_budget_ms": CAP_MS})
    arena = build_agent(AgentConfig.from_dict({"name": f"pentobi_l{level}", "type": "pentobi",
                                               "params": {"level": level}}), seed=seed)
    board = Board()
    player = Player.RED
    compared = mismatches = passes = 0
    served_ms, arena_ms = [], []
    while compared < positions and passes < 4:
        legal = gen.get_legal_moves(board, player)
        if not legal:
            passes += 1
            player = Player((player.value % 4) + 1)
            continue
        passes = 0
        t0 = time.perf_counter()
        arena_move, _ = arena.choose_move(board, player, legal, thinking_time_ms=CAP_MS)
        arena_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        served_move, stats = served.choose_move(board, player, legal, CAP_MS)
        served_ms.append((time.perf_counter() - t0) * 1000)
        same = (served_move.piece_id, served_move.orientation, served_move.anchor_row, served_move.anchor_col) == \
               (arena_move.piece_id, arena_move.orientation, arena_move.anchor_row, arena_move.anchor_col)
        if not same or stats["fallback"]:
            mismatches += 1
        board.place_piece(served_move.get_positions(gen.piece_orientations_cache[served_move.piece_id]),
                          player, served_move.piece_id)
        compared += 1
        player = Player((player.value % 4) + 1)
    served.close()
    arena.close()
    return {"level": level, "seed": seed, "positions": compared, "mismatches": mismatches,
            "served_ms": _summary(served_ms), "arena_ms": _summary(arena_ms), "pass": mismatches == 0}


def latency(level: int, seed: int, out_dir: Path) -> dict:
    os.environ["GAME_LOG_DIR"] = str(out_dir / "game_logs")
    from schemas.game_state import AgentType, GameConfig, Player as SPlayer, PlayerConfig
    from webapi.app import GameManager

    manager = GameManager(app_profile="research")
    game_id = f"m4_latency_l{level}"
    config = GameConfig(game_id=game_id, auto_start=True, scoring_mode="standard", players=[
        PlayerConfig(player=SPlayer(c), agent_type=AgentType.PENTOBI,
                     agent_config={"level": level, "seed": seed + i, "time_budget_ms": CAP_MS})
        for i, c in enumerate(("RED", "BLUE", "GREEN", "YELLOW"))])
    manager.create_game(config)

    turn_ms = []  # whole advance_turn call: search + applying the move + telemetry

    async def play() -> None:
        while manager.games[game_id]["status"].value == "in_progress":
            t0 = time.perf_counter()
            await manager.advance_turn(game_id)
            turn_ms.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    asyncio.run(play())
    total_s = time.perf_counter() - t0
    records = manager.games[game_id]["move_records"]
    moves = [r for r in records if not r.get("isPass")]
    times = [int(r["stats"].get("timeSpentMs", 0)) for r in moves]
    fallbacks = [r for r in moves if r["stats"].get("fallback")]
    game = manager.games[game_id]["game"]
    return {"level": level, "seed": seed, "moves": len(moves), "passes": len(records) - len(moves),
            "game_seconds": round(total_s, 1), "search_ms": _summary(times), "turn_ms": _summary(turn_ms),
            "fallbacks": len(fallbacks),
            "over_cap": sum(1 for t in times if t > CAP_MS),
            "over_gate": sum(1 for t in turn_ms if t > GATE_MAX_MOVE_MS),
            "scores": {p.name: int(game.get_score(p)) for p in Player},
            "game_log": str(out_dir / "game_logs" / f"{game_id}.json"),
            "pass": not fallbacks and max(turn_ms, default=0) <= GATE_MAX_MOVE_MS}


def _summary(values):
    if not values:
        return {}
    s = sorted(values)
    return {"n": len(s), "mean": round(statistics.fmean(s), 1), "p90": round(s[int(0.9 * len(s)) - 1], 1),
            "p99": round(s[max(0, int(0.99 * len(s)) - 1)], 1), "max": round(s[-1], 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="3,7")
    ap.add_argument("--positions", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out", default="training/reports/m4_serving_gate")
    args = ap.parse_args()
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"gate": "M4 serving parity", "cap_ms": CAP_MS, "gate_max_move_ms": GATE_MAX_MOVE_MS,
              "seed": args.seed, "parity": [], "latency": []}
    for level in [int(x) for x in args.levels.split(",")]:
        print(f"[parity] level {level} ...", flush=True)
        report["parity"].append(parity(level, args.positions, args.seed))
        print(json.dumps(report["parity"][-1]), flush=True)
        print(f"[latency] level {level} ...", flush=True)
        report["latency"].append(latency(level, args.seed, out_dir))
        print(json.dumps(report["latency"][-1]), flush=True)
        (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    report["pass"] = all(r["pass"] for r in report["parity"]) and all(r["pass"] for r in report["latency"])
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print("GATE", "PASS" if report["pass"] else "FAIL", "->", out_dir / "report.json")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
