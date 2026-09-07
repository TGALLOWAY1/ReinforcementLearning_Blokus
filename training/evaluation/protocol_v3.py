"""Protocol v3 — an evaluation that can see strength.

Why a new protocol
------------------
The 2026-09 diagnostic assessment (docs/agent-strength-rebuild/
DIAGNOSTIC_ASSESSMENT_2026-09-06.md, findings B4-B8) found that every earlier
evaluation ran at 25-50 iterations per move (where search collapses to the
move-ordering heuristic), replayed two fixed seeds night after night, rated
agents against a pool that copied the champion, and could not reach a
decision inside its game budget. Protocol v3 fixes the harness before any
strength work:

* **Fresh seed per run** — the run seed is derived from the label and a salt
  (by default the launch timestamp) and recorded in the report, so two runs
  never replay the same games unless the same seed is passed on purpose.
* **Round-robin seats** — with four agents and a game count that is a
  multiple of four, every agent sits in every seat equally often.
* **Iteration-pinned search at the serving budget** — agent configs carry
  explicit ``iterations`` (``deterministic_time_budget: false``) and a single
  worker, so results reproduce and no wall-clock override can silently shrink
  the search.
* **Paired statistics with pre-registered n** — every game is a paired
  observation across all four seats; the primary statistic is the per-game
  score difference with a sign-flip permutation test, plus first-place rate,
  average rank and Wilson intervals.
* **Two harness gates** (assessment §4, milestone M1):
  :func:`clone_gate` — an agent byte-identical to the champion must score
  within noise of it; :func:`discrimination_gate` — agents known to differ
  must separate.

Games are independent given their seed, so they are played in a process
pool; the arena's own :func:`run_single_game` is reused unchanged.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from analytics.tournament import gauntlet
from analytics.tournament.arena_runner import (
    RunConfig,
    _seat_assignment_for_game,
    game_seed_from_run_seed,
    run_single_game,
)
from analytics.tournament.statistics import paired_permutation_test

PROTOCOL_VERSION = "protocol_v3"
SEAT_POLICY = "round_robin"
SCORING_MODE = "standard"

# Gate defaults (pre-registered in the assessment, §4 milestone M1).
CLONE_MAX_ABS_DIFF = 3.0     # points per game
CLONE_MIN_P = 0.30
DISCRIMINATION_MAX_P = 0.01
GATE_MIN_GAMES = 100


def derive_run_seed(label: str, salt: str) -> int:
    """Deterministic 31-bit seed from (label, salt); salt defaults to a timestamp."""
    digest = hashlib.sha256(f"{PROTOCOL_VERSION}|{label}|{salt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1) or 1


def fresh_salt() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_config(agents: Sequence[Dict[str, Any]], num_games: int, seed: int,
                out_dir: Path) -> RunConfig:
    return RunConfig.from_dict({
        "agents": list(agents),
        "num_games": int(num_games),
        "seed": int(seed),
        "seat_policy": SEAT_POLICY,
        "output_root": str(out_dir),
        "scoring_mode": SCORING_MODE,
        "snapshots": {"enabled": False},
    })


def _play_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Worker: play one game from a serialisable payload; return the record."""
    run_config = RunConfig.from_dict(payload["run_config"])
    game_index = int(payload["game_index"])
    game_seed = game_seed_from_run_seed(run_config.seed, game_index)
    seat_assignment = _seat_assignment_for_game(
        run_config.agent_names, game_index, game_seed, SEAT_POLICY)
    agent_configs = {a.name: a for a in run_config.agents}
    record, _rows = run_single_game(
        run_id=payload["run_id"], game_index=game_index, game_seed=game_seed,
        run_config=run_config, seat_assignment=seat_assignment,
        agent_configs=agent_configs, verbose=False,
    )
    record["seat_policy"] = SEAT_POLICY
    return record


def play_games(agents: Sequence[Dict[str, Any]], *, num_games: int, seed: int,
               workers: int, out_dir: Path, label: str,
               deadline_minutes: Optional[float] = None,
               progress: bool = False) -> List[Dict[str, Any]]:
    """Play ``num_games`` seat-rotated games in a process pool.

    Writes ``run_config.json`` and ``games.jsonl`` (in game-index order) under
    ``out_dir`` and returns the game records sorted by game index. Reproducible
    for a given (agents, num_games, seed) regardless of ``workers``.
    """
    if len(agents) != 4:
        raise ValueError("protocol v3 tables hold exactly 4 agents")
    if num_games % 4 != 0:
        raise ValueError("num_games must be a multiple of 4 for seat balance")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = _run_config(agents, num_games, seed, out_dir)
    run_id = label
    payload_cfg = run_config.to_dict()
    (out_dir / "run_config.json").write_text(json.dumps({
        **payload_cfg, "run_id": run_id, "protocol": PROTOCOL_VERSION,
        "created_at": fresh_salt(), "workers": int(workers),
    }, indent=2), encoding="utf-8")

    payloads = [{"run_config": payload_cfg, "run_id": run_id, "game_index": i}
                for i in range(num_games)]
    deadline = (time.monotonic() + deadline_minutes * 60.0) if deadline_minutes else None
    records: Dict[int, Dict[str, Any]] = {}
    started = time.monotonic()
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=ctx) as pool:
        futures = {pool.submit(_play_one, p): p["game_index"] for p in payloads}
        pending = set(futures)
        while pending:
            timeout = None
            if deadline is not None:
                timeout = max(0.0, deadline - time.monotonic())
            done, pending = cf.wait(pending, timeout=timeout, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                idx = futures[fut]
                records[idx] = fut.result()
                if progress:
                    rec = records[idx]
                    elapsed = time.monotonic() - started
                    print(f"[{PROTOCOL_VERSION}] game {idx:>3} done "
                          f"({len(records)}/{num_games}, {elapsed / 60:.1f} min) "
                          f"scores={rec.get('agent_scores')}", file=sys.stderr, flush=True)
            if deadline is not None and time.monotonic() >= deadline and pending:
                if progress:
                    print(f"[{PROTOCOL_VERSION}] deadline reached with {len(pending)} "
                          f"game(s) unfinished; cancelling", file=sys.stderr, flush=True)
                for fut in pending:
                    fut.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                break
    ordered = [records[i] for i in sorted(records)]
    with (out_dir / "games.jsonl").open("w", encoding="utf-8") as fh:
        for rec in ordered:
            fh.write(json.dumps(rec, default=str) + "\n")
    return ordered


def _pair_key(pairwise: Dict[str, Any], a: str, b: str) -> Tuple[Optional[Dict[str, Any]], float]:
    """Return (test, sign) where sign flips observed_diff into a-minus-b order."""
    if f"{a} vs {b}" in pairwise:
        return pairwise[f"{a} vs {b}"], 1.0
    if f"{b} vs {a}" in pairwise:
        return pairwise[f"{b} vs {a}"], -1.0
    return None, 1.0


def analyze(games: Sequence[Dict[str, Any]], agents: Sequence[Dict[str, Any]], *,
            label: str, seed: int, stat_seed: int = 20260907,
            out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Leaderboard, paired permutation tests, seat counts and per-move cost."""
    games = [g for g in games if g.get("is_valid_result", True)]
    names = [a["name"] for a in agents]
    pooled = gauntlet.aggregate_summary(
        games, agent_names=sorted(names),
        thinking_time_ms_by_agent={a["name"]: a.get("thinking_time_ms") for a in agents},
        seeds=[seed], seat_policy=SEAT_POLICY,
        run_config={"protocol": PROTOCOL_VERSION, "label": label},
    )
    leaderboard = gauntlet.build_leaderboard(pooled)
    ranks: Dict[str, List[int]] = {n: [] for n in names}
    seat_counts: Dict[str, List[int]] = {n: [0, 0, 0, 0] for n in names}
    first_by_seat: Dict[str, List[int]] = {n: [0, 0, 0, 0] for n in names}
    move_ms: Dict[str, List[float]] = {n: [] for n in names}
    sims: Dict[str, List[float]] = {n: [] for n in names}
    for g in games:
        for name, rank in (g.get("agent_ranks") or {}).items():
            if name in ranks and rank is not None:
                ranks[name].append(int(rank))
        for seat, name in (g.get("seat_assignment") or {}).items():
            if name in seat_counts:
                seat_counts[name][int(seat) - 1] += 1
                if name in (g.get("winner_agents") or []):
                    first_by_seat[name][int(seat) - 1] += 1
        for name, st in (g.get("agent_move_stats") or {}).items():
            if name in move_ms:
                move_ms[name].extend(float(t) for t in (st.get("move_times_ms") or []))
                if st.get("moves_with_simulations"):
                    sims[name].append(float(st.get("avg_simulations_per_move") or 0.0))
    for row in leaderboard:
        r = ranks.get(row["name"], [])
        row["avg_rank"] = (sum(r) / len(r)) if r else None
        row["games"] = len(r)
    pairwise = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairwise[f"{a} vs {b}"] = paired_permutation_test(games, a, b, seed=stat_seed)
    report = {
        "protocol": PROTOCOL_VERSION,
        "label": label,
        "seed": int(seed),
        "seat_policy": SEAT_POLICY,
        "scoring_mode": SCORING_MODE,
        "agents": list(agents),
        "completed_games": len(games),
        "leaderboard": leaderboard,
        "pairwise": pairwise,
        "seat_counts": seat_counts,
        "first_place_by_seat": first_by_seat,
        "move_time_ms": {n: {"mean": (sum(v) / len(v)) if v else None,
                             "max": max(v) if v else None, "moves": len(v)}
                         for n, v in move_ms.items()},
        "avg_simulations_per_move": {n: (sum(v) / len(v)) if v else None for n, v in sims.items()},
        "pooled_summary": pooled,
    }
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def clone_gate(report: Dict[str, Any], champion: str, clone: str, *,
               max_abs_diff: float = CLONE_MAX_ABS_DIFF, min_p: float = CLONE_MIN_P,
               min_games: int = GATE_MIN_GAMES) -> Dict[str, Any]:
    """Calibration: a byte-identical clone must be indistinguishable from the champion."""
    test, sign = _pair_key(report.get("pairwise", {}), champion, clone)
    if test is None:
        return {"passed": False, "reason": f"no paired test for {champion} vs {clone}"}
    diff = sign * float(test["observed_diff"])
    n = int(test["n_games"])
    p = float(test["p_value"])
    checks = {
        "enough_games": n >= min_games,
        "diff_within_noise": abs(diff) < max_abs_diff,
        "not_significant": p > min_p,
    }
    return {"passed": all(checks.values()), "checks": checks, "observed_diff": diff,
            "p_value": p, "n_games": n,
            "thresholds": {"max_abs_diff": max_abs_diff, "min_p": min_p, "min_games": min_games}}


def discrimination_gate(report: Dict[str, Any], pairs: Iterable[Tuple[str, str]], *,
                        max_p: float = DISCRIMINATION_MAX_P,
                        min_games: int = GATE_MIN_GAMES) -> Dict[str, Any]:
    """Every (stronger, weaker) pair must separate: positive diff, p < max_p, n >= min_games."""
    results = {}
    for a, b in pairs:
        test, sign = _pair_key(report.get("pairwise", {}), a, b)
        if test is None:
            results[f"{a} > {b}"] = {"passed": False, "reason": "no paired test"}
            continue
        diff = sign * float(test["observed_diff"])
        n = int(test["n_games"])
        p = float(test["p_value"])
        results[f"{a} > {b}"] = {
            "passed": n >= min_games and diff > 0 and p < max_p,
            "observed_diff": diff, "p_value": p, "n_games": n,
        }
    return {"passed": all(r["passed"] for r in results.values()), "pairs": results,
            "thresholds": {"max_p": max_p, "min_games": min_games}}
