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
* **All 24 seat permutations** — a cyclic round-robin keeps who-moves-after-
  whom fixed (review of the first harness draft found a byte-identical clone
  drifting 7 points because it always followed the champion), so games cycle
  through every permutation of the four agents: with a game count that is a
  multiple of 24, every agent sits in every seat and after every other agent
  equally often.
* **Iteration-pinned search at the serving budget** — agent configs carry
  explicit ``iterations`` (``deterministic_time_budget: false``) and a single
  worker, so results reproduce and no wall-clock override can silently shrink
  the search.
* **Paired statistics with pre-registered n** — every game is a paired
  observation across all four seats; the primary statistic is the per-game
  score difference with a sign-flip permutation test, plus first-place rate,
  average rank and Wilson intervals.
* **Two harness gates** (assessment §4, milestone M1):
  :func:`clone_gate` — equivalence test: the 90% confidence interval of the
  paired score difference between the champion and a byte-identical clone
  must lie within ±4 points (TOST at α = 0.05), the difference must not be
  significant (p > 0.05), no game may have errored, and n ≥ 240 (10 cycles
  of the 24 permutations). With a paired-difference sd of 15-18 points this
  passes an unbiased harness ≥ 93% of the time and passes a 4-point bias
  ≤ 5% of the time. :func:`discrimination_gate` — agents known to differ must
  separate at p < 0.01 over n ≥ 120 (power ≈ 0.99 for a 10-point gap).

Games are independent given their seed, so they are played in a process
pool; the arena's own :func:`run_single_game` is reused unchanged.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
import traceback
import uuid
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
SEAT_POLICY = "all_permutations"
SCORING_MODE = "standard"
PERMUTATIONS = 24  # 4! seatings; num_games must be a multiple of this

# Gate constants (pre-registered; see BENCHMARK_PROTOCOL.md protocol_v3 entry).
# Floors cannot be lowered from the CLI: a run below them is exploratory.
CLONE_EQUIV_BAND = 4.0       # points: 90% CI of the paired difference must sit within +/- band
CLONE_MIN_P = 0.05           # and the difference must not be significant
CLONE_MIN_GAMES = 240        # 10 cycles of the 24 permutations
DISCRIMINATION_MAX_P = 0.01
DISCRIMINATION_MIN_GAMES = 120
GATE_MIN_GAMES = DISCRIMINATION_MIN_GAMES  # kept for callers that import the old name


def derive_run_seed(label: str, salt: str) -> int:
    """Deterministic 31-bit seed from (label, salt); salt defaults to a timestamp."""
    digest = hashlib.sha256(f"{PROTOCOL_VERSION}|{label}|{salt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1) or 1


def fresh_salt() -> str:
    """Launch timestamp + pid + random fragment: distinct even for same-second launches."""
    return (f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}"
            f"|pid{os.getpid()}|{uuid.uuid4().hex[:6]}")


def seat_assignment(agent_names: Sequence[str], game_index: int) -> Dict[str, str]:
    """Seat map for game ``game_index``: the (game_index mod 24)-th permutation
    of the agent list, so every seat AND every successor relation is balanced
    over any multiple of 24 games."""
    names = list(agent_names)
    if len(names) != 4:
        raise ValueError("protocol v3 tables hold exactly 4 agents")
    # Delegates to the arena's own implementation so both paths agree.
    return _seat_assignment_for_game(names, int(game_index), 0, SEAT_POLICY)


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
    seats = seat_assignment(run_config.agent_names, game_index)
    agent_configs = {a.name: a for a in run_config.agents}
    record, _rows = run_single_game(
        run_id=payload["run_id"], game_index=game_index, game_seed=game_seed,
        run_config=run_config, seat_assignment=seats,
        agent_configs=agent_configs, verbose=False,
    )
    record["seat_policy"] = SEAT_POLICY
    record["seat_permutation_index"] = int(game_index) % PERMUTATIONS
    if record.get("error"):
        record["is_valid_result"] = False
        record["invalid_reason"] = "error"
    return record


def _error_record(run_id: str, game_index: int, agent_names: Sequence[str], exc: BaseException) -> Dict[str, Any]:
    """Record for a game whose worker raised before/while playing."""
    return {
        "run_id": run_id, "game_id": f"{run_id}_g{game_index:04d}", "game_index": int(game_index),
        "seat_assignment": seat_assignment(agent_names, game_index),
        "seat_permutation_index": int(game_index) % PERMUTATIONS, "seat_policy": SEAT_POLICY,
        "agent_scores": {}, "agent_ranks": {}, "winner_agents": [], "is_valid_result": False,
        "invalid_reason": "worker_exception",
        "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
    }


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
    if num_games % PERMUTATIONS != 0:
        raise ValueError(f"num_games must be a multiple of {PERMUTATIONS} (all seat permutations)")
    out_dir = Path(out_dir)
    if (out_dir / "games.jsonl").exists():
        raise ValueError(f"{out_dir} already holds a run; choose a new label")
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
    names = [a["name"] for a in agents]
    deadline = (time.monotonic() + deadline_minutes * 60.0) if deadline_minutes else None
    records: Dict[int, Dict[str, Any]] = {}
    started = time.monotonic()
    ctx = mp.get_context("spawn")
    games_path = out_dir / "games.jsonl"

    def _take(fut, idx, fh):
        try:
            rec = fut.result()
        except BaseException as exc:  # a worker died or raised: keep the game as an error record
            rec = _error_record(run_id, idx, names, exc)
        records[idx] = rec
        fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush()
        if progress:
            elapsed = time.monotonic() - started
            status = rec.get("agent_scores") if rec.get("is_valid_result", True) else "ERROR"
            print(f"[{PROTOCOL_VERSION}] game {idx:>3} done ({len(records)}/{num_games}, "
                  f"{elapsed / 60:.1f} min) {status}", file=sys.stderr, flush=True)

    with games_path.open("w", encoding="utf-8") as fh:
        pool = cf.ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=ctx)
        try:
            futures = {pool.submit(_play_one, p): p["game_index"] for p in payloads}
            pending = set(futures)
            while pending:
                timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
                done, pending = cf.wait(pending, timeout=timeout, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    _take(fut, futures[fut], fh)
                if deadline is not None and time.monotonic() >= deadline and pending:
                    if progress:
                        print(f"[{PROTOCOL_VERSION}] deadline reached with {len(pending)} game(s) "
                              f"unfinished; cancelling queued games, draining running ones",
                              file=sys.stderr, flush=True)
                    for fut in pending:
                        fut.cancel()
                    pool.shutdown(wait=True, cancel_futures=True)
                    for fut in pending:
                        if fut.done() and not fut.cancelled():
                            _take(fut, futures[fut], fh)
                    break
        finally:
            pool.shutdown(wait=True)
    ordered = [records[i] for i in sorted(records)]
    # Rewrite in game-index order (records were appended as they completed).
    with games_path.open("w", encoding="utf-8") as fh:
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
    all_games = list(games)
    excluded = [g for g in all_games
                if not g.get("is_valid_result", True) or g.get("error") or g.get("truncated")]
    games = [g for g in all_games if g not in excluded]
    excluded_reasons = {}
    for g in excluded:
        reason = g.get("invalid_reason") or ("error" if g.get("error") else "truncated")
        excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
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
            test = paired_permutation_test(games, a, b, seed=stat_seed)
            diffs = [g["agent_scores"][a] - g["agent_scores"][b] for g in games
                     if a in g.get("agent_scores", {}) and b in g.get("agent_scores", {})]
            test.update(_diff_intervals(diffs))
            pairwise[f"{a} vs {b}"] = test
    report = {
        "protocol": PROTOCOL_VERSION,
        "label": label,
        "seed": int(seed),
        "seat_policy": SEAT_POLICY,
        "scoring_mode": SCORING_MODE,
        "agents": list(agents),
        "completed_games": len(games),
        "excluded_games": len(excluded),
        "excluded_reasons": excluded_reasons,
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


def _diff_intervals(diffs: Sequence[float]) -> Dict[str, Optional[float]]:
    """Mean, sd, se and normal-approximation 90%/95% intervals of paired differences."""
    n = len(diffs)
    if n < 2:
        return {"sd_diff": None, "se_diff": None, "ci90_low": None, "ci90_high": None,
                "ci95_low": None, "ci95_high": None}
    mean = sum(diffs) / n
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(n)
    return {"sd_diff": sd, "se_diff": se,
            "ci90_low": mean - 1.645 * se, "ci90_high": mean + 1.645 * se,
            "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se}


def clone_gate(report: Dict[str, Any], champion: str, clone: str, *,
               band: float = CLONE_EQUIV_BAND, min_p: float = CLONE_MIN_P,
               min_games: int = CLONE_MIN_GAMES) -> Dict[str, Any]:
    """Calibration (equivalence test): a byte-identical clone must be
    indistinguishable from the champion. Floors are hard: ``min_games`` below
    :data:`CLONE_MIN_GAMES` is raised back to it."""
    min_games = max(int(min_games), CLONE_MIN_GAMES)
    test, sign = _pair_key(report.get("pairwise", {}), champion, clone)
    if test is None:
        return {"passed": False, "reason": f"no paired test for {champion} vs {clone}"}
    diff = sign * float(test["observed_diff"])
    lo, hi = sorted([sign * float(test["ci90_low"]), sign * float(test["ci90_high"])])
    n = int(test["n_games"])
    p = float(test["p_value"])
    excluded = int(report.get("excluded_games", 0))
    checks = {
        "enough_games": n >= min_games,
        "ci90_within_band": -band < lo and hi < band,
        "not_significant": p > min_p,
        "no_errored_games": excluded == 0,
    }
    return {"passed": all(checks.values()), "checks": checks, "observed_diff": diff,
            "ci90": [lo, hi], "p_value": p, "n_games": n, "excluded_games": excluded,
            "thresholds": {"band": band, "min_p": min_p, "min_games": min_games}}


def discrimination_gate(report: Dict[str, Any], pairs: Iterable[Tuple[str, str]], *,
                        max_p: float = DISCRIMINATION_MAX_P,
                        min_games: int = DISCRIMINATION_MIN_GAMES) -> Dict[str, Any]:
    """Every (stronger, weaker) pair must separate: positive diff, p < max_p,
    n >= min_games (floor cannot be lowered), and no errored games."""
    min_games = max(int(min_games), DISCRIMINATION_MIN_GAMES)
    excluded = int(report.get("excluded_games", 0))
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
    passed = all(r["passed"] for r in results.values()) and excluded == 0
    return {"passed": passed, "pairs": results, "excluded_games": excluded,
            "thresholds": {"max_p": max_p, "min_games": min_games}}
