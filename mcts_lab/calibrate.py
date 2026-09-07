"""Protocol v3 harness gates (milestone M1 of the 2026-09 assessment).

    python -m mcts_lab.calibrate clone          --games 100 --workers 8
    python -m mcts_lab.calibrate discrimination --games 100 --workers 8

``clone``: the gen140 champion config against a byte-identical copy under a
different name, plus the deterministic greedy baseline and a random agent.
PASS when the 90% confidence interval of the paired score difference lies
within +/-4 points, the difference is not significant (p > 0.05), no game
errored, and n >= 240 (10 cycles of the 24 seat permutations). A FAIL means
the harness itself manufactures differences (seat, seed or bookkeeping bias)
and must be fixed before any strength claim.

``pentobi``: EXP-016 calibration table [gen140, d016_250, pentobi_l3, pentobi_l7];
report only (where do the repo's configurations sit against Pentobi levels?).

``discrimination``: gen140, the served registry-v2 configuration and the
D-016 configuration, all pinned to 250 iterations, plus greedy. PASS when the
pre-registered pairs (gen140 > serving_v2, d016_250 > serving_v2) separate at
p < 0.01 over >= 120 games. Note: serving_v2 here is pinned to 250 iterations
in ONE worker; in production the registry runs 2 root workers x 125
iterations (same total budget, split trees), so this is the v2 search
settings at equal budget, not a replay of the served process.

Seeds are fresh per run (derived from the label and the launch time) unless
``--seed`` is given; the seed is recorded in the report so any run can be
replayed exactly. Every search is iteration-pinned and single-worker.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from training.evaluation import protocol_v3 as p3

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = REPO / "training" / "reports" / "protocol_v3"
SERVING_ITERATIONS = 250  # champion.json: 500 ms x 0.5 iterations/ms

_DROP_KEYS = {"thinking_time_ms", "type", "name"}


def pinned_mcts(name: str, params: Dict[str, Any], iterations: int = SERVING_ITERATIONS) -> Dict[str, Any]:
    """An MCTS agent config pinned to an iteration budget, single worker."""
    p = {k: v for k, v in copy.deepcopy(params).items() if k not in _DROP_KEYS}
    p.update({"deterministic_time_budget": False, "iterations": int(iterations),
              "num_workers": 1, "parallel_strategy": "root"})
    p.pop("time_limit", None)
    return {"name": name, "type": "mcts", "thinking_time_ms": None, "params": p}


def champion_params() -> Dict[str, Any]:
    return json.loads((REPO / "training/state/champion.json").read_text())["flat_config"]


def serving_v2_params() -> Dict[str, Any]:
    reg = json.loads((REPO / "data/champion_registry.json").read_text())
    return reg["versions"][reg["current_version"]]["params"]


def d016_params() -> Dict[str, Any]:
    agents = json.loads((REPO / "training/experiments/exp007_agents.json").read_text())
    vm2 = next(a for a in agents if a["name"] == "vm2_500")
    p = dict(vm2["params"])
    p["value_model_path"] = str(REPO / p["value_model_path"])
    return p


def baseline(name: str, agent_type: str) -> Dict[str, Any]:
    return {"name": name, "type": agent_type, "thinking_time_ms": None, "params": {}}


def clone_table() -> List[Dict[str, Any]]:
    champ = champion_params()
    return [pinned_mcts("champion", champ), pinned_mcts("champion_clone", champ),
            baseline("greedy", "greedy"), baseline("random", "random")]


def discrimination_table() -> List[Dict[str, Any]]:
    return [pinned_mcts("gen140", champion_params()),
            pinned_mcts("serving_v2", serving_v2_params()),
            pinned_mcts("d016_250", d016_params()),
            baseline("greedy", "greedy")]


DISCRIMINATION_PAIRS = [("gen140", "serving_v2"), ("d016_250", "serving_v2")]


def pentobi_agent(name: str, level: int) -> Dict[str, Any]:
    return {"name": name, "type": "pentobi", "thinking_time_ms": None,
            "params": {"level": int(level), "use_book": False}}


def pentobi_table() -> List[Dict[str, Any]]:
    """EXP-016: where do gen140 and D-016 sit against Pentobi levels 3 and 7?"""
    return [pinned_mcts("gen140", champion_params()), pinned_mcts("d016_250", d016_params()),
            pentobi_agent("pentobi_l3", 3), pentobi_agent("pentobi_l7", 7)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gate", choices=["clone", "discrimination", "pentobi"],
                    help="pentobi = EXP-016 calibration table (report only, no pass/fail)")
    ap.add_argument("--games", type=int, default=None,
                    help="default: the gate's pre-registered n (clone 240, discrimination/pentobi 120)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None, help="replay a recorded run seed")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--deadline-minutes", type=float, default=None)
    ap.add_argument("--stat-seed", type=int, default=20260907)
    args = ap.parse_args(argv)

    floors = {"clone": p3.CLONE_MIN_GAMES, "discrimination": p3.DISCRIMINATION_MIN_GAMES,
              "pentobi": p3.DISCRIMINATION_MIN_GAMES}
    if args.games is None:
        args.games = floors[args.gate]
    if args.games < floors[args.gate]:
        print(f"[{p3.PROTOCOL_VERSION}] WARNING: {args.games} games is below the pre-registered "
              f"floor of {floors[args.gate]} for '{args.gate}': exploratory run, the gate cannot pass",
              file=sys.stderr, flush=True)
    label = args.label or f"{args.gate}_{p3.fresh_salt().replace(':', '').replace('-', '')}"
    seed = args.seed if args.seed is not None else p3.derive_run_seed(label, p3.fresh_salt())
    agents = {"clone": clone_table, "discrimination": discrimination_table,
              "pentobi": pentobi_table}[args.gate]()
    out_dir = Path(args.out_root) / label
    print(f"[{p3.PROTOCOL_VERSION}] gate={args.gate} label={label} seed={seed} "
          f"games={args.games} workers={args.workers}", file=sys.stderr, flush=True)
    games = p3.play_games(agents, num_games=args.games, seed=seed, workers=args.workers,
                          out_dir=out_dir, label=label, deadline_minutes=args.deadline_minutes,
                          progress=True)
    report = p3.analyze(games, agents, label=label, seed=seed, stat_seed=args.stat_seed,
                        out_dir=out_dir)
    if args.gate == "clone":
        gate = p3.clone_gate(report, "champion", "champion_clone")
    elif args.gate == "discrimination":
        gate = p3.discrimination_gate(report, DISCRIMINATION_PAIRS)
    else:
        gate = {"passed": True, "note": "calibration table — report only, no pass/fail"}
    report["gate"] = {"name": args.gate, **gate}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\n{args.gate.upper()} GATE: {'PASS' if gate['passed'] else 'FAIL'}")
    print(json.dumps(gate, indent=2))
    print(f"\n{'agent':<16}{'1st%':>7}{'avg rank':>10}{'mean ms/move':>14}{'sims/move':>11}")
    for row in report["leaderboard"]:
        n = row["name"]
        mt = report["move_time_ms"].get(n, {}).get("mean")
        sims = report["avg_simulations_per_move"].get(n)
        print(f"{n:<16}{(row.get('win_rate') or 0) * 100:>6.1f}%{(row.get('avg_rank') or 0):>10.2f}"
              f"{(mt or 0):>14.0f}{(sims or 0):>11.0f}")
    print("\npaired score differences (sign-flip permutation):")
    for pair, t in report["pairwise"].items():
        print(f"  {pair:<32} diff={t['observed_diff']:>7.2f}  p={t['p_value']:.4f}  n={t['n_games']}")
    print(f"\nreport -> {out_dir / 'report.json'}")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
