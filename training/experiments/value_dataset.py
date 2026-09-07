"""Generate a fresh, manifested value-evaluator training dataset (rescue).

Phase 5/6 evaluator track (see docs/agent-strength-rebuild/HANDOFF.md): plays
standard-scored self-play games between four identical PW+rollout MCTS agents
(the strongest validated configuration family from EXP-002; 50 iterations is
statistically equal to 500 there at a fraction of the cost) and appends rich
45-feature TD trajectory rows to a NEW dataset file — never the legacy
data/td_trajectories.csv corpus — together with a manifest (DATA_LINEAGE.md
rules: fresh datasets are manifested and immutable once finalized).

Example:
    python -m training.experiments.value_dataset \
        --games 60 --seed 20260713 --deadline-minutes 150 \
        --out data/value_dataset_v1
"""

from __future__ from engine.board import STATE_SCHEMA_VERSION
import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from training.experiments.search_scaling import MINIMAL_SEARCH_PARAMS, PW_PARAMS
from training.rich_features import FEATURE_SET_VERSION
from training.td_selfplay import collect_trajectories

DATASET_SCHEMA_VERSION = "value_dataset_v1"
TEACHER_ITERATIONS = 50
AGENT_VERSION_TAG = f"rescue_pw{TEACHER_ITERATIONS}_std"


def teacher_agent(name: str) -> dict:
    params = dict(MINIMAL_SEARCH_PARAMS)
    params.update(PW_PARAMS)
    params["iterations"] = TEACHER_ITERATIONS
    return {"name": name, "type": "mcts", "thinking_time_ms": None, "params": params}


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m training.experiments.value_dataset", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--deadline-minutes", type=float, default=150.0)
    parser.add_argument("--min-games", type=int, default=8)
    parser.add_argument("--out", default="data/value_dataset_v1")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "trajectories.csv"
    # Datasets are immutable once written (DATA_LINEAGE.md): the CSV helper is
    # append-only and run ids derive from the seed, so a rerun into the same
    # directory would silently mix duplicate game ids into a dataset whose
    # manifest describes only the last invocation. Refuse instead.
    if csv_path.exists():
        raise SystemExit(
            f"{csv_path} already exists — datasets are immutable. "
            "Pass a new --out directory (e.g. data/value_dataset_v2) or delete "
            "the old directory deliberately before regenerating."
        )
    run_id = args.run_id or f"vds1_s{args.seed}"

    agents = [teacher_agent(f"pw{TEACHER_ITERATIONS}_{i}") for i in range(1, 5)]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    manifest = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "purpose": "value-evaluator training (Phase 5/6 evaluator track)",
        "generating_commit": commit,
        "run_id": run_id,
        "scoring_mode": "standard",
        "engine_state_schema": STATE_SCHEMA_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "agent_version": AGENT_VERSION_TAG,
        "teacher_agents": agents,
        "seed": args.seed,
        "num_games_requested": args.games,
        "capture_every_ply": True,
        "status": "generating",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    deadline = time.monotonic() + args.deadline_minutes * 60.0
    t0 = time.monotonic()
    rows = collect_trajectories(
        agents,
        run_id=run_id,
        num_games=args.games,
        seed=args.seed,
        agent_version=AGENT_VERSION_TAG,
        output_path=csv_path,
        capture_every_ply=True,
        deadline=deadline,
        min_games=args.min_games,
        verbose=args.verbose,
    )
    elapsed = time.monotonic() - t0

    manifest["status"] = "finalized"
    manifest["rows_written"] = int(rows)
    manifest["elapsed_sec"] = round(elapsed, 1)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\n{rows} rows in {elapsed / 60:.1f} min -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
