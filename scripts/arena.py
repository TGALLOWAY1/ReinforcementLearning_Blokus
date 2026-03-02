"""CLI entrypoint for reproducible arena experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.tournament.arena_runner import RunConfig, load_run_config, run_experiment


def _with_overrides(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    payload: Dict[str, Any] = config.to_dict()
    if args.num_games is not None:
        payload["num_games"] = args.num_games
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.seat_policy is not None:
        payload["seat_policy"] = args.seat_policy
    if args.output_root is not None:
        payload["output_root"] = args.output_root
    if args.max_turns is not None:
        payload["max_turns"] = args.max_turns
    if args.notes is not None:
        payload["notes"] = args.notes
    snapshots = dict(payload.get("snapshots") or {})
    if args.snapshots_enabled:
        snapshots["enabled"] = True
    if args.snapshots_disabled:
        snapshots["enabled"] = False
    if args.snapshot_strategy is not None:
        snapshots["strategy"] = args.snapshot_strategy
    if args.snapshot_plys is not None:
        checkpoints = [int(part.strip()) for part in args.snapshot_plys.split(",") if part.strip()]
        snapshots["checkpoints"] = checkpoints
    payload["snapshots"] = snapshots
    return RunConfig.from_dict(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible Blokus arena experiments and emit run artifacts."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/arena_config.json",
        help="Path to run configuration JSON.",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=None,
        help="Override num_games from config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override run seed from config.",
    )
    parser.add_argument(
        "--seat-policy",
        type=str,
        choices=["randomized", "round_robin"],
        default=None,
        help="Override seat policy from config.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override output root directory (default: arena_runs).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override maximum turns per game.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Optional notes to store in run metadata and index.csv.",
    )
    parser.add_argument(
        "--snapshots-enabled",
        action="store_true",
        help="Enable snapshot dataset logging (one row per player per checkpoint).",
    )
    parser.add_argument(
        "--snapshots-disabled",
        action="store_true",
        help="Disable snapshot dataset logging regardless of config.",
    )
    parser.add_argument(
        "--snapshot-strategy",
        type=str,
        choices=["fixed_ply"],
        default=None,
        help="Snapshot checkpoint strategy (currently only fixed_ply).",
    )
    parser.add_argument(
        "--snapshot-plys",
        type=str,
        default=None,
        help="Comma-separated ply checkpoints for snapshots, e.g. '8,16,24,32'.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved run config and exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-game progress.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    run_config = load_run_config(str(config_path))
    run_config = _with_overrides(run_config, args)

    if args.print_config:
        print(json.dumps(run_config.to_dict(), indent=2, sort_keys=True))
        return

    result = run_experiment(run_config, verbose=args.verbose)
    run_dir = result["run_dir"]
    summary = result["summary"]
    print(f"run_id: {result['run_id']}")
    print(f"run_dir: {run_dir}")
    print(
        "completed_games: "
        f"{summary['completed_games']}/{summary['num_games']}, "
        f"errors: {summary['error_games']}"
    )
    print(f"summary_json: {run_dir}/summary.json")
    print(f"summary_md: {run_dir}/summary.md")
    snapshots = summary.get("snapshots")
    if snapshots and snapshots.get("enabled"):
        print(f"snapshots_csv: {run_dir}/snapshots.csv")
        if snapshots.get("parquet_written"):
            print(f"snapshots_parquet: {run_dir}/snapshots.parquet")


if __name__ == "__main__":
    main()
