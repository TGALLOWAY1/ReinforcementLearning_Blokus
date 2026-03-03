"""CLI entrypoint for equal-time MCTS parameter tuning tournaments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.tournament.arena_runner import (
    AgentConfig,
    RunConfig,
    SnapshotConfig,
    _prepare_run_directory,
    _write_json,
    _append_index_row,
    run_single_game,
)
from analytics.tournament.scheduler import generate_matchups, validate_balance
from analytics.tournament.tuning import get_tuning_set
from analytics.tournament.arena_stats import compute_summary
# We will create tuning_stats later, but for now we import the interface.
from analytics.tournament.tuning_stats import compute_tuning_summary, render_tuning_summary_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run equal-time tuning tournament.")
    parser.add_argument("--tunings-set", type=str, required=True, help="Name of TuningSet.")
    parser.add_argument("--num-games", type=int, default=100, help="Total games.")
    parser.add_argument("--thinking-time-ms", type=int, default=100, help="Equal budget per tuning.")
    parser.add_argument("--seed", type=int, default=0, help="Master run seed.")
    parser.add_argument("--seat-policy", type=str, choices=["randomized", "round_robin"], default="randomized")
    parser.add_argument("--agent-backend", type=str, choices=["mcts", "fast_mcts"], default="mcts", help="Base engine implementation to use for all tunings.")
    parser.add_argument("--output-root", type=str, default="arena_runs")
    parser.add_argument("--max-turns", type=int, default=2500)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    tuning_set = get_tuning_set(args.tunings_set)
    tuning_names = [t.name for t in tuning_set.tunings]
    
    # Pre-generate schedule and validate
    matchups = generate_matchups(tuning_names, args.num_games, args.seed, args.seat_policy)
    validate_balance(matchups, tuning_names)
    
    # Build AgentConfigs for each tuning
    agent_configs: Dict[str, AgentConfig] = {}
    for tuning in tuning_set.tunings:
        agent_configs[tuning.name] = AgentConfig(
            name=tuning.name,
            type=args.agent_backend,
            thinking_time_ms=args.thinking_time_ms,
            params=tuning.to_dict()["params"]
        )
        
    # We create a pseudo RunConfig just to track run_id, seed, output_root etc.
    # We spoof 4 agents just so it validates, but we'll use agent_configs dynamically.
    pseudo_agents = list(agent_configs.values())[:4]
    
    run_config = RunConfig(
        agents=pseudo_agents,
        num_games=args.num_games,
        seed=args.seed,
        seat_policy=args.seat_policy,
        output_root=args.output_root,
        max_turns=args.max_turns,
        notes=f"Tuning Set: {args.tunings_set}. " + args.notes,
        snapshots=SnapshotConfig(enabled=False)
    )
    
    if args.print_config:
        print(json.dumps({"tuning_set": args.tunings_set, "matchups": len(matchups)}))
        return
        
    run_id, run_dir = _prepare_run_directory(run_config)
    
    # Dump matchup schedule
    matchups_data = []
    games_scheduled = {name: 0 for name in tuning_names}
    for m in matchups:
        matchups_data.append({
            "game_index": m.game_index,
            "seats": m.seats
        })
        for seat, name in m.seats.items():
            games_scheduled[name] += 1
            
    _write_json(run_dir / "matchups.json", matchups_data)
    
    for name, count in games_scheduled.items():
        if count == 0:
            raise ValueError(f"Tuning '{name}' was scheduled 0 games! Aborting tournament.")
    
    _write_json(run_dir / "tuning_config.json", {
        "tunings_set": args.tunings_set,
        "thinking_time_ms": args.thinking_time_ms,
        "num_games": args.num_games,
        "seed": args.seed,
        "seat_policy": args.seat_policy,
        "tunings": [t.to_dict() for t in tuning_set.tunings]
    })
    
    games_jsonl_path = run_dir / "games.jsonl"
    game_records = []
    
    completed_games = 0
    error_games = 0
    
    with games_jsonl_path.open("w", encoding="utf-8") as handle:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from analytics.tournament.arena_runner import game_seed_from_run_seed
        
        futures = []
        with ProcessPoolExecutor() as executor:
            for idx, matchup in enumerate(matchups):
                game_index = matchup.game_index
                seat_assignment = {str(seat + 1): agent_name for seat, agent_name in matchup.seats.items()}
                
                game_seed = game_seed_from_run_seed(run_config.seed, game_index)
                
                futures.append(executor.submit(
                    run_single_game,
                    run_id=run_id,
                    game_index=game_index,
                    game_seed=game_seed,
                    run_config=run_config,
                    seat_assignment=seat_assignment,
                    agent_configs=agent_configs
                ))
                
            for future in as_completed(futures):
                try:
                    record, _ = future.result()
                    game_records.append(record)
                    handle.write(json.dumps(record, sort_keys=True) + "\\n")
                    handle.flush()
                    
                    if record.get("error"):
                        error_games += 1
                        print(f"Error in game: {record.get('error')}")
                    else:
                        completed_games += 1
                        
                    if args.verbose:
                        print(f"[{completed_games + error_games}/{args.num_games}] "
                              f"winners={record.get('winner_agents')} "
                              f"scores={record.get('agent_scores')}")
                except Exception as e:
                    print(f"Process crashed: {e}")
                    error_games += 1

    # Core statistics (reuse old structure for backwards compat)
    thinking_params = {t.name: args.thinking_time_ms for t in tuning_set.tunings}
    core_summary = compute_summary(
        game_records,
        run_id=run_id,
        run_seed=args.seed,
        seat_policy=args.seat_policy,
        agent_names=tuning_names,
        thinking_time_ms_by_agent=thinking_params,
        run_config=run_config.to_dict()
    )
    
    _write_json(run_dir / "summary.json", core_summary)
    
    # Tournament specific stats (ranking, ablation, budget validation)
    tuning_summary = compute_tuning_summary(
        core_summary, 
        tuning_set,
        args.thinking_time_ms
    )
    
    _write_json(run_dir / "tuning_summary.json", tuning_summary)
    with (run_dir / "tuning_summary.md").open("w", encoding="utf-8") as handle:
        handle.write(render_tuning_summary_markdown(tuning_summary))

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _append_index_row(
        output_root=Path(args.output_root),
        run_id=run_id,
        timestamp=timestamp,
        run_config=run_config,
        completed_games=completed_games
    )
    
    print(f"run_id: {run_id}")
    print(f"Completed: {completed_games}/{args.num_games}")
    print(f"Results: {run_dir}/tuning_summary.md")


if __name__ == "__main__":
    main()

