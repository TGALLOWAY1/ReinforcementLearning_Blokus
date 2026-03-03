"""Summary statistics for arena experiment runs."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Union

import numpy as np


def load_games_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load game records from JSONL."""
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    mean_value = _mean(values)
    if mean_value is None:
        return None
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return float(math.sqrt(variance))


def _percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    if quantile <= 0:
        return float(min(values))
    if quantile >= 1:
        return float(max(values))
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(sorted_values[lower])
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    fraction = index - lower
    return float(lower_value + (upper_value - lower_value) * fraction)


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _init_win_bucket() -> Dict[str, float]:
    return {
        "games_played": 0.0,
        "outright_wins": 0.0,
        "shared_wins": 0.0,
        "win_points": 0.0,
        "win_rate": 0.0,
    }


def _score_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p25": None,
            "p75": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": _mean(values),
        "median": float(median(values)),
        "std": _std(values),
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _collect_score_values(
    games: Iterable[Mapping[str, Any]], agent_names: Sequence[str]
) -> Dict[str, List[float]]:
    score_values: Dict[str, List[float]] = {agent_name: [] for agent_name in agent_names}
    for game in games:
        if game.get("error"):
            continue
        for agent_name, score in (game.get("agent_scores") or {}).items():
            if agent_name in score_values:
                score_values[agent_name].append(float(score))
    return score_values


def _compute_pairwise_counts(
    games: Iterable[Mapping[str, Any]],
    agent_names: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    pairwise: Dict[str, Dict[str, int]] = {}
    for agent_a, agent_b in itertools.combinations(sorted(agent_names), 2):
        key = f"{agent_a}__vs__{agent_b}"
        pairwise[key] = {
            "agent_a": agent_a,
            "agent_b": agent_b,
            "a_beats_b": 0,
            "b_beats_a": 0,
            "tie": 0,
            "total": 0,
        }

    for game in games:
        if game.get("error"):
            continue
        agent_scores = game.get("agent_scores") or {}
        for agent_a, agent_b in itertools.combinations(sorted(agent_names), 2):
            if agent_a not in agent_scores or agent_b not in agent_scores:
                continue
            key = f"{agent_a}__vs__{agent_b}"
            score_a = int(agent_scores[agent_a])
            score_b = int(agent_scores[agent_b])
            pairwise[key]["total"] += 1
            if score_a > score_b:
                pairwise[key]["a_beats_b"] += 1
            elif score_b > score_a:
                pairwise[key]["b_beats_a"] += 1
            else:
                pairwise[key]["tie"] += 1
    return pairwise


def _compute_efficiency(
    games: Iterable[Mapping[str, Any]],
    agent_names: Sequence[str],
    thinking_time_ms_by_agent: Mapping[str, Optional[int]],
    win_stats: Mapping[str, Mapping[str, float]],
    score_stats: Mapping[str, Mapping[str, Optional[float]]],
) -> Dict[str, Dict[str, Optional[float]]]:
    accumulator: Dict[str, Dict[str, Any]] = {
        agent_name: {
            "moves": 0.0,
            "total_time_ms": 0.0,
            "total_simulations": 0.0,
            "moves_with_simulations": 0.0,
            "all_move_times": [],
        }
        for agent_name in agent_names
    }

    for game in games:
        if game.get("error"):
            continue
        per_agent_stats = game.get("agent_move_stats") or {}
        for agent_name, stats in per_agent_stats.items():
            if agent_name not in accumulator:
                continue
            moves = float(stats.get("moves", 0))
            total_time_ms = _to_float(stats.get("total_time_ms")) or 0.0
            simulations = _to_float(stats.get("total_simulations"))
            sim_moves = float(stats.get("moves_with_simulations", 0))
            move_times = stats.get("move_times_ms", [])
            accumulator[agent_name]["moves"] += moves
            accumulator[agent_name]["total_time_ms"] += total_time_ms
            accumulator[agent_name]["all_move_times"].extend(move_times)
            if simulations is not None:
                accumulator[agent_name]["total_simulations"] += simulations
                accumulator[agent_name]["moves_with_simulations"] += sim_moves

    efficiency: Dict[str, Dict[str, Optional[float]]] = {}
    for agent_name in agent_names:
        entry = accumulator[agent_name]
        moves = entry["moves"]
        total_time_ms = entry["total_time_ms"]
        total_simulations = entry["total_simulations"]
        sim_moves = entry["moves_with_simulations"]
        avg_time_ms = _safe_div(total_time_ms, moves)
        avg_simulations_per_move = _safe_div(total_simulations, sim_moves)
        simulations_per_second = _safe_div(total_simulations, total_time_ms / 1000.0)
        thinking_time_ms = thinking_time_ms_by_agent.get(agent_name)
        budget_utilization = (
            _safe_div(avg_time_ms or 0.0, float(thinking_time_ms))
            if thinking_time_ms
            else None
        )
        win_rate = float(win_stats.get(agent_name, {}).get("win_rate", 0.0))
        mean_score = _to_float(score_stats.get(agent_name, {}).get("mean")) or 0.0
        avg_time_seconds = (avg_time_ms or 0.0) / 1000.0
        
        all_times = entry.get("all_move_times", [])
        p50 = float(np.percentile(all_times, 50)) if all_times else None
        p95 = float(np.percentile(all_times, 95)) if all_times else None
        max_time = float(np.max(all_times)) if all_times else None

        efficiency[agent_name] = {
            "moves": moves,
            "total_time_ms": total_time_ms,
            "avg_time_ms_per_move": avg_time_ms,
            "move_time_ms_p50": p50,
            "move_time_ms_p95": p95,
            "move_time_ms_max": max_time,
            "configured_thinking_time_ms": float(thinking_time_ms) if thinking_time_ms is not None else None,
            "avg_budget_utilization": budget_utilization,
            "total_simulations": total_simulations if sim_moves > 0 else None,
            "avg_simulations_per_move": avg_simulations_per_move if sim_moves > 0 else None,
            "simulations_per_second": simulations_per_second if sim_moves > 0 else None,
            "win_rate_per_second": _safe_div(win_rate, avg_time_seconds) if avg_time_seconds > 0 else None,
            "score_per_second": _safe_div(mean_score, avg_time_seconds) if avg_time_seconds > 0 else None,
        }

    return efficiency


def compute_summary(
    games: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    run_seed: int,
    seat_policy: str,
    agent_names: Sequence[str],
    thinking_time_ms_by_agent: Mapping[str, Optional[int]],
    run_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compute aggregate summary from game records."""
    win_stats: Dict[str, Dict[str, float]] = {
        agent_name: _init_win_bucket() for agent_name in agent_names
    }
    wins_by_seat: Dict[str, Dict[str, float]] = {
        agent_name: {str(seat): 0.0 for seat in range(4)} for agent_name in agent_names
    }
    games_by_seat: Dict[str, Dict[str, int]] = {
        agent_name: {str(seat): 0 for seat in range(4)} for agent_name in agent_names
    }

    completed_games = 0
    error_games = 0
    for game in games:
        if game.get("error"):
            error_games += 1
            continue
        completed_games += 1
        seat_assignment = {
            str(player_id): str(agent_name)
            for player_id, agent_name in (game.get("seat_assignment") or {}).items()
        }
        winner_agents = list(game.get("winner_agents") or [])
        winner_share = 1.0 / len(winner_agents) if winner_agents else 0.0
        for player_id, agent_name in seat_assignment.items():
            seat_index = int(player_id) - 1
            if agent_name not in win_stats:
                continue
            win_stats[agent_name]["games_played"] += 1
            games_by_seat[agent_name][str(seat_index)] += 1
            if agent_name in winner_agents:
                if len(winner_agents) == 1:
                    win_stats[agent_name]["outright_wins"] += 1
                else:
                    win_stats[agent_name]["shared_wins"] += 1
                win_stats[agent_name]["win_points"] += winner_share
                wins_by_seat[agent_name][str(seat_index)] += winner_share

    for agent_name in agent_names:
        games_played = win_stats[agent_name]["games_played"]
        win_points = win_stats[agent_name]["win_points"]
        win_stats[agent_name]["win_rate"] = float(win_points / games_played) if games_played else 0.0

    seat_breakdown: Dict[str, Dict[str, Dict[str, float]]] = {}
    for agent_name in agent_names:
        seat_breakdown[agent_name] = {}
        for seat in range(4):
            games_count = games_by_seat[agent_name][str(seat)]
            wins = wins_by_seat[agent_name][str(seat)]
            seat_breakdown[agent_name][str(seat)] = {
                "games": games_count,
                "win_points": wins,
                "win_rate": float(wins / games_count) if games_count else 0.0,
            }

    score_values = _collect_score_values(games, agent_names)
    score_summary = {
        agent_name: _score_summary(values) for agent_name, values in score_values.items()
    }

    pairwise = _compute_pairwise_counts(games, agent_names)
    total_pairwise_comparisons = sum(item["total"] for item in pairwise.values())

    efficiency = _compute_efficiency(
        games,
        agent_names,
        thinking_time_ms_by_agent,
        win_stats,
        score_summary,
    )

    duration_seconds = [
        _to_float(game.get("duration_sec")) or 0.0
        for game in games
        if not game.get("error")
    ]

    summary = {
        "run_id": run_id,
        "seed": run_seed,
        "seat_policy": seat_policy,
        "num_games": len(games),
        "completed_games": completed_games,
        "error_games": error_games,
        "run_config": dict(run_config),
        "win_stats": win_stats,
        "wins_by_seat": seat_breakdown,
        "score_stats": score_summary,
        "pairwise_matchups": pairwise,
        "pairwise_total_comparisons": total_pairwise_comparisons,
        "time_sim_efficiency": efficiency,
        "game_duration_sec": _score_summary(duration_seconds),
    }
    return summary


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    """Render a concise human-readable summary."""
    lines: List[str] = []
    lines.append(f"# Arena Run Summary: {summary['run_id']}")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Seed: `{summary['seed']}`")
    lines.append(f"- Seat policy: `{summary['seat_policy']}`")
    lines.append(f"- Games: `{summary['completed_games']}/{summary['num_games']}` completed")
    lines.append(f"- Error games: `{summary['error_games']}`")
    lines.append("")

    lines.append("## Win Rates by Agent")
    for agent_name, item in sorted(summary["win_stats"].items()):
        lines.append(
            f"- `{agent_name}`: win_rate={item['win_rate']:.3f}, "
            f"win_points={item['win_points']:.2f}, outright={int(item['outright_wins'])}, shared={int(item['shared_wins'])}"
        )
    lines.append("")

    lines.append("## Win Rates by Seat")
    for agent_name, seat_data in sorted(summary["wins_by_seat"].items()):
        seat_text = ", ".join(
            f"seat{seat}: {stats['win_rate']:.3f} ({stats['games']} games)"
            for seat, stats in sorted(seat_data.items(), key=lambda entry: int(entry[0]))
        )
        lines.append(f"- `{agent_name}`: {seat_text}")
    lines.append("")

    lines.append("## Score Stats")
    for agent_name, item in sorted(summary["score_stats"].items()):
        lines.append(
            f"- `{agent_name}`: mean={item['mean']}, median={item['median']}, std={item['std']}, "
            f"p25={item['p25']}, p75={item['p75']}, min={item['min']}, max={item['max']}"
        )
    lines.append("")

    lines.append("## Pairwise Matchups")
    for key, item in sorted(summary["pairwise_matchups"].items()):
        lines.append(
            f"- `{key}`: {item['agent_a']}>{item['agent_b']}={item['a_beats_b']}, "
            f"{item['agent_b']}>{item['agent_a']}={item['b_beats_a']}, tie={item['tie']} (total={item['total']})"
        )
    lines.append("")

    lines.append("## Time and Simulation Efficiency")
    for agent_name, item in sorted(summary["time_sim_efficiency"].items()):
        lines.append(
            f"- `{agent_name}`: avg_time_ms={item['avg_time_ms_per_move']}, "
            f"avg_sims_per_move={item['avg_simulations_per_move']}, sims_per_sec={item['simulations_per_second']}, "
            f"win_rate_per_sec={item['win_rate_per_second']}, score_per_sec={item['score_per_second']}"
        )
    lines.append("")

    return "\n".join(lines).strip() + "\n"
