"""Compute comprehensive statistics for MCTS equal-time parameter tuning tournaments."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, List

from analytics.tournament.tuning import TuningSet


def wilson_score_interval(wins: float, trials: float, z: float = 1.96) -> Dict[str, float]:
    """Calculate the Wilson score interval for a binomial proportion."""
    if trials == 0:
        return {"lower": 0.0, "upper": 0.0, "center": 0.0}
    p = wins / trials
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
    return {
        "lower": max(0.0, center - spread),
        "upper": min(1.0, center + spread),
        "center": p,
    }


def compute_tuning_summary(
    core_summary: Mapping[str, Any],
    tuning_set: TuningSet,
    configured_time_ms: int
) -> Dict[str, Any]:
    """
    Given the base arena stats generated for a tournament, extract parameter-specific insights,
    ranking, time budget compliance, and ablation importance.
    """
    tunings_by_name = {t.name: t for t in tuning_set.tunings}
    win_stats = core_summary.get("win_stats", {})
    score_stats = core_summary.get("score_stats", {})
    efficiency = core_summary.get("time_sim_efficiency", {})

    tuning_ranks: List[Dict[str, Any]] = []

    for agent_name in tunings_by_name.keys():
        ws = win_stats.get(agent_name, {})
        games = ws.get("games_played", 0.0)
        wins = ws.get("win_points", 0.0)
        win_rate = ws.get("win_rate", 0.0)

        ci = wilson_score_interval(wins, games)

        ss = score_stats.get(agent_name, {})
        mean_score = ss.get("mean", 0.0) or 0.0

        eff = efficiency.get(agent_name, {})
        avg_time = eff.get("avg_time_ms_per_move", 0.0)
        p50_time = eff.get("move_time_ms_p50", 0.0)
        p95_time = eff.get("move_time_ms_p95", 0.0)
        max_time = eff.get("move_time_ms_max", 0.0)
        avg_sims = eff.get("avg_simulations_per_move", 0.0)

        # explicit best tuning metric: Rank score prioritizes win rate over mean score
        rank_score = (win_rate * 100.0) + (mean_score * 0.5)

        tuning_ranks.append({
            "name": agent_name,
            "win_rate": win_rate,
            "win_rate_ci_lower": ci["lower"],
            "win_rate_ci_upper": ci["upper"],
            "mean_score": mean_score,
            "games_played": games,
            "avg_time_ms": avg_time,
            "p50_time_ms": p50_time,
            "p95_time_ms": p95_time,
            "max_time_ms": max_time,
            "avg_sims": avg_sims,
            "rank_score": rank_score
        })

    tuning_ranks.sort(key=lambda t: t["rank_score"], reverse=True)

    epsilon_ms = max(25.0, configured_time_ms * 0.25)
    budget_compliance = {}
    validation_errors = []

    for t in tuning_ranks:
        budget_compliance[t["name"]] = {
            "configured_ms": configured_time_ms,
            "actual_ms": t["avg_time_ms"],
            "p50_ms": t["p50_time_ms"],
            "p95_ms": t["p95_time_ms"],
            "max_ms": t["max_time_ms"],
            "diff_ms": (t["avg_time_ms"] or 0.0) - configured_time_ms,
            "simulations_per_move": t["avg_sims"]
        }

        # Hard validation gate
        avg_ms = t["avg_time_ms"] or 0.0
        p95_ms = t["p95_time_ms"] or 0.0

        if avg_ms > configured_time_ms * 1.10:
            validation_errors.append(f"Tuning '{t['name']}' failed! Mean time ({avg_ms:.1f}ms) > {configured_time_ms * 1.10:.1f}ms limit.")
        if p95_ms > configured_time_ms + epsilon_ms:
            validation_errors.append(f"Tuning '{t['name']}' failed! p95 time ({p95_ms:.1f}ms) > {configured_time_ms + epsilon_ms:.1f}ms cap.")

    if validation_errors:
        err_msg = "\\n".join(validation_errors)
        raise ValueError(f"Strict time budget violation detected:\\n{err_msg}")

    # Ablation analysis
    # We heuristically find a "base" tuning. Usually the first one or one with 'base' in name.
    base_name = tuning_set.tunings[0].name
    ablation = {}
    if len(tuning_ranks) > 0:
        base_stats = next(t for t in tuning_ranks if t["name"] == base_name)
        for t in tuning_ranks:
            if t["name"] == base_name:
                continue
            delta_wr = t["win_rate"] - base_stats["win_rate"]
            delta_score = t["mean_score"] - base_stats["mean_score"]
            ablation[t["name"]] = {
                "base_tuning": base_name,
                "delta_win_rate": delta_wr,
                "delta_mean_score": delta_score
            }

    return {
        "tunings_set": tuning_set.name,
        "configured_time_ms": configured_time_ms,
        "rankings": tuning_ranks,
        "budget_compliance": budget_compliance,
        "ablation_vs_baseline": ablation
    }


def render_tuning_summary_markdown(tuning_summary: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Tuning Tournament Summary: {tuning_summary['tunings_set']}")
    lines.append("")

    lines.append("## Overall Rankings")
    for i, t in enumerate(tuning_summary["rankings"]):
        lines.append(f"{i+1}. **{t['name']}**")
        lines.append(f"   - **Win Rate:** {t['win_rate']:.3f} (95% CI: [{t['win_rate_ci_lower']:.3f}, {t['win_rate_ci_upper']:.3f}])")
        lines.append(f"   - **Mean Score:** {t['mean_score']:.1f}")
        lines.append(f"   - Games Played: {t['games_played']}")
        lines.append(f"   - Internal Rank Score: {t['rank_score']:.2f}")
    lines.append("")

    lines.append("## Time Budget Compliance")
    lines.append("Ensuring fair comparison under equal-time constraints.")
    lines.append("| Tuning | Budg (ms) | Avg (ms) | p50 (ms) | p95 (ms) | Max (ms) | Avg Sims |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, item in tuning_summary["budget_compliance"].items():
        name = next(k for k, v in tuning_summary["budget_compliance"].items() if v == item)
        actual_ms = item["actual_ms"]
        p50 = item.get("p50_ms")
        p95 = item.get("p95_ms")
        max_ms = item.get("max_ms")
        sims = item["simulations_per_move"]

        def _f(val): return f"{val:.1f}" if val is not None else "N/A"

        lines.append(f"| {name} | {item['configured_ms']} | {_f(actual_ms)} | {_f(p50)} | {_f(p95)} | {_f(max_ms)} | {_f(sims)} |")
    lines.append("")

    lines.append("## Ablation & Parameter Importance (vs Baseline)")
    lines.append("Changes observed when modifying the baseline tuning.")
    if not tuning_summary["ablation_vs_baseline"]:
        lines.append("No ablation data available (or only 1 tuning).")
    else:
        for name, data in tuning_summary["ablation_vs_baseline"].items():
            lines.append(f"- **{name}** (vs {data['base_tuning']}):")
            lines.append(f"  - Win Rate Δ: {data['delta_win_rate']:+.3f}")
            lines.append(f"  - Score Δ: {data['delta_mean_score']:+.1f}")
    lines.append("")

    return "\\n".join(lines).strip() + "\\n"
