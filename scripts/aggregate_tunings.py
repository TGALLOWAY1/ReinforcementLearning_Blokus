"""Aggregate multiple tuning_summary.json files into a single view."""

import json
from pathlib import Path
from typing import Dict


def wilson_score_interval(wins: float, trials: float, z: float = 1.96) -> Dict[str, float]:
    """Calculate the Wilson score interval for a binomial proportion."""
    import math
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

def main():
    arena_runs = Path("arena_runs")
    summary_files = list(arena_runs.glob("*/tuning_summary.json"))

    if not summary_files:
        print("No tuning_summary.json files found.")
        return

    # Aggregate stats by tuning name
    aggregated = {}
    total_files = 0

    for f in summary_files:
        try:
            with f.open() as p:
                data = json.load(p)

            rankings = data.get("rankings", [])
            for r in rankings:
                name = r["name"]
                if name not in aggregated:
                    aggregated[name] = {
                        "wins": 0.0,
                        "games_played": 0.0,
                        "total_score": 0.0,
                        "time_ms_sum": 0.0,
                        "sims_sum": 0.0,
                        "time_calls": 0,
                    }

                # Reverse engineer wins from win_rate and games_played
                wins = r["win_rate"] * r["games_played"]

                aggregated[name]["wins"] += wins
                aggregated[name]["games_played"] += r["games_played"]
                aggregated[name]["total_score"] += r["mean_score"] * r["games_played"]

                if r.get("avg_time_ms") is not None and r.get("avg_sims") is not None:
                    aggregated[name]["time_ms_sum"] += r.get("avg_time_ms")
                    aggregated[name]["sims_sum"] += r.get("avg_sims")
                    aggregated[name]["time_calls"] += 1
            total_files += 1

        except Exception as e:
            print(f"Error reading {f}: {e}")

    print(f"Aggregated {total_files} tournament instances.\\n")

    final_ranks = []

    for name, stats in aggregated.items():
        if stats["games_played"] == 0:
            continue

        win_rate = stats["wins"] / stats["games_played"]
        ci = wilson_score_interval(stats["wins"], stats["games_played"])
        mean_score = stats["total_score"] / stats["games_played"]

        avg_time = stats["time_ms_sum"] / stats["time_calls"] if stats["time_calls"] > 0 else 0
        avg_sims = stats["sims_sum"] / stats["time_calls"] if stats["time_calls"] > 0 else 0

        # explicit best tuning metric
        rank_score = (win_rate * 100.0) + (mean_score * 0.5)

        final_ranks.append({
            "name": name,
            "win_rate": win_rate,
            "win_rate_ci_lower": ci["lower"],
            "win_rate_ci_upper": ci["upper"],
            "mean_score": mean_score,
            "games_played": stats["games_played"],
            "avg_time_ms": avg_time,
            "avg_sims": avg_sims,
            "rank_score": rank_score
        })

    final_ranks.sort(key=lambda t: t["rank_score"], reverse=True)

    print("## Overall Aggregated Rankings")
    for i, t in enumerate(final_ranks):
        print(f"{i+1}. **{t['name']}**")
        print(f"   - **Win Rate:** {t['win_rate']:.3f} (95% CI: [{t['win_rate_ci_lower']:.3f}, {t['win_rate_ci_upper']:.3f}])")
        print(f"   - **Mean Score:** {t['mean_score']:.1f}")
        print(f"   - Games Played: {t['games_played']}")
        print(f"   - Avg Time/Move: {t['avg_time_ms']:.1f} ms")
        print(f"   - Avg Sims/Move: {t['avg_sims']:.1f}")
        print("")


if __name__ == "__main__":
    main()
