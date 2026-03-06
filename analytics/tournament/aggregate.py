"""Aggregates multiple MCTS tuning tournaments (multi-seed) into a robust summary."""

import json
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np


def t_value(df: int) -> float:
    try:
        from scipy import stats
        return stats.t.ppf(0.975, df)
    except ImportError:
        # Fallbacks for low df when scipy is unavailable
        if df == 1: return 12.7
        if df == 2: return 4.30
        if df == 3: return 3.18
        if df == 4: return 2.78
        if df >= 5: return 2.5
        return 1.96

class SeedRun:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        with (run_dir / "tuning_config.json").open() as f:
            self.config = json.load(f)
        with (run_dir / "tuning_summary.json").open() as f:
            self.summary = json.load(f)

        self.games = []
        games_path = run_dir / "games.jsonl"
        if games_path.exists():
            with games_path.open() as f:
                for line in f:
                    if line.strip():
                        self.games.append(json.loads(line))

        # Extract basic metrics per agent
        self.win_rates = {}
        self.ranks = {}
        self.scores = {}
        self.p95_times = {}
        self.avg_sims = {}
        for row in self.summary.get("rankings", []):
            name = row["name"]
            self.win_rates[name] = row["win_rate"]
            self.ranks[name] = row["rank_score"]
            self.scores[name] = row["mean_score"]
            self.avg_sims[name] = row["avg_sims"]

        for name, comp in self.summary.get("budget_compliance", {}).items():
            self.p95_times[name] = comp["p95_ms"]

        self.tunings = list(self.win_rates.keys())
        self.base_tuning = self.config["tunings"][0]["name"]

        # Calculate pairwise dominance from games
        self.pairwise_wins = defaultdict(lambda: defaultdict(int))
        self.pairwise_losses = defaultdict(lambda: defaultdict(int))
        self.pairwise_ties = defaultdict(lambda: defaultdict(int))

        for g in self.games:
            seat_assign = g["seat_assignment"] # seat id (str) -> tuning
            final_scores = g["final_scores"] # seat id (str) -> score

            tunings_in_game = list(seat_assign.values())
            for i in range(len(tunings_in_game)):
                for j in range(i+1, len(tunings_in_game)):
                    t1 = tunings_in_game[i]
                    t2 = tunings_in_game[j]

                    # find seats
                    s1 = next(s for s, t in seat_assign.items() if t == t1)
                    s2 = next(s for s, t in seat_assign.items() if t == t2)

                    score1 = final_scores[s1]
                    score2 = final_scores[s2]

                    if score1 > score2:
                        self.pairwise_wins[t1][t2] += 1
                        self.pairwise_losses[t2][t1] += 1
                    elif score2 > score1:
                        self.pairwise_wins[t2][t1] += 1
                        self.pairwise_losses[t1][t2] += 1
                    else:
                        self.pairwise_ties[t1][t2] += 1
                        self.pairwise_ties[t2][t1] += 1

        self.pairwise_wpct = {}
        for t in self.tunings:
            wins = sum(self.pairwise_wins[t].values())
            losses = sum(self.pairwise_losses[t].values())
            ties = sum(self.pairwise_ties[t].values())
            total = wins + losses + ties
            self.pairwise_wpct[t] = (wins + 0.5 * ties) / total if total > 0 else 0.0


def aggregate_seed_runs(meta_dir: Path, base_dir: Path, child_run_ids: List[str]):
    runs = [SeedRun(base_dir / rid) for rid in child_run_ids]
    if not runs:
        raise ValueError("No valid child runs found.")

    n_seeds = len(runs)
    underpowered = n_seeds < 3
    tunings = runs[0].tunings
    base_tuning = runs[0].base_tuning

    # Aggregated metrics
    agg = {}
    for t in tunings:
        agg[t] = {
            "win_rates": [r.win_rates[t] for r in runs],
            "ranks": [r.ranks[t] for r in runs],
            "scores": [r.scores[t] for r in runs],
            "p95_times": [r.p95_times.get(t, 0.0) for r in runs],
            "avg_sims": [r.avg_sims.get(t, 0.0) for r in runs],
            "pairwise_wpct": [r.pairwise_wpct.get(t, 0.0) for r in runs]
        }

    summary = {
        "n_seeds": n_seeds,
        "underpowered": underpowered,
        "base_tuning": base_tuning,
        "tunings_set": runs[0].config.get('tunings_set', 'unknown'),
        "metrics": {}
    }

    tv = t_value(max(1, n_seeds - 1))

    for t in tunings:
        wr_mean = float(np.mean(agg[t]["win_rates"]))
        wr_se = float(np.std(agg[t]["win_rates"], ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0

        # Beats baseline
        if t != base_tuning:
            beats = sum(1 for r in runs if (r.win_rates[t] > r.win_rates[base_tuning]))
        else:
            beats = 0

        wpct_mean = float(np.mean(agg[t]["pairwise_wpct"]))
        dominance = sum(1 for other in tunings if other != t and np.mean(agg[t]["pairwise_wpct"]) > np.mean(agg[other]["pairwise_wpct"]))

        summary["metrics"][t] = {
            "win_rate_mean": wr_mean,
            "win_rate_se": wr_se,
            "win_rate_ci_lower": max(0.0, wr_mean - tv * wr_se),
            "win_rate_ci_upper": min(1.0, wr_mean + tv * wr_se),
            "mean_score": float(np.mean(agg[t]["scores"])),
            "mean_rank": float(np.mean(agg[t]["ranks"])),
            "p95_mean": float(np.mean(agg[t]["p95_times"])),
            "p95_max": float(np.max(agg[t]["p95_times"])),
            "sims_mean": float(np.mean(agg[t]["avg_sims"])),
            "beats_baseline_in_seeds": beats,
            "pairwise_wpct_mean": wpct_mean,
            "dominance_score": dominance
        }

    sorted_tunings = sorted(tunings, key=lambda t: summary["metrics"][t]["win_rate_mean"], reverse=True)

    # Save JSON
    with (meta_dir / "aggregate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Render Markdown
    lines = []
    lines.append(f"# Multi-Seed Aggregate Summary: {summary['tunings_set']}")
    if underpowered:
        lines.append("")
        lines.append("> [!WARNING]")
        lines.append(f"> **Underpowered Test**: Only {n_seeds} seeds executed. CI spreads will be artificially large. Recommend N>=3.")

    lines.append("")
    lines.append("## Overall Robust Rankings")
    lines.append("Sorted by Mean Win Rate across seeds.")

    top = sorted_tunings[:3]
    for i, t in enumerate(sorted_tunings):
        m = summary["metrics"][t]
        lines.append(f"{i+1}. **{t}**")
        lines.append(f"   - **Win Rate:** {m['win_rate_mean']:.3f} ± {m['win_rate_se']:.3f} (95% CI: [{m['win_rate_ci_lower']:.3f}, {m['win_rate_ci_upper']:.3f}])")
        lines.append(f"   - **Pairwise WPCT:** {m['pairwise_wpct_mean']:.3f}")
        if t != base_tuning:
            lines.append(f"   - **Beats Baseline Consistency:** {m['beats_baseline_in_seeds']}/{n_seeds} seeds")
        lines.append(f"   - **Dominance Score:** Beats {m['dominance_score']} other tunings pairwise")
        lines.append(f"   - Mean Score: {m['mean_score']:.1f}")

    lines.append("")
    lines.append("## Time Budget Compliance (Aggregate)")
    lines.append("| Tuning | p95 Mean (ms) | p95 Max (ms) | Avg Sims |")
    lines.append("|---|---|---|---|")
    for t in sorted_tunings:
        m = summary["metrics"][t]
        lines.append(f"| {t} | {m['p95_mean']:.1f} | {m['p95_max']:.1f} | {m['sims_mean']:.1f} |")

    with (meta_dir / "aggregate_summary.md").open("w", encoding="utf-8") as f:
        f.write("\\n".join(lines).strip() + "\\n")
