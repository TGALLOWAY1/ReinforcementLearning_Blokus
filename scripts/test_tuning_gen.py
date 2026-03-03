"""Generate tuning stats from existing summary to verify."""

import json
from pathlib import Path
from analytics.tournament.tuning_stats import compute_tuning_summary, render_tuning_summary_markdown
from analytics.tournament.tuning import TuningSet, MctsTuning

def main():
    target = Path("arena_runs/test_det_1/summary.json")
    if not target.exists():
        print("Missing target!")
        return

    with target.open() as f:
        core_summary = json.load(f)

    # The agents in test_det_1 were mcts_25ms, mcts_50ms, mcts_100ms, mcts_200ms
    tunings = [
        MctsTuning("mcts_25ms"),
        MctsTuning("mcts_50ms"),
        MctsTuning("mcts_100ms"),
        MctsTuning("mcts_200ms"),
    ]
    tuning_set = TuningSet("test_ablation", tunings)

    # Hack core_summary to include efficiency for them
    for t in tunings:
        core_summary["time_sim_efficiency"][t.name]["avg_time_ms_per_move"] = 55.0
        core_summary["time_sim_efficiency"][t.name]["avg_simulations_per_move"] = 12.0

    ts = compute_tuning_summary(core_summary, tuning_set, 50)
    md = render_tuning_summary_markdown(ts)
    print(md)

if __name__ == "__main__":
    main()
