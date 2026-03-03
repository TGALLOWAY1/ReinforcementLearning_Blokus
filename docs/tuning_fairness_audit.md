# Tuning Fairness Audit & Reproduction

**Date**: 2026-03-02
**Context**: Investigating why certain MCTS tunings drastically exceed their time budgets during tournament runs, and why certain scheduling combinations result in missing agents (0 games played).

## 1. Time Budget Enforcement Discrepancies

### Mechanism
In `analytics/tournament/arena_runner.py` -> `build_agent`:
- The runner consumes the `--thinking-time-ms` scalar and injects it into agent configuration logic.
- **Python MCTS (`mcts_agent.py`)**: Has two modes. 
  - If `deterministic_time_budget=True`, the time limit is ignored and mathematically converted to an iteration count (defaults to `iterations_per_ms = 10.0`). Given the high overhead of Python evaluations, `10.0 * 200ms = 2000 iterations` can take upwards of 5-10 seconds of true wall-clock time.
  - If `deterministic_time_budget=False`, it uses a standard `while time.time() - start < limit:` loop around the search. However, Python object overhead naturally inflates execution past the theoretical boundary before the condition trips and exits.
- **C++ Native MCTS (`fast_mcts.py`)**: Implements internal high-resolution interrupt timers. It scales much more aggressively and tightly enforces the millisecond cutoff constraint.

### Conclusion
Comparing `MCTSAgent` baselines against evaluated agents (or `fast_mcts` vs standard) creates massive computational differences disguised under the same CLI `thinking_time` argument. **Apples-to-apples fair testing requires strict runtime validation caps and a unified execution backend.**

## 2. Schedule Missingness (0 Games Played Bug)

### Mechanism
The `TournamentScheduler` was initially generating combinations blindly shuffling `N` players into lobby sizes of 4. Certain optimization configurations had a `max(seat_counts) - min(seat_counts) <= 2` validation tolerance, allowing cases where a tuning literally gets shuffled out entirely during 20-game micro-tournaments.

### Conclusion
A strict preprocessing validation must take place checking that `games_scheduled > 0` for all requested tunings, and outputting `matchups.json` *before* the tournament loop begins.

## Next Steps
1. Add strict `move_time_ms_p95` compliance gating in `tuning_stats.py`. Error out if a tuning goes rogue.
2. Force `arena_tuning.py` to write `matchups.json` and validate the 0-game edge cases definitively before kickoff.
3. Add an `--agent-backend` flag to unify testing architectures.
