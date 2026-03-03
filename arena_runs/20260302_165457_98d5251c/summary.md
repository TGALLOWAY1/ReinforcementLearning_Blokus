# Arena Run Summary: 20260302_165457_98d5251c

## Overview
- Seed: `20260301`
- Seat policy: `round_robin`
- Games: `5/5` completed
- Error games: `0`

## Win Rates by Agent
- `mcts_100ms`: win_rate=0.000, win_points=0.00, outright=0, shared=0
- `mcts_200ms`: win_rate=0.200, win_points=1.00, outright=1, shared=0
- `mcts_25ms`: win_rate=0.600, win_points=3.00, outright=3, shared=0
- `mcts_50ms`: win_rate=0.200, win_points=1.00, outright=1, shared=0

## Win Rates by Seat
- `mcts_100ms`: seat0: 0.000 (1 games), seat1: 0.000 (1 games), seat2: 0.000 (2 games), seat3: 0.000 (1 games)
- `mcts_200ms`: seat0: 0.000 (1 games), seat1: 0.000 (1 games), seat2: 1.000 (1 games), seat3: 0.000 (2 games)
- `mcts_25ms`: seat0: 1.000 (2 games), seat1: 1.000 (1 games), seat2: 0.000 (1 games), seat3: 0.000 (1 games)
- `mcts_50ms`: seat0: 0.000 (1 games), seat1: 0.000 (2 games), seat2: 0.000 (1 games), seat3: 1.000 (1 games)

## Score Stats
- `mcts_100ms`: mean=75.4, median=73.0, std=10.091580649234292, p25=71.0, p75=86.0, min=60.0, max=87.0
- `mcts_200ms`: mean=74.8, median=73.0, std=9.368030742904295, p25=66.0, p75=83.0, min=64.0, max=88.0
- `mcts_25ms`: mean=82.8, median=87.0, std=16.33891061240008, p25=80.0, p75=94.0, min=53.0, max=100.0
- `mcts_50ms`: mean=77.6, median=85.0, std=11.146299834474219, p25=64.0, p75=87.0, min=64.0, max=88.0

## Pairwise Matchups
- `mcts_100ms__vs__mcts_200ms`: mcts_100ms>mcts_200ms=3, mcts_200ms>mcts_100ms=2, tie=0 (total=5)
- `mcts_100ms__vs__mcts_25ms`: mcts_100ms>mcts_25ms=2, mcts_25ms>mcts_100ms=3, tie=0 (total=5)
- `mcts_100ms__vs__mcts_50ms`: mcts_100ms>mcts_50ms=2, mcts_50ms>mcts_100ms=3, tie=0 (total=5)
- `mcts_200ms__vs__mcts_25ms`: mcts_200ms>mcts_25ms=2, mcts_25ms>mcts_200ms=3, tie=0 (total=5)
- `mcts_200ms__vs__mcts_50ms`: mcts_200ms>mcts_50ms=2, mcts_50ms>mcts_200ms=3, tie=0 (total=5)
- `mcts_25ms__vs__mcts_50ms`: mcts_25ms>mcts_50ms=4, mcts_50ms>mcts_25ms=1, tie=0 (total=5)

## Time and Simulation Efficiency
- `mcts_100ms`: avg_time_ms=314.9868421052632, avg_sims_per_move=1894.7894736842106, sims_per_sec=6015.455950540958, win_rate_per_sec=0.0, score_per_sec=239.37507832407368
- `mcts_200ms`: avg_time_ms=569.922077922078, avg_sims_per_move=3948.064935064935, sims_per_sec=6927.376720444809, win_rate_per_sec=0.3509251663476438, score_per_sec=131.24601221401878
- `mcts_25ms`: avg_time_ms=41.42857142857143, avg_sims_per_move=470.29761904761904, sims_per_sec=11352.011494252874, win_rate_per_sec=14.482758620689653, score_per_sec=1998.620689655172
- `mcts_50ms`: avg_time_ms=130.35064935064935, avg_sims_per_move=961.077922077922, sims_per_sec=7373.019826641426, win_rate_per_sec=1.534323004881937, score_per_sec=595.3173258941914
