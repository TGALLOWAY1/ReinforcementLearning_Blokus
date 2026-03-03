# Arena Run Summary: 20260301_233100_b5d0b952

## Overview
- Seed: `20260301`
- Seat policy: `round_robin`
- Games: `20/20` completed
- Error games: `0`

## Win Rates by Agent
- `mcts_100ms`: win_rate=0.250, win_points=5.00, outright=5, shared=0
- `mcts_200ms`: win_rate=0.250, win_points=5.00, outright=5, shared=0
- `mcts_25ms`: win_rate=0.225, win_points=4.50, outright=4, shared=1
- `mcts_50ms`: win_rate=0.275, win_points=5.50, outright=5, shared=1

## Win Rates by Seat
- `mcts_100ms`: seat0: 0.000 (5 games), seat1: 0.400 (5 games), seat2: 0.000 (5 games), seat3: 0.600 (5 games)
- `mcts_200ms`: seat0: 0.200 (5 games), seat1: 0.600 (5 games), seat2: 0.200 (5 games), seat3: 0.000 (5 games)
- `mcts_25ms`: seat0: 0.500 (5 games), seat1: 0.200 (5 games), seat2: 0.000 (5 games), seat3: 0.200 (5 games)
- `mcts_50ms`: seat0: 0.200 (5 games), seat1: 0.500 (5 games), seat2: 0.000 (5 games), seat3: 0.400 (5 games)

## Score Stats
- `mcts_100ms`: mean=76.65, median=77.0, std=7.302568041449527, p25=71.0, p75=81.25, min=60.0, max=90.0
- `mcts_200ms`: mean=74.95, median=72.0, std=9.45238065251289, p25=66.75, p75=83.25, min=63.0, max=94.0
- `mcts_25ms`: mean=76.6, median=76.0, std=13.188631468048532, p25=67.75, p75=88.25, min=53.0, max=100.0
- `mcts_50ms`: mean=77.35, median=73.5, std=11.921723868635777, p25=67.5, p75=87.0, min=58.0, max=100.0

## Pairwise Matchups
- `mcts_100ms__vs__mcts_200ms`: mcts_100ms>mcts_200ms=12, mcts_200ms>mcts_100ms=8, tie=0 (total=20)
- `mcts_100ms__vs__mcts_25ms`: mcts_100ms>mcts_25ms=12, mcts_25ms>mcts_100ms=8, tie=0 (total=20)
- `mcts_100ms__vs__mcts_50ms`: mcts_100ms>mcts_50ms=9, mcts_50ms>mcts_100ms=11, tie=0 (total=20)
- `mcts_200ms__vs__mcts_25ms`: mcts_200ms>mcts_25ms=11, mcts_25ms>mcts_200ms=9, tie=0 (total=20)
- `mcts_200ms__vs__mcts_50ms`: mcts_200ms>mcts_50ms=9, mcts_50ms>mcts_200ms=11, tie=0 (total=20)
- `mcts_25ms__vs__mcts_50ms`: mcts_25ms>mcts_50ms=10, mcts_50ms>mcts_25ms=9, tie=1 (total=20)

## Time and Simulation Efficiency
- `mcts_100ms`: avg_time_ms=342.95098039215685, avg_sims_per_move=1928.140522875817, sims_per_sec=5622.204434788409, win_rate_per_sec=0.7289671535976674, score_per_sec=223.50132929304485
- `mcts_200ms`: avg_time_ms=702.7892976588629, avg_sims_per_move=3919.752508361204, sims_per_sec=5577.42202594535, win_rate_per_sec=0.35572539427222627, score_per_sec=106.64647320281344
- `mcts_25ms`: avg_time_ms=48.28064516129032, avg_sims_per_move=487.1225806451613, sims_per_sec=10089.396672679895, win_rate_per_sec=4.660252555622369, score_per_sec=1586.5570922696597
- `mcts_50ms`: avg_time_ms=139.08469055374593, avg_sims_per_move=980.4755700325733, sims_per_sec=7049.485936438793, win_rate_per_sec=1.9772125810908923, score_per_sec=556.1359750813837
