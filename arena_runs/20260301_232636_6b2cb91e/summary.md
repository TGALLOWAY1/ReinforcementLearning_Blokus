# Arena Run Summary: 20260301_232636_6b2cb91e

## Overview
- Seed: `20260301`
- Seat policy: `round_robin`
- Games: `100/100` completed
- Error games: `0`

## Win Rates by Agent
- `mcts_100ms`: win_rate=0.230, win_points=23.00, outright=22, shared=2
- `mcts_200ms`: win_rate=0.165, win_points=16.50, outright=16, shared=1
- `mcts_25ms`: win_rate=0.255, win_points=25.50, outright=24, shared=3
- `mcts_50ms`: win_rate=0.350, win_points=35.00, outright=34, shared=2

## Win Rates by Seat
- `mcts_100ms`: seat0: 0.100 (25 games), seat1: 0.340 (25 games), seat2: 0.160 (25 games), seat3: 0.320 (25 games)
- `mcts_200ms`: seat0: 0.120 (25 games), seat1: 0.340 (25 games), seat2: 0.200 (25 games), seat3: 0.000 (25 games)
- `mcts_25ms`: seat0: 0.460 (25 games), seat1: 0.220 (25 games), seat2: 0.200 (25 games), seat3: 0.140 (25 games)
- `mcts_50ms`: seat0: 0.320 (25 games), seat1: 0.380 (25 games), seat2: 0.340 (25 games), seat3: 0.360 (25 games)

## Score Stats
- `mcts_100ms`: mean=76.45, median=77.0, std=9.96130011594872, p25=69.0, p75=83.0, min=56.0, max=98.0
- `mcts_200ms`: mean=72.33, median=71.5, std=11.643929749015149, p25=64.0, p75=83.0, min=41.0, max=94.0
- `mcts_25ms`: mean=77.58, median=78.0, std=11.02558841967176, p25=71.0, p75=85.0, min=52.0, max=100.0
- `mcts_50ms`: mean=80.09, median=80.0, std=10.93443642809267, p25=73.0, p75=88.0, min=54.0, max=101.0

## Pairwise Matchups
- `mcts_100ms__vs__mcts_200ms`: mcts_100ms>mcts_200ms=59, mcts_200ms>mcts_100ms=38, tie=3 (total=100)
- `mcts_100ms__vs__mcts_25ms`: mcts_100ms>mcts_25ms=46, mcts_25ms>mcts_100ms=52, tie=2 (total=100)
- `mcts_100ms__vs__mcts_50ms`: mcts_100ms>mcts_50ms=40, mcts_50ms>mcts_100ms=59, tie=1 (total=100)
- `mcts_200ms__vs__mcts_25ms`: mcts_200ms>mcts_25ms=37, mcts_25ms>mcts_200ms=60, tie=3 (total=100)
- `mcts_200ms__vs__mcts_50ms`: mcts_200ms>mcts_50ms=30, mcts_50ms>mcts_200ms=68, tie=2 (total=100)
- `mcts_25ms__vs__mcts_50ms`: mcts_25ms>mcts_50ms=43, mcts_50ms>mcts_25ms=54, tie=3 (total=100)

## Time and Simulation Efficiency
- `mcts_100ms`: avg_time_ms=342.4123376623377, avg_sims_per_move=1950.6740259740259, sims_per_sec=5696.856717521784, win_rate_per_sec=0.6717047684970084, score_per_sec=223.2688241373752
- `mcts_200ms`: avg_time_ms=738.3372093023256, avg_sims_per_move=3901.529411764706, sims_per_sec=5284.210740850192, win_rate_per_sec=0.2234751248090461, score_per_sec=97.96336834811093
- `mcts_25ms`: avg_time_ms=49.709884467265724, avg_sims_per_move=492.63350449293966, sims_per_sec=9910.171986365045, win_rate_per_sec=5.1297644871397585, score_per_sec=1560.6554074992252
- `mcts_50ms`: avg_time_ms=145.30486284289276, avg_sims_per_move=980.6926433915212, sims_per_sec=6749.20731628831, win_rate_per_sec=2.408728745564618, score_per_sec=551.1859578064865
