# Arena Run Summary: 20260301_230251_2b489b51

## Overview
- Seed: `2027`
- Seat policy: `round_robin`
- Games: `5/5` completed
- Error games: `0`

## Win Rates by Agent
- `mcts_100ms`: win_rate=0.200, win_points=1.00, outright=1, shared=0
- `mcts_200ms`: win_rate=0.000, win_points=0.00, outright=0, shared=0
- `mcts_25ms`: win_rate=0.600, win_points=3.00, outright=3, shared=0
- `mcts_50ms`: win_rate=0.200, win_points=1.00, outright=1, shared=0

## Win Rates by Seat
- `mcts_100ms`: seat0: 0.000 (1 games), seat1: 0.000 (1 games), seat2: 0.500 (2 games), seat3: 0.000 (1 games)
- `mcts_200ms`: seat0: 0.000 (1 games), seat1: 0.000 (1 games), seat2: 0.000 (1 games), seat3: 0.000 (2 games)
- `mcts_25ms`: seat0: 0.000 (2 games), seat1: 1.000 (1 games), seat2: 1.000 (1 games), seat3: 1.000 (1 games)
- `mcts_50ms`: seat0: 0.000 (1 games), seat1: 0.500 (2 games), seat2: 0.000 (1 games), seat3: 0.000 (1 games)

## Score Stats
- `mcts_100ms`: mean=70.0, median=69.0, std=8.899438184514796, p25=69.0, p75=70.0, min=57.0, max=85.0
- `mcts_200ms`: mean=70.0, median=75.0, std=10.620734437881403, p25=66.0, p75=79.0, min=51.0, max=79.0
- `mcts_25ms`: mean=81.2, median=81.0, std=1.9390719429665317, p25=81.0, p75=82.0, min=78.0, max=84.0
- `mcts_50ms`: mean=72.6, median=73.0, std=6.118823416311342, p25=66.0, p75=76.0, min=66.0, max=82.0

## Pairwise Matchups
- `mcts_100ms__vs__mcts_200ms`: mcts_100ms>mcts_200ms=2, mcts_200ms>mcts_100ms=3, tie=0 (total=5)
- `mcts_100ms__vs__mcts_25ms`: mcts_100ms>mcts_25ms=1, mcts_25ms>mcts_100ms=4, tie=0 (total=5)
- `mcts_100ms__vs__mcts_50ms`: mcts_100ms>mcts_50ms=2, mcts_50ms>mcts_100ms=3, tie=0 (total=5)
- `mcts_200ms__vs__mcts_25ms`: mcts_200ms>mcts_25ms=0, mcts_25ms>mcts_200ms=5, tie=0 (total=5)
- `mcts_200ms__vs__mcts_50ms`: mcts_200ms>mcts_50ms=2, mcts_50ms>mcts_200ms=3, tie=0 (total=5)
- `mcts_25ms__vs__mcts_50ms`: mcts_25ms>mcts_50ms=4, mcts_50ms>mcts_25ms=1, tie=0 (total=5)

## Time and Simulation Efficiency
- `mcts_100ms`: avg_time_ms=360.3380281690141, avg_sims_per_move=1971.8450704225352, sims_per_sec=5472.209193245779, win_rate_per_sec=0.5550343964978112, score_per_sec=194.2620387742339
- `mcts_200ms`: avg_time_ms=713.4857142857143, avg_sims_per_move=4000.0, sims_per_sec=5606.279032516418, win_rate_per_sec=0.0, score_per_sec=98.10988306903732
- `mcts_25ms`: avg_time_ms=53.53333333333333, avg_sims_per_move=500.0, sims_per_sec=9339.975093399751, win_rate_per_sec=11.207970112079702, score_per_sec=1516.8119551681198
- `mcts_50ms`: avg_time_ms=146.88888888888889, avg_sims_per_move=1000.0, sims_per_sec=6807.86686838124, win_rate_per_sec=1.3615733736762483, score_per_sec=494.2511346444781
