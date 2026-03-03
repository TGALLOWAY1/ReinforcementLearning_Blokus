# Arena Run Summary: 20260301_225351_8397d051

## Overview
- Seed: `2026`
- Seat policy: `round_robin`
- Games: `5/5` completed
- Error games: `0`

## Win Rates by Agent
- `mcts_100ms`: win_rate=0.000, win_points=0.00, outright=0, shared=0
- `mcts_200ms`: win_rate=0.200, win_points=1.00, outright=1, shared=0
- `mcts_25ms`: win_rate=0.300, win_points=1.50, outright=1, shared=1
- `mcts_50ms`: win_rate=0.500, win_points=2.50, outright=2, shared=1

## Win Rates by Seat
- `mcts_100ms`: seat0: 0.000 (1 games), seat1: 0.000 (1 games), seat2: 0.000 (2 games), seat3: 0.000 (1 games)
- `mcts_200ms`: seat0: 0.000 (1 games), seat1: 0.000 (1 games), seat2: 0.000 (1 games), seat3: 0.500 (2 games)
- `mcts_25ms`: seat0: 0.000 (2 games), seat1: 1.000 (1 games), seat2: 0.000 (1 games), seat3: 0.500 (1 games)
- `mcts_50ms`: seat0: 0.500 (1 games), seat1: 0.500 (2 games), seat2: 0.000 (1 games), seat3: 1.000 (1 games)

## Score Stats
- `mcts_100ms`: mean=70.0, median=67.0, std=5.621387729022079, p25=66.0, p75=74.0, min=64.0, max=79.0
- `mcts_200ms`: mean=78.0, median=78.0, std=9.695359714832659, p25=69.0, p75=85.0, min=66.0, max=92.0
- `mcts_25ms`: mean=82.0, median=78.0, std=10.658330075579382, p25=76.0, p75=90.0, min=68.0, max=98.0
- `mcts_50ms`: mean=79.4, median=79.0, std=4.223742416388575, p25=78.0, p75=81.0, min=73.0, max=86.0

## Pairwise Matchups
- `mcts_100ms__vs__mcts_200ms`: mcts_100ms>mcts_200ms=1, mcts_200ms>mcts_100ms=4, tie=0 (total=5)
- `mcts_100ms__vs__mcts_25ms`: mcts_100ms>mcts_25ms=2, mcts_25ms>mcts_100ms=3, tie=0 (total=5)
- `mcts_100ms__vs__mcts_50ms`: mcts_100ms>mcts_50ms=0, mcts_50ms>mcts_100ms=5, tie=0 (total=5)
- `mcts_200ms__vs__mcts_25ms`: mcts_200ms>mcts_25ms=2, mcts_25ms>mcts_200ms=3, tie=0 (total=5)
- `mcts_200ms__vs__mcts_50ms`: mcts_200ms>mcts_50ms=2, mcts_50ms>mcts_200ms=3, tie=0 (total=5)
- `mcts_25ms__vs__mcts_50ms`: mcts_25ms>mcts_50ms=2, mcts_50ms>mcts_25ms=2, tie=1 (total=5)

## Time and Simulation Efficiency
- `mcts_100ms`: avg_time_ms=91.93055555555556, avg_sims_per_move=1850.0, sims_per_sec=20123.88578335096, win_rate_per_sec=0.0, score_per_sec=761.4443269376038
- `mcts_200ms`: avg_time_ms=169.20253164556962, avg_sims_per_move=2528.772151898734, sims_per_sec=14945.238273359766, win_rate_per_sec=1.1820154110870054, score_per_sec=460.9860103239321
- `mcts_25ms`: avg_time_ms=26.180722891566266, avg_sims_per_move=671.8915662650602, sims_per_sec=25663.598711458813, win_rate_per_sec=11.45881270133456, score_per_sec=3132.0754716981132
- `mcts_50ms`: avg_time_ms=47.44871794871795, avg_sims_per_move=1212.8333333333333, sims_per_sec=25560.92947851932, win_rate_per_sec=10.537692515536342, score_per_sec=1673.3855714671713
