# Arena Run Summary: 20260301_225314_8397d051

## Overview
- Seed: `2026`
- Seat policy: `round_robin`
- Games: `5/5` completed
- Error games: `0`

## Win Rates by Agent
- `mcts_100ms`: win_rate=0.400, win_points=2.00, outright=2, shared=0
- `mcts_200ms`: win_rate=0.000, win_points=0.00, outright=0, shared=0
- `mcts_25ms`: win_rate=0.400, win_points=2.00, outright=2, shared=0
- `mcts_50ms`: win_rate=0.200, win_points=1.00, outright=1, shared=0

## Win Rates by Seat
- `mcts_100ms`: seat0: 1.000 (1 games), seat1: 1.000 (1 games), seat2: 0.000 (2 games), seat3: 0.000 (1 games)
- `mcts_200ms`: seat0: 0.000 (1 games), seat1: 0.000 (1 games), seat2: 0.000 (1 games), seat3: 0.000 (2 games)
- `mcts_25ms`: seat0: 0.500 (2 games), seat1: 1.000 (1 games), seat2: 0.000 (1 games), seat3: 0.000 (1 games)
- `mcts_50ms`: seat0: 0.000 (1 games), seat1: 0.500 (2 games), seat2: 0.000 (1 games), seat3: 0.000 (1 games)

## Score Stats
- `mcts_100ms`: mean=71.2, median=68.0, std=7.782030583337488, p25=64.0, p75=80.0, min=63.0, max=81.0
- `mcts_200ms`: mean=66.2, median=66.0, std=1.7204650534085253, p25=65.0, p75=67.0, min=64.0, max=69.0
- `mcts_25ms`: mean=81.6, median=79.0, std=12.86234815265082, p25=74.0, p75=83.0, min=67.0, max=105.0
- `mcts_50ms`: mean=78.4, median=77.0, std=4.127953488110059, p25=76.0, p75=79.0, min=74.0, max=86.0

## Pairwise Matchups
- `mcts_100ms__vs__mcts_200ms`: mcts_100ms>mcts_200ms=3, mcts_200ms>mcts_100ms=2, tie=0 (total=5)
- `mcts_100ms__vs__mcts_25ms`: mcts_100ms>mcts_25ms=2, mcts_25ms>mcts_100ms=3, tie=0 (total=5)
- `mcts_100ms__vs__mcts_50ms`: mcts_100ms>mcts_50ms=2, mcts_50ms>mcts_100ms=3, tie=0 (total=5)
- `mcts_200ms__vs__mcts_25ms`: mcts_200ms>mcts_25ms=0, mcts_25ms>mcts_200ms=5, tie=0 (total=5)
- `mcts_200ms__vs__mcts_50ms`: mcts_200ms>mcts_50ms=0, mcts_50ms>mcts_200ms=5, tie=0 (total=5)
- `mcts_25ms__vs__mcts_50ms`: mcts_25ms>mcts_50ms=2, mcts_50ms>mcts_25ms=3, tie=0 (total=5)

## Time and Simulation Efficiency
- `mcts_100ms`: avg_time_ms=92.45945945945945, avg_sims_per_move=1870.5405405405406, sims_per_sec=20230.926629640457, win_rate_per_sec=4.326220403390822, score_per_sec=770.0672318035663
- `mcts_200ms`: avg_time_ms=167.7, avg_sims_per_move=2496.3571428571427, sims_per_sec=14885.850583525002, win_rate_per_sec=0.0, score_per_sec=394.75253428741803
- `mcts_25ms`: avg_time_ms=29.58108108108108, avg_sims_per_move=538.8108108108108, sims_per_sec=18214.709913202376, win_rate_per_sec=13.522156235724076, score_per_sec=2758.519872087711
- `mcts_50ms`: avg_time_ms=46.80487804878049, avg_sims_per_move=1046.0365853658536, sims_per_sec=22348.879624804584, win_rate_per_sec=4.273058884835852, score_per_sec=1675.0390828556542
