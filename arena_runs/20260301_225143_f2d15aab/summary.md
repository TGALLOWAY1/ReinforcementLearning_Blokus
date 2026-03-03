# Arena Run Summary: 20260301_225143_f2d15aab

## Overview
- Seed: `12345`
- Seat policy: `round_robin`
- Games: `10/10` completed
- Error games: `0`

## Win Rates by Agent
- `mcts_100ms`: win_rate=0.050, win_points=0.50, outright=0, shared=1
- `mcts_200ms`: win_rate=0.100, win_points=1.00, outright=1, shared=0
- `mcts_25ms`: win_rate=0.700, win_points=7.00, outright=6, shared=2
- `mcts_50ms`: win_rate=0.150, win_points=1.50, outright=1, shared=1

## Win Rates by Seat
- `mcts_100ms`: seat0: 0.250 (2 games), seat1: 0.000 (3 games), seat2: 0.000 (3 games), seat3: 0.000 (2 games)
- `mcts_200ms`: seat0: 0.500 (2 games), seat1: 0.000 (2 games), seat2: 0.000 (3 games), seat3: 0.000 (3 games)
- `mcts_25ms`: seat0: 0.833 (3 games), seat1: 0.500 (2 games), seat2: 0.750 (2 games), seat3: 0.667 (3 games)
- `mcts_50ms`: seat0: 0.333 (3 games), seat1: 0.167 (3 games), seat2: 0.000 (2 games), seat3: 0.000 (2 games)

## Score Stats
- `mcts_100ms`: mean=72.8, median=74.0, std=7.3999999999999995, p25=68.75, p75=76.75, min=60.0, max=84.0
- `mcts_200ms`: mean=73.7, median=71.5, std=7.523961722390671, p25=68.5, p75=74.75, min=67.0, max=94.0
- `mcts_25ms`: mean=87.7, median=88.5, std=5.933801479658718, p25=83.25, p75=92.75, min=77.0, max=96.0
- `mcts_50ms`: mean=75.0, median=75.5, std=8.660254037844387, p25=68.5, p75=80.25, min=63.0, max=88.0

## Pairwise Matchups
- `mcts_100ms__vs__mcts_200ms`: mcts_100ms>mcts_200ms=3, mcts_200ms>mcts_100ms=6, tie=1 (total=10)
- `mcts_100ms__vs__mcts_25ms`: mcts_100ms>mcts_25ms=0, mcts_25ms>mcts_100ms=9, tie=1 (total=10)
- `mcts_100ms__vs__mcts_50ms`: mcts_100ms>mcts_50ms=4, mcts_50ms>mcts_100ms=6, tie=0 (total=10)
- `mcts_200ms__vs__mcts_25ms`: mcts_200ms>mcts_25ms=1, mcts_25ms>mcts_200ms=9, tie=0 (total=10)
- `mcts_200ms__vs__mcts_50ms`: mcts_200ms>mcts_50ms=4, mcts_50ms>mcts_200ms=6, tie=0 (total=10)
- `mcts_25ms__vs__mcts_50ms`: mcts_25ms>mcts_50ms=8, mcts_50ms>mcts_25ms=1, tie=1 (total=10)

## Time and Simulation Efficiency
- `mcts_100ms`: avg_time_ms=88.32, avg_sims_per_move=1937.8466666666666, sims_per_sec=21941.198671497586, win_rate_per_sec=0.5661231884057971, score_per_sec=824.2753623188406
- `mcts_200ms`: avg_time_ms=163.88079470198676, avg_sims_per_move=2526.456953642384, sims_per_sec=15416.43093833347, win_rate_per_sec=0.6101996282227431, score_per_sec=449.7171260001616
- `mcts_25ms`: avg_time_ms=27.355029585798817, avg_sims_per_move=579.5917159763313, sims_per_sec=21187.756867834738, win_rate_per_sec=25.589444083928182, score_per_sec=3205.991780229288
- `mcts_50ms`: avg_time_ms=47.416666666666664, avg_sims_per_move=1221.3846153846155, sims_per_sec=25758.550763823172, win_rate_per_sec=3.163444639718805, score_per_sec=1581.7223198594027
