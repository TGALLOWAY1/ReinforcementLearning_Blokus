# Arena Run Summary: 20260301_230108_fb33677b

## Overview
- Seed: `2026`
- Seat policy: `round_robin`
- Games: `5/5` completed
- Error games: `0`

## Win Rates by Agent
- `mcts_100ms`: win_rate=0.200, win_points=1.00, outright=1, shared=0
- `mcts_200ms`: win_rate=0.400, win_points=2.00, outright=2, shared=0
- `mcts_25ms`: win_rate=0.000, win_points=0.00, outright=0, shared=0
- `mcts_50ms`: win_rate=0.400, win_points=2.00, outright=2, shared=0

## Win Rates by Seat
- `mcts_100ms`: seat0: 0.000 (1 games), seat1: 0.000 (1 games), seat2: 0.500 (2 games), seat3: 0.000 (1 games)
- `mcts_200ms`: seat0: 1.000 (1 games), seat1: 1.000 (1 games), seat2: 0.000 (1 games), seat3: 0.000 (2 games)
- `mcts_25ms`: seat0: 0.000 (2 games), seat1: 0.000 (1 games), seat2: 0.000 (1 games), seat3: 0.000 (1 games)
- `mcts_50ms`: seat0: 1.000 (1 games), seat1: 0.500 (2 games), seat2: 0.000 (1 games), seat3: 0.000 (1 games)

## Score Stats
- `mcts_100ms`: mean=72.6, median=73.0, std=8.260750571225353, p25=65.0, p75=76.0, min=63.0, max=86.0
- `mcts_200ms`: mean=67.8, median=65.0, std=12.08966500776593, p25=60.0, p75=80.0, min=51.0, max=83.0
- `mcts_25ms`: mean=66.8, median=70.0, std=7.626270385975047, p25=63.0, p75=71.0, min=54.0, max=76.0
- `mcts_50ms`: mean=76.0, median=79.0, std=7.3484692283495345, p25=76.0, p75=80.0, min=62.0, max=83.0

## Pairwise Matchups
- `mcts_100ms__vs__mcts_200ms`: mcts_100ms>mcts_200ms=2, mcts_200ms>mcts_100ms=2, tie=1 (total=5)
- `mcts_100ms__vs__mcts_25ms`: mcts_100ms>mcts_25ms=3, mcts_25ms>mcts_100ms=2, tie=0 (total=5)
- `mcts_100ms__vs__mcts_50ms`: mcts_100ms>mcts_50ms=2, mcts_50ms>mcts_100ms=3, tie=0 (total=5)
- `mcts_200ms__vs__mcts_25ms`: mcts_200ms>mcts_25ms=3, mcts_25ms>mcts_200ms=2, tie=0 (total=5)
- `mcts_200ms__vs__mcts_50ms`: mcts_200ms>mcts_50ms=2, mcts_50ms>mcts_200ms=3, tie=0 (total=5)
- `mcts_25ms__vs__mcts_50ms`: mcts_25ms>mcts_50ms=0, mcts_50ms>mcts_25ms=5, tie=0 (total=5)

## Time and Simulation Efficiency
- `mcts_100ms`: avg_time_ms=341.0, avg_sims_per_move=2000.0, sims_per_sec=5865.102639296188, win_rate_per_sec=0.5865102639296188, score_per_sec=212.9032258064516
- `mcts_200ms`: avg_time_ms=657.4264705882352, avg_sims_per_move=4000.0, sims_per_sec=6084.33061178839, win_rate_per_sec=0.6084330611788391, score_per_sec=103.12940386981323
- `mcts_25ms`: avg_time_ms=53.25, avg_sims_per_move=492.6617647058824, sims_per_sec=9251.864125932063, win_rate_per_sec=0.0, score_per_sec=1254.4600938967135
- `mcts_50ms`: avg_time_ms=153.46478873239437, avg_sims_per_move=1000.0, sims_per_sec=6516.152716593245, win_rate_per_sec=2.6064610866372986, score_per_sec=495.2276064610867
