# Arena Run Summary: 20260301_225550_824b529c

## Overview
- Seed: `12345`
- Seat policy: `round_robin`
- Games: `10/10` completed
- Error games: `0`

## Win Rates by Agent
- `mcts_100ms`: win_rate=0.200, win_points=2.00, outright=2, shared=0
- `mcts_200ms`: win_rate=0.000, win_points=0.00, outright=0, shared=0
- `mcts_25ms`: win_rate=0.400, win_points=4.00, outright=4, shared=0
- `mcts_50ms`: win_rate=0.400, win_points=4.00, outright=4, shared=0

## Win Rates by Seat
- `mcts_100ms`: seat0: 0.000 (2 games), seat1: 0.333 (3 games), seat2: 0.000 (3 games), seat3: 0.500 (2 games)
- `mcts_200ms`: seat0: 0.000 (2 games), seat1: 0.000 (2 games), seat2: 0.000 (3 games), seat3: 0.000 (3 games)
- `mcts_25ms`: seat0: 0.667 (3 games), seat1: 0.000 (2 games), seat2: 0.500 (2 games), seat3: 0.333 (3 games)
- `mcts_50ms`: seat0: 0.333 (3 games), seat1: 0.333 (3 games), seat2: 0.500 (2 games), seat3: 0.500 (2 games)

## Score Stats
- `mcts_100ms`: mean=72.3, median=71.5, std=9.839207285142438, p25=63.75, p75=79.75, min=59.0, max=88.0
- `mcts_200ms`: mean=71.9, median=72.5, std=7.647875521999557, p25=66.75, p75=78.75, min=58.0, max=83.0
- `mcts_25ms`: mean=80.7, median=81.0, std=11.9, p25=72.5, p75=89.0, min=60.0, max=100.0
- `mcts_50ms`: mean=83.1, median=83.5, std=5.5578772926361015, p25=81.0, p75=86.75, min=72.0, max=92.0

## Pairwise Matchups
- `mcts_100ms__vs__mcts_200ms`: mcts_100ms>mcts_200ms=5, mcts_200ms>mcts_100ms=4, tie=1 (total=10)
- `mcts_100ms__vs__mcts_25ms`: mcts_100ms>mcts_25ms=3, mcts_25ms>mcts_100ms=6, tie=1 (total=10)
- `mcts_100ms__vs__mcts_50ms`: mcts_100ms>mcts_50ms=3, mcts_50ms>mcts_100ms=7, tie=0 (total=10)
- `mcts_200ms__vs__mcts_25ms`: mcts_200ms>mcts_25ms=2, mcts_25ms>mcts_200ms=7, tie=1 (total=10)
- `mcts_200ms__vs__mcts_50ms`: mcts_200ms>mcts_50ms=1, mcts_50ms>mcts_200ms=9, tie=0 (total=10)
- `mcts_25ms__vs__mcts_50ms`: mcts_25ms>mcts_50ms=5, mcts_50ms>mcts_25ms=4, tie=1 (total=10)

## Time and Simulation Efficiency
- `mcts_100ms`: avg_time_ms=366.2068965517241, avg_sims_per_move=1972.4275862068966, sims_per_sec=5386.1016949152545, win_rate_per_sec=0.5461393596986818, score_per_sec=197.42937853107344
- `mcts_200ms`: avg_time_ms=729.3355704697987, avg_sims_per_move=3892.6442953020132, sims_per_sec=5337.247287684847, win_rate_per_sec=0.0, score_per_sec=98.58287859686577
- `mcts_25ms`: avg_time_ms=48.01840490797546, avg_sims_per_move=487.7546012269939, sims_per_sec=10157.659384182956, win_rate_per_sec=8.3301392615306, score_per_sec=1680.6055960137985
- `mcts_50ms`: avg_time_ms=130.5670731707317, avg_sims_per_move=975.6341463414634, sims_per_sec=7472.283192453183, win_rate_per_sec=3.063559519917807, score_per_sec=636.4544902629243
