# End-Game and Block Risk Metrics

This document outlines heuristics used beyond the immediate frontier and territory mapping—focusing specifically on preventing self-inflicted damage to optionality, and measuring end-game lock risk.

## Metrics Exposed

### Piece_Lock_Risk
`PieceLockRisk` is a single integer value per player representing the number of their *remaining*, *unplayed* pieces that currently have **zero valid placements on the board**. 
*   **Interpretation**: As the game progresses, the board fills up, and larger pieces become completely impossible to place. A rising `PieceLockRisk` indicates a player is being squeezed out of the game. A zero risk means all unplaced pieces still have at least one valid spot (though not necessarily an optimal one).
*   **Computation**: Calculated cheaply during standard legal move generation by simply checking which pieces do not appear in any valid move's `piece_id`.

```json
"piece_lock_risk": int
``` 

### Self_Block_Risk (Top Moves)
`SelfBlockRisk` evaluates each immediate legal move for how likely it is to "collapse your own future options." It provides a heuristic estimate without requiring an expensive apply/undo forward simulation.

*   **Interpretation**: Moves with high risk scores utilize multiple frontier redundancy clusters at once. This implies they 'bridge' or consume multiple distinct avenues of expansion, shutting down optionality.
*   **Computation**: 
    1.  Determine which specific frontier points the move connects to.
    2.  Lookup the cluster IDs for those points (from the redundancy clustering graph).
    3.  Compute `Risk = 2*clusters_touched + 1*frontier_points_used`.
*   **Telemetry Structure**: Returns the top 10 riskiest moves descending.

```json
"self_block_risk": {
  "top_moves": [
    {
      "piece_id": int,
      "orientation": int,
      "anchor_row": int,
      "anchor_col": int,
      "risk": int,
      "clusters_touched": int,
      "frontier_points_used": int
    },
    ...
  ]
}
```
