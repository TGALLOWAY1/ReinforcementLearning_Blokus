# Frontier Usefulness Metrics

This document defines the new frontier-point metrics available in the Blokus engine telemetry. These metrics help identify the most critical expansion points for a player.

## Metrics

### Utility(f)
The number of current-player legal moves that “use” frontier point `f`.
*   **Interpretation**: High utility means many of your possible moves depend on this specific frontier point. Losing this point would significantly restrict your options.
*   **Computation**: When iterating the current player's legal moves, `Utility[f]` is incremented if the move's placed piece connects diagonally to the frontier point `f`.

### BlockPressure(f)
The number of opponent legal moves (in the next 1-ply) that would occupy or deny the frontier point `f`.
*   **Interpretation**: High block pressure means opponents are actively threatening this frontier point. They have multiple moves that would place a piece on this exact cell, stealing it or blocking your expansion.
*   **Computation**: Computed by generating opponent legal moves and marking any frontier cell `f` that an opponent's move would occupy. 

### Urgency(f)
A combined metric highlighting frontier points that are both highly useful and highly threatened.
*   **Formula**: `Urgency(f) = Utility(f) * (1 + BlockPressure(f))`
*   **Interpretation**: 
    *   If `Utility` is 0, the point is useless right now, so `Urgency` is 0.
    *   If `BlockPressure` is 0, the point is safe, so `Urgency` equals `Utility`.
    *   If `BlockPressure` > 0, the `Urgency` multiplies the `Utility`, highlighting critical contested points that need immediate attention.

## Usage
These metrics are calculated during `get_state()` in the WebWorker and are returned under the `frontier_metrics` key in the JSON payload. They are designed for UI visualization or heuristic evaluation without requiring deep MCTS rollouts.
