# Piece Placement Translation & Dashboard Restructuring

## Overview
This PR addresses crucial bugs in how piece orientations were communicated between the frontend and the Pyodide Web Worker, and restructures the Telemetry Dashboard to better accommodate the predictive charts.

## Changes Included

### 🐛 Bug Fixes: Frontend-Backend Orientation Translation (Web Worker)
*   **The Issue**: The frontend UI cycled through a fixed 8-state index (0-7) for rotations and flips. However, the Blokus engine dynamically deduplicates shapes to optimize generation. This means symmetrical pieces (like the 1-square or a 2x2 square) expected fewer than 8 orientation indices, immediately leading to out-of-bounds errors or incorrect shape matching when a player rotated a piece.
*   **The Fix**: Implemented a bidirectional translation layer in `browser_python/worker_bridge.py`. 
    *   **Inbound**: Translates the frontend's raw 0-7 rotation index to the exact matching `orientation_id` the engine expects based on normalized shape offsets.
    *   **Outbound**: Translates the backend's unique `orientation_id` back to the frontend's 0-7 index for legal move highlights, heatmap overlays, and game history restoration.

### 🎨 Feature: Dashboard Layout Restructuring
*   **Chart Resizing**: Transformed the `FrontierMap` and `DeadZoneMap` cells in `AnalysisDashboard.tsx` into strict squares to match the actual Blokus board aspect ratio.
*   **Layout Adjustments**: Modifed the dashboard layout to place the predictive line charts (Frontier Size and Corner Differential) in a wider 6-column container, providing more horizontal space. The spatial grid components were stacked vertically into a 3-column container alongside them.
*   **Tab Simplification**: Removed the redundant "Events" and "Charts / Dashboard" sub-tabs inside `TelemetryPanel.tsx`. Selecting the main "Dashboard" tab now immediately renders the Analysis Dashboard.
*   **Strategy Documentation**: Created a detailed `strategy_metrics.md` guide documenting the calculated metrics and outlining potential new MCTS agent "personalities" (e.g., Expansionist, Survivor, Aggressor).

## How to Test
1.  Start a fresh Local WebWorker game.
2.  Pick up an asymmetrical piece (like Piece #9 or #10). Rotate (`R`) and Flip (`F`) the piece.
3.  Attempt to place the piece on valid anchor points. Confirm placement succeeds without the previous Warning #9 or #10 console messages, and that the piece visually matches what you placed.
4.  Navigate to the `Dashboard` tab next to `Game` and observe the new, streamlined view with square spatial grides and wider predictive charts.
