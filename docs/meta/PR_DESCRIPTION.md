# Dashboard Visual Overhaul and Frontend Metrics Calculation

## Overview
This PR completely overhauls the game analysis dashboard, providing detailed spatial metrics and predictive visualizations based on the current board state and historical game progression. All advanced metrics are now calculated locally on the frontend, ensuring compatibility with previously exported game history JSON files.

## Key Features & Changes

### Frontend Metric Calculations (`dashboardMetrics.ts`)
Calculates heavy spatial data purely on the client side:
- **True Dead-Zones**: Implements an 8-way (Chebyshev) BFS to detect empty cells fully disconnected by pieces from ever being reached by specific players.
- **Influence Maps (Voronoi)**: Computes a true multi-source distance map to show territory dominance and highly contested areas.
- **Win Probability Heuristic**: Real-time evaluation combining board score (pieces placed), spatial influence, and frontier mobility. 

### `AnalysisDashboard.tsx` 3-Column Redesign
Restructured into a responsive layout with cohesive aesthetic styling:
*   **Predictive Line Charts**:
    - Corner Differential (Mobility over time)
    - Mobility (Frontier Size over time)
*   **Spatial Analysis**:
    - Combined side-by-side view for `Frontier` maps and `Endgame Dead-Zones`.
    - Integrated a unified player-color toggle to dynamically analyze spaces globally for any given player.
*   **Game State & Player Status**:
    - Replaced the redundant piece tray view with a compact *Pieces Remaining By Size* table.
    - Added a *Territory Control* pie chart showing current influence distributions.
    - Updated the layout and colors to match the neon palette and deep charcoal background used in the primary game grid. 

### Game State Persistence & UI Refinements
- **Save/Load Functionality**: Added "Save Game" functionality to the main `Play.tsx` header (and Game Over screen) that exports `game_history` as a serialized JSON file.
- **Hydration**: Added a hidden "Load Game from File" input natively into `GameConfigModal.tsx` under custom options and deployment profiles. Uploading a previous JSON file perfectly hydrates the new dashboard data.
- **Resizable Right Panel**: Implemented a draggable col-resize handle between the main game board and the right dashboard/telemetry drawer for customizable workspace real estate.

## Technical Notes
- Removed the old `Live Legal Moves` chart to condense real estate in favor of broader strategic insight tools.
- Ensured all color references use the unified neon theme tokens from `gameConstants.ts`.
