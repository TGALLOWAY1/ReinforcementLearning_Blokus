# Move Delta Dashboard

The Move Delta Dashboard is a per-move analysis tool that lets you analyze *why* a move was good or bad by breaking it down into measurable game-state changes.

## Accessing the Dashboard

1. Start or load a game from the Play page.
2. Click the **"Move Impact"** tab in the right panel.
3. Play or watch at least one move — the panel will populate automatically.

---

## How to Navigate

The **Move Selector** at the top of the panel controls which ply is being analysed.

| Control | Effect |
|---|---|
| **← Prev / Next →** | Step one move at a time |
| **Slider** | Jump to any move in the game |
| **Preset dropdown** | Switch scoring weight profile (see below) |
| **Raw / Normalized** toggle | Show raw delta values or z-score normalised values |
| **Agg. Opp / Per Opponent** | Switch diverging bar and radar charts between aggregate and per-player views |
| **🗺 Overlay** | Highlight the cells placed by the selected move on the main board |

---

## Weight Presets

Each preset adjusts how much each metric contributes to the composite **Impact Score**:

| Preset | Favours |
|---|---|
| **Balanced** | Equal mix of frontier expansion, mobility, and blocking |
| **Aggressive Blocking** | Dead-space suppression and piece-lock risk reduction |
| **Expansion** | Frontier size and center control |
| **Late-game** | Mobility maximization and lock-risk mitigation |

Switching presets re-ranks the Top Moves leaderboard in real time.

---

## Charts

### 1. Diverging Bar Chart
Shows `deltaSelf` (your gain) vs `deltaOppTotal` (opponent loss) per metric. Bars extending right = improved your position; left = the opponent benefited.

### 2. Move Impact Waterfall
Breaks the composite **Impact Score** into per-metric contributions, coloured green (positive) or red (negative). The footer shows the final normalized score.

### 3. Radar / Before–After Shape
Two overlapping spider polygons: your player's metric profile **before** the move (dashed) vs **after** (solid). A large expansion of the polygon = strong move.

### 4. Cumulative Timeline
Tracks each player's cumulative impact over the full game (ply on X-axis). The yellow vertical line marks the currently selected move.

### 5. Top Moves Leaderboard
Ranks the winner's (or selected player's) top 10 moves by Impact Score. **Click any row** to jump the move selector to that ply.

### 6. Strategy Mix
Shows what fraction of the selected player's total impact came from each metric category (frontier, mobility, dead space, etc.). Use the **phase tabs** (Opening / Mid-game / End-game) to see how the strategy evolved throughout the game.

### 7. Opponent Suppression
One small area-chart per opponent, showing their Frontier Size, Mobility, and Dead Space trajectory across the game. The yellow line marks the currently selected move so you can see when they were suppressed most.

---

## How to Interpret Move Deltas

**A move was strong if:**
- `deltaSelf.frontierSize` is large and positive — you opened new territory.
- `deltaOppTotal.deadSpace` is large and positive — you densified opponents' dead zones.
- `deltaSelf.mobility` is positive or stable — you maintained piece-play options.
- The **Waterfall** shows mostly green bars.
- The Impact Score (bottom of Waterfall) is in the top quartile.

**A move was weak if:**
- `deltaSelf.frontierSize` is near zero or negative — you closed yourself in.
- `deltaOppTotal.frontierSize` is positive — you opened territory for opponents instead.
- The **Radar chart** shows little change between Before and After.

**Strategy Mix tells you style:**
- Frontier-dominant player → expansion strategy.
- Dead-space dominant → aggressive blocker.
- Mobility-dominant late-game → optimizing for end-score.

---

## Data Flow

```
Engine: BlokusGame.make_move()
  └─ collect_all_player_metrics (before)
  └─ place piece
  └─ collect_all_player_metrics (after)
  └─ compute_move_telemetry_delta()
  └─ store in game_history[n].telemetry
       ├─ deltaSelf, deltaOppTotal, deltaOppByPlayer
       └─ before[], after[] (raw snapshots)

WebAPI /api/games/{id}/history
  └─ includes telemetry dict in each history entry

Frontend gameStore.gameState.game_history
  └─ MoveDeltaPanel reads telemetry per entry
       ├─ DivergingBarChart, RadarDeltaChart, CumulativeTimelineChart
       ├─ MoveImpactWaterfall (via moveImpactScore.ts)
       ├─ TopMovesLeaderboard (ranks all moves by Impact Score)
       ├─ StrategyMixPanel (via strategyMix.ts phase segments)
       └─ OpponentSuppressionMultiples (from after[] snapshots)
```

---

## Files Reference

| Area | File |
|---|---|
| TS types | `frontend/src/types/telemetry.ts` |
| Impact Score | `frontend/src/utils/moveImpactScore.ts` |
| Strategy Mix | `frontend/src/utils/strategyMix.ts` |
| Panel entry | `frontend/src/components/telemetry/MoveDeltaPanel.tsx` |
| Diverging bar | `frontend/src/components/telemetry/charts/DivergingBarChart.tsx` |
| Radar chart | `frontend/src/components/telemetry/charts/RadarDeltaChart.tsx` |
| Timeline | `frontend/src/components/telemetry/charts/CumulativeTimelineChart.tsx` |
| Waterfall | `frontend/src/components/telemetry/charts/MoveImpactWaterfall.tsx` |
| Leaderboard | `frontend/src/components/telemetry/charts/TopMovesLeaderboard.tsx` |
| Opp suppression | `frontend/src/components/telemetry/charts/OpponentSuppressionMultiples.tsx` |
| Strategy Mix UI | `frontend/src/components/telemetry/StrategyMixPanel.tsx` |
| Engine telemetry | `engine/telemetry.py` |
