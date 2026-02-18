
# Phase Filter Rail
High-resolution desktop UI mockup of a minimal left sidebar filter rail for a Blokus RL “Strategy Analytics” dashboard. Clean modern research-tool aesthetic. White background, light gray cards, charcoal text, cool blue accents. Crisp typography, generous spacing, subtle shadows. No clutter.

Sidebar title: “Phase”
Three large segmented buttons: Opening / Midgame / Endgame. Opening selected (blue highlight).
Under it: small helper text “All metrics and plots update by phase.”
Optional toggle row: “Compare Winners vs Losers” (switch).
Optional toggle row: “Normalize (z-score)” (switch).
No other filters. Keep it minimal and elegant.

# Top Metrics That Drive Winning (Opening)
High-resolution desktop UI mockup of a dashboard section titled “Top Metrics That Drive Winning (Opening)”. Clean modern analytics UI, white/charcoal/cool blue palette, crisp typography, subtle shadows, high clarity.

Panel contents:
- A ranked list of 8 metrics with rows:
  metric name (e.g., Corner Gain, Mobility Delta, Corner Block, Center Gain, Center Distance, Piece Tempo, Compactness, Proximity Risk)
  a small right-aligned sparkline showing trend across turns
  an impact indicator: “+12% win rate” or “-8% win rate”
  a small confidence badge (e.g., “High”, “Med”) and sample size (“n=12,340”)
- Hover tooltip example shown on one row:
  “Corner Gain: +1.8 corners per move in wins; strongest effect in opening by turn 5.”

Make it feel like a polished product module, not a full page. No other panels.

# Win Rate vs Metric (Opening)
High-resolution desktop UI mockup of a single dashboard module titled “Win Rate vs Metric (Opening)”. Clean research-tool analytics look. White background, charcoal text, cool blue accents. Not cluttered.

Module layout:
- Tabs at top for three metrics: Corner Gain / Mobility After / Center Gain (Corner Gain selected).
- Main chart: a smooth binned curve (line) showing win rate on Y-axis and metric value on X-axis.
- A shaded “Recommended Range” band (light blue) labeled “Target Zone”.
- Vertical marker for “Your average” with label.
- Small legend: “bins = 20; n=12,340 games; 95% CI shown as light shading around the curve.”
- Under chart: short callout sentence:
  “In the opening, win rate rises sharply when Corner Gain ≥ 2 by move 5.”

Focus on clarity and instructional feel. No other charts.

# Winners vs Losers (Opening)
High-resolution desktop UI mockup of a module titled “Winners vs Losers (Opening)”. Clean analytics UI, white/charcoal/cool blue palette, crisp typography, subtle shadows.

Show 5 metrics stacked vertically. For each metric, show a compact distribution comparison:
- Either violin plots or ridgeline plots (choose one style and keep consistent).
- Two distributions per metric: Winners (blue) vs Losers (gray).
- Show median lines and an effect label on the right like “Δ +1.6” or “Separation: High”.
Metrics to include: Corner Gain, Mobility After, Corner Block, Center Distance, Avg Turn of Big-5 Pieces.

Include a small note at the bottom: “Phase: Opening | n=12,340 games”.
Make it a standalone panel, not a full page.

# Piece Economy (Opening)
High-resolution desktop UI mockup of a focused module titled “Piece Economy (Opening)”. Modern research-tool aesthetic, white background, charcoal text, cool blue accents.

Split the module into two columns:
Left: “Big Piece Timing”
- Histogram showing distribution of the turn index when the top 5 largest pieces are played.
- Two overlays: Winners vs Losers (winners in blue, losers in gray).
- A small annotation: “Winners play 5-square pieces by turn ~4–6.”

Right: “Endgame Traps”
- Bar chart listing the 6 most frequently unplayed pieces in losses (piece icons + names like P12, P7).
- Each bar shows “% unplayed in losses” with a small comparison dot for winners.

Include small icon strip at the bottom: remaining pieces grouped by size (visual legend).
Keep it clean and instructional.

# Strategy Archetypes
High-resolution desktop UI mockup of a module titled “Strategy Archetypes”. Clean research-tool style, white background, charcoal text, cool blue accents, subtle shadows.

Display 4 archetype cards in a 2x2 grid:
- The Bully
- The Turtle
- Center-Rusher
- Tempo-Heavy

Each card includes:
- 3 “signature metric” badges (e.g., Corner Block ↑, Proximity ↓, MobilityCare ↓)
- 1 “weakness” badge (e.g., “Mobility collapses in midgame”)
- A mini 3-segment phase recipe strip:
  Opening: “target center_gain”
  Midgame: “deny corners”
  Endgame: “protect mobility”
- A “View exemplar games” link and a small “Win rate vs field” number.

No radar charts unless they’re tiny; prioritize readable recipes and badges.


# Corner & Mobility Advantage (Why You Won/Lost)
High-resolution desktop UI mockup of a single analytics module titled “Corner & Mobility Advantage (Why You Won/Lost)”. Clean modern research-tool UI. White background, charcoal text, cool blue accents, subtle shadows, crisp typography, generous spacing. Make the module feel instructional and message-driven.

Replace “two lines for me vs opponents” with two stacked charts that each show ADVANTAGE:
1) Chart A: “Corner Advantage” line = Corners(me) − Corners(opponents sum)
2) Chart B: “Mobility Advantage” line = LegalMoves(me) − LegalMoves(opponents sum)

Both charts include:
- A bold zero baseline labeled “Even”
- Clear y-axis labels like “Advantage (+ / −)”
- Phase shading bands (Opening / Mid / End) as very subtle background blocks
- A highlighted “Danger Zone” band on Mobility Advantage (e.g., below −10) labeled “Danger: you’re getting squeezed”
- A highlighted “Target Zone” band on Corner Advantage (e.g., above +3) labeled “Target: strong growth options”

Add one primary narrative annotation (large callout bubble) anchored to the most critical event:
- Example: “Mobility crash here (Turn 6): you fell into danger zone; win rate typically collapses after this.”
Also show a smaller secondary annotation:
- Example: “Corner advantage plateaued: you stopped generating new anchors.”

Include a vertical cursor at Turn 6 with a compact tooltip showing deltas:
“Turn 6: corner_adv +3 (corner_block 4), mobility_adv −18 (mobility_me_delta −18)”

Footer controls:
- Toggle: “Show averages” (selected) / “Show per-game”
- Small pill tag right side: “Phase: Opening” or “All phases”

Make the charts visually communicate a clear lesson at a glance: advantage rising is good, crossing danger zone is bad.
No clutter, no extra panels.


