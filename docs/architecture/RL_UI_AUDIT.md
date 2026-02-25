# RL/Training UI Layout Audit

## Entry Point
- **Main Page**: `frontend/src/pages/Play.tsx` (not TrainEval.tsx, which is placeholder content)
- **Route**: `/play` (default route `/` also points to Play)

## Component Mapping

### Section → Component → File

| Section | Component | File | Location in Layout |
|---------|-----------|------|-------------------|
| **Environment Controls** | `ResearchSidebar` | `frontend/src/components/ResearchSidebar.tsx` | Left sidebar (w-80, fixed width) |
| **Training Parameters** | `ResearchSidebar` | `frontend/src/components/ResearchSidebar.tsx` | Left sidebar (w-80, fixed width) |
| **Model Status** | `ResearchSidebar` | `frontend/src/components/ResearchSidebar.tsx` | Left sidebar (w-80, fixed width) |
| **Policy** | `AgentVisualizations` (tab: 'policy') | `frontend/src/components/AgentVisualizations.tsx` | Right sidebar (w-96, fixed width) |
| **Value** | `AgentVisualizations` (tab: 'value') | `frontend/src/components/AgentVisualizations.tsx` | Right sidebar (w-96, fixed width) |
| **Tree** | `AgentVisualizations` (tab: 'tree') | `frontend/src/components/AgentVisualizations.tsx` | Right sidebar (w-96, fixed width) |
| **PieceTray** | `AgentVisualizations` (tab: 'pieces') | `frontend/src/components/AgentVisualizations.tsx` | Right sidebar (w-96, fixed width) |
| **PieceTray (standalone)** | `PieceTray` | `frontend/src/components/PieceTray.tsx` | **Not currently used in Play.tsx** |

## Current High-Level Layout

### Layout Structure (Play.tsx)
```
<div className="fixed h-screen w-screen bg-charcoal-900 flex overflow-hidden">
  ├── Left Column (w-80, fixed)
  │   └── ResearchSidebar
  │       ├── Environment Controls (section 1)
  │       ├── Training Parameters (section 2)
  │       └── Model Status (section 3)
  │
  ├── Center Column (flex-1, flexible)
  │   ├── Turn Indicator Bar (top)
  │   ├── Error Display (conditional)
  │   └── Board Container (flex-1, centered)
  │
  └── Right Column (w-96, fixed)
      └── AgentVisualizations
          ├── Tab Navigation (Policy | Value | Tree | Pieces)
          └── Tab Content (scrollable)
              ├── Policy tab: Policy Heatmap
              ├── Value tab: Value Function
              ├── Tree tab: MCTS Tree (placeholder)
              └── Pieces tab: Piece Selection
```

### Layout Primitives
- **Outer container**: Flexbox (`flex`, `overflow-hidden`)
- **Left sidebar**: Fixed width (`w-80` = 320px)
- **Center**: Flexible (`flex-1`)
- **Right sidebar**: Fixed width (`w-96` = 384px)
- **No CSS Grid**: Pure flexbox layout
- **No split-panel library**: Custom flexbox implementation

### Component Details

#### ResearchSidebar (Left Column)
- **File**: `frontend/src/components/ResearchSidebar.tsx`
- **Layout**: Vertical flex column (`flex flex-col space-y-6`)
- **Sections**: Stacked vertically, all always visible
  - Environment Controls: Buttons (Start Episode, Reset, Auto-Step)
  - Training Parameters: Input fields (Learning Rate, Exploration, Discount Factor, Agent Type)
  - Model Status: Stats (Episode, Win Rate) + Sparkline chart
- **No collapsible sections**: All sections always expanded

#### AgentVisualizations (Right Column)
- **File**: `frontend/src/components/AgentVisualizations.tsx`
- **Layout**: Vertical flex column with tabs
- **Tabs**: Policy, Value, Tree, Pieces (horizontal tab bar)
- **Content**: Scrollable (`overflow-y-auto`) tab content area
- **Policy tab**: Board heatmap overlay showing policy probabilities
- **Value tab**: Win probability bar chart
- **Tree tab**: Placeholder ("Tree visualization coming soon")
- **Pieces tab**: List of available pieces with selection

#### PieceTray Component (Unused)
- **File**: `frontend/src/components/PieceTray.tsx`
- **Status**: Exists but not imported/used in Play.tsx
- **Design**: Grid layout (6 columns) with modern tile styling
- **Note**: Piece selection is currently handled via AgentVisualizations 'pieces' tab

## Existing Expand/Collapse Patterns

### Reusable Component: **None Found**
- No reusable Accordion, CollapsibleSection, Panel, or Disclosure component exists

### Inline Collapsible Pattern: **Sidebar.tsx**
- **File**: `frontend/src/components/Sidebar.tsx`
- **Pattern**: Uses local state (`useState`) with conditional rendering
- **Example**:
  ```tsx
  const [isStatusExpanded, setIsStatusExpanded] = useState(false);
  // ... button with onClick to toggle
  {isStatusExpanded && <div>Content</div>}
  ```
- **Note**: This component is not used in Play.tsx (it's a separate component)

### ResearchSidebar: **No Collapsible Sections**
- All three sections (Environment Controls, Training Parameters, Model Status) are always visible
- No expand/collapse functionality currently implemented

## Summary

### Current Layout Description
**Three-column flexbox layout:**
- **Left (320px)**: ResearchSidebar with Environment Controls, Training Parameters, and Model Status stacked vertically (always visible)
- **Center (flexible)**: Game board with turn indicator and error display
- **Right (384px)**: AgentVisualizations with tabbed interface showing Policy, Value, Tree, and Pieces visualizations

### Key Observations
1. **No reusable collapsible component** - would need to implement one
2. **All sections always visible** - no accordion/collapse functionality in ResearchSidebar
3. **Fixed-width sidebars** - left (320px) and right (384px) are fixed, center is flexible
4. **PieceTray component exists but unused** - piece selection handled via AgentVisualizations tabs
5. **Pure flexbox layout** - no CSS Grid or split-panel libraries
6. **TrainEval.tsx is placeholder** - actual RL UI is in Play.tsx

