# PR Description: Dashboard Restoration & Repo Maintenance

This pull request encapsulates two major phases of work tailored to improving developer velocity, reducing technical debt, and synchronizing project documentation with the current repository state.

## 🚀 Phase 1: Dashboard Restoration & Game Persistence

**Goal**: Restore the visibility and functionality of the analysis charts on the main game screen, and implement a save/load game feature for easier development iteration.

### Key Changes
- **Analysis Charts Restoration**: Extracted individual chart components (`CornerChart`, `FrontierChart`, `LegalMovesBarChart`) and integrated them directly into `RightPanel.tsx`. Charts now render dynamically alongside the board from the very first turn.
- **Engine Bug Fixes**: Corrected a bug in `engine/game.py` where `corner_count` mistakenly returned the frontier size rather than placement corners, fixing `AttributeError` tracebacks during validation.
- **Save/Load Game**: Added a new persistence capability to the development workflow.
  - Implemented a `saveGame` and `loadGame` method in `gameStore.ts`.
  - Added message passing for `load_game` in the Pyodide `blokusWorker.ts` and `worker_bridge.py` allowing instant parsing and recreation of a historical game timeline.
  - Added UI hooks in `ResearchSidebar.tsx`.

## 🧹 Phase 2: Repo Maintenance & Documentation Sync

**Goal**: Synchronize documentation with the project state, consolidate markdown into `/docs`, analyze warnings, and formalize tech debt priorities.

### Key Changes
- **Docs Refresh & Consolidation**: Audited all ~40 Markdown files (`frontend/README.md`, `webapi/README.md`, etc.), moving them into the structured `/docs/` directory. Updated root `README.md` to function as an index with correct internal paths.
- **Comprehensive `.gitignore`**: Added a thorough root-level `.gitignore` catching system artifacts, node modules, Python bytecode, and local environment files.
- **Warnings Cleanup**: 
  - Found and fixed React structural warnings (conditional `useMemo` hooks, unused variables) in `Play.tsx`, `gameStore.ts`, and `LegalMovesBarChart.tsx`.
  - Ran `ruff` to identify and auto-format ~300 mechanical lints in the backend (unused imports, typing deprecations, `dict`/`list` generic refactors).
  - Documented intentional unaddressed typings (`any` interfaces in TypeScript) in `/docs/development/warnings.md`.
- **TODO Audit**: Searched the entire codebase for scattered `TODO`, `FIXME`, and `HACK` directives, aggregating them into a single, prioritized tech-debt tracking document at `/docs/development/todos.md`. 

---

### Verification
- **Automated Tests**: TypeScript Compilation (`tsc --noEmit`) passes. Frontend linting verifies strictly. `ruff check .` confirms all obvious Python structural errors are squashed.
- **Manual Verification**: Chart data reacts instantly to slider state shifts and live play. "Save Game" exports validated JSON. "Load Game" rehydrates full game timelines flawlessly via Pyodide.
