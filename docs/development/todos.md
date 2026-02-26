# Codebase TODO Audit

Last verified: February 2026.

This document centralizes `TODO`, `FIXME`, and `HACK` comments found across the repository, categorized by criticality to help prioritize technical debt and feature completion.

---

## P0 (Critical)
*None found.* No immediate data loss, security risks, or crashing bugs were flagged by TODOs in the source.

---

## P1 (High)
*Issues that affect training accuracy, break major UX flows, or block core features.*

### 1. Proper Win Detection in Training
- **Locations**: Referenced extensively in `docs/engine/win_detection_notes.md`, `docs/vecenv-integration-plan.md`, and `docs/rl_current_state.md`.
- **Context**: The RL environment or training loop is currently hardcoding `win = None`.
- **Reasoning for Priority**: Without proper win state logging, it's impossible to correctly measure agent win-rates in TensorBoard/MongoDB during the RL loop.

### 2. Track Last Move in Game Manager
- **Location**: `webapi/game_manager.py:315` (`last_move=None, # TODO: Track last move`)
- **Context**: Setting the `last_move` attribute on state tracking.
- **Reasoning for Priority**: The RL agents' observation space often includes a "last move info" channel. If this is `None`, the agent loses critical context about the opponent's prior action.

---

## P2 (Medium)
*Incomplete features, UI gaps, or architectural debt that significantly slows work.*

### 1. Training/Evaluation MongoDB Logging
- **Locations**: `docs/mongodb.md:232`, `docs/webapi/README.md`
- **Context**: `### TODO: TrainingRun Logging` and hooking EvaluationRun logging into the execution loop.
- **Reasoning for Priority**: The MongoDB persistence models exist, but the hooks to actually populate them during local/cluster training are missing, requiring manual tracking of runs.

### 2. Research Sidebar Hooks
- **Locations**: `frontend/src/components/ResearchSidebar.tsx:14` (episode start) and `:22` (reset logic).
- **Context**: UI buttons for "Start Episode" and "Reset Env" are currently stubbed.
- **Reasoning for Priority**: Breaks the intended research workflow UI, forcing researchers to use CLI or other tools to control episodes.

---

## P3 (Low)
*Nice-to-haves, code cleanup, and minor automation.*

### 1. Dynamic Version Extraction for Reproducibility
- **Location**: `training/reproducibility.py:110` 
- **Context**: `"code_version": "1.0.0", # TODO: Extract from pyproject.toml or __version__`
- **Reasoning for Priority**: Currently hardcoded. Would be better to dynamically parse `pyproject.toml` to ensure exact run artifacts, but 1.0.0 serves as an acceptable manual placeholder for now.
