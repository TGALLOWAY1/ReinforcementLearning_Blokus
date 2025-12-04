# Blokus RL Project - Current State Report

**Generated:** 2025-01-XX  
**Focus:** RL Training Pipeline, Game Engine, Logging & Performance

---

## Table of Contents

1. [RL Training Pipeline](#rl-training-pipeline)
2. [Training Entrypoints](#training-entrypoints)
3. [Game Engine & Move Generator](#game-engine--move-generator)
4. [Logging & Metrics](#logging--metrics)
5. [Performance & Benchmarking](#performance--benchmarking)
6. [Known Issues & TODOs](#known-issues--todos)
7. [Key Files Reference](#key-files-reference)

---

## RL Training Pipeline

### Architecture Overview

The RL training system uses **Stable-Baselines3 (SB3)** with **MaskablePPO** for training agents on the Blokus environment. The architecture follows a standard RL setup:

```
TrainingConfig → EnvFactory → BlokusEnv → ActionMasker → MaskablePPO
     ↓              ↓            ↓            ↓              ↓
  CLI/YAML    Single/VecEnv  PettingZoo   Gymnasium     Training Loop
```

### Components

#### 1. **Training Entry Point**
- **File:** `training/trainer.py`
- **Function:** `train(config: TrainingConfig)`
- **Main entry:** `python -m training.trainer` or via CLI
- **Features:**
  - Supports single-env and vectorized environments (DummyVecEnv, SubprocVecEnv)
  - Action masking via `ActionMasker` wrapper
  - Episode limits, step limits, checkpointing
  - Smoke-test mode for quick verification
  - Resume from checkpoint functionality

#### 2. **Configuration System**
- **File:** `training/config.py`
- **Class:** `TrainingConfig` (dataclass)
- **Features:**
  - YAML/JSON config file support
  - CLI argument parsing
  - Smoke-test vs full training modes
  - Hyperparameter configuration
  - Environment vectorization settings (`num_envs`, `vec_env_type`)

#### 3. **Environment Factory**
- **File:** `training/env_factory.py`
- **Function:** `make_training_env(config, mask_fn)`
- **Features:**
  - Creates single or vectorized environments
  - Handles ActionMasker wrapping at single-env level (not VecEnv level)
  - SubprocVecEnv pickling compatibility
  - Seed management for parallel envs

#### 4. **Environment Implementation**
- **File:** `envs/blokus_v0.py`
- **Class:** `BlokusEnv` (PettingZoo AECEnv)
- **Wrapper:** `GymnasiumBlokusWrapper` (Gymnasium compatibility)
- **Features:**
  - Discrete action space: ~36,400 actions (21 pieces × orientations × 20×20 positions)
  - Multi-channel observations: board (5 channels) + remaining pieces (21) + last move (4) = 30 channels
  - Action masking: `legal_action_mask` in info dict
  - Dense rewards: score delta per move
  - Episode termination: game over or max steps

#### 5. **Agent Configuration**
- **File:** `training/agent_config.py`
- **Class:** `AgentConfig`
- **Features:**
  - Hyperparameter management (learning rate, gamma, network architecture)
  - YAML/JSON config files in `config/agents/`
  - Network architecture specification (MLP layers, activation functions)
  - PPO-specific parameters (clip_range, ent_coef, vf_coef)
  - Sweep variant support for hyperparameter tuning

#### 6. **Training Callback**
- **File:** `training/trainer.py`
- **Class:** `TrainingCallback` (SB3 BaseCallback)
- **Features:**
  - Per-environment episode tracking (supports VecEnv)
  - Episode reward/length logging
  - Periodic checkpointing (configurable interval)
  - Sanity checks (NaN/Inf detection)
  - Detailed logging in smoke-test mode
  - MongoDB integration for episode metrics

### Training Flow

1. **Initialization:**
   - Load config (file or CLI args)
   - Load agent config (optional)
   - Set seeds (reproducibility)
   - Create environment(s)
   - Initialize MaskablePPO model

2. **Training Loop:**
   - SB3's `model.learn()` handles rollouts
   - `TrainingCallback._on_step()` tracks episodes
   - Action masking ensures only legal moves
   - Checkpoints saved periodically

3. **Completion:**
   - Final checkpoint saved
   - Training summary logged
   - MongoDB status updated

### Current Status

✅ **Fully Implemented:**
- Single-env and VecEnv support
- Action masking with MaskablePPO
- Checkpointing and resume
- Agent config system
- Episode/step tracking
- MongoDB logging integration

⚠️ **Partially Implemented:**
- Win detection (TODO: implement proper win detection based on game outcome)
- Multi-agent training (currently single-agent focus on player_0)

---

## Training Entrypoints

### Quick Reference

**Main Script:** `training/trainer.py`

**Basic Commands:**
```bash
# Smoke test (quick verification)
python training/trainer.py --mode smoke

# Full training
python training/trainer.py --mode full --total-timesteps 1000000

# Using config file
python training/trainer.py --config training/config_smoke.yaml
```

### Configuration

Training can be configured via:
1. **CLI arguments** (highest priority)
2. **Config files** (YAML/JSON in `training/` directory)
3. **Agent configs** (hyperparameters in `config/agents/`)

### Key Configuration Options

- **Mode:** `--mode {smoke,full}` - Training mode (smoke for quick tests, full for production)
- **Environments:** `--num-envs N` - Number of parallel environments (default: 1)
- **Timesteps:** `--total-timesteps N` - Total training steps (default: 100000)
- **Checkpoints:** `--checkpoint-interval-episodes N` - Save every N episodes (default: 50)
- **Agent Config:** `--agent-config PATH` - Hyperparameter config file
- **Resume:** `--resume-from-checkpoint PATH` - Resume from saved checkpoint

### Environment Construction

- **Single env** (`num_envs=1`): `ActionMasker(GymnasiumBlokusWrapper(BlokusEnv))`
- **VecEnv** (`num_envs>1`): `DummyVecEnv` or `SubprocVecEnv` containing multiple wrapped envs
- **Seeding:** Controlled via `--seed`, `--env-seed`, `--agent-seed`
- **Logging:** Python logging, TensorBoard (automatic), MongoDB (optional)

### Full Documentation

For complete details on:
- All CLI arguments and defaults
- Config file formats
- Environment construction details
- Debug modes and environment variables
- Example commands and workflows

See: **[Training Entrypoints Guide](training_entrypoints.md)**

---

## Game Engine & Move Generator

### Architecture Overview

The game engine consists of:
- **Board:** 20×20 grid with bitboard representation
- **Move Generator:** Frontier-based + bitboard legality checks
- **Game:** High-level game state management
- **Pieces:** Piece definitions and orientations

### Recent Speed Improvements

#### 1. **Frontier-Based Move Generation** (M6)

**Status:** ✅ Implemented and enabled by default

**What it does:**
- Instead of scanning all 400 board cells, only considers "frontier cells"
- Frontier = empty cells diagonally adjacent to player's pieces (but not orthogonally adjacent)
- Dramatically reduces search space, especially in mid/late game

**Performance:**
- Early game: 2-3× speedup
- Mid-game: 5-7× speedup  
- Late-game: 10-20× speedup

**Configuration:**
- `BLOKUS_USE_FRONTIER_MOVEGEN=1` (default: True)
- Can disable for debugging: `BLOKUS_USE_FRONTIER_MOVEGEN=0`

**Files:**
- `engine/board.py`: Frontier tracking (`get_frontier()`, `update_frontier_after_move()`)
- `engine/move_generator.py`: `_get_legal_moves_frontier()`

#### 2. **Bitboard Legality Checks** (M6)

**Status:** ✅ Implemented and enabled by default

**What it does:**
- Represents board state as bitmasks (400 bits for 20×20 board)
- O(1) overlap checks: `shape_mask & occupied_bits != 0`
- O(1) adjacency checks: `diag_mask & player_bits != 0`
- Replaces O(N) grid scans with bitwise operations

**Performance:**
- Combined with frontier: 10-20× speedup in late game
- Bitboard alone: ~2× speedup over grid-based checks

**Configuration:**
- `BLOKUS_USE_BITBOARD_LEGALITY=1` (default: True)
- Can disable for debugging: `BLOKUS_USE_BITBOARD_LEGALITY=0`

**Files:**
- `engine/bitboard.py`: Bitmask utilities (`coord_to_bit()`, `coords_to_mask()`, `shift_mask()`)
- `engine/board.py`: Bitboard state (`occupied_bits`, `player_bits`)
- `engine/move_generator.py`: `is_placement_legal_bitboard_coords()`

#### 3. **Optimization Techniques**

**Caching:**
- Piece orientations pre-computed and cached
- Piece position lists cached per orientation
- Per-call cache for failed (piece, orientation, frontier) combinations

**Code Optimizations:**
- Direct numpy grid access (avoid Position object creation)
- Inline adjacency checking
- Early exits for bounds/overlap checks
- Hoisted lookups outside loops

**Files:**
- `engine/move_generator.py`: `_cache_piece_orientations()`, cached positions
- `engine/board.py`: `_check_adjacency_rules_fast()` (inline, direct grid access)

### Performance Results

From `PERFORMANCE_OPTIMIZATION_RESULTS.md` and benchmarks:

**Phase 1 Optimizations (pre-frontier/bitboard):**
- Move processing: ~3s → ~500ms (82% improvement)

**M6 Optimizations (frontier + bitboard):**
- Early game: ~50-100ms (naive) → ~10-20ms (5-10× speedup)
- Mid-game: ~500-2000ms (naive) → ~50-150ms (10-20× speedup)
- Late-game: ~2000-5000ms (naive) → ~100-250ms (20× speedup)

### Current Status

✅ **Fully Implemented:**
- Frontier-based generation (default)
- Bitboard legality checks (default)
- Incremental frontier updates
- Piece orientation caching
- Grid-based fallback (for debugging)

⚠️ **Partially Implemented:**
- Heuristic anchor selection (`BLOKUS_USE_HEURISTIC_ANCHORS`, default: False)
  - Has per-orientation fallback to ensure correctness
  - Can provide additional speedup but disabled for safety

---

## Logging & Metrics

### Logging Infrastructure

#### 1. **Python Logging**
- **Standard library:** `logging` module
- **Levels:** ERROR, INFO, DEBUG (configurable via `logging_verbosity`)
- **Format:** Timestamp, logger name, level, message
- **Files:**
  - `training/trainer.py`: Main training logs
  - `envs/blokus_v0.py`: Environment diagnostics
  - `engine/move_generator.py`: Move generation timing (when `BLOKUS_MOVEGEN_DEBUG=1`)

#### 2. **TensorBoard Logging**
- **Directory:** `./logs/` (configurable via `tensorboard_log_dir`)
- **Integration:** Automatic via SB3's `tensorboard_log` parameter
- **Metrics:** Standard SB3 metrics (policy loss, value loss, entropy, etc.)
- **Files:** `logs/PPO_*/events.out.tfevents.*`

#### 3. **MongoDB Logging**
- **File:** `training/run_logger.py`
- **Class:** `TrainingRunLogger`
- **Database:** MongoDB (via `webapi/db/mongo.py`)
- **Models:** `TrainingRun`, `EpisodeMetric`, `RollingWinRate`, `CheckpointPath`
- **Status:** ✅ Implemented, gracefully degrades if MongoDB unavailable

**Logged Data:**
- Training run metadata (config, hyperparameters, start/end time)
- Episode metrics (episode number, total reward, steps, win status)
- Rolling win rate (last 100 episodes)
- Checkpoint paths

**Usage:**
```python
run_logger = create_training_run_logger(config_dict, agent_id, algorithm)
run_logger.log_episode(episode, total_reward, steps, win)
run_logger.log_checkpoint(episode, checkpoint_path)
run_logger.update_status("completed")
```

### Metrics Currently Logged

#### Training Metrics (via TrainingCallback)
- **Episode count:** Total episodes across all envs
- **Episode rewards:** Per-episode total reward
- **Episode lengths:** Steps per episode
- **Step count:** Total training steps
- **Per-env stats:** When using VecEnv, tracks per-environment metrics

#### Environment Metrics (via info dict)
- **Legal moves count:** `info["legal_moves_count"]`
- **Score:** `info["score"]`
- **Pieces used:** `info["pieces_used"]`
- **Pieces remaining:** `info["pieces_remaining"]`
- **Can move:** `info["can_move"]`

#### Move Generation Profiling (optional)
- **Enabled:** `BLOKUS_PROFILE_MOVEGEN=1`
- **Metrics:** Total time, call count, average time, max time
- **Logging:** Every N episodes (default: 10)
- **File:** `envs/blokus_v0.py` (lines 94-100, 160-178)

### Logging Frequency

- **Episode completion:** Every episode (INFO level)
- **Periodic stats:** Every 10 episodes (INFO level)
- **Checkpoints:** Every N episodes (configurable via `checkpoint_interval_episodes`)
- **Detailed step info:** First 3 episodes, first 10 steps (smoke-test mode only)
- **Move generation profiling:** Every 10 episodes (if enabled)

### Current Status

✅ **Fully Implemented:**
- Python logging (standard library)
- TensorBoard integration (SB3 automatic)
- MongoDB training run logging
- Episode metrics tracking
- Checkpoint logging

⚠️ **Partially Implemented:**
- Win detection (currently `win=None` in episode logs)
- Move generation profiling (optional, via env var)

---

## Performance & Benchmarking

### Benchmark Scripts

#### 1. **Move Generation Benchmark**
- **File:** `benchmarks/benchmark_move_generation.py`
- **Purpose:** Compare naive vs frontier vs bitboard move generation
- **Metrics:** Average time (ms), number of moves, frontier size
- **Usage:** `python benchmarks/benchmark_move_generation.py`
- **Results:** Shows speedup for different game states (early, mid, late)

#### 2. **VecEnv Benchmark**
- **File:** `scripts/benchmark_vecenv.py`
- **Purpose:** Compare single-env vs multi-env training speed
- **Metrics:** Wall-clock time, episodes completed
- **Usage:** `python scripts/benchmark_vecenv.py`
- **Results:** Shows speedup from parallel environments

### Performance Monitoring

#### Move Generation Timing
- **Location:** `engine/move_generator.py`
- **Enabled:** `BLOKUS_MOVEGEN_DEBUG=1`
- **Output:** Per-call timing (ms), piece-level timing, frontier size
- **Log level:** INFO (when enabled)

#### Environment Profiling
- **Location:** `envs/blokus_v0.py`
- **Enabled:** `BLOKUS_PROFILE_MOVEGEN=1`
- **Output:** Episode-level summary (total time, calls, avg/max time)
- **Log level:** INFO (when enabled)

### Known Performance Bottlenecks

1. **Move Generation:**
   - Still requires checking thousands of candidate placements
   - Adjacency checking is computationally expensive (40 grid accesses per 5-square piece)
   - Frontier-based generation helps but doesn't eliminate the fundamental complexity

2. **Action Mask Construction:**
   - Must map all legal moves to action space indices
   - Action space is large (~36,400 actions)
   - Mask construction happens every step

3. **Environment Step:**
   - Calls `get_legal_moves()` for all agents every step
   - Updates observations and infos for all agents
   - Multi-agent overhead (currently focused on single-agent training)

### Potential Future Optimizations

1. **Incremental legal move caching:** Cache moves and invalidate on board changes
2. **Spatial indexing:** Use spatial data structures for adjacency checks
3. **Parallel candidate evaluation:** Multiprocessing for move generation
4. **Move validation skip:** Skip redundant validation when move is known legal
5. **Heuristic anchor optimization:** Enable `USE_HEURISTIC_ANCHORS` once validated

---

## Known Issues & TODOs

### High Priority

1. **Win Detection** ✅ **RESOLVED**
   - **Location:** `training/trainer.py:452` (was)
   - **Issue:** `win = None` in episode logs (TODO: implement proper win detection)
   - **Status:** ✅ Implemented in WD-04 - win detection now extracts from terminal step info
   - **Impact:** Rolling win rate calculation now works correctly

2. **Agent ID/Algorithm Hardcoded** ✅ **RESOLVED**
   - **Location:** `training/run_logger.py:50-51` (was)
   - **Issue:** `agent_id` and `algorithm` were hardcoded
   - **Status:** ✅ Resolved - now parameterized via TrainingConfig
   - **Impact:** Agent metadata now configurable via config files

### Medium Priority

3. **Code Version Extraction**
   - **Location:** `training/reproducibility.py:110`
   - **Issue:** Code version hardcoded as "1.0.0"
   - **Impact:** Reproducibility metadata incomplete

4. **Frontend Integration TODOs**
   - **Location:** `frontend/src/components/ResearchSidebar.tsx`
   - **Issue:** Episode start/reset logic not connected
   - **Impact:** Frontend features incomplete

### Low Priority

5. **Game Manager Last Move Tracking**
   - **Location:** `webapi/game_manager.py:314`
   - **Issue:** `last_move=None` (TODO: Track last move)
   - **Impact:** Game state tracking incomplete

6. **MongoDB Evaluation Run Logging**
   - **Location:** `docs/mongodb.md:284`
   - **Issue:** EvaluationRun logging not implemented
   - **Impact:** Evaluation metrics not persisted

### Experimental/Partial Features

1. **Heuristic Anchor Selection**
   - **Status:** Implemented but disabled by default
   - **Reason:** Safety - exact mode ensures correctness
   - **Future:** Enable once validated with fallback

2. **SubprocVecEnv Support**
   - **Status:** Implemented but not extensively tested
   - **Reason:** Requires picklable factory functions (implemented)
   - **Future:** Benchmark and validate performance

---

## Key Files Reference

### RL Training
- `training/trainer.py` - Main training script, MaskablePPO setup, callbacks
- `training/config.py` - Training configuration dataclass and CLI parser
- `training/env_factory.py` - Environment factory (single/VecEnv)
- `training/agent_config.py` - Agent hyperparameter configuration
- `training/run_logger.py` - MongoDB training run logging
- `training/checkpoints.py` - Checkpoint save/load utilities
- `training/reproducibility.py` - Seed management and reproducibility metadata

### Environment
- `envs/blokus_v0.py` - PettingZoo AECEnv implementation, Gymnasium wrapper
- `envs/blokus_env.py` - (Empty placeholder)

### Game Engine
- `engine/board.py` - Board state, frontier tracking, bitboard representation
- `engine/move_generator.py` - Legal move generation (naive, frontier, bitboard)
- `engine/bitboard.py` - Bitmask utilities (coord conversion, shifting)
- `engine/game.py` - High-level game management, scoring
- `engine/pieces.py` - Piece definitions and orientations

### Agents
- `agents/base_agent.py` - (Empty placeholder)
- `agents/random_agent.py` - Random move selection
- `agents/heuristic_agent.py` - Heuristic-based agent
- `agents/mcts_agent.py` - MCTS agent
- `agents/fast_mcts_agent.py` - Optimized MCTS agent

### Benchmarking
- `benchmarks/benchmark_move_generation.py` - Move generation performance comparison
- `scripts/benchmark_vecenv.py` - VecEnv training speed comparison

### Documentation
- `docs/move-generation-optimization.md` - M6 optimization details
- `docs/move-generation-notes.md` - Move generation implementation notes
- `docs/training-architecture.md` - Training system architecture
- `GAME_ENGINE_OPTIMIZATION.md` - Phase 1 optimization results
- `PERFORMANCE_OPTIMIZATION_RESULTS.md` - Performance improvement summary

### Configuration
- `config/agents/` - Agent hyperparameter config files (YAML/JSON)
- `checkpoints/` - Saved model checkpoints
- `logs/` - TensorBoard log files

---

## Summary

### Strengths

1. **Well-structured RL pipeline:** Clean separation of concerns, configurable, supports both single and vectorized environments
2. **Significant performance improvements:** Frontier + bitboard optimizations provide 10-20× speedup in move generation
3. **Comprehensive logging:** Python logging, TensorBoard, MongoDB integration
4. **Reproducibility:** Seed management, checkpointing, config persistence
5. **Action masking:** Properly integrated with MaskablePPO

### Areas for Improvement

1. **Win detection:** Need to implement proper win/loss tracking for episode metrics
2. **Multi-agent training:** Currently focused on single-agent (player_0)
3. **Performance:** Still above 100-150ms target for move generation in some cases
4. **Heuristic optimizations:** Can enable heuristic anchors once validated

### Ready for Training?

✅ **Yes, with caveats:**
- Core training pipeline is functional
- Environment is stable and performant
- Logging and checkpointing work
- Action masking is correct

⚠️ **Before full production training:**
- Implement win detection for proper metrics
- Validate VecEnv performance with multiple environments
- Consider enabling heuristic anchors if speed is still an issue
- Run extended smoke tests to verify stability

---

**End of Report**

