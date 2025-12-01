# Training Architecture Summary

## Training Entrypoint

**File:** `training/trainer.py`  
**Function:** `main()` (lines 791-800)

The training script is launched via:
```bash
python training/trainer.py [options]
```

The `main()` function:
1. Creates an argument parser using `create_arg_parser()` from `training.config`
2. Parses CLI arguments
3. Converts arguments to a `TrainingConfig` object via `parse_args_to_config()`
4. Calls `train(config)` to start training

**Alternative entrypoints:**
- `training/run_sweep.py` - For hyperparameter sweeps across multiple agent configs
- Direct import: `from training.trainer import train; train(config)`

---

## Core Training Loop

**File:** `training/trainer.py`  
**Function:** `train(config: TrainingConfig)` (lines 382-789)

The `train()` function orchestrates the entire training process:

### High-Level Control Flow

1. **Configuration & Logging Setup** (lines 389-471)
   - Sets logging verbosity based on config
   - Loads agent hyperparameter config (if provided via `--agent-config`)
   - Creates `TrainingRunLogger` for MongoDB integration (optional)
   - Logs reproducibility metadata (seeds, versions, etc.)

2. **Seed Initialization** (lines 473-479)
   - Sets random seeds for reproducibility (Python, NumPy, PyTorch, environment)

3. **Environment Instantiation** (lines 481-494)
   - Creates Blokus environment via `make_gymnasium_env()` from `envs.blokus_v0`
   - Wraps with `ActionMasker` wrapper for action masking support
   - Resets environment with seed if provided

4. **Model Initialization** (lines 511-571)
   - **If resuming:** Loads checkpoint via `load_checkpoint()` from `training.checkpoints`
   - **If new:** Creates `MaskablePPO` model from `sb3_contrib` with:
     - Policy network architecture (from agent config or defaults)
     - Hyperparameters (learning rate, gamma, batch size, etc.)
     - TensorBoard logging directory

5. **Callback Setup** (lines 573-583)
   - Creates `TrainingCallback` instance to track:
     - Episode count, step count
     - Episode rewards and lengths
     - Sanity checks (NaN/Inf detection)
     - Checkpoint saving at intervals

6. **Training Execution** (lines 703-729)
   - Calls `model.learn(total_timesteps=config.total_timesteps, callback=callback)`
   - This is where Stable-Baselines3's internal training loop runs
   - The callback's `_on_step()` method is invoked at each environment step
   - Training stops when:
     - `total_timesteps` reached
     - `max_episodes` limit hit (via callback)
     - Keyboard interrupt
     - Exception (re-raised in smoke mode)

7. **Final Summary & Checkpointing** (lines 731-788)
   - Logs "Training Summary" with statistics (lines 732-743)
   - Saves final model checkpoint
   - Saves training config to YAML file

---

## Training Summary Logging

**Location:** `training/trainer.py`, lines 731-743

The "Training Summary" block is printed in the `train()` function after training completes:

```python
logger.info("Training Summary")
logger.info(f"Total episodes: {callback.episode_count}")
logger.info(f"Total steps: {callback.step_count}")
logger.info(f"Average reward: {np.mean(callback.episode_rewards):.2f}")
logger.info(f"Best episode reward: {np.max(callback.episode_rewards):.2f}")
logger.info(f"Worst episode reward: {np.min(callback.episode_rewards):.2f}")
```

Statistics are collected by the `TrainingCallback` class (lines 179-380), which tracks:
- Episode rewards in `self.episode_rewards` list
- Episode lengths in `self.episode_lengths` list
- Step count in `self.step_count`
- Episode count in `self.episode_count`

---

## Key Dependencies

### Environment Classes
- **`envs.blokus_v0.BlokusEnv`** - PettingZoo AEC environment for Blokus
- **`envs.blokus_v0.GymnasiumBlokusWrapper`** - Wrapper for Gymnasium/Stable-Baselines3 compatibility
- **`envs.blokus_v0.make_gymnasium_env()`** - Factory function to create wrapped environment
- **`sb3_contrib.common.wrappers.ActionMasker`** - Wrapper that extracts action masks for MaskablePPO

### Agent/Algorithm Classes
- **`sb3_contrib.MaskablePPO`** - Proximal Policy Optimization with action masking support
- **`training.agent_config.AgentConfig`** - Hyperparameter configuration loader
- **`training.agent_config.load_agent_config()`** - Loads agent config from YAML/JSON files

### Configuration System
- **`training.config.TrainingConfig`** - Dataclass containing all training parameters
- **`training.config.create_arg_parser()`** - Creates CLI argument parser
- **`training.config.parse_args_to_config()`** - Converts CLI args to TrainingConfig
- **`training.config.TrainingConfig.from_file()`** - Loads config from YAML/JSON files

### Checkpointing System
- **`training.checkpoints.save_checkpoint()`** - Saves model checkpoint with metadata
- **`training.checkpoints.load_checkpoint()`** - Loads checkpoint for resuming
- **`training.checkpoints.get_checkpoint_path()`** - Generates structured checkpoint paths
- **`training.checkpoints.cleanup_old_checkpoints()`** - Removes old checkpoints (keeps last N)

### Logging & Monitoring
- **`training.run_logger.TrainingRunLogger`** - MongoDB logger for training runs
- **`training.run_logger.create_training_run_logger()`** - Factory for creating logger
- **`stable_baselines3.common.logger.configure()`** - Configures TensorBoard logging

### Reproducibility
- **`training.seeds.set_seed()`** - Sets seeds for all random number generators
- **`training.reproducibility.get_reproducibility_metadata()`** - Collects system/environment metadata
- **`training.reproducibility.log_reproducibility_info()`** - Logs reproducibility info

### Game Engine (used by environment)
- **`engine.game.BlokusGame`** - Core game logic
- **`engine.board.Board`** - Board state representation
- **`engine.move_generator.LegalMoveGenerator`** - Generates legal moves
- **`engine.pieces.PieceGenerator`** - Piece shape definitions

---

## Training Episode Sequence Diagram

### One Training Episode Flow

```
1. [train()] Initialize environment
   └─> make_gymnasium_env() creates BlokusEnv
   └─> Wrap with ActionMasker for masking support
   └─> env.reset(seed) → returns initial observation + info

2. [train()] Initialize/load model
   └─> Create MaskablePPO with policy network
   └─> OR load from checkpoint if resuming

3. [train()] Create TrainingCallback
   └─> Initialize tracking variables (episode_count=0, step_count=0, etc.)

4. [train()] Call model.learn(total_timesteps, callback)
   │
   ├─> [SB3 Internal Loop] For each rollout collection:
   │   │
   │   ├─> [SB3] env.reset() → obs, info
   │   │   └─> [BlokusEnv] Reset game state, return observation
   │   │
   │   ├─> [SB3] For each step in rollout (n_steps):
   │   │   │
   │   │   ├─> [SB3] Get action mask via mask_fn(env)
   │   │   │   └─> [mask_fn] Extract legal_action_mask from env.infos[agent_name]
   │   │   │
   │   │   ├─> [SB3] Policy predicts action (MaskablePPO uses masked distribution)
   │   │   │   └─> [MaskablePPO] Samples action from masked categorical distribution
   │   │   │
   │   │   ├─> [SB3] env.step(action) → obs, reward, done, info
   │   │   │   └─> [BlokusEnv] Execute move, update game state
   │   │   │   └─> Calculate reward (dense + sparse)
   │   │   │   └─> Check termination/truncation conditions
   │   │   │
   │   │   ├─> [SB3] Store (obs, action, reward, done, info) in rollout buffer
   │   │   │
   │   │   └─> [TrainingCallback._on_step()] Called by SB3
   │   │       ├─> Increment step_count, current_episode_length
   │   │       ├─> Accumulate current_episode_reward += reward
   │   │       ├─> Run sanity checks (if enabled)
   │   │       ├─> Log step details (if in smoke mode)
   │   │       └─> If done: call _on_episode_end()
   │   │
   │   └─> [SB3] Update policy using collected rollouts
   │       ├─> Compute advantages (GAE)
   │       ├─> Update policy network (PPO clipped objective)
   │       └─> Update value network
   │
   └─> [TrainingCallback._on_episode_end()] When episode completes
       ├─> Increment episode_count
       ├─> Append episode_reward to episode_rewards list
       ├─> Append episode_length to episode_lengths list
       ├─> Log episode to MongoDB (if run_logger available)
       ├─> Save checkpoint (if checkpoint_interval_episodes reached)
       └─> Reset current_episode_reward and current_episode_length

5. [train()] After model.learn() completes
   ├─> Log "Training Summary" with statistics
   ├─> Save final checkpoint
   └─> Save training config to YAML
```

### Key Interactions

- **SB3's `model.learn()`** is the main training loop that:
  - Collects rollouts by interacting with the environment
  - Updates the policy using PPO algorithm
  - Calls the callback at each step

- **`TrainingCallback._on_step()`** is invoked by SB3 at every environment step to:
  - Track metrics
  - Perform sanity checks
  - Detect episode completion
  - Save periodic checkpoints

- **`mask_fn()`** is called by `ActionMasker` wrapper to extract legal action masks for MaskablePPO

- **Environment** (`BlokusEnv`) handles:
  - Game state management
  - Move validation
  - Reward calculation
  - Termination detection

---

## File Structure Summary

```
training/
├── trainer.py              # Main training script (entrypoint + train() function)
├── config.py               # TrainingConfig class + CLI argument parsing
├── checkpoints.py          # Checkpoint save/load utilities
├── agent_config.py         # Agent hyperparameter config loader
├── run_logger.py           # MongoDB training run logger
├── seeds.py                # Seed management for reproducibility
├── reproducibility.py      # System metadata collection
├── run_sweep.py            # Hyperparameter sweep runner
└── evaluate_agent.py       # Agent evaluation script

envs/
└── blokus_v0.py            # Blokus environment (BlokusEnv, GymnasiumBlokusWrapper, make_gymnasium_env)

engine/
├── game.py                 # BlokusGame - core game logic
├── board.py                # Board state representation
├── move_generator.py       # LegalMoveGenerator - move validation
└── pieces.py               # PieceGenerator - piece definitions
```

---

## Configuration Flow

1. **CLI Arguments** → `create_arg_parser()` → `parse_args_to_config()` → `TrainingConfig`
2. **Config File** → `TrainingConfig.from_file()` → `TrainingConfig`
3. **Agent Config** → `load_agent_config()` → `AgentConfig` → Merged into `TrainingConfig`
4. **TrainingConfig** → Used throughout `train()` function to configure:
   - Environment parameters (max_steps_per_episode)
   - Model hyperparameters (learning_rate, batch_size, etc.)
   - Training limits (max_episodes, total_timesteps)
   - Logging settings (verbosity, checkpoint intervals)
   - Reproducibility (seeds)

---

## Notes

- The actual RL training loop (rollout collection + policy updates) is handled internally by Stable-Baselines3's `MaskablePPO.learn()` method
- The `TrainingCallback` provides hooks into SB3's training loop for custom tracking and checkpointing
- Action masking ensures only legal moves are selected by the policy
- The environment is a PettingZoo AEC environment wrapped for Gymnasium compatibility
- Training supports both "smoke" mode (quick verification) and "full" mode (production training)

