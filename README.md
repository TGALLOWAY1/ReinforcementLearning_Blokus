# Reinforcement Learning Environment for Blokus

A comprehensive reinforcement learning research environment for the Blokus board game, featuring a complete game engine, multiple AI agents, web interface, and PettingZoo/Gymnasium compatibility.

<img width="1795" height="865" alt="image" src="https://github.com/user-attachments/assets/751e771f-ce00-45b8-8289-6086f760cd7d" />

## 🎯 Overview

This project provides a full-stack implementation of Blokus with:
- **Complete game engine** implementing official Blokus rules
- **Multiple AI agents** (Random, Heuristic, MCTS)
- **Reinforcement learning environment** compatible with PettingZoo and Gymnasium
- **Web interface** for interactive gameplay
- **REST API** for programmatic access and research tooling
- **Arena system** for agent evaluation and tournaments

## ✨ Features

### Game Engine
- Full Blokus rule implementation (20x20 board, 4 players, 21 pieces per player)
- Legal move generation with optimized caching (frontier-based + bitboard)
- Scoring system with bonuses (corner control, center control, piece completion)
- Game state management and history tracking
- **Move Generation**: For details on the optimized move generation system (frontier-based generation, bitboard legality, performance optimizations), see [docs/engine/move-generation-optimization.md](docs/engine/move-generation-optimization.md)

### AI Agents
- **Random Agent**: Baseline agent making random legal moves
- **Heuristic Agent**: Rule-based agent with configurable weights
- **MCTS Agent**: Monte Carlo Tree Search with transposition tables
- **Fast MCTS Agent**: Optimized MCTS for real-time gameplay

<img width="712" height="213" alt="image" src="https://github.com/user-attachments/assets/32be3357-c4cf-4b89-8954-90f6c6a8b075" />

### RL Environment
- **PettingZoo AEC** environment for multi-agent RL
- **Gymnasium compatibility** for single-agent training
- Action masking for legal moves only
- Multi-channel observations (board state, remaining pieces, last move)
- Dense and sparse reward signals

### Web Interface
- React + TypeScript frontend
- Real-time game visualization with SVG
- Interactive piece placement (drag, rotate, flip)
- In-browser Pyodide worker for local gameplay and fast iteration
- Color-blind friendly design
- Agent visualization and research tools

### API & Backend
- FastAPI REST API
- Automatic turn management for AI agents via `advance_turn`
- Human pass/move support via REST endpoints
- Game state persistence and management

### Evaluation Tools
- Arena system for round-robin tournaments
- Agent performance statistics
- Match result logging and analysis
- Snapshot dataset export (`snapshots.parquet` / `snapshots.csv`) for ML modeling
- Win-probability training scripts:
  - `scripts/train_winprob_v1.py` (pairwise logistic regression, calibrated baseline)
  - `scripts/train_winprob_v2.py` (phase-aware gradient boosting)

## Stage 3 Self-Play League (Checkpoint-Only)

Stage 3 is a GPU-first self-play regime where the learning agent plays only against a league of its own prior checkpoints. No MCTS or random opponents are used during training.

Quick start:
1. Produce Stage 2 checkpoints (e.g., `configs/v1_rl_vs_mcts.yaml`).
2. Set `stage3_league.seed_dir` to the Stage 2 checkpoint directory and `stage3_league.league_dir` to a new Stage 3 output directory.
3. Run Stage 3:

```bash
PYTHONPATH=. python rl/train.py --config configs/stage3_selfplay.yaml
```

GPU-first (DummyVecEnv) variant:

```bash
PYTHONPATH=. python rl/train.py --config configs/stage3_selfplay_gpu.yaml
```

On macOS, GPU acceleration uses `mps`. The `stage3_selfplay_gpu.yaml` config is set up for MPS (DummyVecEnv + `device: mps`, opponent_device `mps`).

Additional Stage 3 configs:
1. `configs/stage3_selfplay_gpu_small.yaml`: smaller policy net for throughput testing.
2. `configs/stage3_selfplay_subproc.yaml`: SubprocVecEnv for parallel env stepping (opponents on CPU).
3. `configs/stage3_selfplay_fast.yaml`: reduced eval overhead (SubprocVecEnv).
4. `configs/stage3_selfplay_gpu_fast.yaml`: reduced eval overhead (DummyVecEnv + MPS).

Profiling Stage 3 rollout:

```bash
PYTHONPATH=. python benchmarks/profile_stage3.py --config configs/stage3_selfplay_gpu.yaml --steps 200
```

Scan Stage 3 throughput across vecenv/num_envs combinations:

```bash
PYTHONPATH=. python benchmarks/scan_stage3_envs.py --config configs/stage3_selfplay_gpu.yaml --steps 500 --num-envs 2,4,8 --vecenvs dummy,subproc
```

Key config fields in `configs/stage3_selfplay.yaml`:
1. `training_stage: 3`
2. `stage3_league.seed_dir`: where to discover prior checkpoints (Stage 2 output)
3. `stage3_league.league_dir`: where Stage 3 checkpoints + registry live
4. `stage3_league.save_every_steps`: how often to register new checkpoints into the league
5. `stage3_league.max_checkpoints_to_keep`: retention cap for league snapshots
6. `stage3_league.window_schedule`: progressive window shrink schedule (recent-focus over time)
7. `stage3_league.sampling`: band weights for old/mid/recent checkpoints
8. `stage3_league.vecenv_mode`: optional override for Stage 3 vec env (`dummy` or `subproc`)
9. `stage3_league.strict_resume`: require RNG + step metadata when resuming Stage 3
10. `device`: training device (`auto`, `cuda`, `mps`, `cpu`)

League metadata:
1. Registry file: `stage3_league.league_dir/league_registry.jsonl`
2. State file: `stage3_league.league_dir/league_state.json`

Note: when using `vec_env_type: subproc`, Stage 3 auto-resolves `opponent_device` to `cpu` to avoid multi-process GPU memory duplication. Use `vec_env_type: dummy` if you want opponents on GPU.

## Benchmarks

Compare Stage 2 (MCTS baseline) vs Stage 3 (checkpoint league) rollout throughput:

```bash
PYTHONPATH=. python benchmarks/bench_selfplay_league.py \\
  --stage2-config configs/v1_rl_vs_mcts.yaml \\
  --stage3-config configs/stage3_selfplay.yaml \\
  --steps 5000
```

Results are saved to `benchmarks/results/*.json`.

macOS MPS fast path:

```bash
PYTHONPATH=. python benchmarks/bench_selfplay_league.py \\
  --stage2-config configs/v1_rl_vs_mcts.yaml \\
  --stage2-vecenv dummy \\
  --stage3-config configs/stage3_selfplay_gpu.yaml \\
  --stage3-vecenv dummy \\
  --stage3-opponent-device mps \\
  --steps 2000
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **Node.js 16+** (for frontend)
- **pip** and **npm**

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd blokus_rl
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   # Or using pyproject.toml
   pip install -e .
   ```

3. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Running the Application

1. **Start the backend server**:
   ```bash
   python run_server.py
   ```
   Server runs at `http://localhost:8000`
   - API docs: `http://localhost:8000/docs`

2. **Start the frontend** (in a new terminal):
   ```bash
   cd frontend
   npm run dev
   ```
   Frontend runs at `http://localhost:5173`

3. **Open your browser** and navigate to `http://localhost:5173`

### Arena + Win-Probability Workflow

Run a reproducible 100-game arena benchmark:

```bash
python scripts/arena.py --config scripts/arena_config.json
```

Run fair-time benchmark (equal think times):

```bash
python scripts/arena.py --config scripts/arena_config_fair_time.json
```

Train v1/v2 win-probability models from a completed run:

```bash
python scripts/train_winprob_v1.py --snapshots arena_runs/<run_id>
python scripts/train_winprob_v2.py --snapshots arena_runs/<run_id>
```

Detailed arena/modeling docs: [`docs/arena.md`](docs/arena.md), [`docs/datasets.md`](docs/datasets.md)

## 📖 Usage

### Playing a Game via Web Interface

1. Navigate to the home page
2. Configure game settings:
   - Select agents for each player (Random, Heuristic, MCTS, or Human)
   - Set game parameters
   - Click "Start New Game"
3. Play:
   - Select a piece from your tray
   - Use **R** to rotate, **F** to flip
   - Click on the board to place the piece
   - Watch AI agents play automatically

### Using the API

#### Create a Game
```python
import requests

response = requests.post("http://localhost:8000/api/games", json={
    "players": [
        {"player": "RED", "agent_type": "human"},
        {"player": "BLUE", "agent_type": "random"},
        {"player": "YELLOW", "agent_type": "heuristic"},
        {"player": "GREEN", "agent_type": "mcts"}
    ],
    "auto_start": True
})

game_id = response.json()["game_id"]
```

#### Get Game State
```python
response = requests.get(f"http://localhost:8000/api/games/{game_id}")
game_state = response.json()
```

#### Make a Move
```python
response = requests.post(
    f"http://localhost:8000/api/games/{game_id}/move",
    json={
        "player": "RED",
        "move": {
            "piece_id": 1,
            "orientation": 0,
            "anchor_row": 0,
            "anchor_col": 0
        }
    }
)
```

### Using the RL Environment

#### PettingZoo AEC Environment
```python
from envs.blokus_v0 import env

# Create environment
blokus_env = env(render_mode="human", max_episode_steps=1000)

# Reset
blokus_env.reset()

# Step through game
while not blokus_env.terminations[blokus_env.agent_selection]:
    agent = blokus_env.agent_selection
    obs = blokus_env.observe(agent)
    info = blokus_env.infos[agent]
    
    # Get legal moves from info
    legal_mask = info["legal_action_mask"]
    legal_actions = [i for i, legal in enumerate(legal_mask) if legal]
    
    # Select action (example: random legal action)
    import random
    action = random.choice(legal_actions)
    
    # Step
    blokus_env.step(action)
```

#### Gymnasium Compatibility
```python
from envs.blokus_v0 import make_gymnasium_env

env = make_gymnasium_env(render_mode="human")
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Replace with your agent
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

### Self-Play Training (MaskablePPO)
Run an overnight training job with periodic Elo evaluation:
```bash
python -m rl.train --config configs/overnight.yaml
```

Run a quick smoke test (~1 minute) that trains briefly and updates Elo across 10 matches:
```bash
python -m rl.smoke_test
```

### Running Agent Arena

Run a reproducible multi-agent arena experiment:

```bash
python scripts/arena.py --config scripts/arena_config.json
```

Configuration file format:
```json
{
  "agents": [
    {
      "name": "mcts_25ms",
      "type": "gameplay_fast_mcts",
      "thinking_time_ms": 25,
      "params": {
        "deterministic_time_budget": true,
        "iterations": 5000,
        "iterations_per_ms": 20.0,
        "exploration_constant": 1.414
      }
    },
    {
      "name": "mcts_50ms",
      "type": "gameplay_fast_mcts",
      "thinking_time_ms": 50,
      "params": {
        "deterministic_time_budget": true,
        "iterations": 5000,
        "iterations_per_ms": 20.0,
        "exploration_constant": 1.414
      }
    },
    {
      "name": "mcts_100ms",
      "type": "gameplay_fast_mcts",
      "thinking_time_ms": 100,
      "params": {
        "deterministic_time_budget": true,
        "iterations": 5000,
        "iterations_per_ms": 20.0,
        "exploration_constant": 1.414
      }
    },
    {
      "name": "mcts_200ms",
      "type": "gameplay_fast_mcts",
      "thinking_time_ms": 200,
      "params": {
        "deterministic_time_budget": true,
        "iterations": 5000,
        "iterations_per_ms": 20.0,
        "exploration_constant": 1.414
      }
    }
  ],
  "num_games": 100,
  "seed": 20260301,
  "seat_policy": "round_robin",
  "output_root": "arena_runs",
  "max_turns": 2500
}
```

See `docs/arena.md` for full run schema and output artifacts.

## 🏗️ Project Structure

```
blokus_rl/
├── agents/              # AI agent implementations
│   ├── base_agent.py
│   ├── random_agent.py
│   ├── heuristic_agent.py
│   └── fast_mcts_agent.py
├── analytics/           # Jupyter notebooks and data analysis
├── browser_python/      # Pyodide WebWorker for in-browser Python execution
├── checkpoints/         # Saved model weights
├── config/              # Centralized YAML configurations
├── docs/                # Project documentation
├── engine/              # Core game engine
│   ├── board.py         # Board state management
│   ├── game.py          # Game logic and scoring
│   ├── pieces.py        # Piece definitions and generation
│   └── move_generator.py # Legal move generation
├── envs/                # RL environments
│   ├── blokus_env.py
│   └── blokus_v0.py     # PettingZoo AEC environment
├── frontend/            # React / Vite frontend
├── logs/                # TensorBoard logs and run metrics
├── mcts/                # MCTS implementation
├── rl/                  # Multi-stage PettingZoo self-play training
├── schemas/             # Pydantic definitions
├── scripts/             # Utility and evaluation scripts
├── tests/               # Test suite
├── training/            # SB3 training code and legacy trainers
├── webapi/              # FastAPI backend
├── pyproject.toml       # Python package configuration
└── requirements.txt     # Python dependencies
```

## 🔧 Key Components

### Game Engine (`engine/`)

- **Board**: 20x20 grid with player tracking, move validation
- **Game**: Main game logic, scoring, turn management
- **Pieces**: All 21 Blokus pieces with rotation/flip support
- **MoveGenerator**: Efficient legal move generation with caching

### Agents (`agents/`)

All agents implement a common interface:
```python
def select_action(board: Board, player: Player, legal_moves: List[Move]) -> Move
```

- **RandomAgent**: Baseline random selection
- **HeuristicAgent**: Evaluates moves using configurable heuristics
- **MCTSAgent**: Full Monte Carlo Tree Search
- **FastMCTSAgent**: Optimized MCTS for real-time play

### RL Environment (`envs/blokus_v0.py`)

- **Action Space**: Discrete actions mapped to (piece_id, orientation, row, col)
- **Observation Space**: Multi-channel tensor with:
  - Board state (5 channels: empty + 4 players)
  - Remaining pieces (21 channels)
  - Last move info (4 channels)
- **Rewards**: Score-based rewards with win/tie bonuses
- **Action Masking**: Info dict includes legal action mask

### Web API (`webapi/app.py`)

- **REST Endpoints**: Create games, get state, make moves, pass turns, list agents
- **Turn Loop**: Automatic agent moves with human player support via REST
- **Game Manager**: Session management and state persistence

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Test coverage includes:
- Game engine functionality
- Move generation and validation
- Agent behavior
- Environment compatibility

## 🎓 Training RL Agents

The environment is compatible with:
- **Stable-Baselines3**: Use `make_gymnasium_env()` wrapper
- **PettingZoo**: Use `env()` directly for multi-agent training
- **Custom training**: See `training/trainer.py` for examples

### Quick Start: Smoke Test

Before running long training jobs, verify everything works with a smoke test:

```bash
# Quick verification run (5 episodes, detailed logging)
python training/trainer.py --mode smoke

# Or use a config file
python training/trainer.py --config training/config_smoke.yaml
```

### Full Training

Once smoke test passes, run full training:

```bash
# Full training run
python training/trainer.py --mode full --total-timesteps 1000000

# Or use a config file
python training/trainer.py --config training/config_full.yaml
```

### Training Features

The training system includes:
- **Smoke-test mode**: Quick verification with small episode counts and detailed logging
- **Full mode**: Production training with optimized settings
- **Config files**: YAML/JSON configuration support
- **Seed control**: Reproducible training runs
- **Sanity checks**: Automatic detection of NaN/Inf values and other issues
- **Checkpointing**: Periodic checkpoint saving with resume capability
- **Training History**: Automatic logging of all training runs to MongoDB with web interface

See `training/README.md` for complete documentation.

### Training History

All training runs are automatically logged to MongoDB and can be viewed in the web interface:

1. **Start a training run** (runs are automatically logged)
2. **View Training History**: Navigate to `/training` in the web app
3. **View Run Details**: Click any run to see metrics, charts, and configuration

Features:
- **List view**: Filter by agent, status, view key metrics
- **Detail view**: Charts (reward over time, win rate), statistics, full configuration
- **Checkpoints**: View all saved checkpoints with resume commands
- **API access**: REST endpoints for programmatic access

See `docs/training-history.md` for complete documentation.

### Checkpointing and Resume

The training system includes automatic checkpointing:

- **Periodic checkpoints**: Save checkpoints every N episodes (default: 50)
- **Automatic cleanup**: Keep only the most recent N checkpoints (default: 3)
- **Resume training**: Continue from any saved checkpoint
- **UI integration**: View checkpoints and copy resume commands from web interface

Example:
```bash
# Start training with checkpointing
python training/trainer.py --mode full --checkpoint-interval-episodes 50

# Resume from checkpoint
python training/trainer.py --resume-from-checkpoint checkpoints/ppo_agent/run123/ep000100.zip
```

See `docs/checkpoints.md` for complete checkpointing documentation.

### Hyperparameter Configuration and Sweeps

The system supports structured hyperparameter management:

- **Agent configs**: Versioned hyperparameter files in `config/agents/`
- **Config includes**: Learning rate, gamma, network architecture, PPO parameters
- **Quick sweeps**: Test multiple configs with short runs before long training
- **UI integration**: View config names and hyperparameters in Training History

Example:
```bash
# Use specific agent config
python training/trainer.py --agent-config config/agents/ppo_agent_v1.yaml

# Run hyperparameter sweep
python training/run_sweep.py config/agents/ppo_agent_sweep_*.yaml --episodes 100
```

See `docs/hyperparams.md` for complete hyperparameter documentation.

### Evaluation and Baselines

The system includes evaluation protocols to assess trained agents:

- **Baseline Agents**: RandomAgent and HeuristicAgent for comparison
- **Evaluation Script**: Test trained checkpoints against baselines
- **Metrics**: Win rate, average reward, game length
- **UI Integration**: View evaluation results in Training History

Example:
```bash
# Evaluate a checkpoint
python training/evaluate_agent.py checkpoints/ppo_agent/run123/ep000100.zip --num-games 100
```

See `docs/evaluation.md` for complete evaluation documentation.

### Example with Stable-Baselines3 (Direct)

For custom training scripts:

```python
from envs.blokus_v0 import make_gymnasium_env
from stable_baselines3 import PPO

env = make_gymnasium_env()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## 🛠️ Development

### Code Quality

- **Linting**: `ruff` (configured in `pyproject.toml`)
- **Type Checking**: `mypy` (configured in `pyproject.toml`)
- **Testing**: `pytest` with `pytest-asyncio`

### Adding New Agents

1. Create a new agent class in `agents/`
2. Inherit from base agent interface
3. Implement `select_action()` method
4. Register in API agent creation paths (for example `webapi/app.py`)
5. Register in arena builder (`analytics/tournament/arena_runner.py`) if you want arena support

### Adding New Features

- **Frontend**: Add components in `frontend/src/components/`
- **Backend**: Add endpoints in `webapi/app.py`
- **Engine**: Extend classes in `engine/`

## 📚 API Documentation

Full API documentation available at `http://localhost:8000/docs` when server is running.

### Key Endpoints

- `POST /api/games` - Create new game
- `GET /api/games/{game_id}` - Get game state
- `POST /api/games/{game_id}/move` - Make a move
- `POST /api/games/{game_id}/pass` - Pass a human turn
- `POST /api/games/{game_id}/advance_turn` - Advance one AI turn
- `GET /api/agents` - List available agents
- `GET /api/games` - List all games
- `GET /api/analysis/{game_id}` - Game analysis (research profile)
- `GET /api/analysis/{game_id}/replay?move_index=N` - Move-by-move replay (research profile)
- `GET /api/history` - List recently finished games (research profile)

## 🗄️ Game Persistence and Replay

Games are stored in MongoDB when they end, but **only in research profile** (`APP_PROFILE=research`). In deploy profile (e.g. Vercel), MongoDB is skipped and games are not persisted.

### Setup for Persistence

1. **Run in research profile with MongoDB**:
   ```bash
   APP_PROFILE=research MONGODB_URI=mongodb://localhost:27017 python3 -m uvicorn webapi.app:app --reload
   ```
   Or use your main webapi entrypoint if you have one.

2. **Configure MongoDB** (see `docs/mongodb.md`):
   - `MONGODB_URI`: Connection string (default: `mongodb://localhost:27017`)
   - `MONGODB_DB_NAME`: Database name (default: `blokusdb`)

### Move-by-Move Replay

Use the replay API to inspect board state at any point in a finished game:

```bash
# Initial board
curl "http://localhost:8000/api/analysis/{game_id}/replay?move_index=0"

# After 5th event (move or pass)
curl "http://localhost:8000/api/analysis/{game_id}/replay?move_index=5"

# Final state
curl "http://localhost:8000/api/analysis/{game_id}/replay?move_index=-1"
```

**Response format:**
```json
{
  "game_id": "...",
  "move_index": 5,
  "total_events": 42,
  "board": [[0,0,...], ...],
  "current_player": "BLUE",
  "scores": {"RED": 10, "BLUE": 15, ...},
  "pieces_used": {"RED": [1,2,3], ...},
  "move_count": 4,
  "game_over": false,
  "event": { "player": "RED", "move": {...}, ... }
}
```

### List Finished Games

```bash
curl "http://localhost:8000/api/history?limit=20"
```

### Deploy Profile

In deploy profile (e.g. Vercel), MongoDB is not used. For persistence there you would need either MongoDB Atlas (or similar) with `MONGODB_URI` set in Vercel env, or a different storage backend. See `docs/mongodb.md` for details.

## 🎮 Game Rules

Blokus is a strategy board game where players take turns placing pieces on a 20x20 board:

- **Starting**: Each player starts from their corner
- **Placement**: Pieces must touch at least one corner of your own pieces, but cannot share edges
- **Pieces**: 21 unique polyomino pieces (1-5 squares)
- **Scoring**: 1 point per square, +15 bonus for using all pieces, bonuses for corner/center control
- **End**: Game ends when no player can make a legal move

## 🤝 Contributing

1. Follow existing code style (ruff formatting)
2. Add tests for new features
3. Update documentation as needed
4. Use type hints for Python code
5. Use TypeScript for frontend code

## 📝 License

This project is part of a reinforcement learning research environment.

## 🔗 Additional Resources

- **Frontend README**: See `docs/frontend/README.md` for frontend-specific details
- **API README**: See `docs/webapi/README.md` for API documentation
- **PettingZoo**: https://pettingzoo.farama.org/
- **Gymnasium**: https://gymnasium.farama.org/
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**: Change port in `run_server.py` or `frontend/vite.config.ts`
2. **API connection fails**: Ensure backend is running on port 8000 for server-backed routes
3. **Import errors**: Ensure you've installed dependencies with `pip install -e .`
4. **Frontend build errors**: Run `npm install` in `frontend/` directory

### Debugging Invalid Moves

If the game reports "Invalid move" for moves you believe are legal:

**Where validation happens:**
- Frontend sends `{ piece_id, orientation, anchor_row, anchor_col }` (clicked cell)
- Backend validates via `game.make_move()` → `move_generator.is_move_legal()`
- Engine checks piece usage, orientation, and `board.can_place_piece()`

**Likely causes:**
1. **Anchor convention**: The engine expects the anchor to be the **top-left** of the piece. The frontend treats the clicked cell as the anchor. If you click a cell that is not the top-left of the piece, the move will be wrong.
2. **Piece shape/orientation mismatch**: Frontend and backend may use different orientation conventions.
3. **State desync**: Board state may differ between frontend and backend (e.g. stale local state).

**Debugging steps:**

1. **Browser console**: The frontend logs the move being sent. Check `[UI] Sending move` when a move fails.

2. **Check if the move is in `legal_moves`**: The game state includes `legal_moves`. Add a check before sending—if the move is in `legal_moves` but the backend rejects it, the bug is likely on the backend or in state sync.

3. **Run engine tests**:
   ```bash
   PYTHONPATH=. python3 -m pytest tests/test_move_generation_equivalence.py tests/test_engine.py -v
   ```

4. **Verify legal move logic**:
   ```bash
   PYTHONPATH=. python3 -m pytest tests/test_move_generation_equivalence.py -v
   ```

5. **Use the heatmap**: The game state includes a `heatmap` where `1.0` marks cells that are part of legal moves. Use this to verify where legal placements are.

### Getting Help

- Check API docs at `http://localhost:8000/docs`
- Review test files for usage examples
- Check console logs for detailed error messages

---

**Version**: 0.1.0  
**Python**: 3.9+  
**Node.js**: 16+


Old Mockups 
<img width="2816" height="1536" alt="BlokusRL" src="https://github.com/user-attachments/assets/93e85cd8-c5fe-4785-ae13-810327a1aa07" />
