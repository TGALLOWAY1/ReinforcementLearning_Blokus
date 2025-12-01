# Blokus RL - Reinforcement Learning Environment for Blokus

A comprehensive reinforcement learning research environment for the Blokus board game, featuring a complete game engine, multiple AI agents, web interface, and PettingZoo/Gymnasium compatibility.

<img width="1795" height="865" alt="image" src="https://github.com/user-attachments/assets/751e771f-ce00-45b8-8289-6086f760cd7d" />

## 🎯 Overview

This project provides a full-stack implementation of Blokus with:
- **Complete game engine** implementing official Blokus rules
- **Multiple AI agents** (Random, Heuristic, MCTS)
- **Reinforcement learning environment** compatible with PettingZoo and Gymnasium
- **Web interface** for interactive gameplay
- **REST API & WebSocket** for programmatic access
- **Arena system** for agent evaluation and tournaments

## ✨ Features

### Game Engine
- Full Blokus rule implementation (20x20 board, 4 players, 21 pieces per player)
- Legal move generation with optimized caching
- Scoring system with bonuses (corner control, center control, piece completion)
- Game state management and history tracking

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
- WebSocket integration for live updates
- Color-blind friendly design
- Agent visualization and research tools

### API & Backend
- FastAPI REST API
- WebSocket support for real-time gameplay
- Automatic turn management for AI agents
- Human player support via WebSocket
- Game state persistence and management

### Evaluation Tools
- Arena system for round-robin tournaments
- Agent performance statistics
- Match result logging and analysis

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

### Running Agent Arena

Run round-robin tournaments between agents:

```bash
python scripts/arena.py --config scripts/arena_config.json --rounds 5 --verbose
```

Configuration file format:
```json
{
  "RandomAgent": {
    "type": "random",
    "seed": 42
  },
  "HeuristicAgent": {
    "type": "heuristic",
    "seed": 43,
    "weights": {
      "piece_size": 1.0,
      "corner_creation": 2.0,
      "edge_avoidance": -1.5,
      "center_preference": 0.5
    }
  },
  "MCTSAgent": {
    "type": "mcts",
    "iterations": 500,
    "exploration_constant": 1.414,
    "use_transposition_table": true,
    "seed": 44
  }
}
```

## 🏗️ Project Structure

```
blokus_rl/
├── agents/              # AI agent implementations
│   ├── base_agent.py
│   ├── random_agent.py
│   ├── heuristic_agent.py
│   └── fast_mcts_agent.py
├── engine/              # Core game engine
│   ├── board.py         # Board state management
│   ├── game.py          # Game logic and scoring
│   ├── pieces.py        # Piece definitions and generation
│   └── move_generator.py # Legal move generation
├── envs/                # RL environments
│   ├── blokus_env.py
│   └── blokus_v0.py     # PettingZoo AEC environment
├── mcts/                # MCTS implementation
│   ├── mcts.py
│   ├── mcts_agent.py
│   └── zobrist.py       # Zobrist hashing
├── webapi/              # Backend API
│   ├── app.py           # FastAPI application
│   └── game_manager.py  # Game session management
├── frontend/            # React frontend
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Page components
│   │   ├── store/       # State management
│   │   └── utils/       # Utilities
│   └── package.json
├── schemas/             # Pydantic schemas
│   ├── game_config.py
│   ├── game_state.py
│   └── move.py
├── scripts/             # Utility scripts
│   ├── arena.py         # Tournament runner
│   └── arena_config.json
├── training/            # RL training code
│   └── trainer.py
├── tests/               # Test suite
│   ├── test_engine.py
│   └── test_blokus_env.py
├── run_server.py        # Server entry point
├── requirements.txt     # Python dependencies
└── pyproject.toml       # Project configuration
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

- **REST Endpoints**: Create games, get state, make moves, list agents
- **WebSocket**: Real-time game updates and human player input
- **Turn Loop**: Automatic agent moves with human player support
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
4. Register in `webapi/app.py` and `scripts/arena.py`

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
- `GET /api/agents` - List available agents
- `GET /api/games` - List all games
- `WS /ws/games/{game_id}` - WebSocket connection

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

- **Frontend README**: See `frontend/README.md` for frontend-specific details
- **API README**: See `webapi/README.md` for API documentation
- **PettingZoo**: https://pettingzoo.farama.org/
- **Gymnasium**: https://gymnasium.farama.org/
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**: Change port in `run_server.py` or `frontend/vite.config.ts`
2. **WebSocket connection fails**: Ensure backend is running on port 8000
3. **Import errors**: Ensure you've installed dependencies with `pip install -e .`
4. **Frontend build errors**: Run `npm install` in `frontend/` directory

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

