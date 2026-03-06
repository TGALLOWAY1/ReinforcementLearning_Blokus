# Reinforcement Learning Environment for Blokus

A comprehensive reinforcement learning research environment for the Blokus board game, featuring a complete game engine, multiple AI agents, web interface,.

<img width="1795" height="865" alt="image" src="https://github.com/user-attachments/assets/751e771f-ce00-45b8-8289-6086f760cd7d" />

## 🎯 Overview

> **Note**: The reinforcement learning (RL) agents and training pipeline have been archived to consolidate the project. To access the PyTorch/Stable-Baselines3 training code or PettingZoo environments, check out the `archive/rl-agents` branch:
> ```bash
> git fetch && git checkout archive/rl-agents
> ```


This project provides a full-stack implementation of Blokus with:
- **Complete game engine** implementing official Blokus rules
- **Multiple AI agents** (Random, Heuristic, MCTS)
- **Web interface** for interactive gameplay
- **REST API** for programmatic access and research tooling
- **Arena system** for agent evaluation and tournaments

---

## 👋 For Recruiters & Hiring Managers

**Welcome!** This project demonstrates a production-ready stack for Reinforcement Learning and full-stack web development. 

### ⏱️ 60-Second Architecture Summary
- **Game Engine (Python)**: High-performance bitboard and frontier-based move generation, capable of thousands of simulations per second.
- **AI Agents**: Implementations ranging from simple heuristics to an optimized Monte Carlo Tree Search (MCTS) algorithm with UCB1 and transposition tables.
- **Frontend (React/TypeScript)**: A responsive, color-blind friendly SPA.
- **Deployment Topology**: The frontend uses `Pyodide` to run the Python core engine directly in the browser via WebWorkers. This allows the heavy MCTS simulations to run locally on the client's machine with zero backend server scaling required, creating a highly portable and free-to-host architecture.

### 🚀 How to Run the Demo
1. Open the live deployment (or run frontend locally via `npm run dev`).
2. Click **Run Demo Game** on the home page.
3. The game will automatically start an AI vs. AI match.
4. Use the **Pause/Step** controls at the top left to freeze the game.
5. Watch the **Explain This Move** panel on the right side to see the MCTS agent's thought process, including its top evaluated candidates, simulation counts, and Q-values.
6. Click **AI Scoreboard** to view a 1,500-game statistically significant evaluation matrix mapping out the strength hierarchy of different agent hyperparameters.

---

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
