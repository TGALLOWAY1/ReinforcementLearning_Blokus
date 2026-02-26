# Blokus RL Web API

A comprehensive web API for the Blokus RL project with REST endpoints and WebSocket support for real-time game interaction.

## Features

- **REST API**: Create games, get game state, make moves, list agents
- **WebSocket Support**: Real-time game state streaming and human player interaction
- **Turn Loop**: Automatic agent moves with human player support via WebSocket
- **Multiple Agent Types**: Random, Heuristic, MCTS, and Human players
- **Pydantic Schemas**: Type-safe request/response models
- **MongoDB Persistence**: Centralized database layer for training runs, evaluation results, and game data

## MongoDB Setup

The API includes a MongoDB persistence layer for storing training runs, evaluation results, and other game data.

### Configuration

MongoDB connection is configured via environment variables:

- `MONGODB_URI`: MongoDB connection string (default: `mongodb://localhost:27017`)
- `MONGODB_DB_NAME`: Database name (default: `blokus_rl`)

**Example local connection:**
```bash
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DB_NAME="blokus_rl"
```

**Example production connection:**
```bash
export MONGODB_URI="mongodb://username:password@host:port/database?authSource=admin"
export MONGODB_DB_NAME="blokus_rl"
```

### Connection Module

The MongoDB connection is managed in `webapi/db/mongo.py`:
- Singleton connection pattern (connects once at startup)
- Async Motor driver for FastAPI compatibility
- Automatic connection handling in FastAPI lifespan
- Graceful shutdown on server stop

### Models

The following MongoDB models are defined in `webapi/db/models.py`:

- **TrainingRun**: Stores RL training session data
  - Fields: `run_id`, `agent_id`, `algorithm`, `config`, `status`, `metrics`, `checkpoint_paths`, etc.
- **EvaluationRun**: Stores model evaluation results
  - Fields: `training_run_id`, `checkpoint_path`, `opponent_type`, `win_rate`, `avg_reward`, etc.

### Health Check

Test MongoDB connectivity:
```http
GET /api/health/db
```

Response:
```json
{
  "ok": true,
  "db": "connected",
  "database": "blokus_rl"
}
```

### Usage in Code

```python
from db.mongo import get_database
from db.models import TrainingRun

# Get database instance
db = get_database()

# Access collections
training_runs = db.training_runs
evaluation_runs = db.evaluation_runs

# Example: Insert a training run
training_run = TrainingRun(
    run_id="550e8400-e29b-41d4-a716-446655440000",
    agent_id="ppo_agent",
    algorithm="PPO",
    config={"learning_rate": 3e-4},
    status="running"
)
result = await training_runs.insert_one(training_run.dict(by_alias=True))
```

### TODO: Future Integration

- [ ] Hook TrainingRun creation & updates into the RL training loop
- [ ] Implement EvaluationRun logging when running evaluation scripts
- [ ] Add Training History UI to display stored training runs

## Quick Start

1. **Start the server**:
   ```bash
   cd blokus_rl
   
   ```

2. **View API documentation**: http://localhost:8000/docs

3. **Test the API**:
   ```bash
   python example_usage.py
   ```

## API Endpoints

### REST Endpoints

#### Create Game
```http
POST /api/games
Content-Type: application/json

{
  "players": [
    {"player": "RED", "agent_type": "human"},
    {"player": "BLUE", "agent_type": "random"}
  ],
  "auto_start": true
}
```

#### Get Game State
```http
GET /api/games/{game_id}
```

#### Make Move
```http
POST /api/games/{game_id}/move
Content-Type: application/json

{
  "move": {
    "piece_id": 1,
    "orientation": 0,
    "anchor_row": 0,
    "anchor_col": 0
  }
}
```

#### List Agents
```http
GET /api/agents
```

#### List Games
```http
GET /api/games
```

### WebSocket Endpoint

Connect to: `ws://localhost:8000/ws/games/{game_id}`

#### Message Types

**From Server:**
- `game_state`: Current game state update
- `move_made`: Move was made
- `game_over`: Game has ended
- `error`: Error occurred

**To Server:**
- `make_move`: Make a move (for human players)
- `ping`: Ping for connection testing

#### Example WebSocket Usage

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/games/game-id');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data.type);
  
  if (data.type === 'game_state') {
    const gameState = data.data.game_state;
    console.log('Current player:', gameState.current_player);
    console.log('Legal moves:', gameState.legal_moves.length);
  }
};

// Make a move
ws.send(JSON.stringify({
  type: 'make_move',
  move: {
    piece_id: 1,
    orientation: 0,
    anchor_row: 0,
    anchor_col: 0
  }
}));
```

## Game Configuration

### Player Configuration
```python
PlayerConfig(
    player="RED",           # Player color: RED, BLUE, YELLOW, GREEN
    agent_type="human",     # Agent type: random, heuristic, mcts, human
    agent_config={}         # Optional agent-specific configuration
)
```

### Game Configuration
```python
GameConfig(
    players=[...],          # List of PlayerConfig (2-4 players)
    game_id=None,           # Optional custom game ID
    auto_start=True         # Whether to start game automatically
)
```

## Agent Types

### Random Agent
- Makes random legal moves
- Good baseline for testing

### Heuristic Agent
- Uses strategic preferences:
  - Prefers larger pieces
  - Creates new corners for future moves
  - Avoids edges early in game
  - Prefers center positions

### MCTS Agent
- Monte Carlo Tree Search with UCT
- Uses heuristic rollouts
- Configurable iterations/time limit
- Transposition table for efficiency

### Human Agent
- Controlled via WebSocket
- Waits for move input from client

## Game Flow

1. **Game Creation**: Create game with player configurations
2. **Auto Start**: Game starts automatically if `auto_start=True`
3. **Turn Loop**: 
   - Agent players make moves automatically
   - Human players wait for WebSocket input
   - Game state broadcasted to all WebSocket connections
4. **Game End**: Game ends when no player can make legal moves

## Error Handling

The API includes comprehensive error handling:

- **HTTP Errors**: Proper status codes and error messages
- **WebSocket Errors**: Error messages sent to clients
- **Game Errors**: Invalid moves, game not found, etc.
- **Agent Errors**: Agent failures are logged and handled gracefully

## Development

### Project Structure
```
webapi/
├── app.py              # Main FastAPI application
├── run_server.py       # Server startup script
├── example_usage.py    # Usage examples
└── README.md          # This file
```

### Dependencies
- FastAPI: Web framework
- Uvicorn: ASGI server
- Pydantic: Data validation
- WebSockets: Real-time communication

### Running in Development
```bash
# With auto-reload
python run_server.py

# Or directly with uvicorn
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Example Games

### Human vs Random Agent
```python
config = GameConfig(
    players=[
        PlayerConfig(player="RED", agent_type="human"),
        PlayerConfig(player="BLUE", agent_type="random")
    ]
)
```

### Tournament (All Agents)
```python
config = GameConfig(
    players=[
        PlayerConfig(player="RED", agent_type="random"),
        PlayerConfig(player="BLUE", agent_type="heuristic"),
        PlayerConfig(player="YELLOW", agent_type="mcts"),
        PlayerConfig(player="GREEN", agent_type="human")
    ]
)
```

## Performance Notes

- Games run asynchronously with automatic turn loops
- WebSocket connections are managed efficiently
- Agent moves are computed in background tasks
- Transposition tables used for MCTS efficiency
- Memory cleanup for completed games

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Agent Errors**: Check agent implementations have `select_action` method
3. **WebSocket Disconnections**: Handle reconnection in client code
4. **Game Not Found**: Ensure game ID exists before connecting

### Debug Mode
Set log level to debug for detailed logging:
```python
uvicorn.run(app, log_level="debug")
```