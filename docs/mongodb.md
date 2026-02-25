# MongoDB Persistence Layer

This document describes the MongoDB persistence layer for the Blokus RL project.

## Overview

The MongoDB persistence layer provides centralized storage for:
- **Game Records**: Completed games with per-move history for replay and debugging
- **TrainingRun**: Records of RL training sessions with metrics, checkpoints, and configuration
- **EvaluationRun**: Records of model evaluation sessions with performance metrics

### Game Persistence (Research Profile Only)

Games are persisted to MongoDB when they end. **MongoDB is only used in research profile** (`APP_PROFILE=research`). In deploy profile (e.g. Vercel), MongoDB is skipped and games are not stored.

**Collections:**
- `game_records`: Game metadata (game_id, winner, scores, players, created_at, finished_at)
- `move_records`: Per-move/event history (piece placements and passes) for replay

**Move-by-move replay:** Use `GET /api/analysis/{game_id}/replay?move_index=N` to get board state at event N (0=initial, 1=after first move/pass, -1=final).

The connection is managed as a singleton, established once at application startup and reused across all requests.

## Configuration

### Environment Variables

The MongoDB connection is configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URI` | MongoDB connection string | `mongodb://localhost:27017` |
| `MONGODB_DB_NAME` | Database name | `blokus_rl` |

### Setting Environment Variables

**Option 1: Default (Local MongoDB)**
If MongoDB is running locally on the default port, no setup is needed. The code uses:
- `MONGODB_URI`: `mongodb://localhost:27017` (default)
- `MONGODB_DB_NAME`: `blokus_rl` (default)

**Option 2: Export in Terminal (Temporary)**
```bash
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DB_NAME="blokus_rl"
python run_server.py
```

**Option 3: Use .env File (Recommended for Development)**
1. Create a `.env` file in the project root:
   ```bash
   # .env
   MONGODB_URI=mongodb://localhost:27017
   MONGODB_DB_NAME=blokus_rl
   ```
2. The code automatically loads `.env` files if `python-dotenv` is installed (included in requirements.txt)

**Option 4: Set Inline with Command**
```bash
MONGODB_URI="mongodb://localhost:27017" MONGODB_DB_NAME="blokus_rl" python run_server.py
```

### Example Connection Strings

**Local Development (default):**
```bash
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=blokus_rl
```

**Local with Authentication:**
```bash
MONGODB_URI=mongodb://username:password@localhost:27017/blokus_rl?authSource=admin
MONGODB_DB_NAME=blokus_rl
```

**MongoDB Atlas (cloud):**
```bash
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/blokus_rl?retryWrites=true&w=majority
MONGODB_DB_NAME=blokus_rl
```

**Production (remote server):**
```bash
MONGODB_URI=mongodb://username:password@host:port/blokus_rl?authSource=admin
MONGODB_DB_NAME=blokus_rl
```

## Architecture

### Connection Module (`webapi/db/mongo.py`)

The connection module provides:
- `connect_to_mongo()`: Establishes connection at startup
- `close_mongo_connection()`: Closes connection at shutdown
- `get_database()`: Returns the database instance for queries
- `get_client()`: Returns the MongoDB client (for advanced use)

The connection is integrated into FastAPI's lifespan manager, ensuring:
- Connection is established before the server starts accepting requests
- Connection is closed gracefully when the server shuts down
- Errors during connection are logged but don't prevent server startup (non-blocking)

### Models (`webapi/db/models.py`)

#### TrainingRun Model

Stores information about RL training sessions:

```python
{
    "_id": ObjectId,
    "run_id": str,              # Unique identifier (UUID)
    "agent_id": str,             # Agent identifier (e.g., "ppo_agent")
    "algorithm": str,            # RL algorithm (e.g., "PPO", "DQN")
    "config": dict,              # Hyperparameters and training config
    "status": str,               # "running" | "completed" | "stopped" | "failed"
    "start_time": datetime,      # Training start time
    "end_time": datetime | None, # Training end time (null if running)
    "metrics": {
        "episodes": [
            {
                "episode": int,
                "total_reward": float,
                "steps": int,
                "win": bool | None,
                "epsilon": float | None
            }
        ],
        "rolling_win_rate": [
            {
                "episode": int,
                "win_rate": float
            }
        ]
    },
    "checkpoint_paths": [
        {
            "episode": int,
            "path": str
        }
    ],
    "metadata": dict             # Additional metadata (git hash, env version, etc.)
}
```

#### EvaluationRun Model

Stores information about model evaluation sessions:

```python
{
    "_id": ObjectId,
    "training_run_id": str,     # Reference to TrainingRun
    "checkpoint_path": str,      # Path to model checkpoint
    "opponent_type": str,       # "random" | "heuristic" | "self_play"
    "games_played": int,         # Number of games played
    "win_rate": float,          # Win rate (0.0 to 1.0)
    "avg_reward": float,        # Average reward per game
    "avg_game_length": float,   # Average game length in steps
    "created_at": datetime       # Evaluation run creation time
}
```

## Usage

### Basic Usage

```python
from db.mongo import get_database
from db.models import TrainingRun

# Get database instance
db = get_database()

# Access collections
training_runs = db.training_runs
evaluation_runs = db.evaluation_runs

# Create a training run
training_run = TrainingRun(
    run_id="550e8400-e29b-41d4-a716-446655440000",
    agent_id="ppo_agent",
    algorithm="PPO",
    config={"learning_rate": 3e-4, "gamma": 0.99},
    status="running"
)

# Insert into database
result = await training_runs.insert_one(training_run.dict(by_alias=True))
print(f"Inserted training run with ID: {result.inserted_id}")

# Query training runs
async for run in training_runs.find({"status": "completed"}):
    print(f"Found completed run: {run['run_id']}")

# Update a training run
await training_runs.update_one(
    {"run_id": "550e8400-e29b-41d4-a716-446655440000"},
    {"$set": {"status": "completed", "end_time": datetime.utcnow()}}
)
```

### Health Check

The API provides a health check endpoint to verify MongoDB connectivity:

```http
GET /api/health/db
```

**Success Response (200):**
```json
{
  "ok": true,
  "db": "connected",
  "database": "blokus_rl"
}
```

**Failure Response (503):**
```json
{
  "ok": false,
  "error": "Database connection failed",
  "message": "Unable to connect to MongoDB"
}
```

## Integration with Training Loop

### TODO: TrainingRun Logging

The following integration points need to be implemented:

1. **Training Start**: Create a TrainingRun document when training begins
   ```python
   # In training/trainer.py
   from db.mongo import get_database
   from db.models import TrainingRun
   
   training_run = TrainingRun(
       run_id=str(uuid.uuid4()),
       agent_id="ppo_agent",
       algorithm="PPO",
       config=config.to_dict(),
       status="running"
   )
   db = get_database()
   await db.training_runs.insert_one(training_run.dict(by_alias=True))
   ```

2. **Episode Metrics**: Update metrics after each episode
   ```python
   # After each episode
   episode_metric = {
       "episode": episode_num,
       "total_reward": episode_reward,
       "steps": episode_steps,
       "win": episode_win
   }
   await db.training_runs.update_one(
       {"run_id": training_run_id},
       {"$push": {"metrics.episodes": episode_metric}}
   )
   ```

3. **Checkpoint Saving**: Record checkpoint paths
   ```python
   # When saving checkpoint
   checkpoint = CheckpointPath(
       episode=episode_num,
       path=checkpoint_path
   )
   await db.training_runs.update_one(
       {"run_id": training_run_id},
       {"$push": {"checkpoint_paths": checkpoint.dict()}}
   )
   ```

4. **Training Completion**: Update status and end_time
   ```python
   # When training completes
   await db.training_runs.update_one(
       {"run_id": training_run_id},
       {
           "$set": {
               "status": "completed",
               "end_time": datetime.utcnow()
           }
       }
   )
   ```

### TODO: EvaluationRun Logging

When running evaluation scripts:

```python
# In evaluation script
from db.mongo import get_database
from db.models import EvaluationRun

evaluation_run = EvaluationRun(
    training_run_id=training_run_id,
    checkpoint_path=checkpoint_path,
    opponent_type="random",
    games_played=100,
    win_rate=0.75,
    avg_reward=15.3,
    avg_game_length=45.2
)

db = get_database()
await db.evaluation_runs.insert_one(evaluation_run.dict(by_alias=True))
```

## Troubleshooting

### Connection Issues

1. **MongoDB not running**: Ensure MongoDB is running locally or the connection string is correct
   ```bash
   # Check if MongoDB is running
   mongosh --eval "db.adminCommand('ping')"
   ```

2. **Connection timeout**: Check network connectivity and firewall settings
   ```bash
   # Test connection
   mongosh "mongodb://localhost:27017"
   ```

3. **Authentication errors**: Verify credentials in connection string
   ```bash
   # Test with credentials
   mongosh "mongodb://username:password@host:port/database"
   ```

### Server Startup

If MongoDB connection fails during startup:
- The server will log an error but continue to start
- The `/api/health/db` endpoint will return a 503 status
- Features requiring MongoDB will be unavailable

To ensure MongoDB is available:
- Start MongoDB before starting the API server
- Verify environment variables are set correctly
- Check MongoDB logs for connection attempts

## Dependencies

The MongoDB persistence layer requires:
- `motor`: Async MongoDB driver for Python
- `pymongo`: Synchronous MongoDB driver (dependency of motor)

Install with:
```bash
pip install motor pymongo
```

## Future Enhancements

Potential improvements:
- [ ] Add indexes for common queries (run_id, status, start_time)
- [ ] Implement data migration utilities
- [ ] Add backup/restore functionality
- [ ] Create aggregation pipelines for analytics
- [ ] Add data validation at the database level
- [ ] Implement connection pooling configuration
- [ ] Add retry logic for transient connection failures

