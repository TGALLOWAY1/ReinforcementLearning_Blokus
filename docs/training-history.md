# Training History Documentation

## Overview

The Training History feature provides a comprehensive system for logging, tracking, and visualizing RL training runs. Every training session is automatically logged to MongoDB, and you can view detailed metrics, charts, and configuration through the web interface.

## What is a TrainingRun?

A `TrainingRun` is a record that captures all information about a single RL training session, including:

- **Metadata**: Run ID, agent ID, algorithm type, status, timestamps
- **Configuration**: All hyperparameters and training settings
- **Metrics**: Per-episode rewards, steps, win rates, and other performance indicators
- **Checkpoints**: Saved model checkpoints with episode numbers
- **Status**: Current state (running, completed, stopped, failed)

## How Training Runs are Created

Training runs are automatically created when you start a training session using the training script:

```bash
python training/trainer.py --mode smoke
# or
python training/trainer.py --mode full --total-timesteps 1000000
```

The training script:
1. **Creates a TrainingRun** at the start with status "running"
2. **Logs each episode** with metrics (reward, steps, etc.)
3. **Updates status** to "completed", "stopped", or "failed" at the end
4. **Logs checkpoints** when models are saved

### Training Run Logger

The `TrainingRunLogger` class (`training/run_logger.py`) handles all MongoDB interactions:

- Creates run records
- Logs episode metrics
- Calculates rolling win rates
- Updates status
- Tracks checkpoints

The logger is integrated into the training callback, so metrics are automatically logged after each episode.

## Viewing Training History

### Web Interface

Navigate to the Training History page:

1. **From the Training & Evaluation page**: Click "View Training History"
2. **Direct URL**: `http://localhost:5173/training`

### List View Features

The Training History list view shows:

- **Run ID**: Unique identifier (truncated for display)
- **Agent / Algorithm**: Agent ID and algorithm type
- **Start Time**: When training started
- **Duration**: How long training ran (or "Running..." if still active)
- **Episodes**: Number of episodes completed
- **Final Metric**: Average reward or win rate
- **Status**: Current status with color-coded badges

### Filtering

You can filter training runs by:

- **Agent ID**: Show only runs for a specific agent
- **Status**: Filter by running, completed, stopped, or failed

### Detail View

Click "View Details" on any run to see:

- **Run Information**: Full metadata, timestamps, duration
- **Statistics Summary**: Total episodes, average/max reward, final win rate
- **Charts**:
  - Episode Reward: Line chart showing reward over time
  - Rolling Win Rate: Win rate progression (if available)
- **Configuration**: Full training configuration (hyperparameters)
- **Checkpoints**: List of saved model checkpoints

## API Endpoints

### List Training Runs

```
GET /api/training-runs?agent_id=<agent_id>&status=<status>&limit=<limit>
```

**Query Parameters:**
- `agent_id` (optional): Filter by agent ID
- `status` (optional): Filter by status (running, completed, stopped, failed)
- `limit` (optional, default: 50): Maximum number of runs to return

**Response:**
```json
[
  {
    "id": "...",
    "run_id": "550e8400-e29b-41d4-a716-446655440000",
    "agent_id": "ppo_agent",
    "algorithm": "MaskablePPO",
    "status": "completed",
    "start_time": "2024-01-01T00:00:00Z",
    "end_time": "2024-01-01T02:30:00Z",
    "metrics": {
      "episodes": [...],
      "rolling_win_rate": [...]
    }
  }
]
```

### Get Training Run Details

```
GET /api/training-runs/{run_id}
```

**Response:**
```json
{
  "id": "...",
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "agent_id": "ppo_agent",
  "algorithm": "MaskablePPO",
  "status": "completed",
  "start_time": "2024-01-01T00:00:00Z",
  "end_time": "2024-01-01T02:30:00Z",
  "config": {
    "mode": "full",
    "total_timesteps": 1000000,
    "learning_rate": 0.0003,
    ...
  },
  "metrics": {
    "episodes": [
      {
        "episode": 1,
        "total_reward": 10.5,
        "steps": 50,
        "win": true
      }
    ],
    "rolling_win_rate": [
      {
        "episode": 1,
        "win_rate": 0.5
      }
    ]
  },
  "checkpoint_paths": [
    {
      "episode": 100,
      "path": "checkpoints/ppo_blokus"
    }
  ]
}
```

### List Agents

```
GET /api/training-runs/agents/list
```

**Response:**
```json
{
  "agent_ids": ["ppo_agent", "dqn_agent", ...]
}
```

## MongoDB Schema

Training runs are stored in the `training_runs` collection with the following structure:

```javascript
{
  "_id": ObjectId("..."),
  "run_id": "550e8400-e29b-41d4-a716-446655440000",  // UUID
  "agent_id": "ppo_agent",
  "algorithm": "MaskablePPO",
  "config": {
    // Training configuration dictionary
  },
  "status": "running" | "completed" | "stopped" | "failed",
  "start_time": ISODate("2024-01-01T00:00:00Z"),
  "end_time": ISODate("2024-01-01T02:30:00Z"),  // null if still running
  "metrics": {
    "episodes": [
      {
        "episode": 1,
        "total_reward": 10.5,
        "steps": 50,
        "win": true,  // optional
        "epsilon": 0.1  // optional
      }
    ],
    "rolling_win_rate": [
      {
        "episode": 1,
        "win_rate": 0.5
      }
    ]
  },
  "checkpoint_paths": [
    {
      "episode": 100,
      "path": "checkpoints/ppo_blokus"
    }
  ],
  "metadata": {
    // Additional metadata (error messages, etc.)
  }
}
```

## Usage Examples

### Starting a Training Run

```bash
# Smoke test (quick verification)
python training/trainer.py --mode smoke

# Full training
python training/trainer.py --mode full --total-timesteps 1000000 --seed 42
```

The training script will:
1. Create a TrainingRun record in MongoDB
2. Log each episode's metrics
3. Update status when training completes or fails

### Viewing Runs via API

```python
import requests

# List all runs
response = requests.get("http://localhost:8000/api/training-runs")
runs = response.json()

# Get specific run
run_id = runs[0]["run_id"]
response = requests.get(f"http://localhost:8000/api/training-runs/{run_id}")
run_details = response.json()

# Filter by agent
response = requests.get(
    "http://localhost:8000/api/training-runs",
    params={"agent_id": "ppo_agent", "status": "completed"}
)
filtered_runs = response.json()
```

### Accessing from Frontend

The Training History page automatically:
- Fetches runs on load
- Updates when filters change
- Navigates to detail view on click
- Handles loading and error states

## Troubleshooting

### Training Runs Not Appearing

1. **Check MongoDB connection**: Ensure MongoDB is running and the API can connect
2. **Check training logs**: Look for errors in the training script output
3. **Verify logger initialization**: Check that `TrainingRunLogger` is created successfully

### Missing Metrics

- **No episodes logged**: Training may have failed before completing an episode
- **No win rates**: Win detection may not be implemented for your environment
- **Incomplete data**: Training may have been interrupted

### Database Issues

If MongoDB is unavailable:
- Training will continue but runs won't be logged
- API endpoints will return 503 errors
- Frontend will show error messages

To fix:
1. Ensure MongoDB is running: `mongosh --eval "db.adminCommand('ping')"`
2. Check connection string: `MONGODB_URI` environment variable
3. Verify database name: `MONGODB_DB_NAME` environment variable

## Future Enhancements

Potential improvements:

- [ ] Real-time updates for running training sessions
- [ ] Comparison view for multiple runs
- [ ] Export metrics to CSV/JSON
- [ ] Advanced filtering and search
- [ ] Automated evaluation after training
- [ ] Integration with TensorBoard logs
- [ ] Performance benchmarking across runs

## Related Documentation

- [Training Configuration](./training/README.md): Training config system and smoke-test mode
- [MongoDB Setup](../webapi/README.md): MongoDB connection and setup
- [API Documentation](../webapi/README.md): Complete API reference

