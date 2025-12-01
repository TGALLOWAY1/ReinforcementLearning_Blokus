# Evaluation Protocol Documentation

## Overview

The evaluation protocol provides a systematic way to assess trained RL agents by comparing them against baseline agents. This helps validate that training is producing agents that learn meaningful strategies, not just random behavior.

## Baseline Agents

### RandomAgent

**Purpose**: Provides a minimal baseline - any trained agent should significantly outperform random play.

**Behavior**: 
- Selects uniformly from all legal moves at each step
- No strategic thinking
- Reproducible with seed control

**When to use**: Always evaluate against random to establish a minimum performance bar.

### HeuristicAgent

**Purpose**: Provides a rule-based baseline that demonstrates basic strategic thinking.

**Behavior**:
- Prefers larger pieces (more tiles placed)
- Creates new corners for future moves
- Avoids edges early in the game
- Prefers center positions

**When to use**: Evaluate against heuristic to see if RL agent learns strategies beyond simple rules.

## Evaluation Procedure

### Running Evaluation

Evaluate a trained checkpoint:

```bash
# Basic evaluation (50 games vs random and heuristic)
python training/evaluate_agent.py checkpoints/ppo_agent/run123/ep000100.zip

# More games for better statistics
python training/evaluate_agent.py checkpoints/ppo_agent/run123/ep000100.zip --num-games 100

# Specific opponents
python training/evaluate_agent.py checkpoints/ppo_agent/run123/ep000100.zip \
  --opponents random heuristic

# With seed for reproducibility
python training/evaluate_agent.py checkpoints/ppo_agent/run123/ep000100.zip \
  --num-games 50 --seed 42

# Link to training run
python training/evaluate_agent.py checkpoints/ppo_agent/run123/ep000100.zip \
  --training-run-id 550e8400-e29b-41d4-a716-446655440000
```

### Evaluation Metrics

For each opponent type, the evaluation computes:

1. **Win Rate**: Percentage of games won (0.0 to 1.0)
   - Higher is better
   - > 0.5 indicates agent outperforms opponent
   - > 0.7 indicates strong performance

2. **Average Reward**: Mean score per game
   - Higher is better
   - Compare to baseline scores (random ~10-15, heuristic ~15-25)

3. **Average Game Length**: Mean number of steps per game
   - Indicates game complexity
   - Longer games may indicate more strategic play

4. **Score Distribution**: Individual game scores
   - Shows consistency vs variance
   - Stable high scores > variable scores

### Interpreting Results

**Good Signs:**
- Win rate > 0.6 vs random
- Win rate > 0.5 vs heuristic
- Consistent high scores (low variance)
- Steady improvement over training

**Warning Signs:**
- Win rate < 0.5 vs random (agent not learning)
- High variance in scores (unstable)
- No improvement over training episodes

## Evaluation vs Training Metrics

### Training Metrics (from Training History)

- **Episode Reward**: Reward during training episodes
- **Learning Curve**: How reward changes over time
- **Training Loss**: Policy/value function loss

**Use for**: Monitoring training progress, detecting overfitting, tuning hyperparameters

### Evaluation Metrics (from Evaluation Protocol)

- **Win Rate**: Performance against fixed opponents
- **Average Reward**: Score in actual games
- **Game Length**: Complexity of games played

**Use for**: Validating that training produced a good agent, comparing different training runs, demonstrating progress

### Why Both?

- **Training metrics** can be misleading (reward shaping, exploration, etc.)
- **Evaluation metrics** show real-world performance
- **Together** they provide a complete picture

## Evaluation Workflow

### Step 1: Train Agent

```bash
python training/trainer.py --agent-config config/agents/ppo_agent_v1.yaml --mode full
```

### Step 2: Evaluate Checkpoint

```bash
# Evaluate final checkpoint
python training/evaluate_agent.py checkpoints/ppo_agent/run123/ep000200.zip --num-games 100
```

### Step 3: View Results

1. Open Training History: `http://localhost:5173/training`
2. Click on training run
3. Scroll to "Evaluation Results" section
4. See win rates, average rewards, and game lengths

### Step 4: Compare Runs

Compare evaluation results across different training runs:
- Different hyperparameters
- Different training durations
- Different agent configs

## EvaluationRun Records

Each evaluation creates `EvaluationRun` records in MongoDB:

- **training_run_id**: Links to the TrainingRun
- **checkpoint_path**: Which checkpoint was evaluated
- **opponent_type**: Type of opponent tested
- **games_played**: Number of games
- **win_rate**: Win rate (0.0 to 1.0)
- **avg_reward**: Average reward
- **avg_game_length**: Average game length
- **created_at**: When evaluation was run

## Best Practices

### When to Evaluate

1. **After Training**: Evaluate final checkpoint
2. **During Training**: Periodically evaluate checkpoints to track progress
3. **Before Long Runs**: Quick evaluation to verify training is working
4. **After Hyperparameter Changes**: Compare new configs to previous

### How Many Games?

- **Quick Check**: 20-50 games (fast, less accurate)
- **Standard**: 50-100 games (good balance)
- **Thorough**: 100-200 games (high confidence)
- **Publication**: 200+ games (statistical significance)

### Reproducibility

Always use seeds for reproducible evaluations:

```bash
python training/evaluate_agent.py checkpoint.zip --seed 42
```

This ensures:
- Same game sequences
- Comparable results across runs
- Debuggable failures

## Tradeoffs and Limitations

### Why Compare to Baselines?

**Random Agent:**
- Establishes minimum performance bar
- Any learning should beat random
- Fast to evaluate against

**Heuristic Agent:**
- Shows if RL learns beyond simple rules
- Demonstrates strategic improvement
- More challenging opponent

**Self-Play:**
- Tests agent against itself
- Useful for symmetric games
- May not reflect real-world performance

### Limitations

1. **2-Player Simplification**: Current evaluation uses 2-player games, but Blokus is 4-player
   - Future: Extend to 4-player evaluation

2. **RL Agent Wrapper**: Current wrapper is simplified
   - May not perfectly match training environment
   - Future: Improve observation conversion

3. **Limited Opponents**: Only random and heuristic baselines
   - Future: Add more sophisticated baselines (MCTS, etc.)

4. **Fixed Evaluation**: Doesn't adapt to agent strength
   - Future: Adaptive evaluation protocols

## Integration with Training Workflow

### Development Workflow

1. **Train**: Run training with agent config
2. **Evaluate**: Test checkpoint against baselines
3. **Compare**: View results in Training History
4. **Iterate**: Adjust hyperparameters based on results
5. **Repeat**: Continue until satisfactory performance

### Example Workflow

```bash
# 1. Train agent
python training/trainer.py --agent-config config/agents/ppo_agent_v1.yaml

# 2. Evaluate checkpoint
python training/evaluate_agent.py checkpoints/ppo_agent/run123/ep000100.zip \
  --training-run-id <run_id> --num-games 100

# 3. View in UI
# Navigate to Training History and click on run

# 4. If results are good, continue training
# If results are poor, adjust hyperparameters and retrain
```

## Related Documentation

- [Training Configuration](./training/README.md): How to train agents
- [Hyperparameters](./hyperparams.md): Agent configs and sweeps
- [Training History](./training-history.md): Viewing training runs
- [Checkpoints](./checkpoints.md): Saving and loading models

