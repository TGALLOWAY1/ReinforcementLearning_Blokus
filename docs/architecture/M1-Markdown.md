# M1 Branch - Major Changes Documentation

## Overview

The M1 branch (`M1---create-new-front-end`) introduces significant enhancements to the Blokus RL project, focusing on a new React-based frontend, improved training infrastructure, and enhanced game state management. This document outlines all major changes implemented in this branch.

---

## 1. Frontend Architecture Overhaul

### 1.1 New React/TypeScript Frontend

A complete frontend rewrite using modern web technologies:

- **Technology Stack**: React 18, TypeScript, Vite, Tailwind CSS
- **Architecture**: Component-based architecture with Zustand state management
- **Routing**: React Router for multi-page navigation

#### Key Frontend Files:
- `frontend/src/App.tsx` - Main application component
- `frontend/src/main.tsx` - Application entry point
- `frontend/src/pages/` - Page components (Home, Play, TrainEval)
- `frontend/src/store/gameStore.ts` - Centralized state management

### 1.2 New Frontend Components

#### AgentVisualizations Component
**Location**: `frontend/src/components/AgentVisualizations.tsx`

A comprehensive visualization component for RL agent insights:

- **Policy Heatmap**: Visual representation of agent's policy over the game board
  - Binary heatmap mode for legal/illegal moves
  - Continuous policy visualization with probability gradients
  - Color-coded overlays (neon red for high probability, neon blue for low)
  
- **Value Function Display**: Win probability visualization
  - Gradient bar chart showing estimated win probability
  - Real-time updates based on game state
  
- **MCTS Tree Visualization**: Placeholder for future tree structure visualization

- **Piece Selection Interface**: Interactive piece management
  - Visual piece representations with rotation/flip support
  - Keyboard shortcuts (R for rotate, F for flip)
  - Piece availability tracking
  - Used piece indicators

#### ResearchSidebar Component
**Location**: `frontend/src/components/ResearchSidebar.tsx`

A research-focused sidebar for training and experimentation:

- **Environment Controls**:
  - Start Episode button
  - Reset functionality
  - Auto-step toggle for automated gameplay
  
- **Training Parameters**:
  - Learning rate configuration
  - Exploration (ε) parameter
  - Discount factor (γ) setting
  - Agent type selection (DQN, PPO, MCTS)
  
- **Model Status Dashboard**:
  - Episode counter
  - Win rate statistics
  - Reward trend sparkline chart
  - Real-time training metrics

#### LogConsole Component
**Location**: `frontend/src/components/LogConsole.tsx`

A terminal-style console for game logs:

- Auto-scrolling log display
- Timestamp and log level formatting
- Real-time log updates from game state
- Terminal-style dark theme

#### Enhanced Board Component
**Location**: `frontend/src/components/Board.tsx`

Improvements to the game board visualization:

- Enhanced cell interaction handling
- Better visual feedback for legal moves
- Improved piece placement visualization
- Heatmap overlay support

### 1.3 Frontend State Management

**Location**: `frontend/src/store/gameStore.ts`

Centralized state management using Zustand:

- **Game State**: Current board state, players, scores, pieces used
- **WebSocket Integration**: Real-time game updates via WebSocket
- **Move Management**: Piece selection, orientation, move execution
- **Error Handling**: Centralized error state management
- **Log Management**: Game event logging system

### 1.4 Frontend Styling

**Location**: `frontend/src/index.css`, `frontend/tailwind.config.js`

- **Dark Theme**: Cyberpunk-inspired dark color scheme
- **Neon Accents**: Neon blue, red, and green for highlights
- **Responsive Design**: Mobile-friendly layouts
- **Custom Tailwind Configuration**: Extended color palette and utilities

---

## 2. Training Infrastructure Improvements

### 2.1 Enhanced Training Script

**Location**: `training/trainer.py`

Complete rewrite of the training script with robust features:

#### Key Improvements:
- **MaskablePPO Integration**: Proper integration with `sb3_contrib.MaskablePPO`
- **ActionMasker Wrapper**: Automatic action masking for legal moves only
- **Progress Bar Support**: `progress_bar=True` to prevent UI hanging
- **Command-Line Arguments**: Flexible training configuration via argparse
  - `--total_timesteps`: Total training duration (default: 100,000)
  - `--n_steps`: Steps per update (default: 2048, can be lowered to 512 for faster first update)
  - `--lr`: Learning rate (default: 3e-4)
- **Tensorboard Logging**: Automatic logging to `./logs` directory
- **Model Checkpointing**: Saves trained models to `checkpoints/ppo_blokus`

#### Action Masking Implementation:
```python
def mask_fn(env):
    """Extracts legal action mask from environment"""
    blokus_env = env.env
    agent_name = env.agent_name
    mask = blokus_env.infos[agent_name]["legal_action_mask"]
    return np.asarray(mask, dtype=np.bool_)
```

### 2.2 Dependencies Update

**Location**: `requirements.txt`

Added new dependencies for RL training:

- **`tqdm`**: Progress bar library for training visualization
- **`sb3-contrib`**: Already present, required for MaskablePPO
- **`shimmy`**: Already present, required for PettingZoo/Gymnasium compatibility

---

## 3. Web API Enhancements

### 3.1 Game Manager Updates

**Location**: `webapi/game_manager.py`

Improvements to game session management:

- Enhanced game state tracking
- Better player management
- Improved move validation
- WebSocket connection handling

### 3.2 API Endpoint Updates

**Location**: `webapi/app.py`

- Enhanced REST endpoints for game management
- Improved WebSocket message handling
- Better error responses
- Real-time state synchronization

---

## 4. Schema Updates

### 4.1 Game State Schema

**Location**: `schemas/game_state.py`

Enhanced game state representation:

- **Heatmap Support**: Added `heatmap` field for policy visualization
- **Player Information**: Enhanced player configuration tracking
- **Pieces Tracking**: Improved pieces_used tracking per player
- **State Metadata**: Additional metadata for frontend rendering

### 4.2 State Update Schema

**Location**: `schemas/state_update.py`

Improved state update messages:

- Real-time game state updates
- Move validation responses
- Error message formatting
- WebSocket message structure

---

## 5. Game Constants and Configuration

### 5.1 Frontend Game Constants

**Location**: `frontend/src/constants/gameConstants.ts`

Centralized game configuration:

- Board dimensions (20x20)
- Cell size for rendering
- Piece names mapping
- Color schemes
- UI constants

---

## 6. Testing Infrastructure

### 6.1 New Test Files

- **`tests/smoke_test_env.py`**: Environment smoke tests
- **`tests/smoke_test_game.py`**: Game engine smoke tests
- **`tests/verify_engine.py`**: Engine verification tests
- **`tests/test_blokus_env.py`**: Comprehensive environment tests

---

## 7. Build and Development Tools

### 7.1 Frontend Build Configuration

- **Vite Configuration**: Fast build tool and dev server
- **TypeScript Configuration**: Strict type checking
- **Tailwind CSS**: Utility-first CSS framework
- **PostCSS**: CSS processing pipeline

### 7.2 Setup Scripts

**Location**: `setup_frontend.sh`

Automated frontend setup script for:
- Node.js dependency installation
- Development server configuration
- Build process setup

---

## 8. Key Features Summary

### 8.1 User Interface Features

1. **Interactive Game Board**: Click-to-place piece interface
2. **Real-time Visualizations**: Policy heatmaps and value functions
3. **Piece Management**: Visual piece tray with rotation/flip
4. **Research Tools**: Training parameter controls and metrics
5. **Log Console**: Real-time game event logging
6. **Error Handling**: User-friendly error messages

### 8.2 Training Features

1. **Progress Visualization**: Real-time training progress bars
2. **Action Masking**: Automatic filtering of illegal moves
3. **Flexible Configuration**: Command-line parameter tuning
4. **Model Persistence**: Automatic checkpoint saving
5. **Tensorboard Integration**: Training metrics visualization

### 8.3 Technical Improvements

1. **Type Safety**: Full TypeScript implementation
2. **State Management**: Centralized Zustand store
3. **WebSocket Integration**: Real-time bidirectional communication
4. **Error Handling**: Comprehensive error management
5. **Code Organization**: Modular component architecture

---

## 9. Migration Notes

### 9.1 Breaking Changes

- Frontend completely rewritten (no backward compatibility with old frontend)
- Training script API changed (new command-line arguments)
- Game state schema extended (new fields may be required)

### 9.2 New Requirements

- Node.js 16+ for frontend development
- Python 3.7+ with updated dependencies
- Modern browser with WebSocket support

### 9.3 Setup Instructions

1. **Backend Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **Training**:
   ```bash
   python training/trainer.py --total_timesteps 10000 --n_steps 512
   ```

---

## 10. Future Enhancements

### Planned Features

1. **MCTS Tree Visualization**: Complete tree structure rendering
2. **Real-time Training Metrics**: Live training dashboard
3. **Agent Comparison Tools**: Side-by-side agent evaluation
4. **Replay System**: Game replay and analysis
5. **Export Functionality**: Model and game state export

---

## 11. File Structure Changes

### New Directories
- `frontend/` - Complete React frontend application
- `docs/` - Project documentation
- `logs/` - Tensorboard log files
- `checkpoints/` - Saved model checkpoints

### Modified Files
- `training/trainer.py` - Complete rewrite
- `requirements.txt` - Added tqdm
- `webapi/app.py` - Enhanced API endpoints
- `webapi/game_manager.py` - Improved game management
- `schemas/game_state.py` - Extended schema
- `schemas/state_update.py` - Enhanced update messages

---

## 12. Performance Considerations

### Frontend Performance
- Vite for fast development and optimized builds
- React component memoization where appropriate
- Efficient state updates via Zustand
- WebSocket connection pooling

### Training Performance
- Configurable batch sizes for memory management
- Progress bar to prevent UI blocking
- Efficient action masking
- Tensorboard logging for monitoring

---

## 13. Known Issues and Limitations

1. **Action Masking**: Some edge cases in multi-agent environment wrapping may cause issues
2. **MCTS Visualization**: Tree visualization is placeholder (not yet implemented)
3. **Training Stability**: Initial training may require tuning of n_steps parameter
4. **Browser Compatibility**: Requires modern browser with WebSocket support

---

## 14. Contributors and Credits

This branch represents a major milestone in the Blokus RL project, introducing:
- Modern frontend architecture
- Enhanced training capabilities
- Improved user experience
- Better development tooling

---

## 15. Conclusion

The M1 branch successfully delivers a comprehensive frontend rewrite and training infrastructure improvements. The new architecture provides a solid foundation for future development, with enhanced visualization capabilities, better user experience, and more robust training tools.

**Key Achievements**:
- ✅ Complete React/TypeScript frontend
- ✅ Enhanced training script with progress bars
- ✅ Real-time game visualization
- ✅ Improved state management
- ✅ Better error handling
- ✅ Comprehensive testing infrastructure

---

*Document generated for M1 branch (`M1---create-new-front-end`)*
*Last updated: Based on commit history and current branch state*


