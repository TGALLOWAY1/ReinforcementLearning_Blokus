# Blokus RL Frontend

A React-based web interface for playing Blokus against AI agents.

## Features

- **Visual Game Board**: 20x20 SVG grid with color-blind friendly palette
- **Interactive Piece Selection**: Draggable, rotatable pieces with keyboard shortcuts
- **Real-time Updates**: WebSocket integration for live game state
- **Agent Controls**: Configure different AI opponents
- **Responsive Design**: Works on desktop and mobile devices

## Setup Instructions

### Prerequisites

1. **Install Node.js**: Download from [nodejs.org](https://nodejs.org/) or use a package manager:
   ```bash
   # macOS with Homebrew
   brew install node
   
   # Ubuntu/Debian
   sudo apt install nodejs npm
   
   # Windows with Chocolatey
   choco install nodejs
   ```

2. **Verify Installation**:
   ```bash
   node --version
   npm --version
   ```

### Installation

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```

4. **Open your browser** to `http://localhost:5173`

## Usage

### Playing a Game

1. **Start the Backend**: Make sure the Blokus API server is running:
   ```bash
   cd ..  # Go back to project root
   python3 run_server.py
   ```

2. **Create a Game**: On the home page, configure your game:
   - Choose opponents (Random, Heuristic, MCTS agents)
   - Set game parameters
   - Click "Start New Game"

3. **Play**: 
   - Select a piece from your tray
   - Use **R** to rotate, **F** to flip
   - Click on the board to place the piece
   - Watch agents play automatically

### Keyboard Shortcuts

- **R**: Rotate selected piece
- **F**: Flip selected piece
- **Click**: Place piece on board

### Color-Blind Friendly Design

The interface uses a carefully chosen color palette:
- **Red**: #E53E3E (High contrast)
- **Blue**: #3182CE (Distinct from red)
- **Green**: #38A169 (Clear differentiation)
- **Yellow**: #D69E2E (Orange-tinted for visibility)

## Project Structure

```
frontend/src/
├── components/
│   ├── Board.tsx          # 20x20 SVG game board
│   ├── PieceTray.tsx      # Draggable piece selection
│   └── Sidebar.tsx        # Game info and controls
├── pages/
│   ├── Home.tsx           # Game creation page
│   ├── Play.tsx           # Main game interface
│   └── TrainEval.tsx      # Training/evaluation dashboard
├── store/
│   └── gameStore.ts       # Zustand state management
└── App.tsx                # Main app with routing
```

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Adding New Features

1. **New Components**: Add to `src/components/`
2. **New Pages**: Add to `src/pages/` and update `App.tsx`
3. **State Management**: Extend `src/store/gameStore.ts`
4. **Styling**: Use Tailwind CSS classes

### WebSocket Integration

The frontend connects to the backend via WebSocket for real-time updates:

```typescript
// Connect to game
const { connect } = useGameStore();
await connect(gameId);

// Make moves
const { makeMove } = useGameStore();
await makeMove({
  player: 'red',
  piece_id: 1,
  orientation: 0,
  anchor_row: 0,
  anchor_col: 0
});
```

## Troubleshooting

### Common Issues

1. **"Module not found" errors**: Run `npm install` to ensure all dependencies are installed

2. **WebSocket connection fails**: 
   - Ensure the backend server is running on `http://localhost:8000`
   - Check browser console for connection errors

3. **Pieces not displaying**: 
   - Verify the game state is loaded
   - Check that legal moves are available

4. **Styling issues**: 
   - Ensure Tailwind CSS is properly configured
   - Check that PostCSS is processing styles

### Browser Compatibility

- **Chrome**: Full support
- **Firefox**: Full support  
- **Safari**: Full support
- **Edge**: Full support

## Contributing

1. Follow the existing code style
2. Use TypeScript for type safety
3. Add proper error handling
4. Test on multiple browsers
5. Update documentation as needed

## License

This project is part of the Blokus RL research environment.
