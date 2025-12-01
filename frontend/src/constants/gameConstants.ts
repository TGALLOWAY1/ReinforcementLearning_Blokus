// Shared game constants to avoid duplication

export const API_BASE = 'http://localhost:8000';
export const WS_BASE = 'ws://localhost:8000';

// Piece definitions with their shapes
export const PIECE_SHAPES: { [key: number]: number[][] } = {
  1: [[1]], // Monomino
  2: [[1, 1]], // Domino
  3: [[1, 1, 1]], // Tromino I
  4: [[1, 0], [1, 1]], // Tromino L
  5: [[1, 1, 1, 1]], // Tetromino I
  6: [[1, 1], [1, 1]], // Tetromino O
  7: [[1, 1, 1], [0, 1, 0]], // Tetromino T
  8: [[1, 0], [1, 0], [1, 1]], // Tetromino L
  9: [[0, 1, 1], [1, 1, 0]], // Tetromino S
  10: [[1, 1, 0], [0, 1, 1]], // Tetromino Z
  11: [[0, 1, 1], [1, 1, 0], [0, 1, 0]], // Pentomino F
  12: [[1, 1, 1, 1, 1]], // Pentomino I
  13: [[1, 0], [1, 0], [1, 0], [1, 1]], // Pentomino L
  14: [[1, 0], [1, 1], [0, 1], [0, 1]], // Pentomino N
  15: [[1, 1], [1, 1], [1, 0]], // Pentomino P
  16: [[1, 1, 1], [0, 1, 0], [0, 1, 0]], // Pentomino T
  17: [[1, 0, 1], [1, 1, 1]], // Pentomino U
  18: [[1, 0, 0], [1, 0, 0], [1, 1, 1]], // Pentomino V
  19: [[1, 0, 0], [1, 1, 0], [0, 1, 1]], // Pentomino W
  20: [[0, 1, 0], [1, 1, 1], [0, 1, 0]], // Pentomino X
  21: [[1, 1, 1, 1], [0, 1, 0, 0]] // Pentomino Y
};

export const PIECE_NAMES: { [key: number]: string } = {
  1: 'Monomino',
  2: 'Domino', 
  3: 'Tromino I',
  4: 'Tromino L',
  5: 'Tetromino I',
  6: 'Tetromino O',
  7: 'Tetromino T',
  8: 'Tetromino L',
  9: 'Tetromino S',
  10: 'Tetromino Z',
  11: 'Pentomino F',
  12: 'Pentomino I',
  13: 'Pentomino L',
  14: 'Pentomino N',
  15: 'Pentomino P',
  16: 'Pentomino T',
  17: 'Pentomino U',
  18: 'Pentomino V',
  19: 'Pentomino W',
  20: 'Pentomino X',
  21: 'Pentomino Y'
};

// Modern color-blind friendly palette
export const PLAYER_COLORS = {
  red: '#EF4444',       // Modern red
  blue: '#3B82F6',      // Modern blue  
  green: '#10B981',     // Modern green
  yellow: '#F59E0B',    // Modern amber
  empty: '#F8FAFC',     // Slate-50
  hover: '#E2E8F0',     // Slate-200
  grid: '#CBD5E1',      // Slate-300
  preview: '#3B82F6',   // Blue for piece preview
  previewBg: '#DBEAFE'  // Blue-100 for preview background
};

export const BOARD_SIZE = 20;
export const CELL_SIZE = 20;
export const PIECE_SIZE = 12;
