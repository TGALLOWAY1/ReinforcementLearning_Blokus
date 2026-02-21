// Shared game constants to avoid duplication
// Environment variables are loaded from .env file (development) or OS env (production)
// Vite requires VITE_ prefix for environment variables exposed to client code

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const APP_PROFILE_RAW = (import.meta.env.VITE_APP_PROFILE || 'research').toLowerCase();

export const APP_PROFILE: 'research' | 'deploy' = APP_PROFILE_RAW === 'deploy' ? 'deploy' : 'research';
export const IS_DEPLOY_PROFILE = APP_PROFILE === 'deploy';

export const DEPLOY_MCTS_PRESETS = {
  easy: 200,
  medium: 450,
  hard: 900,
} as const;

// Derive WebSocket URL from API URL if VITE_WS_URL is not set
// Convert http:// to ws:// and https:// to wss://
const getWebSocketURL = (): string => {
  if (import.meta.env.VITE_WS_URL) {
    return import.meta.env.VITE_WS_URL;
  }
  // Convert http://localhost:8000 to ws://localhost:8000
  // Convert https://example.com to wss://example.com
  return API_URL.replace(/^http:/, 'ws:').replace(/^https:/, 'wss:');
};

// API_BASE: Empty string means use relative URLs (works with Vite proxy in dev)
// In production, this should be the full API URL
export const API_BASE = import.meta.env.PROD ? API_URL : '';

// WS_BASE: WebSocket URL (always needs full URL)
export const WS_BASE = getWebSocketURL();

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
  21: [[1, 0], [1, 1], [1, 0], [1, 0]] // Pentomino Y
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

// Neon color palette for dark research aesthetic
export const PLAYER_COLORS = {
  red: '#FF4D4D',
  blue: '#00F0FF',
  green: '#00FF9D',
  yellow: '#FFE600',
  empty: 'transparent',
  grid: '#3E3E42',
  preview: '#00F0FF',
};

export const BOARD_SIZE = 20;
export const CELL_SIZE = 24;
export const PIECE_SIZE = 12;
