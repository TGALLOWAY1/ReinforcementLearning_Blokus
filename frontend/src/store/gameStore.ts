import { useMemo } from 'react';
import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { ENABLE_DEBUG_UI } from '../constants/gameConstants';
import BlokusWorker from './blokusWorker?worker';

// Types
export interface Position {
  row: number;
  col: number;
}

export interface LegalMove {
  piece_id: number;
  orientation: number;
  anchor_row: number;
  anchor_col: number;
  positions: Position[];
}

export interface PlayerState {
  player: string;
  score: number;
  pieces_used: number[];
  pieces_remaining: number[];
  is_active: boolean;
}

export interface BoardState {
  board: number[][];
  move_count: number;
}

export interface GameState {
  game_id: string;
  status: string;
  current_player: string;
  board: number[][];
  scores: { [key: string]: number };
  pieces_used: { [key: string]: number[] };
  move_count: number;
  game_over: boolean;
  winner: string | null;
  legal_moves: LegalMove[];
  created_at: string;
  updated_at: string;
  players?: Array<{
    player: string;
    agent_type: string;
    agent_config: any;
  }>;
  heatmap?: number[][];
  mobility_metrics?: PlayerMobilityMetrics;
  mcts_top_moves?: MctsTopMove[];
  influence_map?: number[][];
  dead_zones?: boolean[][];
  advanced_metrics?: {
    [player: string]: {
      corner_differential: number;
      territory_ratio: number;
      piece_penalty: number;
      center_proximity: number;
      opponent_adjacency: number;
    }
  };
}

export interface MctsTopMove {
  piece_id: number;
  orientation: number;
  anchor_row: number;
  anchor_col: number;
  visits: number;
  q_value: number;
}

export interface MoveRequest {
  player: string;
  piece_id: number;
  orientation: number;
  anchor_row: number;
  anchor_col: number;
}

export interface MoveResponse {
  success: boolean;
  message: string;
  new_score?: number;
  game_over: boolean;
  winner?: string;
  game_state?: GameState;
}

export interface LogEntry {
  timestamp: string;
  message: string;
  level?: 'INFO' | 'WARN' | 'ERROR';
}

import { computeMobilityMetrics, type PlayerMobilityMetrics } from '../utils/mobilityMetrics';
import { useDebugLogStore } from './debugLogStore';

export interface LegalMovesHistoryEntry {
  turn: number;
  byPlayer: Record<string, PlayerMobilityMetrics>;
  advanced?: {
    [player: string]: {
      corner_differential: number;
      territory_ratio: number;
      piece_penalty: number;
      center_proximity: number;
      opponent_adjacency: number;
    }
  };
}

// Store interface
interface GameStore {
  gameState: GameState | null;
  legalMovesHistory: LegalMovesHistoryEntry[];
  connectionStatus: 'disconnected' | 'connecting' | 'connected';
  selectedPiece: number | null;
  pieceOrientation: number;
  previewMove: MctsTopMove | null;
  pollIntervalId: ReturnType<typeof setInterval> | null;
  isAdvancingTurn: boolean;
  error: string | null;
  logs: LogEntry[];
  activeRightTab: 'main' | 'telemetry';

  connect: (gameId: string) => Promise<void>;
  disconnect: () => void;
  selectPiece: (pieceId: number | null) => void;
  setPieceOrientation: (orientation: number) => void;
  setPreviewMove: (move: MctsTopMove | null) => void;
  makeMove: (move: MoveRequest) => Promise<MoveResponse>;
  passTurn: (player: string) => Promise<MoveResponse>;
  createGame: (config: any) => Promise<string>;
  setError: (error: string | null) => void;
  setGameState: (gameState: GameState | null) => void;
  addLog: (message: string, level?: 'INFO' | 'WARN' | 'ERROR') => void;
  clearLogs: () => void;
  setActiveRightTab: (tab: 'main' | 'telemetry') => void;
}

let workerInstance: Worker | null = null;
let initResolver: ((val: string) => void) | null = null;
let moveResolver: ((val: MoveResponse) => void) | null = null;
let isWorkerReady = false;

function setupWorker() {
  if (workerInstance) return workerInstance;
  workerInstance = new BlokusWorker();

  workerInstance.onmessage = (e) => {
    const data = e.data;
    if (data.type === 'ready') {
      isWorkerReady = true;
      useGameStore.getState().addLog("WebWorker & Pyodide Ready", "INFO");
    } else if (data.type === 'state_update') {
      useGameStore.getState().setGameState(data.state);
      if (initResolver) {
        initResolver(data.state.game_id);
        initResolver = null;
      }
      triggerAgentTurnIfNeeded();
    } else if (data.type === 'move_response') {
      const resp = data.response;
      useGameStore.getState().setGameState(resp.game_state);
      useGameStore.setState({ isAdvancingTurn: false });
      if (moveResolver) {
        moveResolver(resp);
        moveResolver = null;
      }
      triggerAgentTurnIfNeeded();
    } else if (data.type === 'error' || data.type === 'init_error') {
      console.error("Worker Error:", data.error);
      useGameStore.getState().setError(data.error);
      useGameStore.getState().addLog("Worker Error: " + data.error, "ERROR");
    }
  };
  return workerInstance;
}

function triggerAgentTurnIfNeeded() {
  const state = useGameStore.getState();
  const gameState = state.gameState;
  if (!gameState || gameState.game_over || gameState.status !== 'in_progress') return;

  const currentPlayer = gameState.current_player;
  const pConf = gameState.players?.find(p => p.player === currentPlayer);

  if (pConf && pConf.agent_type !== 'human' && !state.isAdvancingTurn) {
    useGameStore.setState({ isAdvancingTurn: true });
    workerInstance?.postMessage({ type: 'advance_turn' });
  }
}

export const useGameStore = create<GameStore>()(
  subscribeWithSelector((set, get) => ({
    gameState: null,
    legalMovesHistory: [],
    connectionStatus: 'disconnected',
    selectedPiece: null,
    pieceOrientation: 0,
    previewMove: null,
    pollIntervalId: null,
    isAdvancingTurn: false,
    error: null,
    logs: [],
    activeRightTab: 'main',

    connect: async (gameId: string): Promise<void> => {
      // With Pyodide, state is local. If we reload the page, state is gone.
      // So connect only works if we already just created the game locally in this session.
      set({ connectionStatus: 'connected', error: null, isAdvancingTurn: false });
      get().addLog(`Connected to local Pyodide game ${gameId}`, 'INFO');
      triggerAgentTurnIfNeeded();
    },

    disconnect: () => {
      set({
        connectionStatus: 'disconnected',
        pollIntervalId: null,
        gameState: null,
        legalMovesHistory: [],
        isAdvancingTurn: false,
      });
    },

    selectPiece: (pieceId: number | null) => {
      set({ selectedPiece: pieceId, pieceOrientation: 0 });
    },

    setPieceOrientation: (orientation: number) => {
      set({ pieceOrientation: orientation });
    },

    setPreviewMove: (move: MctsTopMove | null) => {
      set({ previewMove: move });
    },

    passTurn: async (_player: string): Promise<MoveResponse> => {
      return new Promise((resolve) => {
        moveResolver = resolve;
        if (!workerInstance) setupWorker();
        workerInstance!.postMessage({ type: 'pass_turn' });
      });
    },

    makeMove: async (move: MoveRequest): Promise<MoveResponse> => {
      return new Promise((resolve) => {
        moveResolver = resolve;
        if (!workerInstance) setupWorker();
        workerInstance!.postMessage({
          type: 'make_move',
          move: {
            piece_id: move.piece_id,
            orientation: move.orientation,
            anchor_row: move.anchor_row,
            anchor_col: move.anchor_col
          }
        });
      });
    },

    createGame: async (config: any): Promise<string> => {
      setupWorker(); // Ensure worker is running

      const waitForWorker = async () => {
        while (!isWorkerReady) {
          await new Promise(r => setTimeout(r, 100));
        }
      };
      await waitForWorker();

      return new Promise((resolve) => {
        initResolver = resolve;
        config.game_id = "local-" + Math.floor(Math.random() * 1000000);
        workerInstance!.postMessage({ type: 'init_game', config });
      });
    },

    setError: (error: string | null) => {
      set({ error });
    },

    setGameState: (gameState: GameState | null) => {
      set((state) => {
        if (!gameState) {
          if (ENABLE_DEBUG_UI) useDebugLogStore.getState().setStateDiff(state.gameState, null);
          return { gameState: null, legalMovesHistory: [], previewMove: null };
        }
        const prev = state.gameState;
        if (ENABLE_DEBUG_UI) useDebugLogStore.getState().setStateDiff(prev, gameState);
        const piecesUsed = gameState.pieces_used?.[gameState.current_player] ?? [];
        const metrics =
          gameState.mobility_metrics ??
          computeMobilityMetrics(gameState.legal_moves ?? [], piecesUsed);

        if (prev?.game_id !== gameState.game_id) {
          const entry: LegalMovesHistoryEntry = {
            turn: 0,
            byPlayer: { [gameState.current_player]: metrics },
            advanced: gameState.advanced_metrics,
          };
          return { gameState, legalMovesHistory: [entry], previewMove: null };
        }

        const prevKey = prev ? `${prev.move_count}-${prev.current_player}` : '';
        const newKey = `${gameState.move_count}-${gameState.current_player}`;
        if (prevKey === newKey) return { gameState };

        const entry: LegalMovesHistoryEntry = {
          turn: state.legalMovesHistory.length,
          byPlayer: { [gameState.current_player]: metrics },
          advanced: gameState.advanced_metrics,
        };
        return {
          gameState,
          previewMove: null,
          legalMovesHistory: [...state.legalMovesHistory, entry],
        };
      });
    },

    addLog: (message: string, level: 'INFO' | 'WARN' | 'ERROR' = 'INFO') => {
      const now = new Date();
      const timestamp = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
      const logEntry: LogEntry = {
        timestamp,
        message,
        level,
      };
      set((state) => ({
        logs: [...state.logs, logEntry].slice(-1000),
      }));
    },

    clearLogs: () => { set({ logs: [] }); },
    setActiveRightTab: (tab: 'main' | 'telemetry') => { set({ activeRightTab: tab }); },
  }))
);

export const useLegalMoves = () => {
  const gameState = useGameStore(state => state.gameState);
  return gameState?.legal_moves || [];
};

export function computeLegalMovesByPiece(gameState: GameState | null): { pieceId: number; count: number }[] {
  if (!gameState) return [];
  const legalMoves = gameState.legal_moves || [];
  const currentPlayer = gameState.current_player;
  const piecesUsed = currentPlayer ? (gameState.pieces_used?.[currentPlayer] || []) : [];
  const counts: Record<number, number> = {};
  for (const m of legalMoves) {
    const pid = m.piece_id ?? (m as any).pieceId;
    if (pid != null) counts[pid] = (counts[pid] ?? 0) + 1;
  }
  return Array.from({ length: 21 }, (_, i) => i + 1).map((pieceId) => ({
    pieceId,
    count: piecesUsed.includes(pieceId) ? 0 : (counts[pieceId] ?? 0),
  }));
}

export const useLegalMovesByPiece = (): { pieceId: number; count: number }[] => {
  const gameState = useGameStore(state => state.gameState);
  const legalMoves = gameState?.legal_moves || [];
  const currentPlayer = gameState?.current_player;
  const piecesUsed = currentPlayer ? (gameState?.pieces_used?.[currentPlayer] || []) : [];

  return useMemo(() => {
    const counts: Record<number, number> = {};
    for (const m of legalMoves) {
      const pid = m.piece_id ?? (m as any).pieceId;
      if (pid != null) {
        counts[pid] = (counts[pid] ?? 0) + 1;
      }
    }
    return Array.from({ length: 21 }, (_, i) => i + 1).map((pieceId) => ({
      pieceId,
      count: piecesUsed.includes(pieceId) ? 0 : (counts[pieceId] ?? 0),
    }));
  }, [legalMoves, piecesUsed]);
};

export const useCurrentPlayer = () => {
  const gameState = useGameStore(state => state.gameState);
  return gameState?.current_player || null;
};

export const useIsMyTurn = () => {
  const currentPlayer = useCurrentPlayer();
  const gameState = useGameStore(state => state.gameState);
  if (!gameState || !currentPlayer) return false;
  return gameState?.current_player === currentPlayer;
};
