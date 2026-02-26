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
  game_history?: GameHistoryEntry[];
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

import { type PlayerMobilityMetrics } from '../utils/mobilityMetrics';
import { useDebugLogStore } from './debugLogStore';

export interface GameHistoryEntry {
  turn_number: number;
  player_to_move: string;
  action: {
    piece_id: number;
    orientation: number;
    anchor_row: number;
    anchor_col: number;
  };
  board_state: number[][];
  metrics: {
    corner_count: Record<string, number>;
    frontier_size: Record<string, number>;
    difficult_piece_penalty: Record<string, number>;
    remaining_pieces: Record<string, number[]>;
    influence_map: number[][] | null;
  };
}

// Store interface
interface GameStore {
  gameState: GameState | null;
  currentSliderTurn: number | null;
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
  setCurrentSliderTurn: (turn: number | null) => void;
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
  saveGame: () => void;
  loadGame: (history: GameHistoryEntry[]) => Promise<void>;
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
    currentSliderTurn: null,
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
        currentSliderTurn: null,
        isAdvancingTurn: false,
      });
    },

    setCurrentSliderTurn: (turn: number | null) => {
      set({ currentSliderTurn: turn });
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
          return { gameState: null, previewMove: null, currentSliderTurn: null };
        }
        const prev = state.gameState;
        if (ENABLE_DEBUG_UI) useDebugLogStore.getState().setStateDiff(prev, gameState);

        let nextSliderTurn = state.currentSliderTurn;
        const totalTurns = gameState.game_history?.length || 0;
        const prevTotalTurns = prev?.game_history?.length || 0;

        if (state.currentSliderTurn === null || state.currentSliderTurn >= prevTotalTurns || prevTotalTurns === 0) {
          nextSliderTurn = totalTurns;
        }

        return {
          gameState,
          previewMove: null,
          currentSliderTurn: nextSliderTurn,
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

    saveGame: () => {
      const state = get().gameState;
      if (!state || !state.game_history) return;

      const blob = new Blob([JSON.stringify(state.game_history, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `blokus_game_${state.game_id}_${new Date().toISOString()}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    },

    loadGame: async (history: GameHistoryEntry[]) => {
      setupWorker();
      const waitForWorker = async () => {
        while (!isWorkerReady) {
          await new Promise(r => setTimeout(r, 100));
        }
      };
      await waitForWorker();

      return new Promise((resolve) => {
        set({ connectionStatus: 'connecting' });
        workerInstance!.postMessage({ type: 'load_game', history });
        // The worker will respond with state_update which calls setGameState
        // and resolve the promise implicitly via message listener if needed, 
        // but for now we just wait for the update to happen.
        resolve();
      });
    },
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
