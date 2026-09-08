import { useMemo } from 'react';
import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { ENABLE_DEBUG_UI } from '../constants/gameConstants';
import { SearchTraceV1 } from '../types/mcts';
import BlokusWorker from './blokusWorker?worker';
import { createBackendGame, deleteBackendGame, isBackendConfig, postAdvanceTurn, postMove, postPass } from './backendTransport';

/** Where a game runs: the in-browser Pyodide worker, or the FastAPI backend (natively served agents). */
export type GameTransport = 'worker' | 'backend';

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
  scoring_mode?: string;
  legal_moves: LegalMove[];
  /** Orientation index space of legal_moves/last_move ('engine' or 'frontend'); backend games use 'frontend'. */
  orientation_space?: 'engine' | 'frontend';
  /** Last placed move as reported by the backend transport (absent in worker games). */
  last_move?: {
    player: string;
    piece_id: number;
    orientation: number;
    engine_orientation?: number;
    anchor_row: number;
    anchor_col: number;
    is_human?: boolean;
    stats?: Record<string, any>;
  } | null;
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
  mcts_stats?: {
    timeSpentMs?: number;
    timeBudgetMs?: number;
    nodesEvaluated?: number;
    maxDepthReached?: number;
    iterations_run?: number;
    budgetTier?: 'trivial' | 'normal' | 'critical' | string;
    budgetCapMs?: number;
    budgetReasons?: string[];
    earlyStopReason?: string | null;
    visit_entropy?: number;
    regret_gap?: number | null;
  };
  mcts_diagnostics?: MctsDiagnosticsV1;
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
  frontier_metrics?: Record<string, {
    utility: Record<string, number>;
    block_pressure: Record<string, number>;
    urgency: Record<string, number>;
  }>;
  frontier_clusters?: Record<string, {
    cluster_id: Record<string, number>;
    cluster_sizes: number[];
    num_clusters: number;
  }>;
  piece_lock_risk?: Record<string, number>;
  search_trace?: SearchTraceV1;
  self_block_risk?: {
    top_moves: Array<{
      piece_id: number;
      orientation: number;
      anchor_row: number;
      anchor_col: number;
      risk: number;
      clusters_touched: number;
      frontier_points_used: number;
    }>;
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
import { type MoveTelemetryDelta } from '../types/telemetry';
import { type MctsDiagnosticsV1 } from '../types/mcts';
import { type ChampionMetadata } from '../utils/championConfig';

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
    frontier_metrics?: Record<string, {
      utility: Record<string, number>;
      block_pressure: Record<string, number>;
      urgency: Record<string, number>;
    }>;
    frontier_clusters?: Record<string, {
      cluster_id: Record<string, number>;
      cluster_sizes: number[];
      num_clusters: number;
    }>;
    piece_lock_risk?: Record<string, number>;
    self_block_risk?: any;
  };
  telemetry?: MoveTelemetryDelta;
  mcts_diagnostics?: MctsDiagnosticsV1;
}

export interface PendingPlacement {
  player: string;
  piece_id: number;
  orientation: number;
  anchor_row: number;
  anchor_col: number;
}

// Store interface
interface GameStore {
  gameState: GameState | null;
  currentSliderTurn: number | null;
  connectionStatus: 'disconnected' | 'connecting' | 'connected';
  selectedPiece: number | null;
  pieceOrientation: number;
  previewMove: MctsTopMove | null;
  pendingPlacement: PendingPlacement | null;
  pollIntervalId: ReturnType<typeof setInterval> | null;
  isAdvancingTurn: boolean;
  isPaused: boolean;
  error: string | null;
  logs: LogEntry[];
  activeRightTab: 'main' | 'telemetry' | 'explanation' | 'moveDelta' | 'analysis' | 'mcts_analysis';
  boardOverlay: Record<string, { color: string; opacity?: number }> | null;
  analysisModeEnabled: boolean;
  /** Set when the current game is a "Play the Champion" game; powers the banner. */
  championMeta: ChampionMetadata | null;
  /** Transport of the current game (see GameTransport). */
  transport: GameTransport;

  connect: (gameId: string) => Promise<void>;
  /** Advance exactly one agent turn (used by the Step button while paused). */
  advanceTurn: () => void;
  disconnect: () => void;
  togglePause: () => void;
  setAnalysisModeEnabled: (enabled: boolean) => void;
  setCurrentSliderTurn: (turn: number | null) => void;
  selectPiece: (pieceId: number | null) => void;
  setPieceOrientation: (orientation: number) => void;
  setPreviewMove: (move: MctsTopMove | null) => void;
  setPendingPlacement: (placement: PendingPlacement | null) => void;
  makeMove: (move: MoveRequest) => Promise<MoveResponse>;
  passTurn: () => Promise<MoveResponse>;
  createGame: (config: any) => Promise<string>;
  setError: (error: string | null) => void;
  setGameState: (gameState: GameState | null) => void;
  addLog: (message: string, level?: 'INFO' | 'WARN' | 'ERROR') => void;
  clearLogs: () => void;
  setActiveRightTab: (tab: 'main' | 'telemetry' | 'explanation' | 'moveDelta' | 'analysis' | 'mcts_analysis') => void;
  setBoardOverlay: (overlay: Record<string, { color: string; opacity?: number }> | null) => void;
  setChampionMeta: (champion: ChampionMetadata | null) => void;
  saveGame: () => void;
  loadGame: (history: GameHistoryEntry[]) => Promise<void>;
}

let workerInstance: Worker | null = null;
let initResolver: ((val: string) => void) | null = null;
let moveResolver: ((val: MoveResponse) => void) | null = null;
let isWorkerReady = false;
/**
 * Bumped whenever a game starts or is loaded/disconnected. Async replies (a
 * worker message, a backend response) carry the generation they were issued
 * under and are dropped if the game has changed meanwhile, so a late reply from
 * a previous game can never clobber the current one.
 */
let gameGeneration = 0;
const nextGeneration = () => { gameGeneration += 1; return gameGeneration; };
const isCurrent = (generation: number) => generation === gameGeneration;

function setupWorker() {
  if (workerInstance) return workerInstance;
  workerInstance = new BlokusWorker();

  // Expose it so Play.tsx step button can access it directly to force manual advance
  (window as any)._blokusWorkerInstance = workerInstance;

  workerInstance.onmessage = (e) => {
    const data = e.data;
    if (data.type === 'ready') {
      isWorkerReady = true;
      useGameStore.getState().addLog("WebWorker & Pyodide Ready", "INFO");
    } else if (data.type === 'state_update') {
      if (useGameStore.getState().transport !== 'worker') return; // stale: a backend game took over
      useGameStore.getState().setGameState(data.state);
      if (initResolver) {
        initResolver(data.state.game_id);
        initResolver = null;
      }
      triggerAgentTurnIfNeeded();
    } else if (data.type === 'move_response') {
      const resp = data.response;
      if (useGameStore.getState().transport !== 'worker') return; // stale reply from a replaced worker game
      useGameStore.getState().setGameState(resp.game_state);
      useGameStore.setState({ isAdvancingTurn: false, pendingPlacement: null });
      if (moveResolver) {
        moveResolver(resp);
        moveResolver = null;
      }
      triggerAgentTurnIfNeeded();
    } else if (data.type === 'error' || data.type === 'init_error') {
      console.error("Worker Error:", data.error);
      useGameStore.getState().setError(data.error);
      useGameStore.setState({ pendingPlacement: null });
      useGameStore.getState().addLog("Worker Error: " + data.error, "ERROR");
    }
  };
  return workerInstance;
}

function describeLastMove(gameState: GameState, previous: GameState | null): string | null {
  const last = gameState.last_move as
    | { player?: string; is_human?: boolean; stats?: { level?: number; timeSpentMs?: number; fallback?: string | null; engine?: string } }
    | null
    | undefined;
  if (!last || last.is_human) return null;
  // A pass leaves last_move untouched: only describe a move when a piece was placed.
  if (previous && previous.move_count === gameState.move_count) return null;
  const stats = last.stats || {};
  const who = stats.engine === 'pentobi' && stats.level != null ? `${last.player} (Pentobi L${stats.level})` : `${last.player}`;
  const secs = stats.timeSpentMs != null ? `${(stats.timeSpentMs / 1000).toFixed(1)} s` : '';
  const fallback = stats.fallback ? ` — FALLBACK: ${stats.fallback}` : '';
  return `${who} moved${secs ? ` in ${secs}` : ''}${fallback}`;
}

async function runBackendAgentTurn(): Promise<void> {
  const store = useGameStore.getState();
  const gameId = store.gameState?.game_id;
  if (!gameId) return;
  const generation = gameGeneration;
  const before = store.gameState;
  useGameStore.setState({ isAdvancingTurn: true });
  try {
    const resp = await postAdvanceTurn<GameState>(gameId);
    if (!isCurrent(generation)) return; // the game changed while the backend was thinking
    if (resp.game_state) {
      useGameStore.getState().setGameState(resp.game_state);
      const line = describeLastMove(resp.game_state, before);
      if (line) useGameStore.getState().addLog(line, line.includes('FALLBACK') ? 'WARN' : 'INFO');
    }
    if (!resp.success) useGameStore.getState().addLog(`Backend: ${resp.message}`, 'WARN');
  } catch (err) {
    if (!isCurrent(generation)) return;
    const message = err instanceof Error ? err.message : String(err);
    useGameStore.getState().setError(message);
    useGameStore.getState().addLog(`Backend error: ${message}. The AI turn was not completed: use Pause, then Step, to retry.`, 'ERROR');
    useGameStore.setState({ isAdvancingTurn: false });
    return;
  }
  useGameStore.setState({ isAdvancingTurn: false });
  triggerAgentTurnIfNeeded();
}

function triggerAgentTurnIfNeeded() {
  const state = useGameStore.getState();
  const gameState = state.gameState;
  if (!gameState || gameState.game_over || gameState.status !== 'in_progress' || state.isPaused) return;

  const currentPlayer = gameState.current_player;
  const pConf = gameState.players?.find(p => p.player === currentPlayer);

  if (pConf && pConf.agent_type !== 'human' && !state.isAdvancingTurn) {
    if (state.transport === 'backend') {
      void runBackendAgentTurn();
      return;
    }
    useGameStore.setState({ isAdvancingTurn: true });
    const enableDiagnostics = state.analysisModeEnabled || false;
    workerInstance?.postMessage({ type: 'advance_turn', enableDiagnostics });
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
    pendingPlacement: null,
    pollIntervalId: null,
    isAdvancingTurn: false,
    isPaused: false,
    error: null,
    logs: [],
    activeRightTab: 'main' as const,
    boardOverlay: null,
    analysisModeEnabled: false,
    championMeta: null,
    transport: 'worker',

    advanceTurn: () => {
      const state = get();
      if (state.isAdvancingTurn) return;
      if (state.transport === 'backend') {
        void runBackendAgentTurn();
        return;
      }
      if (!workerInstance) return;
      set({ isAdvancingTurn: true });
      workerInstance.postMessage({ type: 'advance_turn' });
    },

    setAnalysisModeEnabled: (enabled: boolean) => {
      set({ analysisModeEnabled: enabled });
    },

    connect: async (gameId: string): Promise<void> => {
      // With Pyodide, state is local. If we reload the page, state is gone.
      // So connect only works if we already just created the game locally in this session.
      set({ connectionStatus: 'connected', error: null, isAdvancingTurn: false, isPaused: false });
      const where = get().transport === 'backend' ? 'backend-served' : 'local Pyodide';
      get().addLog(`Connected to ${where} game ${gameId}`, 'INFO');
      triggerAgentTurnIfNeeded();
    },

    togglePause: () => {
      const nextPaused = !get().isPaused;
      set({ isPaused: nextPaused });
      if (!nextPaused) {
        triggerAgentTurnIfNeeded();
      }
    },

    disconnect: () => {
      nextGeneration();
      set({
        connectionStatus: 'disconnected',
        pollIntervalId: null,
        gameState: null,
        currentSliderTurn: null,
        isAdvancingTurn: false,
        isPaused: false,
        pendingPlacement: null,
        transport: 'worker',
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

    setPendingPlacement: (placement: PendingPlacement | null) => {
      set({ pendingPlacement: placement });
    },

    passTurn: async (): Promise<MoveResponse> => {
      if (get().transport === 'backend') {
        const gameState = get().gameState;
        if (!gameState) return { success: false, message: 'No game', game_over: false };
        const generation = gameGeneration;
        try {
          const resp = await postPass<GameState>(gameState.game_id, gameState.current_player);
          if (!isCurrent(generation)) return { success: false, message: 'Game changed', game_over: false };
          if (resp.game_state) get().setGameState(resp.game_state);
          set({ isAdvancingTurn: false, pendingPlacement: null });
          triggerAgentTurnIfNeeded();
          return { success: resp.success, message: resp.message, game_over: Boolean(resp.game_state?.game_over), game_state: resp.game_state };
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          if (isCurrent(generation)) set({ pendingPlacement: null });
          return { success: false, message, game_over: false };
        }
      }
      return new Promise((resolve) => {
        moveResolver = resolve;
        if (!workerInstance) setupWorker();
        workerInstance!.postMessage({ type: 'pass_turn' });
      });
    },

    makeMove: async (move: MoveRequest): Promise<MoveResponse> => {
      if (get().transport === 'backend') {
        const gameState = get().gameState;
        if (!gameState) return { success: false, message: 'No game', game_over: false };
        const generation = gameGeneration;
        try {
          const resp = await postMove<GameState>(gameState.game_id, move);
          if (!isCurrent(generation)) return { success: false, message: 'Game changed', game_over: false };
          if (resp.game_state) get().setGameState(resp.game_state);
          set({ isAdvancingTurn: false, pendingPlacement: null });
          triggerAgentTurnIfNeeded();
          return { success: resp.success, message: resp.message, game_over: Boolean(resp.game_state?.game_over), game_state: resp.game_state };
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          if (isCurrent(generation)) set({ pendingPlacement: null });
          return { success: false, message, game_over: false };
        }
      }
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
      const previous = get().gameState;
      const previousTransport = get().transport;
      nextGeneration();
      set({ isAdvancingTurn: false, pendingPlacement: null });
      if (previousTransport === 'backend' && previous && !previous.game_over) {
        // Release the abandoned game's engine processes; best effort.
        void deleteBackendGame(previous.game_id).catch(() => undefined);
      }
      if (isBackendConfig(config)) {
        const created = await createBackendGame<GameState>(config);
        set({ transport: 'backend', error: null });
        get().setGameState(created.game_state);
        get().addLog(`Backend game ${created.game_id} created`, 'INFO');
        return created.game_id;
      }
      set({ transport: 'worker' });
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
    setActiveRightTab: (tab: 'main' | 'telemetry' | 'explanation' | 'moveDelta' | 'analysis' | 'mcts_analysis') => { set({ activeRightTab: tab }); },
    setBoardOverlay: (overlay) => { set({ boardOverlay: overlay }); },
    setChampionMeta: (champion) => { set({ championMeta: champion }); },

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
      nextGeneration();
      set({ transport: 'worker', isAdvancingTurn: false, pendingPlacement: null });
      setupWorker();
      const waitForWorker = async () => {
        while (!isWorkerReady) {
          await new Promise(r => setTimeout(r, 100));
        }
      };
      await waitForWorker();

      return new Promise<void>((resolve) => {
        // Wire initResolver so we wait for the worker's state_update before resolving.
        // This fixes the bug where the modal closed before the game state was loaded.
        initResolver = (_gameId: string) => resolve();
        set({ connectionStatus: 'connecting' });
        // Strip each entry to only action + player_to_move — the fields load_game actually
        // needs to replay moves. This avoids expensive Pyodide conversion of nested
        // metrics objects (frontier_metrics, board_state, etc.) that load_game ignores.
        const stripped = history.map(e => ({
          player_to_move: e.player_to_move,
          action: e.action,
        }));
        workerInstance!.postMessage({ type: 'load_game', history: stripped });
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

  return useMemo(() => {
    const legalMoves = gameState?.legal_moves || [];
    const currentPlayer = gameState?.current_player;
    const piecesUsed = currentPlayer ? (gameState?.pieces_used?.[currentPlayer] || []) : [];

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
  }, [gameState?.legal_moves, gameState?.current_player, gameState?.pieces_used]);
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
