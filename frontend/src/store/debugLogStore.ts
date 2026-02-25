/**
 * Debug log store for in-game telemetry and event streaming.
 * Gated by ENABLE_DEBUG_UI; no-op when disabled.
 */

import { create } from 'zustand';

/** Minimal shape for diff computation; avoids circular import from gameStore */
interface GameStateShape {
  current_player: string;
  move_count: number;
  pieces_used?: Record<string, number[]>;
  legal_moves?: unknown[];
  board?: number[][];
  heatmap?: number[][];
}

export const DEBUG_EVENT_TYPES = [
  'WS_GAME_STATE_RECEIVED',
  'MOVE_SUBMITTED',
  'MOVE_ACCEPTED',
  'MOVE_REJECTED',
  'TURN_ADVANCED',
  'AGENT_THINK_END',
  'REPLAY_STEP_LOADED',
] as const;

export type DebugEventType = (typeof DEBUG_EVENT_TYPES)[number];

export interface DebugEvent {
  id: string;
  timestamp: number;
  type: DebugEventType;
  payload: Record<string, unknown>;
}

export interface StateDiffSummary {
  current_player?: { prev: string; next: string };
  move_count?: { prev: number; next: number };
  pieces_used_delta?: Array<{ player: string; piece_id: number }>;
  board_changed_count?: number;
  board_changed_cells?: Array<[number, number]>;
  legal_moves_count?: { prev: number; next: number };
  heatmap_present?: boolean;
  heatmap_min_max?: [number, number];
}

const MAX_EVENTS = 300;
const MAX_BOARD_DIFF_CELLS = 50;

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/** Compute compact diff between prev and next game state. Avoids full 20x20 scan when possible. */
export function computeStateDiffSummary(
  prev: GameStateShape | null,
  next: GameStateShape | null
): StateDiffSummary | null {
  if (!prev || !next) return null;
  const diff: StateDiffSummary = {};

  if (prev.current_player !== next.current_player) {
    diff.current_player = { prev: prev.current_player, next: next.current_player };
  }
  if (prev.move_count !== next.move_count) {
    diff.move_count = { prev: prev.move_count, next: next.move_count };
  }

  const prevPieces = prev.pieces_used || {};
  const nextPieces = next.pieces_used || {};
  const piecesDelta: Array<{ player: string; piece_id: number }> = [];
  for (const player of ['RED', 'BLUE', 'YELLOW', 'GREEN'] as const) {
    const prevUsed = new Set(prevPieces[player] || []);
    const nextUsed = new Set(nextPieces[player] || []);
    for (const pid of nextUsed) {
      if (!prevUsed.has(pid)) piecesDelta.push({ player, piece_id: pid });
    }
  }
  if (piecesDelta.length > 0) diff.pieces_used_delta = piecesDelta;

  const prevLegal = prev.legal_moves?.length ?? 0;
  const nextLegal = next.legal_moves?.length ?? 0;
  if (prevLegal !== nextLegal) {
    diff.legal_moves_count = { prev: prevLegal, next: nextLegal };
  }

  if (prev.board && next.board && prev.board.length === next.board.length) {
    const changed: Array<[number, number]> = [];
    for (let r = 0; r < prev.board.length && changed.length < MAX_BOARD_DIFF_CELLS; r++) {
      const prow = prev.board[r];
      const nrow = next.board[r];
      if (!prow || !nrow) continue;
      for (let c = 0; c < prow.length && changed.length < MAX_BOARD_DIFF_CELLS; c++) {
        if (prow[c] !== nrow[c]) changed.push([r, c]);
      }
    }
    if (changed.length > 0) {
      diff.board_changed_count = changed.length;
      if (changed.length <= MAX_BOARD_DIFF_CELLS) {
        diff.board_changed_cells = changed;
      }
    }
  }

  if (next.heatmap) {
    diff.heatmap_present = true;
    let min = Infinity;
    let max = -Infinity;
    for (const row of next.heatmap) {
      for (const v of row) {
        if (typeof v === 'number') {
          min = Math.min(min, v);
          max = Math.max(max, v);
        }
      }
    }
    if (min !== Infinity) diff.heatmap_min_max = [min, max];
  }

  return Object.keys(diff).length > 0 ? diff : null;
}

interface DebugLogStore {
  events: DebugEvent[];
  lastWsTimestamp: number;
  lastStateDiff: StateDiffSummary | null;
  prevGameState: GameStateShape | null;
  paused: boolean;
  autoScroll: boolean;
  typeFilters: Set<DebugEventType>;
  searchText: string;

  addEvent: (type: DebugEventType, payload: Record<string, unknown>) => void;
  setLastWsTimestamp: (ts: number) => void;
  setStateDiff: (prev: GameStateShape | null, next: GameStateShape | null) => void;
  clear: () => void;
  setPaused: (paused: boolean) => void;
  setAutoScroll: (v: boolean) => void;
  setTypeFilter: (type: DebugEventType, enabled: boolean) => void;
  setSearchText: (text: string) => void;
  toggleTypeFilter: (type: DebugEventType) => void;
}

export const useDebugLogStore = create<DebugLogStore>((set, get) => ({
  events: [],
  lastWsTimestamp: 0,
  lastStateDiff: null,
  prevGameState: null,
  paused: false,
  autoScroll: true,
  typeFilters: new Set(DEBUG_EVENT_TYPES),
  searchText: '',

  addEvent: (type, payload) => {
    if (get().paused) return;
    const event: DebugEvent = {
      id: generateId(),
      timestamp: Date.now(),
      type,
      payload,
    };
    set((s) => ({
      events: [event, ...s.events].slice(0, MAX_EVENTS),
    }));
  },

  setLastWsTimestamp: (ts) => set({ lastWsTimestamp: ts }),

  setStateDiff: (prev: GameStateShape | null, next: GameStateShape | null) => {
    const diff = computeStateDiffSummary(prev, next);
    set({ lastStateDiff: diff, prevGameState: next });
  },

  clear: () =>
    set({
      events: [],
      lastStateDiff: null,
      prevGameState: null,
    }),

  setPaused: (paused) => set({ paused }),
  setAutoScroll: (autoScroll) => set({ autoScroll }),
  setSearchText: (searchText) => set({ searchText }),

  setTypeFilter: (type, enabled) =>
    set((s) => {
      const next = new Set(s.typeFilters);
      if (enabled) next.add(type);
      else next.delete(type);
      return { typeFilters: next };
    }),

  toggleTypeFilter: (type) =>
    set((s) => {
      const next = new Set(s.typeFilters);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return { typeFilters: next };
    }),
}));

/** No-op when ENABLE_DEBUG_UI is false. Call from instrumentation points. */
export function debugAddEvent(
  enabled: boolean,
  type: DebugEventType,
  payload: Record<string, unknown>
): void {
  if (enabled) useDebugLogStore.getState().addEvent(type, payload);
}
