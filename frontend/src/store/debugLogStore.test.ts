/**
 * Unit tests for debug log store: diff summarizer and buffer cap.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { useDebugLogStore, computeStateDiffSummary } from './debugLogStore';

describe('debugLogStore', () => {
  beforeEach(() => {
    useDebugLogStore.getState().clear();
  });

  describe('computeStateDiffSummary', () => {
    it('returns null when prev or next is null', () => {
      expect(computeStateDiffSummary(null, null)).toBeNull();
      expect(
        computeStateDiffSummary(null, {
          current_player: 'RED',
          move_count: 0,
          board: [],
        } as any)
      ).toBeNull();
      expect(
        computeStateDiffSummary(
          { current_player: 'RED', move_count: 0, board: [] } as any,
          null
        )
      ).toBeNull();
    });

    it('detects current_player change', () => {
      const prev = { current_player: 'RED', move_count: 1, board: [] } as any;
      const next = { current_player: 'BLUE', move_count: 1, board: [] } as any;
      const diff = computeStateDiffSummary(prev, next);
      expect(diff?.current_player).toEqual({ prev: 'RED', next: 'BLUE' });
    });

    it('detects move_count change', () => {
      const prev = { current_player: 'RED', move_count: 1, board: [] } as any;
      const next = { current_player: 'RED', move_count: 2, board: [] } as any;
      const diff = computeStateDiffSummary(prev, next);
      expect(diff?.move_count).toEqual({ prev: 1, next: 2 });
    });

    it('detects legal_moves count change', () => {
      const prev = { current_player: 'RED', move_count: 1, legal_moves: [1, 2, 3] } as any;
      const next = { current_player: 'RED', move_count: 1, legal_moves: [1, 2] } as any;
      const diff = computeStateDiffSummary(prev, next);
      expect(diff?.legal_moves_count).toEqual({ prev: 3, next: 2 });
    });

    it('returns null when no changes', () => {
      const state = { current_player: 'RED', move_count: 1, board: [] } as any;
      expect(computeStateDiffSummary(state, { ...state })).toBeNull();
    });
  });

  describe('log buffer cap', () => {
    it('caps events at 300', () => {
      const store = useDebugLogStore.getState();
      for (let i = 0; i < 350; i++) {
        store.addEvent('WS_GAME_STATE_RECEIVED', { i });
      }
      expect(useDebugLogStore.getState().events.length).toBe(300);
    });

    it('newest events are kept', () => {
      const store = useDebugLogStore.getState();
      for (let i = 0; i < 310; i++) {
        store.addEvent('WS_GAME_STATE_RECEIVED', { index: i });
      }
      const events = useDebugLogStore.getState().events;
      expect(events[0].payload).toEqual({ index: 309 });
      expect(events[events.length - 1].payload).toEqual({ index: 10 });
    });
  });

  describe('pause', () => {
    it('does not add events when paused', () => {
      const store = useDebugLogStore.getState();
      store.setPaused(true);
      store.addEvent('MOVE_SUBMITTED', {});
      expect(useDebugLogStore.getState().events.length).toBe(0);
      store.setPaused(false);
      store.addEvent('MOVE_SUBMITTED', {});
      expect(useDebugLogStore.getState().events.length).toBe(1);
    });
  });
});
