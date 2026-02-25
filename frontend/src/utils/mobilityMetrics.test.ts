/**
 * Unit tests for mobility metrics per docs/metrics/mobility.md.
 * Ensures frontend and backend produce identical results for the same inputs.
 */
import { describe, it, expect } from 'vitest';
import {
  computeMobilityMetrics,
  getOrientationCount,
  getPieceSize,
} from './mobilityMetrics';

// Shared fixture (matches tests/fixtures/mobility_metrics_fixture.json)
const FIXTURE = {
  legal_moves: [
    { piece_id: 1, orientation: 0, anchor_row: 0, anchor_col: 0 },
    { piece_id: 1, orientation: 0, anchor_row: 1, anchor_col: 0 },
    { piece_id: 1, orientation: 0, anchor_row: 2, anchor_col: 0 },
    { piece_id: 2, orientation: 0, anchor_row: 0, anchor_col: 0 },
    { piece_id: 2, orientation: 1, anchor_row: 0, anchor_col: 1 },
  ],
  pieces_used: [] as number[],
  expected: {
    totalPlacements: 5,
    totalOrientationNormalized: 4,
    totalCellWeighted: 5,
    buckets: { 1: 3, 2: 2, 3: 0, 4: 0, 5: 0 } as Record<1 | 2 | 3 | 4 | 5, number>,
  },
};

describe('mobilityMetrics (docs/metrics/mobility.md)', () => {
  it('fixture matches expected (frontend = backend)', () => {
    const m = computeMobilityMetrics(FIXTURE.legal_moves, FIXTURE.pieces_used);
    expect(m.totalPlacements).toBe(FIXTURE.expected.totalPlacements);
    expect(m.totalOrientationNormalized).toBe(FIXTURE.expected.totalOrientationNormalized);
    expect(m.totalCellWeighted).toBe(FIXTURE.expected.totalCellWeighted);
    expect(m.buckets[1]).toBe(FIXTURE.expected.buckets[1]);
    expect(m.buckets[2]).toBe(FIXTURE.expected.buckets[2]);
    expect(m.buckets[3]).toBe(FIXTURE.expected.buckets[3]);
    expect(m.buckets[4]).toBe(FIXTURE.expected.buckets[4]);
    expect(m.buckets[5]).toBe(FIXTURE.expected.buckets[5]);
  });

  it('accepts pieceId (camelCase) for API compatibility', () => {
    const moves = [{ pieceId: 1 }, { pieceId: 1 }];
    const m = computeMobilityMetrics(moves, []);
    expect(m.totalPlacements).toBe(2);
  });

  it('pieces_used excludes pieces from metrics', () => {
    const moves = [{ piece_id: 1 }, { piece_id: 1 }];
    const m = computeMobilityMetrics(moves, [1]);
    expect(m.totalPlacements).toBe(0);
    expect(m.totalCellWeighted).toBe(0);
  });

  it('empty moves yields zeros', () => {
    const m = computeMobilityMetrics([], []);
    expect(m.totalPlacements).toBe(0);
    expect(m.totalOrientationNormalized).toBe(0);
    expect(m.totalCellWeighted).toBe(0);
    expect(m.buckets).toEqual({ 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 });
  });

  it('O_i matches backend for monomino and domino', () => {
    expect(getOrientationCount(1)).toBe(1);
    expect(getOrientationCount(2)).toBe(2);
  });

  it('S_i matches backend for monomino and domino', () => {
    expect(getPieceSize(1)).toBe(1);
    expect(getPieceSize(2)).toBe(2);
  });
});
