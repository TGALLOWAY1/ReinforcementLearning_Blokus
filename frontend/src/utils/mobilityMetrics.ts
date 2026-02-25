import { getPieceShape } from './pieceUtils';
import { PIECE_SHAPES } from '../constants/gameConstants';

/**
 * O_i: Unique orientation count per piece. Cached per piece id (static).
 * Computed by generating all 8 orientations (0-7) via getPieceShape, deduping by
 * normalized shape string. Matches backend engine/move_generator deduplication.
 */
const ORIENTATION_COUNT_CACHE: Record<number, number> = {};

function shapeKey(shape: number[][]): string {
  return JSON.stringify(shape);
}

export function getOrientationCount(pieceId: number): number {
  if (ORIENTATION_COUNT_CACHE[pieceId] !== undefined) {
    return ORIENTATION_COUNT_CACHE[pieceId];
  }
  const seen = new Set<string>();
  for (let o = 0; o < 8; o++) {
    const shape = getPieceShape(pieceId, o);
    if (shape.length > 0) {
      seen.add(shapeKey(shape));
    }
  }
  const count = seen.size;
  ORIENTATION_COUNT_CACHE[pieceId] = count;
  return count;
}

/** S_i: Piece size (number of squares). From PIECE_SHAPES. */
export function getPieceSize(pieceId: number): number {
  const shape = PIECE_SHAPES[pieceId];
  if (!shape) return 0;
  return shape.flat().filter((c) => c === 1).length;
}

export type SizeBucket = 1 | 2 | 3 | 4 | 5;

export interface PlayerMobilityMetrics {
  totalPlacements: number;
  totalOrientationNormalized: number;
  totalCellWeighted: number;
  buckets: Record<SizeBucket, number>;
  centerControl?: number;
  frontierSize?: number;
}

/**
 * Compute mobility metrics from legal_moves (backend: engine/move_generator.get_legal_moves).
 * P_i = placements per piece, O_i = unique orientations, S_i = piece size.
 * PN_i = P_i/O_i, MW_i = PN_i*S_i.
 */
export function computeMobilityMetrics(
  legalMoves: Array<{ piece_id?: number; pieceId?: number }>,
  piecesUsed: number[]
): PlayerMobilityMetrics {
  const P: Record<number, number> = {};
  for (const m of legalMoves) {
    const pid = m.piece_id ?? m.pieceId;
    if (pid != null) {
      P[pid] = (P[pid] ?? 0) + 1;
    }
  }

  let totalPlacements = 0;
  let totalOrientationNormalized = 0;
  let totalCellWeighted = 0;
  const buckets: Record<SizeBucket, number> = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };

  for (let pieceId = 1; pieceId <= 21; pieceId++) {
    if (piecesUsed.includes(pieceId)) continue;
    const Pi = Math.max(0, P[pieceId] ?? 0);
    const Oi = getOrientationCount(pieceId);
    const Si = getPieceSize(pieceId) as SizeBucket;
    if (Si < 1 || Si > 5) continue;

    const PNi = Oi > 0 ? Pi / Oi : 0;
    const MWi = PNi * Si;

    totalPlacements += Pi;
    totalOrientationNormalized += PNi;
    totalCellWeighted += MWi;
    buckets[Si] = (buckets[Si] ?? 0) + MWi;
  }

  return {
    totalPlacements,
    totalOrientationNormalized,
    totalCellWeighted,
    buckets,
  };
}
