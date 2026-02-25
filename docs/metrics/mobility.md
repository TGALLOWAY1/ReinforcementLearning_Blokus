# Mobility Metrics Specification

**Status:** Canonical definition for frontend and backend. Both implementations MUST produce identical results.

---

## Definitions

### Input

- **Legal moves:** List of moves from `engine/move_generator.get_legal_moves()`. Each move has `(piece_id, orientation, anchor_row, anchor_col)`.
- **Pieces used:** List of piece IDs already placed by the player. These pieces are excluded from metrics.

### Per-Piece Counts

| Symbol | Name | Definition |
|--------|------|------------|
| **P_i** | Placements | Number of legal moves for piece `i` (count of moves with `piece_id == i`). Each move is a distinct `(piece_id, orientation, anchor_row, anchor_col)` tuple. |
| **O_i** | Unique orientations | Number of distinct orientations for piece `i` after symmetry deduplication. Matches `engine/pieces.py` and `LegalMoveGenerator.piece_orientations_cache`. |
| **S_i** | Piece size | Number of cells in piece `i` (1–5). |

### Placements (P_i) Clarification

- **Each legal move is a distinct placement.** Count every move in the list; no deduplication.

- A move is uniquely identified by `(piece_id, orientation, anchor_row, anchor_col)`. Two moves with the same piece_id but different orientations or anchors are counted separately.

- **Duplicates:** If the backend returns duplicate moves (same piece, orientation, anchor), each is counted. In practice the engine does not return duplicates.

### Derived Metrics

| Symbol | Formula | Notes |
|--------|---------|-------|
| **PN_i** | P_i / O_i | Orientation-normalized placements. If O_i = 0, PN_i = 0. |
| **MW_i** | (P_i × S_i) / O_i = PN_i × S_i | Cell-weighted mobility. |

### Aggregates

| Metric | Formula |
|--------|---------|
| **totalPlacements** | Σ P_i (over pieces 1–21, excluding pieces_used) |
| **totalOrientationNormalized** | Σ PN_i |
| **totalCellWeighted** | Σ MW_i |

### Size Buckets

| Bucket | Size | Pieces |
|--------|------|--------|
| 1 | Monomino | 1 |
| 2 | Domino | 2 |
| 3 | Tromino | 3, 4 |
| 4 | Tetromino | 5–10 |
| 5 | Pentomino | 11–21 |

**Bucket sum:** For each size `s` in {1, 2, 3, 4, 5}, `buckets[s] = Σ MW_i` over all pieces `i` with `S_i == s`.

Pieces with `S_i` outside 1–5 are skipped (should not occur for standard Blokus).

---

## Orientation Count (O_i)

- **Source:** Same as `LegalMoveGenerator.piece_orientations_cache` — `PieceGenerator.get_piece_rotations_and_reflections(piece)` with deduplication via `np.array_equal`.

- **Frontend:** Generate 8 orientations (0–7) via `getPieceShape`, deduplicate by normalized shape string. Must match backend count.

---

## Output Schema

```ts
interface PlayerMobilityMetrics {
  totalPlacements: number;
  totalOrientationNormalized: number;
  totalCellWeighted: number;
  buckets: Record<1|2|3|4|5, number>;
}
```

---

## Implementation Notes

- **Integer arithmetic:** P_i, O_i, S_i are integers. PN_i and MW_i are floats.

- **Exclusivity:** Only include pieces not in `pieces_used`.

- **Piece IDs:** 1–21. Skip invalid IDs.
