/**
 * Top Moves (MCTS) table - shows visits and Q for agent's candidate moves.
 * Sortable by visits or Q. Optional click to preview move on board.
 */

import React, { useState, useMemo } from 'react';
import { useGameStore, type MctsTopMove } from '../store/gameStore';
import { PIECE_NAMES } from '../constants/gameConstants';

type SortKey = 'visits' | 'q_value';

export const MctsTopMovesTable: React.FC = () => {
  const mctsTopMoves = useGameStore((s) => s.gameState?.mcts_top_moves);
  const previewMove = useGameStore((s) => s.previewMove);
  const setPreviewMove = useGameStore((s) => s.setPreviewMove);

  const [sortKey, setSortKey] = useState<SortKey>('visits');
  const [sortDesc, setSortDesc] = useState(true);

  const sorted = useMemo(() => {
    if (!mctsTopMoves?.length) return [];
    const copy = [...mctsTopMoves];
    copy.sort((a, b) => {
      const va = a[sortKey] ?? 0;
      const vb = b[sortKey] ?? 0;
      return sortDesc ? vb - va : va - vb;
    });
    return copy;
  }, [mctsTopMoves, sortKey, sortDesc]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDesc((d) => !d);
    else {
      setSortKey(key);
      setSortDesc(true);
    }
  };

  const handleRowClick = (move: MctsTopMove) => {
    const isSame =
      previewMove &&
      previewMove.piece_id === move.piece_id &&
      previewMove.orientation === move.orientation &&
      previewMove.anchor_row === move.anchor_row &&
      previewMove.anchor_col === move.anchor_col;
    setPreviewMove(isSame ? null : move);
  };

  if (!mctsTopMoves?.length) {
    return (
      <div className="px-3 py-3 border-t border-charcoal-700">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Top Moves (MCTS)
        </h3>
        <div className="text-xs text-gray-500 py-4 text-center bg-charcoal-800/50 rounded">
          No MCTS data (human turn or no agent move yet)
        </div>
      </div>
    );
  }

  return (
    <div className="px-3 py-3 border-t border-charcoal-700">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Top Moves (MCTS)
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-gray-500 border-b border-charcoal-600">
              <th className="py-1.5 pr-2">Piece</th>
              <th className="py-1.5 pr-2">Anchor</th>
              <th
                className="py-1.5 pr-2 cursor-pointer hover:text-neon-blue"
                onClick={() => handleSort('visits')}
              >
                Visits {sortKey === 'visits' ? (sortDesc ? '↓' : '↑') : ''}
              </th>
              <th
                className="py-1.5 pr-2 cursor-pointer hover:text-neon-blue"
                onClick={() => handleSort('q_value')}
              >
                Q {sortKey === 'q_value' ? (sortDesc ? '↓' : '↑') : ''}
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((m, i) => (
              <tr
                key={i}
                className="border-b border-charcoal-700/50 hover:bg-charcoal-800 cursor-pointer"
                onClick={() => handleRowClick(m)}
              >
                <td className="py-1.5 pr-2 text-gray-300">
                  {PIECE_NAMES[m.piece_id] ?? `#${m.piece_id}`}
                </td>
                <td className="py-1.5 pr-2 text-gray-400 font-mono">
                  ({m.anchor_row},{m.anchor_col})
                </td>
                <td className="py-1.5 pr-2 text-gray-300">{m.visits}</td>
                <td className="py-1.5 pr-2 text-neon-blue font-mono">
                  {typeof m.q_value === 'number' ? m.q_value.toFixed(3) : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[10px] text-gray-500 mt-1">Click row to preview on board</p>
    </div>
  );
};
