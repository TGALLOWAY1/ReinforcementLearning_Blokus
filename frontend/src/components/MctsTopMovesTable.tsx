/**
 * Top Moves (MCTS) table - shows visits and Q for agent's candidate moves.
 * Sortable by visits or Q. Optional click to preview move on board.
 */

import React, { useState, useMemo } from 'react';
import { useGameStore, type MctsTopMove } from '../store/gameStore';
import { PIECE_NAMES } from '../constants/gameConstants';

type SortKey = 'visits' | 'q_value';

export const MctsTopMovesTable: React.FC = () => {
  const gameState = useGameStore((s) => s.gameState);
  const mctsTopMoves = gameState?.mcts_top_moves;
  const mctsStats = gameState?.mcts_stats;
  const previewMove = useGameStore((s) => s.previewMove);
  const setPreviewMove = useGameStore((s) => s.setPreviewMove);

  const playerConfig = useMemo(() => {
    if (!gameState?.current_player || !gameState?.players) return null;
    return gameState.players.find(p => p.player === gameState.current_player);
  }, [gameState?.current_player, gameState?.players]);

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
          MCTS Analytics
        </h3>
        <div className="text-xs text-gray-500 py-4 text-center bg-charcoal-800/50 rounded border border-charcoal-700/50">
          No MCTS data available
        </div>
      </div>
    );
  }

  return (
    <div className="px-3 py-3 border-t border-charcoal-700">
      <div className="flex justify-between items-center mb-2">
        <div className="flex flex-col">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            MCTS Analytics
          </h3>
          {playerConfig && (
            <div className="flex items-center gap-2 mt-0.5">
              <span className="text-[10px] font-bold text-slate-500 uppercase">{gameState?.current_player}</span>
              <span className={`text-[9px] px-1 rounded-sm font-bold uppercase ${playerConfig.agent_config?.difficulty === 'hard' ? 'text-red-400 bg-red-400/10' :
                playerConfig.agent_config?.difficulty === 'medium' ? 'text-orange-400 bg-orange-400/10' : 'text-green-400 bg-green-400/10'
                }`}>
                {playerConfig.agent_config?.difficulty || 'Standard'}
              </span>
            </div>
          )}
        </div>
        {mctsStats && (
          <div className="text-[10px] text-gray-500 font-mono flex gap-2">
            <span>{mctsStats.timeSpentMs}ms</span>
            <span className="opacity-30">|</span>
            <span>{mctsStats.nodesEvaluated.toLocaleString()} sim</span>
            <span className="opacity-30">|</span>
            <span className="text-neon-blue">{Math.round(mctsStats.nodesEvaluated / (mctsStats.timeSpentMs / 1000 || 1)).toLocaleString()} s/s</span>
          </div>
        )}
      </div>

      {mctsStats && (
        <div className="mb-3 grid grid-cols-2 gap-2">
          <div className="bg-charcoal-800/80 p-1.5 rounded border border-charcoal-700/50">
            <div className="text-[9px] text-gray-500 uppercase">Utilized Budget</div>
            <div className="h-1 bg-charcoal-900 rounded-full mt-1 overflow-hidden">
              <div
                className="h-full bg-neon-blue shadow-[0_0_5px_rgba(0,243,255,0.5)]"
                style={{ width: `${Math.min(100, (mctsStats.timeSpentMs / mctsStats.timeBudgetMs) * 100)}%` }}
              />
            </div>
            <div className="text-[10px] text-gray-400 mt-1">
              {mctsStats.timeSpentMs} / {mctsStats.timeBudgetMs} ms
            </div>
          </div>
          <div className="bg-charcoal-800/80 p-1.5 rounded border border-charcoal-700/50">
            <div className="text-[9px] text-gray-500 uppercase">Search Effort</div>
            <div className="text-[11px] text-neon-blue font-bold">
              {mctsStats.nodesEvaluated.toLocaleString()}
            </div>
            <div className="text-[9px] text-gray-500">simulations</div>
          </div>
        </div>
      )}
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
