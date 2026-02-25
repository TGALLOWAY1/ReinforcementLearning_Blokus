import React, { useMemo } from 'react';
import { useGameStore, type LegalMovesHistoryEntry } from '../store/gameStore';
import { PLAYER_COLORS } from '../constants/gameConstants';

const PLAYER_ORDER = ['RED', 'BLUE', 'GREEN', 'YELLOW'] as const;
const PLAYER_COLOR_MAP: Record<string, string> = {
  RED: PLAYER_COLORS.red,
  BLUE: PLAYER_COLORS.blue,
  GREEN: PLAYER_COLORS.green,
  YELLOW: PLAYER_COLORS.yellow,
};

/**
 * Multi-line time-series plot of cell-weighted mobility per player per turn.
 * Uses totalCellWeighted (Σ MW_i = Σ P_i*S_i/O_i) for orientation-normalized mobility.
 * Data: legalMovesHistory from gameStore, or overrideHistory when provided (e.g. backend logs).
 */
export const LegalMovesPerTurnPlot: React.FC<{
  overrideHistory?: LegalMovesHistoryEntry[];
}> = ({ overrideHistory }) => {
  const storeHistory = useGameStore((s) => s.legalMovesHistory);
  const legalMovesHistory = overrideHistory !== undefined ? overrideHistory : storeHistory;
  const gameState = useGameStore((s) => s.gameState);
  const winner = gameState?.winner ?? null;

  const { lines, maxTurn, maxCount, width, height, pad, plotW, plotH } = useMemo(() => {
    const w = 740;
    const h = 180;
    const pad = { top: 16, right: 12, bottom: 32, left: 36 };

    if (legalMovesHistory.length === 0) {
      return { lines: [], maxTurn: 1, maxCount: 1, width: w, height: h, pad, plotW: w - pad.left - pad.right, plotH: h - pad.top - pad.bottom };
    }

    const maxTurn = legalMovesHistory.length - 1;
    let maxCount = 1;
    for (const e of legalMovesHistory) {
      for (const m of Object.values(e.byPlayer)) {
        const v = typeof m === 'number' ? m : (m as { totalCellWeighted?: number }).totalCellWeighted ?? 0;
        maxCount = Math.max(maxCount, Math.max(0, v));
      }
    }

    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    const toX = (turn: number) => pad.left + (turn / Math.max(1, maxTurn)) * plotW;
    const toY = (count: number) => pad.top + plotH - (count / maxCount) * plotH;

    const lines = PLAYER_ORDER.map((player) => {
      const points: { turn: number; count: number }[] = [];
      legalMovesHistory.forEach((e: LegalMovesHistoryEntry, turn) => {
        const m = e.byPlayer[player];
        const v = m !== undefined ? (typeof m === 'number' ? m : m.totalCellWeighted ?? 0) : undefined;
        if (v !== undefined) {
          points.push({ turn, count: Math.max(0, v) });
        }
      });
      const path =
        points.length > 0
          ? points
            .map((p, i) => `${i === 0 ? 'M' : 'L'} ${toX(p.turn)} ${toY(p.count)}`)
            .join(' ')
          : '';
      const isWinner = winner === player;
      return {
        player,
        path,
        points,
        color: PLAYER_COLOR_MAP[player] ?? '#94a3b8',
        strokeWidth: isWinner ? 2.5 : 1.5,
      };
    });

    return { lines, maxTurn: Math.max(1, maxTurn), maxCount, width: w, height: h, pad, plotW, plotH };
  }, [legalMovesHistory, winner]);

  if (legalMovesHistory.length === 0) {
    return (
      <div className="px-3 py-3 border-t border-charcoal-700">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Mobility Over Time
        </h3>
        <div className="text-xs text-gray-500 h-24 flex items-center justify-center bg-charcoal-800/50 rounded">
          No turn data yet
        </div>
      </div>
    );
  }

  return (
    <div className="px-3 py-3 border-t border-charcoal-700">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Mobility Over Time
      </h3>
      <svg width={width} height={height} className="overflow-visible">
        {/* Y-axis label */}
        <text
          x={12}
          y={height / 2}
          fill="#94a3b8"
          fontSize={9}
          textAnchor="middle"
          transform={`rotate(-90, 12, ${height / 2})`}
        >
          Mobility (cell-weighted)
        </text>
        {/* X-axis label */}
        <text x={pad.left + plotW / 2} y={height - 6} fill="#94a3b8" fontSize={9} textAnchor="middle">
          Turn #
        </text>
        {/* Grid lines */}
        {[0.25, 0.5, 0.75].map((t) => (
          <line
            key={`v-${t}`}
            x1={pad.left + t * plotW}
            y1={pad.top}
            x2={pad.left + t * plotW}
            y2={pad.top + plotH}
            stroke="#334155"
            strokeWidth={0.5}
          />
        ))}
        {[0.25, 0.5, 0.75].map((t) => (
          <line
            key={`h-${t}`}
            x1={pad.left}
            y1={pad.top + (1 - t) * plotH}
            x2={pad.left + plotW}
            y2={pad.top + (1 - t) * plotH}
            stroke="#334155"
            strokeWidth={0.5}
          />
        ))}
        {/* Lines */}
        {lines.map((l) =>
          l.path ? (
            <path
              key={l.player}
              d={l.path}
              fill="none"
              stroke={l.color}
              strokeWidth={l.strokeWidth}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ) : null
        )}
        {/* Markers */}
        {lines.map((l) =>
          l.points.map((p, i) => {
            const x = pad.left + (p.turn / maxTurn) * plotW;
            const y = pad.top + plotH - (p.count / maxCount) * plotH;
            return (
              <circle
                key={`${l.player}-${i}`}
                cx={x}
                cy={y}
                r={2}
                fill={l.color}
                stroke="#1e293b"
                strokeWidth={0.5}
              />
            );
          })
        )}
      </svg>
      {/* Legend */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
        {PLAYER_ORDER.map((p) => (
          <div key={p} className="flex items-center gap-1.5">
            <div
              className="w-2.5 h-0.5 rounded"
              style={{ backgroundColor: PLAYER_COLOR_MAP[p] ?? '#94a3b8' }}
            />
            <span className="text-xs text-gray-400">{p}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
