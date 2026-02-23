import React, { useMemo, useState } from 'react';
import { useGameStore } from '../store/gameStore';

const PLAYER_ORDER = ['RED', 'BLUE', 'GREEN', 'YELLOW'] as const;

/** High-contrast colors for each size bucket (mono → pento) */
const BUCKET_COLORS: Record<number, string> = {
  1: '#00F0FF', // cyan - monomino
  2: '#00FF9D', // green - domino
  3: '#FFE600', // yellow - tromino
  4: '#FF9500', // orange - tetromino
  5: '#FF4D4D', // red - pentomino
};

const SIZE_LABELS: Record<number, string> = {
  1: 'Monomino',
  2: 'Domino',
  3: 'Tromino',
  4: 'Tetromino',
  5: 'Pentomino',
};

type SizeBucket = 1 | 2 | 3 | 4 | 5;

// Match LegalMovesPerTurnPlot dimensions for aligned X axis (Turn #)
const CHART_HEIGHT = 56;
const CHART_WIDTH = 340;
const PAD = { top: 8, right: 12, bottom: 20, left: 36 };

/**
 * Single-line mini chart with normalized y-axis (0–1).
 * Uses same width and X scale as Mobility Over Time chart.
 */
const NormalizedMiniChart: React.FC<{
  size: SizeBucket;
  points: { turn: number; val: number }[];
  maxTurn: number;
}> = ({ size, points, maxTurn }) => {
  const color = BUCKET_COLORS[size] ?? '#94a3b8';
  const maxVal = Math.max(1, ...points.map((p) => p.val));
  const plotW = CHART_WIDTH - PAD.left - PAD.right;
  const plotH = CHART_HEIGHT - PAD.top - PAD.bottom;

  const toX = (turn: number) => PAD.left + (turn / Math.max(1, maxTurn)) * plotW;
  const toY = (val: number) => PAD.top + plotH - (val / maxVal) * plotH;

  const path =
    points.length > 0
      ? points
          .map((p, i) => `${i === 0 ? 'M' : 'L'} ${toX(p.turn)} ${toY(p.val)}`)
          .join(' ')
      : '';

  return (
    <div className="py-1 min-w-0 w-full max-w-[340px]">
      <div className="text-xs mb-0.5" style={{ color }}>
        {SIZE_LABELS[size]}
      </div>
      <svg
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        preserveAspectRatio="xMidYMid meet"
        className="block w-full"
        style={{ height: CHART_HEIGHT }}
      >
        <path
          d={path}
          fill="none"
          stroke={color}
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
};

/**
 * Five separate mini-charts of mobility by piece size, each with normalized y-axis.
 * Makes it easy to compare curve shapes across sizes (e.g. pentomino peaks early, monomino peaks late).
 */
export const MobilityBucketsChart: React.FC = () => {
  const legalMovesHistory = useGameStore((s) => s.legalMovesHistory);
  const [selectedPlayer, setSelectedPlayer] = useState<string>('RED');

  const { series, maxTurn, playersWithData } = useMemo(() => {
    if (legalMovesHistory.length === 0) {
      return {
        series: { 1: [], 2: [], 3: [], 4: [], 5: [] } as Record<
          SizeBucket,
          { turn: number; val: number }[]
        >,
        maxTurn: 1,
        playersWithData: [] as string[],
      };
    }

    const playersWithData = [...new Set(legalMovesHistory.flatMap((e) => Object.keys(e.byPlayer)))];
    const maxTurn = legalMovesHistory.length - 1;

    const series: Record<SizeBucket, { turn: number; val: number }[]> = {
      1: [],
      2: [],
      3: [],
      4: [],
      5: [],
    };
    // Carry forward last known value when it's not this player's turn (avoids drop-to-zero spikes)
    const lastVal: Record<SizeBucket, number> = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };

    for (let t = 0; t < legalMovesHistory.length; t++) {
      const m = legalMovesHistory[t].byPlayer[selectedPlayer];
      if (!m || typeof m === 'number') {
        ([1, 2, 3, 4, 5] as const).forEach((s) => {
          series[s].push({ turn: t, val: lastVal[s] });
        });
        continue;
      }
      const b = m.buckets ?? {};
      for (const s of [1, 2, 3, 4, 5] as const) {
        const val = Math.max(0, b[s] ?? 0);
        lastVal[s] = val;
        series[s].push({ turn: t, val });
      }
    }

    return {
      series,
      maxTurn: Math.max(1, maxTurn),
      playersWithData: playersWithData.length > 0 ? playersWithData : PLAYER_ORDER.slice(),
    };
  }, [legalMovesHistory, selectedPlayer]);

  if (legalMovesHistory.length === 0) {
    return (
      <div className="px-3 py-3 border-t border-charcoal-700">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Mobility by Piece Size
        </h3>
        <div className="text-xs text-gray-500 h-20 flex items-center justify-center bg-charcoal-800/50 rounded">
          No turn data yet
        </div>
      </div>
    );
  }

  return (
    <div className="px-3 py-3 border-t border-charcoal-700">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Mobility by Piece Size
        </h3>
        <select
          value={selectedPlayer}
          onChange={(e) => setSelectedPlayer(e.target.value)}
          className="text-xs bg-charcoal-800 border border-charcoal-600 text-gray-200 rounded px-2 py-1"
        >
          {playersWithData.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </div>
      <p className="text-[10px] text-gray-500 mb-2">Each chart: 0–100% of max for that size</p>
      <div className="flex flex-col gap-0 overflow-hidden min-w-0">
        {([1, 2, 3, 4, 5] as const).map((s) => (
          <NormalizedMiniChart key={s} size={s} points={series[s]} maxTurn={maxTurn} />
        ))}
      </div>
    </div>
  );
};
