import React, { useMemo, useState } from 'react';
import { useLegalMovesByPiece } from '../store/gameStore';

/**
 * Bar chart of legal moves per piece.
 * Data comes from gameState.legal_moves (backend: engine/move_generator.get_legal_moves).
 * Counts are grouped by piece_id for available pieces only.
 * When overrideData is provided (e.g. freeze mode), uses it instead of store.
 */
export const LegalMovesBarChart: React.FC<{
  overrideData?: { pieceId: number; count: number }[];
}> = ({ overrideData }) => {
  const storeData = useLegalMovesByPiece();
  const data = overrideData ?? storeData;
  const [hoveredBar, setHoveredBar] = useState<number | null>(null);

  const { bars, width, height, maxCount, barMaxH, pad } = useMemo(() => {
    const w = 360;
    const h = 160;
    const p = { top: 24, right: 24, bottom: 28, left: 48 };
    const n = 21; // Fixed 21 pieces, spaced as if each label is two digits
    const slotWidth = (w - p.left - p.right) / n;
    const barGap = 4;
    const barW = Math.max(2, slotWidth - barGap);
    const max = Math.max(1, ...data.map((d) => d.count));
    const barMaxH = h - p.top - p.bottom;

    const bars = data.map((d, i) => {
      const x = p.left + i * slotWidth + (slotWidth - barW) / 2;
      const barHeight = max > 0 ? (d.count / max) * barMaxH : 0;
      const y = p.top + barMaxH - barHeight;
      return { ...d, x, y, width: barW, height: barHeight, slotCenter: p.left + (i + 0.5) * slotWidth };
    });

    return { bars, width: w, height: h, maxCount: max, barMaxH, pad: p };
  }, [data]);

  if (!data.length) {
    return (
      <div className="px-3 py-3 border-t border-charcoal-700">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Legal moves by piece
        </h3>
        <div className="text-xs text-gray-500 h-24 flex items-center justify-center bg-charcoal-800/50 rounded">
          No available pieces or no legal moves
        </div>
      </div>
    );
  }

  // Y-axis ticks: 0 and a few values up to maxCount
  const yTicks = useMemo(() => {
    if (maxCount <= 0) return [0];
    const ticks: number[] = [0];
    if (maxCount <= 5) {
      for (let i = 1; i <= maxCount; i++) ticks.push(i);
    } else {
      const step = Math.ceil(maxCount / 4);
      for (let v = step; v < maxCount; v += step) ticks.push(v);
      ticks.push(maxCount);
    }
    return [...new Set(ticks)].sort((a, b) => a - b);
  }, [maxCount]);

  return (
    <div className="px-3 py-3 border-t border-charcoal-700 relative">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Legal moves by piece
      </h3>
      <svg width={width} height={height} className="overflow-visible">
        {/* Y-axis label */}
        <text
          x={16}
          y={height / 2}
          fill="#94a3b8"
          fontSize={10}
          textAnchor="middle"
          transform={`rotate(-90, 16, ${height / 2})`}
        >
          Legal moves
        </text>
        {/* Y-axis ticks and labels */}
        {yTicks.map((tickVal) => {
          const y = pad.top + barMaxH - (tickVal / maxCount) * barMaxH;
          return (
            <g key={tickVal}>
              <line
                x1={pad.left}
                y1={y}
                x2={pad.left - 4}
                y2={y}
                stroke="#64748b"
                strokeWidth={1}
              />
              <text
                x={pad.left - 8}
                y={y}
                fill="#94a3b8"
                fontSize={9}
                textAnchor="end"
                dominantBaseline="middle"
              >
                {tickVal}
              </text>
            </g>
          );
        })}
        {/* Bars with hover */}
        {bars.map((b) => {
          const isHovered = hoveredBar === b.pieceId;
          return (
            <g
              key={b.pieceId}
              onMouseEnter={() => setHoveredBar(b.pieceId)}
              onMouseLeave={() => setHoveredBar(null)}
            >
              <rect
                x={b.x}
                y={b.y}
                width={b.width}
                height={b.height}
                fill={isHovered ? '#4ade80' : '#22c55e'}
                rx={1}
                className="transition-colors"
              />
              {/* Invisible larger hit area for easier hover */}
              <rect
                x={b.x - 2}
                y={pad.top}
                width={b.width + 4}
                height={barMaxH}
                fill="transparent"
              />
            </g>
          );
        })}
        {/* X-axis ticks - spaced for two-digit width, display as 1..21 */}
        {bars.map((b) => (
          <text
            key={`tick-${b.pieceId}`}
            x={b.slotCenter}
            y={height - 6}
            fill="#94a3b8"
            fontSize={10}
            textAnchor="middle"
          >
            {b.pieceId}
          </text>
        ))}
      </svg>
      {/* Hover tooltip */}
      {hoveredBar !== null && (() => {
        const b = bars.find((x) => x.pieceId === hoveredBar);
        if (!b) return null;
        return (
          <div
            className="absolute z-10 px-2 py-1 text-xs font-medium bg-charcoal-800 border border-charcoal-600 rounded shadow-lg pointer-events-none whitespace-nowrap"
            style={{
              left: b.slotCenter,
              top: Math.max(4, b.y - 22),
              transform: 'translateX(-50%)',
            }}
          >
            Piece {b.pieceId}: {b.count} legal move{b.count !== 1 ? 's' : ''}
          </div>
        );
      })()}
    </div>
  );
};
