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

export const OpponentAdjacencyPlot: React.FC<{
    overrideHistory?: LegalMovesHistoryEntry[];
}> = ({ overrideHistory }) => {
    const storeHistory = useGameStore((s) => s.legalMovesHistory);
    const legalMovesHistory = overrideHistory !== undefined ? overrideHistory : storeHistory;
    const gameState = useGameStore((s) => s.gameState);
    const winner = gameState?.winner ?? null;

    const { lines, maxTurn, maxCount, width, height, pad, plotW, plotH, toY } = useMemo(() => {
        const w = 740;
        const h = 180;
        const pad = { top: 16, right: 12, bottom: 32, left: 36 };

        const toY_default = () => pad.top + (h - pad.top - pad.bottom) / 2;
        if (legalMovesHistory.length === 0) {
            return { lines: [], maxTurn: 1, maxCount: 1, width: w, height: h, pad, plotW: w - pad.left - pad.right, plotH: h - pad.top - pad.bottom, toY: toY_default };
        }

        const maxTurn = legalMovesHistory.length - 1;
        let maxCount = 1;
        for (const e of legalMovesHistory) {
            for (const p of PLAYER_ORDER) {
                const v = e.advanced?.[p]?.opponent_adjacency;
                if (v !== undefined) {
                    maxCount = Math.max(maxCount, v);
                }
            }
        }
        maxCount = Math.ceil(maxCount);

        const plotW = w - pad.left - pad.right;
        const plotH = h - pad.top - pad.bottom;

        const toX = (turn: number) => pad.left + (turn / Math.max(1, maxTurn)) * plotW;
        const toY = (count: number) => pad.top + plotH - (count / Math.max(0.1, maxCount)) * plotH;

        const lines = PLAYER_ORDER.map((player) => {
            const points: { turn: number; count: number }[] = [];
            legalMovesHistory.forEach((e: LegalMovesHistoryEntry, turn) => {
                const v = e.advanced?.[player]?.opponent_adjacency;
                if (v !== undefined) {
                    points.push({ turn, count: v });
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

        return { lines, maxTurn: Math.max(1, maxTurn), maxCount, width: w, height: h, pad, plotW, plotH, toY };
    }, [legalMovesHistory, winner]);

    if (legalMovesHistory.length === 0) {
        return (
            <div className="px-3 py-3 border-t border-charcoal-700">
                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                    Opponent Adjacency Over Time
                </h3>
                <div className="text-xs text-gray-500 h-24 flex items-center justify-center bg-charcoal-800/50 rounded">
                    No turn data yet
                </div>
            </div>
        );
    }

    const displayMax = maxCount;
    return (
        <div className="px-3 py-3 border-t border-charcoal-700">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                Opponent Adjacency (Blocking)
            </h3>
            <svg width={width} height={height} className="overflow-visible">
                {/* Y-axis label */}
                <text x={12} y={height / 2} fill="#94a3b8" fontSize={9} textAnchor="middle" transform={`rotate(-90, 12, ${height / 2})`}>
                    Interactions
                </text>
                {/* X-axis label */}
                <text x={pad.left + plotW / 2} y={height - 6} fill="#94a3b8" fontSize={9} textAnchor="middle">
                    Turn #
                </text>

                {/* Grid lines (horizontal) */}
                {[0.25, 0.5, 0.75].map((t) => (
                    <line key={`h-${t}`} x1={pad.left} y1={pad.top + (1 - t) * plotH} x2={pad.left + plotW} y2={pad.top + (1 - t) * plotH} stroke="#334155" strokeWidth={0.5} />
                ))}

                {/* Vertical grid lines */}
                {[0.25, 0.5, 0.75].map((t) => (
                    <line key={`v-${t}`} x1={pad.left + t * plotW} y1={pad.top} x2={pad.left + t * plotW} y2={pad.top + plotH} stroke="#334155" strokeWidth={0.5} />
                ))}

                {/* Y-axis Ticks */}
                {[0, Math.floor(displayMax / 2), displayMax].map((tickVal) => {
                    const y = toY(tickVal);
                    return (
                        <g key={`ytick-${tickVal}`}>
                            <line x1={pad.left} y1={y} x2={pad.left - 4} y2={y} stroke="#64748b" strokeWidth={1} />
                            <text x={pad.left - 6} y={y} fill="#94a3b8" fontSize={9} textAnchor="end" dominantBaseline="middle">
                                {tickVal}
                            </text>
                        </g>
                    );
                })}

                {/* Lines */}
                {lines.map((l) =>
                    l.path ? (
                        <path key={l.player} d={l.path} fill="none" stroke={l.color} strokeWidth={l.strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
                    ) : null
                )}
                {/* Markers */}
                {lines.map((l) =>
                    l.points.map((p, i) => {
                        const x = pad.left + (p.turn / maxTurn) * plotW;
                        const y = toY(p.count);
                        return <circle key={`${l.player}-${i}`} cx={x} cy={y} r={2} fill={l.color} stroke="#1e293b" strokeWidth={0.5} />;
                    })
                )}
            </svg>
        </div>
    );
};
