import React, { useMemo } from 'react';
import { MoveTelemetryDelta } from '../../../types/telemetry';
import {
    computeGameImpactScores,
    WeightPreset,
    NormalizationMethod,
} from '../../../utils/moveImpactScore';

interface TopMovesLeaderboardProps {
    allMoves: MoveTelemetryDelta[];
    preset: WeightPreset;
    normalization: NormalizationMethod;
    winnerId?: string;
    selectedPly: number;
    onSelectPly: (ply: number) => void;
}

const PLAYER_COLORS: Record<string, string> = {
    RED: '#ef4444',
    BLUE: '#3b82f6',
    GREEN: '#22c55e',
    YELLOW: '#eab308',
};

export const TopMovesLeaderboard: React.FC<TopMovesLeaderboardProps> = ({
    allMoves,
    preset,
    normalization,
    winnerId,
    selectedPly,
    onSelectPly,
}) => {
    const ranked = useMemo(() => {
        const scores = computeGameImpactScores(allMoves, preset, normalization);
        return allMoves
            .map((m, i) => ({ ...m, score: scores[i].total }))
            // Filter to winner's moves if winnerId is known
            .filter(m => !winnerId || m.moverId === winnerId)
            .sort((a, b) => b.score - a.score)
            .slice(0, 10); // Top 10
    }, [allMoves, preset, normalization, winnerId]);

    if (!ranked.length) return null;

    return (
        <div className="w-full flex flex-col">
            <h3 className="text-sm font-semibold text-gray-300 mb-2">
                Top Moves {winnerId ? `(${winnerId})` : '(All Players)'}
            </h3>
            <div className="space-y-1">
                {ranked.map((m, rank) => {
                    const color = PLAYER_COLORS[m.moverId] || '#9ca3af';
                    const isSelected = m.ply === selectedPly;

                    // Mini sparkline bar 
                    const barWidth = Math.min(100, Math.abs(m.score) * 40); // approximate scaling

                    return (
                        <button
                            key={m.ply}
                            onClick={() => onSelectPly(m.ply)}
                            className={`w-full text-left px-3 py-2 rounded flex items-center gap-3 transition-colors group ${isSelected
                                    ? 'bg-charcoal-600 ring-1 ring-neon-blue'
                                    : 'bg-charcoal-800 hover:bg-charcoal-700'
                                }`}
                        >
                            {/* Rank */}
                            <span className="text-xs text-gray-500 w-5 shrink-0 text-center">
                                #{rank + 1}
                            </span>

                            {/* Player dot */}
                            <span
                                className="w-2 h-2 rounded-full shrink-0"
                                style={{ backgroundColor: color }}
                            />

                            {/* Ply + piece info */}
                            <span className="text-xs text-gray-300 flex-1 truncate">
                                Ply {m.ply}{m.moveId ? ` · ${m.moveId.split('-')[0]}` : ''}
                            </span>

                            {/* Mini bar */}
                            <div className="w-16 h-2 bg-charcoal-700 rounded overflow-hidden">
                                <div
                                    className="h-full rounded"
                                    style={{
                                        width: `${barWidth}%`,
                                        backgroundColor: m.score >= 0 ? '#22c55e' : '#ef4444',
                                    }}
                                />
                            </div>

                            {/* Score */}
                            <span className={`text-xs font-mono w-14 text-right shrink-0 ${m.score >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {m.score >= 0 ? '+' : ''}{m.score.toFixed(2)}
                            </span>
                        </button>
                    );
                })}
            </div>
        </div>
    );
};
