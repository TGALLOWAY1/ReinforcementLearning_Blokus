import React, { useState, useMemo } from 'react';
import { MoveTelemetryDelta } from '../../types/telemetry';
import {
    computeStrategyMix,
    Phase,
    StrategyMixResult,
} from '../../utils/strategyMix';
import { WeightPreset, NormalizationMethod } from '../../utils/moveImpactScore';

interface StrategyMixPanelProps {
    allMoves: MoveTelemetryDelta[];
    playerId: string;
    preset: WeightPreset;
    normalization: NormalizationMethod;
    onSelectPly?: (ply: number) => void;
}

const METRIC_LABELS: Record<string, string> = {
    frontierSize: 'Frontier Size',
    mobility: 'Mobility',
    deadSpace: 'Dead Space',
    centerControl: 'Center Control',
    pieceLockRisk: 'Piece Lock Risk',
};

const METRIC_COLORS: Record<string, string> = {
    frontierSize: '#60a5fa',
    mobility: '#34d399',
    deadSpace: '#f87171',
    centerControl: '#a78bfa',
    pieceLockRisk: '#fbbf24',
};

const PHASE_ORDER: Phase[] = ['opening', 'mid', 'endgame'];
const PHASE_COLORS: Record<Phase, string> = {
    opening: '#60a5fa',
    mid: '#a78bfa',
    endgame: '#f97316',
};

export const StrategyMixPanel: React.FC<StrategyMixPanelProps> = ({
    allMoves,
    playerId,
    preset,
    normalization,
    onSelectPly,
}) => {
    const [activePhase, setActivePhase] = useState<Phase>('opening');

    const mix: StrategyMixResult = useMemo(
        () => computeStrategyMix(allMoves, playerId, preset, normalization),
        [allMoves, playerId, preset, normalization]
    );

    const phaseData = mix.phases.find(p => p.phase === activePhase);

    // Metric contributions sorted by % desc
    const sortedMetrics = (Object.entries(mix.metricContributions) as [string, number][])
        .sort(([, a], [, b]) => b - a);

    return (
        <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-gray-300">
                    Strategy Mix
                    <span className="ml-2 text-xs text-gray-500">({playerId})</span>
                </h3>
                <span className={`text-xs font-mono font-bold ${mix.totalImpact >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    Total: {mix.totalImpact >= 0 ? '+' : ''}{mix.totalImpact.toFixed(2)}
                </span>
            </div>

            {/* Metric contribution bars */}
            <div className="space-y-1.5">
                {sortedMetrics.map(([metric, pct]) => (
                    <div key={metric} className="flex items-center gap-2">
                        <div className="w-24 text-[10px] text-gray-400 truncate shrink-0">
                            {METRIC_LABELS[metric] || metric}
                        </div>
                        <div className="flex-1 h-2 bg-charcoal-700 rounded overflow-hidden">
                            <div
                                className="h-full rounded transition-all"
                                style={{
                                    width: `${pct.toFixed(1)}%`,
                                    backgroundColor: METRIC_COLORS[metric] || '#9ca3af',
                                }}
                            />
                        </div>
                        <span className="text-[10px] text-gray-400 w-9 text-right shrink-0">
                            {pct.toFixed(0)}%
                        </span>
                    </div>
                ))}
            </div>

            {/* Phase tabs */}
            <div className="flex gap-1 mt-1">
                {PHASE_ORDER.map(phase => {
                    const phaseInfo = mix.phases.find(p => p.phase === phase);
                    const active = activePhase === phase;
                    return (
                        <button
                            key={phase}
                            onClick={() => setActivePhase(phase)}
                            className={`flex-1 py-1 text-xs rounded transition-colors font-medium ${active
                                ? 'text-charcoal-900 font-bold'
                                : 'bg-charcoal-700 text-gray-400 hover:bg-charcoal-600'
                                }`}
                            style={active ? { backgroundColor: PHASE_COLORS[phase] } : {}}
                        >
                            {phaseInfo?.label} ({phaseInfo?.moves.length ?? 0})
                        </button>
                    );
                })}
            </div>

            {/* Phase stats */}
            {phaseData && (
                <div className="bg-charcoal-900 rounded-lg border border-charcoal-700 p-3 space-y-2">
                    <div className="flex justify-between text-xs">
                        <span className="text-gray-400">Avg Impact</span>
                        <span className={`font-mono font-bold ${phaseData.avgImpact >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {phaseData.avgImpact >= 0 ? '+' : ''}{phaseData.avgImpact.toFixed(3)}
                        </span>
                    </div>
                    <div className="flex justify-between text-xs">
                        <span className="text-gray-400">Dominant Metric</span>
                        <span className="text-gray-200 font-medium" style={{ color: METRIC_COLORS[phaseData.dominantMetric] || '#9ca3af' }}>
                            {METRIC_LABELS[phaseData.dominantMetric] || phaseData.dominantMetric}
                        </span>
                    </div>
                    {phaseData.biggestTurningMove && (
                        <div className="flex justify-between items-center text-xs">
                            <span className="text-gray-400">Biggest Move</span>
                            <button
                                onClick={() => onSelectPly?.(phaseData.biggestTurningMove!.delta.ply)}
                                className="text-blue-400 hover:text-blue-300 font-mono underline underline-offset-2"
                            >
                                Ply {phaseData.biggestTurningMove.delta.ply} ({phaseData.biggestTurningMove.score.total >= 0 ? '+' : ''}{phaseData.biggestTurningMove.score.total.toFixed(2)})
                            </button>
                        </div>
                    )}
                    {phaseData.moves.length === 0 && (
                        <p className="text-gray-500 text-xs">No moves in this phase.</p>
                    )}
                </div>
            )}
        </div>
    );
};
