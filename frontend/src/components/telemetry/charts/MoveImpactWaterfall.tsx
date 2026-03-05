import React, { useMemo } from 'react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Cell,
    ReferenceLine,
} from 'recharts';
import { MoveTelemetryDelta } from '../../../types/telemetry';
import {
    computeGameImpactScores,
    WeightPreset,
    NormalizationMethod,
    WEIGHT_PRESETS,
} from '../../../utils/moveImpactScore';

interface MoveImpactWaterfallProps {
    telemetry: MoveTelemetryDelta;
    preset: WeightPreset;
    normalization: NormalizationMethod;
    allMoves: MoveTelemetryDelta[]; // needed  for normalization context
}

const METRIC_LABELS: Record<string, string> = {
    frontierSize: 'Frontier',
    mobility: 'Mobility',
    deadSpace: 'Dead Space',
    centerControl: 'Center Ctrl',
    pieceLockRisk: 'Lock Risk'
};

export const MoveImpactWaterfall: React.FC<MoveImpactWaterfallProps> = ({
    telemetry,
    preset,
    normalization,
    allMoves,
}) => {
    const impactData = useMemo(() => {
        const weights = WEIGHT_PRESETS[preset];
        const contributions = Object.entries(weights).map(([metric, weight]) => {
            const selfVal = telemetry.deltaSelf[metric] ?? 0;
            const oppVal = telemetry.deltaOppTotal?.[metric] ?? 0;
            const value = weight * (selfVal - oppVal);
            return {
                metric: METRIC_LABELS[metric] || metric,
                value,
                positive: value >= 0,
            };
        });

        // Compute total from normalization context
        const scores = computeGameImpactScores(allMoves, preset, normalization);
        const idx = allMoves.findIndex(m => m.ply === telemetry.ply);
        const total = idx >= 0 ? scores[idx].total : 0;

        return { contributions, total };
    }, [telemetry, preset, normalization, allMoves]);

    return (
        <div className="w-full h-full min-h-[300px] flex flex-col">
            <h3 className="text-sm font-semibold text-gray-300 mb-2 truncate">
                Move Impact Waterfall ({preset})
            </h3>

            <div className="flex-1">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        data={impactData.contributions}
                        layout="vertical"
                        margin={{ top: 5, right: 30, left: 70, bottom: 5 }}
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" horizontal={false} />
                        <XAxis
                            type="number"
                            stroke="#9ca3af"
                            fontSize={11}
                            tickFormatter={(v: any) => v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2)}
                        />
                        <YAxis
                            dataKey="metric"
                            type="category"
                            stroke="#9ca3af"
                            fontSize={11}
                            width={68}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', color: '#f3f4f6' }}
                            itemStyle={{ color: '#e5e7eb' }}
                            formatter={(v: any) => [v > 0 ? `+${v.toFixed(3)}` : v.toFixed(3), 'Contribution']}
                        />
                        <ReferenceLine x={0} stroke="#9ca3af" />
                        <Bar dataKey="value" name="Impact">
                            {impactData.contributions.map((entry, index) => (
                                <Cell
                                    key={index}
                                    fill={entry.positive ? '#22c55e' : '#ef4444'}
                                    opacity={0.8}
                                />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>

            <div className="mt-2 pt-2 border-t border-charcoal-700 flex items-center justify-between px-1">
                <span className="text-xs text-gray-400">Total Impact Score</span>
                <span className={`text-sm font-bold font-mono ${impactData.total >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {impactData.total >= 0 ? '+' : ''}{impactData.total.toFixed(3)}
                </span>
            </div>
        </div>
    );
};
