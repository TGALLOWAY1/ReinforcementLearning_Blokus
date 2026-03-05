import React, { useMemo } from 'react';
import {
    Radar,
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    ResponsiveContainer,
    Tooltip,
    Legend
} from 'recharts';
import { MoveTelemetryDelta } from '../../../types/telemetry';

interface RadarDeltaChartProps {
    telemetry: MoveTelemetryDelta;
    showOpponents?: boolean;
}

const METRIC_LABELS: Record<string, string> = {
    frontierSize: 'Frontier Size',
    mobility: 'Mobility',
    deadSpace: 'Dead Space',
    centerControl: 'Center Control',
    pieceLockRisk: 'Piece Lock Risk'
};

const PLAYER_COLORS: Record<string, string> = {
    RED: '#ef4444',
    BLUE: '#3b82f6',
    GREEN: '#22c55e',
    YELLOW: '#eab308'
};

export const RadarDeltaChart: React.FC<RadarDeltaChartProps> = ({ telemetry, showOpponents = false }) => {
    const chartData = useMemo(() => {
        if (!telemetry || !telemetry.before || !telemetry.after) return [];

        // Find the mover's snapshots before and after
        const moverBefore = telemetry.before.find((p: any) => p.playerId === telemetry.moverId);
        const moverAfter = telemetry.after.find((p: any) => p.playerId === telemetry.moverId);

        if (!moverBefore || !moverAfter) return [];

        const metricsKeys = Object.keys(moverBefore.metrics);

        return metricsKeys.map(metric => {
            const entry: any = {
                metric: METRIC_LABELS[metric] || metric,
                Before: moverBefore.metrics[metric] || 0,
                After: moverAfter.metrics[metric] || 0,
            };

            if (showOpponents) {
                // Aggregate opponents Before
                let oppBeforeTotal = 0;
                let oppAfterTotal = 0;

                telemetry.before!.forEach((p: any) => {
                    if (p.playerId !== telemetry.moverId) oppBeforeTotal += (p.metrics[metric] || 0);
                });

                telemetry.after!.forEach((p: any) => {
                    if (p.playerId !== telemetry.moverId) oppAfterTotal += (p.metrics[metric] || 0);
                });

                entry.OppBefore = oppBeforeTotal;
                entry.OppAfter = oppAfterTotal;
            }

            return entry;
        });
    }, [telemetry, showOpponents]);

    if (!chartData.length) return null;

    const moverColor = PLAYER_COLORS[telemetry.moverId] || '#8884d8';

    return (
        <div className="w-full h-full min-h-[300px] flex flex-col items-center">
            <h3 className="text-sm font-semibold text-gray-300 mb-2 truncate w-full text-left">
                Move Shape (Before vs After)
            </h3>
            <div className="flex-1 w-full max-w-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                    <RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData}>
                        <PolarGrid stroke="#4a5568" />
                        <PolarAngleAxis dataKey="metric" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                        <PolarRadiusAxis angle={30} domain={['auto', 'auto']} tick={{ fill: '#6b7280', fontSize: 10 }} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', color: '#f3f4f6' }}
                            itemStyle={{ color: '#e5e7eb' }}
                        />
                        <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />

                        {showOpponents && (
                            <>
                                <Radar name="Opponents (Before)" dataKey="OppBefore" stroke="#6b7280" fill="#6b7280" fillOpacity={0.1} />
                                <Radar name="Opponents (After)" dataKey="OppAfter" stroke="#9ca3af" fill="#9ca3af" fillOpacity={0.4} />
                            </>
                        )}

                        <Radar name={`${telemetry.moverId} (Before)`} dataKey="Before" stroke={moverColor} fill={moverColor} fillOpacity={0.1} strokeDasharray="3 3" />
                        <Radar name={`${telemetry.moverId} (After)`} dataKey="After" stroke={moverColor} fill={moverColor} fillOpacity={0.5} />
                    </RadarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
