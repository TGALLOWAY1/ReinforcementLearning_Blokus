import React, { useMemo } from 'react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
} from 'recharts';
import { MoveTelemetryDelta } from '../../../types/telemetry';

interface DivergingBarChartProps {
    telemetry: MoveTelemetryDelta;
    showRaw: boolean;
    perOpponent: boolean;
    normalizationOptions?: any; // To be added later
}

const PLAYER_COLORS: Record<string, string> = {
    RED: '#ef4444',
    BLUE: '#3b82f6',
    GREEN: '#22c55e',
    YELLOW: '#eab308'
};

const METRIC_LABELS: Record<string, string> = {
    frontierSize: 'Frontier Size',
    mobility: 'Mobility',
    deadSpace: 'Dead Space',
    centerControl: 'Center Control',
    pieceLockRisk: 'Piece Lock Risk'
};

export const DivergingBarChart: React.FC<DivergingBarChartProps> = ({ telemetry, perOpponent }) => {
    const chartData = useMemo(() => {
        if (!telemetry) return [];

        // Convert the delta structures into an array for Recharts
        const metrics = Object.keys(telemetry.deltaSelf);

        return metrics.map(metric => {
            const selfVal = telemetry.deltaSelf[metric] || 0;
            const entry: any = {
                metric: METRIC_LABELS[metric] || metric,
                [telemetry.moverId]: selfVal,
            };

            if (perOpponent && telemetry.deltaOppByPlayer) {
                // Stacked bar approach for each opponent
                Object.keys(telemetry.deltaOppByPlayer).forEach(opp => {
                    entry[`${opp}_Opp`] = telemetry.deltaOppByPlayer![opp][metric] || 0;
                });
            } else if (telemetry.deltaOppTotal) {
                // Aggregate opponent approach
                entry.OpponentAggregate = telemetry.deltaOppTotal[metric] || 0;
            }

            return entry;
        });
    }, [telemetry, perOpponent]);

    if (!chartData.length) return null;

    const opponentKeys = perOpponent && telemetry.deltaOppByPlayer
        ? Object.keys(telemetry.deltaOppByPlayer)
        : [];

    return (
        <div className="w-full h-full min-h-[300px] flex flex-col">
            <h3 className="text-sm font-semibold text-gray-300 mb-2 truncate">
                Metric Deltas (Self vs Opponents)
            </h3>
            <div className="flex-1">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        data={chartData}
                        layout="vertical"
                        margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                        stackOffset="sign"
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" horizontal={false} />
                        <XAxis type="number" stroke="#9ca3af" fontSize={12} tickFormatter={(val) => val > 0 ? `+${val}` : val} />
                        <YAxis dataKey="metric" type="category" stroke="#9ca3af" fontSize={11} width={100} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', color: '#f3f4f6' }}
                            itemStyle={{ color: '#e5e7eb' }}
                            formatter={(value: any, name: string | undefined) => [value > 0 ? `+${value}` : value, (name || '').replace('_Opp', '')]}
                        />
                        <ReferenceLine x={0} stroke="#9ca3af" />

                        {/* Mover's Bar */}
                        <Bar
                            dataKey={telemetry.moverId}
                            fill={PLAYER_COLORS[telemetry.moverId] || '#8884d8'}
                            stackId="stack"
                            name={`${telemetry.moverId} (Self)`}
                        />

                        {/* Opponent Bars (Stacked or Aggregate) */}
                        {perOpponent ? (
                            opponentKeys.map(opp => (
                                <Bar
                                    key={opp}
                                    dataKey={`${opp}_Opp`}
                                    fill={PLAYER_COLORS[opp] || '#82ca9d'}
                                    stackId="stack"
                                    name={`${opp} (Opp)`}
                                    opacity={0.8}
                                />
                            ))
                        ) : (
                            <Bar
                                dataKey="OpponentAggregate"
                                fill="#9ca3af"
                                stackId="stack"
                                name="Opponents (Total)"
                            />
                        )}
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
