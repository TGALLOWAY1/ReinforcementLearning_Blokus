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
    isAdvantageMode?: boolean;
    advantageKeys?: string[];
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
    pieceLockRisk: 'Piece Lock Risk',

    // V2 labels
    winProxy: 'Win Proxy Score',
    mobilityNextP10Adv: 'Stability (P10) Adv',
    effectiveFrontierAdv: 'Effective Frontier Adv',
    lockedAreaAdv: 'Locked Area Adv',
    remainingAreaAdv: 'Remaining Area Adv',
    centerControlWeightedAdv: 'Center Control Adv',
    bottleneckScoreAdv: 'Bottleneck Score Adv',
};

export const DivergingBarChart: React.FC<DivergingBarChartProps> = ({ telemetry, perOpponent, isAdvantageMode, advantageKeys }) => {
    const chartData = useMemo(() => {
        if (!telemetry) return [];

        if (isAdvantageMode) {
            // Read from deltaAdvantage instead 
            const source = telemetry.deltaAdvantage || {};
            let metrics = Object.keys(source);

            if (advantageKeys && advantageKeys.length > 0) {
                metrics = advantageKeys.filter(k => k in source);
            }

            return metrics.map(metric => ({
                metric: METRIC_LABELS[metric] || metric,
                [telemetry.moverId]: source[metric] || 0,
            }));
        }

        // Classic Self vs Opponent diverging logic
        const source = telemetry.deltaSelf || {};
        let metrics = Object.keys(source);

        // Filter out the overly complicated raw V2 metrics by default if showing components
        const coreKeys = ['frontierSize', 'effectiveFrontier', 'mobility', 'deadSpace', 'centerControl'];
        metrics = metrics.filter(m => coreKeys.includes(m));

        return metrics.map(metric => {
            const selfVal = telemetry.deltaSelf[metric] || 0;
            const entry: any = {
                metric: METRIC_LABELS[metric] || metric,
                [telemetry.moverId]: selfVal,
            };

            if (perOpponent && telemetry.deltaOppByPlayer) {
                Object.keys(telemetry.deltaOppByPlayer).forEach(opp => {
                    entry[`${opp}_Opp`] = telemetry.deltaOppByPlayer![opp][metric] || 0;
                });
            } else if (telemetry.deltaOppTotal) {
                entry.OpponentAggregate = telemetry.deltaOppTotal[metric] || 0;
            }

            return entry;
        });
    }, [telemetry, perOpponent, isAdvantageMode, advantageKeys]);

    if (!chartData.length) return null;

    const opponentKeys = perOpponent && telemetry.deltaOppByPlayer
        ? Object.keys(telemetry.deltaOppByPlayer)
        : [];

    // Smart formatting: if every numeric value is a whole number, skip decimals
    const allNums = chartData.flatMap(d => Object.values(d).filter(v => typeof v === 'number')) as number[];
    const allInt = allNums.every(v => Number.isInteger(v));
    const fmtVal = (v: number) => {
        const s = allInt ? String(Math.round(Math.abs(v))) : Math.abs(v).toFixed(1);
        return v > 0 ? `+${s}` : v < 0 ? `-${s}` : allInt ? '0' : '0.0';
    };

    return (
        <div className="w-full h-full min-h-[300px] flex flex-col">
            <h3 className="text-sm font-semibold text-gray-300 mb-2 truncate">
                {isAdvantageMode ? "Predictive Advantage Deltas (Higher is Better)" : "Metric Deltas (Self vs Opponents)"}
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
                        <XAxis type="number" stroke="#9ca3af" fontSize={12} tickFormatter={fmtVal} />
                        <YAxis dataKey="metric" type="category" stroke="#9ca3af" fontSize={11} width={130} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', color: '#f3f4f6' }}
                            itemStyle={{ color: '#e5e7eb' }}
                            formatter={(value: any, name: string | undefined) => [fmtVal(Number(value)), (name || '').replace('_Opp', '')]}
                        />
                        <ReferenceLine x={0} stroke="#9ca3af" />

                        {/* Mover's Bar */}
                        <Bar
                            dataKey={telemetry.moverId}
                            fill={PLAYER_COLORS[telemetry.moverId] || '#8884d8'}
                            stackId="stack"
                            name={`${telemetry.moverId} (Self)`}
                            minPointSize={2}
                        />

                        {/* Opponent Bars (Stacked or Aggregate) */}
                        {!isAdvantageMode && (
                            perOpponent ? (
                                opponentKeys.map(opp => (
                                    <Bar
                                        key={opp}
                                        dataKey={`${opp}_Opp`}
                                        fill={PLAYER_COLORS[opp] || '#82ca9d'}
                                        stackId="stack"
                                        name={`${opp} (Opp)`}
                                        opacity={0.8}
                                        minPointSize={2}
                                    />
                                ))
                            ) : (
                                <Bar
                                    dataKey="OpponentAggregate"
                                    fill="#9ca3af"
                                    stackId="stack"
                                    name="Opponents (Total)"
                                    minPointSize={2}
                                />
                            )
                        )}
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
