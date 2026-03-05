import React, { useMemo } from 'react';
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
} from 'recharts';
import { MoveTelemetryDelta } from '../../../types/telemetry';

interface OpponentSuppressionProps {
    allMoves: MoveTelemetryDelta[];
    currentPly: number;
    moverId: string;
    metrics?: string[];
}

const PLAYER_COLORS: Record<string, string> = {
    RED: '#ef4444',
    BLUE: '#3b82f6',
    GREEN: '#22c55e',
    YELLOW: '#eab308',
};

const METRIC_LABELS: Record<string, string> = {
    frontierSize: 'Frontier',
    mobility: 'Mobility',
    deadSpace: 'Dead Space',
};

const DEFAULT_METRICS = ['mobility', 'frontierSize', 'deadSpace'];

/**
 * Small-multiples chart showing opponent metric trajectories over the game.
 * One panel per opponent, showing key suppression metrics over ply number.
 */
export const OpponentSuppressionMultiples: React.FC<OpponentSuppressionProps> = ({
    allMoves,
    currentPly,
    moverId,
    metrics = DEFAULT_METRICS,
}) => {
    // Build per-opponent time series from the cumulative after-snapshots
    const { opponents, timeseriesPerOpponent } = useMemo(() => {
        // Collect opponent IDs
        const oppIds = new Set<string>();
        allMoves.forEach(m => {
            if (m.deltaOppByPlayer) {
                Object.keys(m.deltaOppByPlayer).forEach(id => oppIds.add(id));
            }
        });
        // If no per-player data in deltas, fall back to snapshot `after` array
        if (oppIds.size === 0 && allMoves[0]?.after) {
            allMoves[0].after
                .filter(s => s.playerId !== moverId)
                .forEach(s => oppIds.add(s.playerId));
        }

        const opponentIds = Array.from(oppIds);

        // Build running metric snapshot per opponent using `after` snapshots
        const series: Record<string, { ply: number;[metric: string]: number }[]> = {};
        opponentIds.forEach(id => { series[id] = []; });

        allMoves.forEach(m => {
            const afterSnaps = m.after;
            if (!afterSnaps) return;
            opponentIds.forEach(oppId => {
                const snap = afterSnaps.find(s => s.playerId === oppId);
                if (!snap) return;
                const point: any = { ply: m.ply };
                metrics.forEach(metric => {
                    point[metric] = snap.metrics[metric] ?? 0;
                });
                series[oppId].push(point);
            });
        });

        return { opponents: opponentIds, timeseriesPerOpponent: series };
    }, [allMoves, moverId, metrics]);

    if (opponents.length === 0) {
        return (
            <div className="text-gray-500 text-xs text-center py-4">
                No opponent snapshot data available.
            </div>
        );
    }

    return (
        <div className="flex flex-col gap-3">
            <h3 className="text-sm font-semibold text-gray-300">Opponent Suppression</h3>
            {opponents.map(opp => {
                const color = PLAYER_COLORS[opp] || '#9ca3af';
                const data = timeseriesPerOpponent[opp] || [];

                return (
                    <div key={opp} className="bg-charcoal-900 rounded-lg border border-charcoal-700 p-2">
                        <div className="flex items-center gap-2 mb-1">
                            <span
                                className="w-2 h-2 rounded-full shrink-0"
                                style={{ backgroundColor: color }}
                            />
                            <span className="text-xs font-semibold" style={{ color }}>
                                {opp}
                            </span>
                        </div>
                        <div className="h-[90px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart
                                    data={data}
                                    margin={{ top: 2, right: 8, left: -20, bottom: 2 }}
                                >
                                    <CartesianGrid strokeDasharray="2 2" stroke="#374151" />
                                    <XAxis
                                        dataKey="ply"
                                        type="number"
                                        stroke="#6b7280"
                                        fontSize={9}
                                        domain={['dataMin', 'dataMax']}
                                        tick={false}
                                    />
                                    <YAxis stroke="#6b7280" fontSize={9} width={28} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', color: '#f3f4f6', fontSize: 11 }}
                                        labelFormatter={(l) => `Ply ${l}`}
                                        formatter={(v: any, name: any) => [
                                            typeof v === 'number' ? v.toFixed(1) : v,
                                            METRIC_LABELS[name] || name
                                        ]}
                                    />
                                    {currentPly > 0 && (
                                        <ReferenceLine x={currentPly} stroke="#eab308" strokeDasharray="3 3" />
                                    )}
                                    {metrics.map((metric, i) => {
                                        const metricColors = ['#60a5fa', '#34d399', '#f87171'];
                                        const mc = metricColors[i % metricColors.length];
                                        return (
                                            <Area
                                                key={metric}
                                                type="monotone"
                                                dataKey={metric}
                                                stroke={mc}
                                                fill={mc}
                                                fillOpacity={0.15}
                                                strokeWidth={1.5}
                                                dot={false}
                                                name={metric}
                                            />
                                        );
                                    })}
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                        {/* Mini legend */}
                        <div className="flex gap-3 mt-1">
                            {metrics.map((metric, i) => {
                                const metricColors = ['#60a5fa', '#34d399', '#f87171'];
                                return (
                                    <div key={metric} className="flex items-center gap-1">
                                        <span className="w-2 h-0.5 rounded" style={{ backgroundColor: metricColors[i % 3] }} />
                                        <span className="text-[9px] text-gray-400">{METRIC_LABELS[metric] || metric}</span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                );
            })}
        </div>
    );
};
