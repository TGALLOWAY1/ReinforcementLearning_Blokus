import React, { useMemo } from 'react';
import {
    Radar,
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    ResponsiveContainer,
    Tooltip,
} from 'recharts';
import { MoveTelemetryDelta } from '../../../types/telemetry';

interface RadarDeltaChartProps {
    telemetry: MoveTelemetryDelta;
    showOpponents?: boolean;
}

/** Expansion (attacking) metrics — how much does the move grow options */
const EXPANSION_METRICS: { key: string; label: string }[] = [
    { key: 'frontierSize', label: 'Frontier' },
    { key: 'mobility', label: 'Mobility' },
    { key: 'centerControl', label: 'Center' },
    { key: 'remainingArea', label: 'Open Area' },
    { key: 'effectiveFrontier', label: 'Eff. Frontier' },
];

/** Risk / defensive metrics — exposure and vulnerability */
const RISK_METRICS: { key: string; label: string }[] = [
    { key: 'deadSpace', label: 'Dead Space' },
    { key: 'pieceLockRisk', label: 'Lock Risk' },
    { key: 'mobilityDropRisk', label: 'Mob. Drop' },
    { key: 'bottleneckScore', label: 'Bottleneck' },
    { key: 'lockedArea', label: 'Locked Area' },
];

const PLAYER_COLORS: Record<string, string> = {
    RED: '#ef4444',
    BLUE: '#3b82f6',
    GREEN: '#22c55e',
    YELLOW: '#eab308',
};

function buildRadarData(
    metrics: { key: string; label: string }[],
    moverBefore: any,
    moverAfter: any,
    telemetry: MoveTelemetryDelta,
    showOpponents: boolean,
) {
    return metrics.map(({ key, label }) => {
        const entry: any = {
            metric: label,
            Before: moverBefore?.metrics?.[key] ?? 0,
            After: moverAfter?.metrics?.[key] ?? 0,
        };
        if (showOpponents) {
            let oppBefore = 0;
            let oppAfter = 0;
            telemetry.before?.forEach((p: any) => {
                if (p.playerId !== telemetry.moverId) oppBefore += p.metrics?.[key] ?? 0;
            });
            telemetry.after?.forEach((p: any) => {
                if (p.playerId !== telemetry.moverId) oppAfter += p.metrics?.[key] ?? 0;
            });
            entry.OppBefore = oppBefore;
            entry.OppAfter = oppAfter;
        }
        return entry;
    });
}

/** Smart number formatter: integer if all values are whole numbers */
function fmtRadar(v: number, allVals: number[]): string {
    const allInt = allVals.every(x => Number.isInteger(x));
    return allInt ? String(Math.round(v)) : v.toFixed(1);
}

const MiniRadar: React.FC<{
    title: string;
    data: any[];
    moverColor: string;
    moverId: string;
    showOpponents: boolean;
}> = ({ title, data, moverColor, moverId, showOpponents }) => {
    // Collect all numeric values for integer-detection
    const allVals = data.flatMap(d => [d.Before ?? 0, d.After ?? 0, d.OppBefore ?? 0, d.OppAfter ?? 0]);

    return (
        <div className="flex-1 flex flex-col min-w-0 min-h-0">
            <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest text-center mb-1 shrink-0">
                {title}
            </p>
            {/* Chart — takes remaining height */}
            <div className="flex-1 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <RadarChart cx="50%" cy="50%" outerRadius="65%" data={data}>
                        <PolarGrid stroke="#374151" />
                        <PolarAngleAxis
                            dataKey="metric"
                            tick={{ fill: '#9ca3af', fontSize: 9 }}
                        />
                        <PolarRadiusAxis
                            angle={30}
                            domain={['auto', 'auto']}
                            tick={{ fill: '#6b7280', fontSize: 8 }}
                            tickCount={3}
                            tickFormatter={(v) => fmtRadar(v, allVals)}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', color: '#f3f4f6', fontSize: 11 }}
                            itemStyle={{ color: '#e5e7eb' }}
                            formatter={(v: any) => fmtRadar(Number(v), allVals)}
                        />
                        {showOpponents && (
                            <>
                                <Radar name="Opp Before" dataKey="OppBefore" stroke="#6b7280" fill="#6b7280" fillOpacity={0.08} strokeDasharray="3 3" dot={false} />
                                <Radar name="Opp After" dataKey="OppAfter" stroke="#9ca3af" fill="#9ca3af" fillOpacity={0.2} dot={false} />
                            </>
                        )}
                        <Radar name={`${moverId} Before`} dataKey="Before" stroke={moverColor} fill={moverColor} fillOpacity={0.08} strokeDasharray="4 2" dot={false} />
                        <Radar name={`${moverId} After`} dataKey="After" stroke={moverColor} fill={moverColor} fillOpacity={0.4} dot={false} />
                        {/* No <Legend> — we render a shared legend below both charts */}
                    </RadarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export const RadarDeltaChart: React.FC<RadarDeltaChartProps> = ({ telemetry, showOpponents = false }) => {
    const { expansionData, riskData, moverColor } = useMemo(() => {
        if (!telemetry?.before || !telemetry?.after) {
            return { expansionData: [], riskData: [], moverColor: '#8884d8' };
        }
        const moverBefore = telemetry.before.find((p: any) => p.playerId === telemetry.moverId);
        const moverAfter = telemetry.after.find((p: any) => p.playerId === telemetry.moverId);
        return {
            expansionData: buildRadarData(EXPANSION_METRICS, moverBefore, moverAfter, telemetry, showOpponents),
            riskData: buildRadarData(RISK_METRICS, moverBefore, moverAfter, telemetry, showOpponents),
            moverColor: PLAYER_COLORS[telemetry.moverId] || '#8884d8',
        };
    }, [telemetry, showOpponents]);

    if (!expansionData.length) return null;

    return (
        <div className="w-full h-full flex flex-col min-h-0">
            <h3 className="text-sm font-semibold text-gray-300 mb-1 shrink-0">
                Move Shape (Before vs After)
            </h3>

            {/* Two mini radars side by side */}
            <div className="flex-1 flex gap-2 min-h-0">
                <MiniRadar
                    title="Expansion"
                    data={expansionData}
                    moverColor={moverColor}
                    moverId={telemetry.moverId}
                    showOpponents={showOpponents}
                />
                <div className="w-px bg-charcoal-700 shrink-0" />
                <MiniRadar
                    title="Risk"
                    data={riskData}
                    moverColor={moverColor}
                    moverId={telemetry.moverId}
                    showOpponents={showOpponents}
                />
            </div>

            {/* Shared legend — lives below both charts, no overlap */}
            <div className="shrink-0 flex items-center justify-center gap-4 pt-1 flex-wrap">
                <span className="flex items-center gap-1 text-[10px] text-gray-400">
                    <span className="inline-block w-5 h-0.5 rounded" style={{ borderTop: `1.5px dashed ${moverColor}` }} />
                    {telemetry.moverId} Before
                </span>
                <span className="flex items-center gap-1 text-[10px] text-gray-400">
                    <span className="inline-block w-5 h-1.5 rounded" style={{ backgroundColor: moverColor, opacity: 0.7 }} />
                    {telemetry.moverId} After
                </span>
                {showOpponents && (
                    <>
                        <span className="flex items-center gap-1 text-[10px] text-gray-400">
                            <span className="inline-block w-5 h-0.5 rounded border-t border-dashed border-gray-500" />
                            Opp Before
                        </span>
                        <span className="flex items-center gap-1 text-[10px] text-gray-400">
                            <span className="inline-block w-5 h-1.5 rounded bg-gray-400 opacity-60" />
                            Opp After
                        </span>
                    </>
                )}
            </div>
        </div>
    );
};
