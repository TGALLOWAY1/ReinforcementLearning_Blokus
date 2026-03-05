import React, { useState, useMemo, useEffect } from 'react';
import { useGameStore } from '../../store/gameStore';
import { DivergingBarChart } from './charts/DivergingBarChart';
import { RadarDeltaChart } from './charts/RadarDeltaChart';
import { CumulativeTimelineChart } from './charts/CumulativeTimelineChart';
import { MoveImpactWaterfall } from './charts/MoveImpactWaterfall';
import { TopMovesLeaderboard } from './charts/TopMovesLeaderboard';
import { WeightPreset, NormalizationMethod } from '../../utils/moveImpactScore';
import { MoveTelemetryDelta } from '../../types/telemetry';
import { OpponentSuppressionMultiples } from './charts/OpponentSuppressionMultiples';

import {
    ModuleE_FrontierChart,
    ModuleF_UrgencyChart,
} from '../AnalysisDashboard';

// --- Constants ---
const PLAYER_COLORS: Record<string, string> = {
    RED: '#ef4444',
    BLUE: '#3b82f6',
    GREEN: '#22c55e',
    YELLOW: '#eab308',
};

const PRESETS: { label: string; value: WeightPreset }[] = [
    { label: 'Balanced', value: 'balanced' },
    { label: 'Blocking', value: 'blocking' },
    { label: 'Expansion', value: 'expansion' },
    { label: 'Late-game', value: 'late-game' },
];

// --- Collapsible sub-section ---
const Collapsible: React.FC<{ title: string; defaultOpen?: boolean; children: React.ReactNode; badge?: string }> = ({
    title, defaultOpen = true, children, badge,
}) => {
    const [open, setOpen] = useState(defaultOpen);
    return (
        <div className="bg-charcoal-800 rounded-lg border border-charcoal-700 overflow-hidden">
            <button
                onClick={() => setOpen(!open)}
                className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-charcoal-700 transition-colors"
            >
                <span className="text-xs font-bold text-gray-300 uppercase tracking-widest">{title}</span>
                <div className="flex items-center gap-2">
                    {badge && <span className="text-[10px] bg-blue-900/50 text-blue-400 px-1.5 py-0.5 rounded font-mono">{badge}</span>}
                    <span className="text-neon-blue text-sm">{open ? '−' : '+'}</span>
                </div>
            </button>
            {open && <div className="border-t border-charcoal-700">{children}</div>}
        </div>
    );
};



// ============================================================
// Main Component
// ============================================================
export const MergedAnalysisPanel: React.FC = () => {
    const gameState = useGameStore((s) => s.gameState);
    const currentSliderTurn = useGameStore((s) => s.currentSliderTurn);
    const setCurrentSliderTurn = useGameStore((s) => s.setCurrentSliderTurn);
    const setBoardOverlay = useGameStore((s) => s.setBoardOverlay);

    // Move Impact controls
    const [showRaw] = useState(false);  // retained for DivergingBarChart prop
    const [showAdvantage, setShowAdvantage] = useState(true);
    const [perOpponent, setPerOpponent] = useState(false);
    const [preset, setPreset] = useState<WeightPreset>('balanced');
    const [normalization] = useState<NormalizationMethod>('z-score');
    const [showOverlay, setShowOverlay] = useState(false);

    // Suppression view: player selector
    const [suppressionPlayer, setSuppressionPlayer] = useState<string>('');

    // Chart x-axis mode
    const [xAxisMode, setXAxisMode] = useState<'move' | 'round'>('move');

    if (!gameState) {
        return (
            <div className="h-full flex items-center justify-center p-6 text-center text-gray-500">
                <p>No game active.</p>
            </div>
        );
    }

    const gameHistory = gameState.game_history || [];
    const totalTurns = gameHistory.length;
    const winner: string = (gameState as any).winner ?? '';

    // Board state derived from slider
    const activeTurnIdx = Math.max(0, Math.min(totalTurns - 1, (currentSliderTurn || totalTurns) - 1));
    const activeTurnData = totalTurns > 0 ? gameHistory[activeTurnIdx] : null;

    // Telemetry data
    const movesWithTelemetry = useMemo(() =>
        gameHistory
            .map((entry: any, index: number) => ({ ...entry, originalIndex: index }))
            .filter((entry: any) => entry.telemetry),
        [gameHistory]
    );

    const allTelemetry: MoveTelemetryDelta[] = useMemo(
        () => movesWithTelemetry.map((m: any) => m.telemetry),
        [movesWithTelemetry]
    );

    const actualPly = currentSliderTurn ?? (totalTurns > 0 ? totalTurns : 0);

    // Find the nearest telemetry move at-or-before the current ply
    const safeIndex = useMemo(() => {
        if (!movesWithTelemetry.length) return -1;
        // Prefer exact match
        const exact = movesWithTelemetry.findIndex((m: any) => m.telemetry.ply === actualPly);
        if (exact >= 0) return exact;
        // Fall back to last entry whose ply <= actualPly
        for (let i = movesWithTelemetry.length - 1; i >= 0; i--) {
            if (movesWithTelemetry[i].telemetry.ply <= actualPly) return i;
        }
        return 0;
    }, [movesWithTelemetry, actualPly]);

    const selectedMove = movesWithTelemetry[safeIndex];

    const handlePrev = () => { if (safeIndex > 0) setCurrentSliderTurn(movesWithTelemetry[safeIndex - 1].telemetry.ply); };
    const handleNext = () => { if (safeIndex < movesWithTelemetry.length - 1) setCurrentSliderTurn(movesWithTelemetry[safeIndex + 1].telemetry.ply); };

    const allPlayers = useMemo(() => Array.from(new Set(allTelemetry.map(m => m.moverId))), [allTelemetry]);

    // Board overlay effect
    useEffect(() => {
        if (!showOverlay || !selectedMove) { setBoardOverlay(null); return; }
        const moveIdx = selectedMove.originalIndex;
        const current = gameHistory[moveIdx];
        const prev = moveIdx > 0 ? gameHistory[moveIdx - 1] : null;
        const boardAfter: number[][] = current?.board_state;
        const boardBefore: number[][] = prev?.board_state ?? Array(20).fill(0).map(() => Array(20).fill(0));
        if (!boardAfter || !boardBefore) { setBoardOverlay(null); return; }
        const overlay: Record<string, { color: string; opacity: number }> = {};
        for (let r = 0; r < boardAfter.length; r++) {
            for (let c = 0; c < (boardAfter[r]?.length ?? 0); c++) {
                if ((boardAfter[r]?.[c] ?? 0) !== (boardBefore[r]?.[c] ?? 0) && (boardAfter[r]?.[c] ?? 0) !== 0) {
                    overlay[`${r}-${c}`] = { color: '#ffffff', opacity: 0.5 };
                }
            }
        }
        setBoardOverlay(Object.keys(overlay).length > 0 ? overlay : null);
        return () => setBoardOverlay(null);
    }, [showOverlay, selectedMove, gameHistory, setBoardOverlay]);

    const plyLabel = activeTurnData?.player_to_move
        ? `${activeTurnData.player_to_move} · Move ${currentSliderTurn || totalTurns}/${totalTurns}`
        : `Move ${currentSliderTurn || totalTurns}/${totalTurns}`;

    return (
        <div className="h-full overflow-y-auto custom-scrollbar p-4 space-y-4 bg-charcoal-900">

            {/* ─── GLOBAL MOVE SLIDER ─── */}
            <div className="bg-charcoal-800 border border-charcoal-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2 flex-wrap gap-1">
                    <span className="text-xs font-bold text-gray-300 uppercase tracking-widest">Timeline</span>
                    <div className="flex items-center gap-2">
                        <div className="flex bg-charcoal-900 p-0.5 rounded border border-charcoal-600">
                            <button
                                onClick={() => setXAxisMode('move')}
                                className={`px-2 py-0.5 text-[10px] font-bold rounded transition-colors ${xAxisMode === 'move' ? 'bg-blue-600 text-white' : 'text-slate-500 hover:text-slate-300'}`}
                            >MOVE</button>
                            <button
                                onClick={() => setXAxisMode('round')}
                                className={`px-2 py-0.5 text-[10px] font-bold rounded transition-colors ${xAxisMode === 'round' ? 'bg-blue-600 text-white' : 'text-slate-500 hover:text-slate-300'}`}
                            >ROUND</button>
                        </div>
                        <span className="text-xs font-mono bg-blue-900/50 text-blue-400 px-2 py-0.5 rounded">{plyLabel}</span>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    <span className="text-[10px] font-mono text-slate-500">1</span>
                    <input
                        type="range"
                        min={1}
                        max={totalTurns || 1}
                        value={currentSliderTurn || totalTurns}
                        onChange={(e) => setCurrentSliderTurn(parseInt(e.target.value, 10))}
                        className="flex-1 accent-neon-blue h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <span className="text-[10px] font-mono text-slate-500">{totalTurns}</span>
                </div>
            </div>

            {/* ─── LINE CHARTS ─── */}
            <Collapsible title="Game Trajectory" defaultOpen={true}>
                <div className="p-3 space-y-3">
                    <div className="h-[180px]">
                        <ModuleE_FrontierChart gameHistory={gameHistory} currentTurn={currentSliderTurn || totalTurns} mode={xAxisMode} />
                    </div>
                    <div className="h-[180px]">
                        <ModuleF_UrgencyChart gameHistory={gameHistory} currentTurn={currentSliderTurn || totalTurns} mode={xAxisMode} />
                    </div>
                </div>
            </Collapsible>

            {/* ─── MOVE IMPACT ─── */}
            {movesWithTelemetry.length === 0 ? (
                <div className="bg-charcoal-800 border border-charcoal-700 rounded-lg p-6 text-center text-gray-500 text-sm">
                    Play some moves to see Move Impact telemetry.
                </div>
            ) : (
                <>
                    {/* Move navigator */}
                    <div className="bg-charcoal-800 border border-charcoal-700 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
                            <span className="text-xs font-bold text-gray-300 uppercase tracking-widest">Move Impact</span>
                            <div className="flex items-center gap-1.5 flex-wrap">
                                <select
                                    value={preset}
                                    onChange={(e) => setPreset(e.target.value as WeightPreset)}
                                    className="px-2 py-0.5 text-xs rounded bg-charcoal-700 text-gray-300 border border-charcoal-600 cursor-pointer"
                                >
                                    {PRESETS.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
                                </select>
                                <button
                                    onClick={() => setShowAdvantage(!showAdvantage)}
                                    className={`px-2 py-0.5 text-xs rounded-full transition-colors ${showAdvantage ? 'bg-purple-500 text-white font-bold' : 'bg-charcoal-700 text-gray-300 hover:bg-charcoal-600'}`}
                                >
                                    {showAdvantage ? 'V2 Adv' : 'Raw Δ'}
                                </button>
                                {!showAdvantage && (
                                    <button
                                        onClick={() => setPerOpponent(!perOpponent)}
                                        className={`px-2 py-0.5 text-xs rounded-full transition-colors ${perOpponent ? 'bg-yellow-500 text-charcoal-900 font-bold' : 'bg-charcoal-700 text-gray-300 hover:bg-charcoal-600'}`}
                                    >
                                        {perOpponent ? 'Per Opp' : 'Agg Opp'}
                                    </button>
                                )}
                                <button
                                    onClick={() => setShowOverlay(!showOverlay)}
                                    className={`px-2 py-0.5 text-xs rounded-full transition-colors ${showOverlay ? 'bg-emerald-500 text-white font-bold' : 'bg-charcoal-700 text-gray-300 hover:bg-charcoal-600'}`}
                                    title="Highlight cells changed by this move on the board"
                                >🗺 Overlay</button>
                            </div>
                        </div>

                        {/* Prev / Next navigator — slides via the top Timeline slider */}
                        <div className="flex items-center justify-between">
                            <button onClick={handlePrev} disabled={safeIndex <= 0} className="px-3 py-1 text-xs bg-charcoal-700 rounded hover:bg-charcoal-600 disabled:opacity-30 text-gray-200">← Prev</button>
                            <div className="text-center">
                                <div className="text-sm font-bold" style={{ color: PLAYER_COLORS[selectedMove?.player_to_move] || '#9ca3af' }}>
                                    {selectedMove?.player_to_move}
                                </div>
                                <div className="text-xs text-gray-400">
                                    Ply {selectedMove?.telemetry?.ply}{selectedMove?.action ? ` · Piece ${selectedMove.action.piece_id}` : ' · Pass'}
                                    <span className="ml-1 text-gray-500">({safeIndex + 1}/{movesWithTelemetry.length})</span>
                                </div>
                            </div>
                            <button onClick={handleNext} disabled={safeIndex >= movesWithTelemetry.length - 1} className="px-3 py-1 text-xs bg-charcoal-700 rounded hover:bg-charcoal-600 disabled:opacity-30 text-gray-200">Next →</button>
                        </div>
                    </div>

                    {/* Charts for selected move */}
                    {selectedMove?.telemetry && (
                        <div className="space-y-3">
                            {/* Diverging / Advantage bar */}
                            <div className="bg-charcoal-800 border border-charcoal-700 rounded-lg p-3 h-[280px]">
                                <DivergingBarChart
                                    telemetry={selectedMove.telemetry}
                                    showRaw={showRaw}
                                    perOpponent={perOpponent}
                                    isAdvantageMode={showAdvantage}
                                    advantageKeys={['winProxy', 'mobilityNextP10Adv', 'effectiveFrontierAdv', 'lockedAreaAdv', 'remainingAreaAdv']}
                                />
                            </div>

                            {/* Waterfall */}
                            <div className="bg-charcoal-800 border border-charcoal-700 rounded-lg p-3 h-[260px]">
                                <MoveImpactWaterfall
                                    telemetry={selectedMove.telemetry}
                                    preset={preset}
                                    normalization={normalization}
                                    allMoves={allTelemetry}
                                />
                            </div>

                            {/* Move Shape radar — full width */}
                            <div className="bg-charcoal-800 border border-charcoal-700 rounded-lg p-3 h-[280px]">
                                <RadarDeltaChart telemetry={selectedMove.telemetry} showOpponents={!perOpponent} />
                            </div>

                            {/* Cumulative Move Impact — full width */}
                            <div className="bg-charcoal-800 border border-charcoal-700 rounded-lg p-3 h-[240px]">
                                <CumulativeTimelineChart gameHistory={gameHistory} currentPly={selectedMove.telemetry.ply} />
                            </div>

                            {/* Top Moves — collapsed by default */}
                            <Collapsible title="Top Moves Leaderboard" defaultOpen={false}>
                                <div className="p-3">
                                    <TopMovesLeaderboard
                                        allMoves={allTelemetry}
                                        preset={preset}
                                        normalization={normalization}
                                        winnerId={winner}
                                        selectedPly={selectedMove.telemetry.ply}
                                        onSelectPly={setCurrentSliderTurn}
                                    />
                                </div>
                            </Collapsible>

                            {/* Opponent Suppression — with player selector */}
                            <Collapsible title="Opponent Suppression" defaultOpen={false}>
                                <div className="p-3 space-y-2">
                                    <div className="flex items-center gap-2 flex-wrap">
                                        <span className="text-xs text-gray-400 shrink-0">View from:</span>
                                        {allPlayers.map(pid => (
                                            <button
                                                key={pid}
                                                onClick={() => setSuppressionPlayer(pid)}
                                                className={`px-2 py-0.5 text-xs rounded-full transition-colors font-bold ${(suppressionPlayer || winner || allTelemetry[0]?.moverId) === pid
                                                    ? 'ring-2 ring-white scale-105'
                                                    : 'opacity-60 hover:opacity-100'}`}
                                                style={{ backgroundColor: PLAYER_COLORS[pid] + '33', color: PLAYER_COLORS[pid], border: `1px solid ${PLAYER_COLORS[pid]}` }}
                                            >
                                                {pid}
                                            </button>
                                        ))}
                                    </div>
                                    <OpponentSuppressionMultiples
                                        allMoves={allTelemetry}
                                        currentPly={selectedMove.telemetry.ply}
                                        moverId={suppressionPlayer || winner || selectedMove.telemetry.moverId}
                                    />
                                </div>
                            </Collapsible>
                        </div>
                    )}
                </>
            )}
        </div>
    );
};
