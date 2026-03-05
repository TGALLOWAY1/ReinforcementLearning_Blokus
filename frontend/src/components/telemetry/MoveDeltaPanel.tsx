import React, { useState, useMemo } from 'react';
import { useGameStore } from '../../store/gameStore';
import { DivergingBarChart } from './charts/DivergingBarChart';
import { RadarDeltaChart } from './charts/RadarDeltaChart';
import { CumulativeTimelineChart } from './charts/CumulativeTimelineChart';
import { MoveImpactWaterfall } from './charts/MoveImpactWaterfall';
import { TopMovesLeaderboard } from './charts/TopMovesLeaderboard';
import { WeightPreset, NormalizationMethod } from '../../utils/moveImpactScore';
import { MoveTelemetryDelta } from '../../types/telemetry';

const PRESETS: { label: string; value: WeightPreset }[] = [
    { label: 'Balanced', value: 'balanced' },
    { label: 'Aggressive Blocking', value: 'blocking' },
    { label: 'Expansion', value: 'expansion' },
    { label: 'Late-game', value: 'late-game' },
];

export const MoveDeltaPanel: React.FC = () => {
    const gameState = useGameStore((s) => s.gameState);

    // UI State
    const [selectedPly, setSelectedPly] = useState<number>(0);
    const [showRaw, setShowRaw] = useState<boolean>(false);
    const [perOpponent, setPerOpponent] = useState<boolean>(false);
    const [preset, setPreset] = useState<WeightPreset>('balanced');
    const [normalization] = useState<NormalizationMethod>('z-score');

    if (!gameState) {
        return (
            <div className="h-full flex items-center justify-center p-6 text-center text-gray-500">
                <p>No game active.</p>
            </div>
        );
    }

    const gameHistory = gameState.game_history || [];
    const winner = (gameState as any).winner;

    const movesWithTelemetry = useMemo(() => {
        return gameHistory
            .map((entry: any, index: number) => ({ ...entry, originalIndex: index }))
            .filter((entry: any) => entry.telemetry);
    }, [gameHistory]);

    // Flat array of just the telemetry objects for scoring computations
    const allTelemetry: MoveTelemetryDelta[] = useMemo(
        () => movesWithTelemetry.map((m: any) => m.telemetry),
        [movesWithTelemetry]
    );

    if (movesWithTelemetry.length === 0) {
        return (
            <div className="h-full flex items-center justify-center p-6 text-center text-gray-500">
                <p>Play some moves to see Move Delta Telemetry.</p>
            </div>
        );
    }

    // Default to the latest move
    const actualPly = selectedPly === 0 && movesWithTelemetry.length > 0
        ? movesWithTelemetry[movesWithTelemetry.length - 1].telemetry.ply
        : selectedPly;

    const currentIndex = movesWithTelemetry.findIndex((m: any) => m.telemetry.ply === actualPly);
    const safeIndex = currentIndex >= 0 ? currentIndex : movesWithTelemetry.length - 1;
    const selectedMove = movesWithTelemetry[safeIndex];

    const handlePrev = () => {
        if (safeIndex > 0) setSelectedPly(movesWithTelemetry[safeIndex - 1].telemetry.ply);
    };
    const handleNext = () => {
        if (safeIndex < movesWithTelemetry.length - 1) setSelectedPly(movesWithTelemetry[safeIndex + 1].telemetry.ply);
    };
    const handleSlider = (e: React.ChangeEvent<HTMLInputElement>) => {
        setSelectedPly(movesWithTelemetry[parseInt(e.target.value, 10)].telemetry.ply);
    };

    return (
        <div className="h-full flex flex-col p-4 space-y-4 overflow-y-auto">
            {/* Header + Controls */}
            <div className="flex items-center justify-between border-b border-charcoal-700 pb-3 flex-wrap gap-2">
                <h2 className="text-xl font-bold text-gray-200">Move Impact Delta</h2>
                <div className="flex items-center gap-2 flex-wrap">
                    <select
                        value={preset}
                        onChange={(e) => setPreset(e.target.value as WeightPreset)}
                        className="px-2 py-1 text-xs rounded bg-charcoal-700 text-gray-300 border border-charcoal-600 cursor-pointer"
                    >
                        {PRESETS.map(p => (
                            <option key={p.value} value={p.value}>{p.label}</option>
                        ))}
                    </select>
                    <button
                        onClick={() => setShowRaw(!showRaw)}
                        className={`px-3 py-1 text-xs rounded-full transition-colors ${showRaw ? 'bg-blue-500 text-white font-bold' : 'bg-charcoal-700 text-gray-300 hover:bg-charcoal-600'}`}
                    >
                        {showRaw ? 'Raw' : 'Normalized'}
                    </button>
                    <button
                        onClick={() => setPerOpponent(!perOpponent)}
                        className={`px-3 py-1 text-xs rounded-full transition-colors ${perOpponent ? 'bg-yellow-500 text-charcoal-900 font-bold' : 'bg-charcoal-700 text-gray-300 hover:bg-charcoal-600'}`}
                    >
                        {perOpponent ? 'Per Opponent' : 'Agg. Opp'}
                    </button>
                </div>
            </div>

            {/* Move Selector */}
            <div className="bg-charcoal-800 rounded-lg p-3 border border-charcoal-700">
                <div className="flex items-center justify-between mb-3">
                    <button
                        onClick={handlePrev}
                        disabled={safeIndex === 0}
                        className="p-1 px-3 bg-charcoal-700 rounded hover:bg-charcoal-600 disabled:opacity-30 text-gray-200"
                    >
                        &larr; Prev
                    </button>
                    <div className="text-center">
                        <div className="text-sm font-bold text-blue-400">
                            {selectedMove?.player_to_move}'s Move
                        </div>
                        <div className="text-xs text-gray-400">
                            Ply {selectedMove?.telemetry?.ply}
                            {selectedMove?.action ? ` · Piece ${selectedMove.action.piece_id}` : ' · Pass'}
                        </div>
                    </div>
                    <button
                        onClick={handleNext}
                        disabled={safeIndex === movesWithTelemetry.length - 1}
                        className="p-1 px-3 bg-charcoal-700 rounded hover:bg-charcoal-600 disabled:opacity-30 text-gray-200"
                    >
                        Next &rarr;
                    </button>
                </div>
                <input
                    type="range"
                    min="0"
                    max={movesWithTelemetry.length - 1}
                    value={safeIndex}
                    onChange={handleSlider}
                    className="w-full accent-blue-500 cursor-pointer"
                />
            </div>

            {/* Charts */}
            {selectedMove?.telemetry && (
                <div className="space-y-4">
                    <div className="bg-charcoal-800 rounded-lg border border-charcoal-700 p-3 h-[280px]">
                        <DivergingBarChart
                            telemetry={selectedMove.telemetry}
                            showRaw={showRaw}
                            perOpponent={perOpponent}
                        />
                    </div>

                    <div className="bg-charcoal-800 rounded-lg border border-charcoal-700 p-3 h-[280px]">
                        <MoveImpactWaterfall
                            telemetry={selectedMove.telemetry}
                            preset={preset}
                            normalization={normalization}
                            allMoves={allTelemetry}
                        />
                    </div>

                    <div className="flex gap-3 h-[280px]">
                        <div className="flex-1 bg-charcoal-800 rounded-lg border border-charcoal-700 p-3">
                            <RadarDeltaChart
                                telemetry={selectedMove.telemetry}
                                showOpponents={!perOpponent}
                            />
                        </div>
                        <div className="flex-1 bg-charcoal-800 rounded-lg border border-charcoal-700 p-3">
                            <CumulativeTimelineChart
                                gameHistory={gameHistory}
                                currentPly={selectedMove.telemetry.ply}
                            />
                        </div>
                    </div>

                    <div className="bg-charcoal-800 rounded-lg border border-charcoal-700 p-3">
                        <TopMovesLeaderboard
                            allMoves={allTelemetry}
                            preset={preset}
                            normalization={normalization}
                            winnerId={winner}
                            selectedPly={selectedMove.telemetry.ply}
                            onSelectPly={setSelectedPly}
                        />
                    </div>
                </div>
            )}
        </div>
    );
};
