import React, { useState, useMemo } from 'react';
import { useGameStore } from '../store/gameStore';
import { TerritoryControlPlot } from './TerritoryControlPlot';
import { PiecePenaltyPlot } from './PiecePenaltyPlot';
import { CenterProximityPlot } from './CenterProximityPlot';
import { OpponentAdjacencyPlot } from './OpponentAdjacencyPlot';
import { CornerDiffPlot } from './CornerDiffPlot';
import { FrontierSizePlot } from './FrontierSizePlot';

function formatTs(): string {
    const d = new Date();
    return d.toTimeString().slice(0, 12);
}

interface AnalysisBoardProps {
    onClose: () => void;
}

export const AnalysisBoard: React.FC<AnalysisBoardProps> = ({ onClose }) => {
    const gameState = useGameStore((s) => s.gameState);
    const legalMovesHistory = useGameStore((s) => s.legalMovesHistory);
    const [frozen, setFrozen] = useState(true); // Freeze by default in analysis view

    // Create a memoized snapshot if frozen
    const snapshot = useMemo(() => {
        if (!frozen) return null;
        return {
            gameState: gameState,
            legalMovesHistory: [...legalMovesHistory],
            timestamp: formatTs(),
        };
    }, [frozen, gameState, legalMovesHistory]);

    const effectiveHistory = frozen && snapshot ? snapshot.legalMovesHistory : legalMovesHistory;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
            <div className="flex flex-col w-full max-w-6xl max-h-[90vh] bg-charcoal-900 border border-charcoal-700 rounded-xl shadow-2xl overflow-hidden">

                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-charcoal-700 bg-charcoal-800/80">
                    <div>
                        <h2 className="text-lg font-bold text-white tracking-wide">Post-Game Analysis Board</h2>
                        <p className="text-xs text-gray-400 mt-0.5">Comprehensive timeline of predictive metrics</p>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2">
                            <span className="text-xs text-gray-500">Mode:</span>
                            <button
                                onClick={() => setFrozen(!frozen)}
                                className={`px-3 py-1 text-xs font-semibold rounded ${frozen ? 'bg-neon-yellow text-black' : 'bg-charcoal-700 text-gray-300'}`}
                            >
                                {frozen ? 'Frozen (Snapshot)' : 'Live Update'}
                            </button>
                        </div>
                        <button
                            onClick={onClose}
                            className="p-2 text-gray-400 hover:text-white hover:bg-charcoal-700 rounded-lg transition-colors"
                            aria-label="Close"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                    </div>
                </div>

                {/* Dashboard Grid */}
                <div className="flex-1 overflow-y-auto p-4 bg-charcoal-900">

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

                        {/* Primary Mobility */}
                        <div className="bg-charcoal-800/50 border border-charcoal-700 rounded-lg shadow-sm">
                            <CornerDiffPlot overrideHistory={effectiveHistory || []} />
                        </div>

                        {/* Territory */}
                        <div className="bg-charcoal-800/50 border border-charcoal-700 rounded-lg shadow-sm">
                            <TerritoryControlPlot overrideHistory={effectiveHistory || []} />
                        </div>

                        {/* Piece Penalty */}
                        <div className="bg-charcoal-800/50 border border-charcoal-700 rounded-lg shadow-sm">
                            <PiecePenaltyPlot overrideHistory={effectiveHistory || []} />
                        </div>

                        {/* Adjacency */}
                        <div className="bg-charcoal-800/50 border border-charcoal-700 rounded-lg shadow-sm">
                            <OpponentAdjacencyPlot overrideHistory={effectiveHistory || []} />
                        </div>

                        {/* Center proximity */}
                        <div className="bg-charcoal-800/50 border border-charcoal-700 rounded-lg shadow-sm">
                            <CenterProximityPlot overrideHistory={effectiveHistory || []} />
                        </div>

                        {/* Frontier baseline */}
                        <div className="bg-charcoal-800/50 border border-charcoal-700 rounded-lg shadow-sm">
                            <FrontierSizePlot overrideHistory={effectiveHistory || []} />
                        </div>

                    </div>

                </div>
            </div>
        </div>
    );
};
