import React, { useMemo } from 'react';
import { useGameStore } from '../store/gameStore';
import { calculateDashboardMetrics, calculateWinProbability } from '../utils/dashboardMetrics';
import { getPieceSize } from '../utils/mobilityMetrics';

const PLAYER_COLORS: Record<string, string> = {
    RED: '#ef4444',
    BLUE: '#3b82f6',
    GREEN: '#22c55e',
    YELLOW: '#eab308',
};

const PLAYER_NAMES_BY_ID: Record<number, string> = { 1: 'RED', 2: 'BLUE', 3: 'YELLOW', 4: 'GREEN' };


/**
 * Compact player leaderboard rendered at the bottom of the piece tray.
 * Reads currentSliderTurn from the store so it stays in sync with the
 * Analysis timeline slider.
 */
export const TrayLeaderboard: React.FC = () => {
    const gameState = useGameStore(s => s.gameState);
    const currentSliderTurn = useGameStore(s => s.currentSliderTurn);

    const gameHistory: any[] = gameState?.game_history || [];
    const totalTurns = gameHistory.length;

    const activeTurnIdx = Math.max(0, Math.min(totalTurns - 1, (currentSliderTurn || totalTurns) - 1));
    const activeTurnData = totalTurns > 0 ? gameHistory[activeTurnIdx] : null;
    const currentBoard: number[][] | null = activeTurnData?.board_state || gameState?.board || null;

    // Snapshot of pieces_used at the selected turn (fall back to live state)
    const piecesUsed: Record<string, number[]> = activeTurnData?.pieces_used || gameState?.pieces_used || {};

    const metrics = useMemo(() => {
        if (!currentBoard) return null;
        try { return calculateDashboardMetrics(currentBoard); } catch { return null; }
    }, [currentBoard]);

    const winProbs = useMemo(() => {
        if (!currentBoard || !metrics) return null;
        try { return calculateWinProbability(currentBoard, metrics); } catch { return null; }
    }, [currentBoard, metrics]);

    if (!gameState) return null;

    const players = [1, 2, 3, 4];
    const maxWinProb = winProbs ? Math.max(...(Object.values(winProbs) as number[])) : 0;

    return (
        <div className="border-t border-charcoal-700 bg-charcoal-900 shrink-0">
            <div className="px-3 py-1.5 flex items-center justify-between border-b border-charcoal-700/50">
                <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Standings</span>
                {currentSliderTurn && currentSliderTurn < totalTurns && (
                    <span className="text-[10px] font-mono text-blue-400 bg-blue-900/30 px-1.5 py-0.5 rounded">
                        Move {currentSliderTurn}
                    </span>
                )}
            </div>
            <table className="w-full text-center border-collapse text-[11px] font-mono">
                <thead>
                    <tr className="text-gray-500 text-[9px] uppercase tracking-wider border-b border-charcoal-700/50">
                        <th className="py-1.5 px-2 text-left">Player</th>
                        <th className="py-1.5 px-1" title="Score (squares placed)">Score</th>
                        <th className="py-1.5 px-1" title="Pieces remaining in hand">Left</th>
                        <th className="py-1.5 px-2 text-right" title="Win probability estimate">Win%</th>
                    </tr>
                </thead>
                <tbody>
                    {players.map(p => {
                        const pName = PLAYER_NAMES_BY_ID[p];
                        const used: number[] = piecesUsed[pName] || [];
                        const score = used.reduce((s, id) => s + getPieceSize(id), 0);
                        const remaining = 21 - used.length;
                        const winProb = winProbs ? (winProbs[p] * 100) : null;
                        const isLeader = winProbs ? winProbs[p] === maxWinProb : false;
                        const color = PLAYER_COLORS[pName];

                        return (
                            <tr
                                key={p}
                                className={`border-b border-charcoal-700/30 ${isLeader ? 'bg-white/5' : ''}`}
                            >
                                <td className="py-1.5 px-2 text-left font-bold flex items-center gap-1" style={{ color }}>
                                    {isLeader && <span className="text-yellow-400 text-[9px]">★</span>}
                                    {pName}
                                </td>
                                <td className="py-1.5 px-1 text-white font-bold">{score}</td>
                                <td className="py-1.5 px-1 text-slate-400">{remaining}</td>
                                <td className={`py-1.5 px-2 text-right font-bold ${isLeader ? 'text-green-400' : 'text-slate-500'}`}>
                                    {winProb !== null ? `${winProb.toFixed(0)}%` : '—'}
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
};
