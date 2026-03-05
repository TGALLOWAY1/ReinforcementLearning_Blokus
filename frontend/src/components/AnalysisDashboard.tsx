import React, { useEffect, useMemo, useState } from 'react';
import { useGameStore } from '../store/gameStore';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { calculateDashboardMetrics, calculateWinProbability, DashboardMetrics } from '../utils/dashboardMetrics';

import { PLAYER_COLORS as GAME_COLORS } from '../constants/gameConstants';

const PLAYER_COLORS: Record<number, string> = {
    1: GAME_COLORS.red,    // '#FF4D4D'
    2: GAME_COLORS.blue,   // '#00F0FF'
    3: GAME_COLORS.yellow, // '#FFE600'
    4: GAME_COLORS.green   // '#00FF9D'
};

const PLAYER_NAMES: Record<number, string> = {
    1: 'RED',
    2: 'BLUE',
    3: 'YELLOW',
    4: 'GREEN'
};

const PLAYER_KEYS: Record<string, number> = {
    'RED': 1,
    'BLUE': 2,
    'YELLOW': 3,
    'GREEN': 4
};

export const AnalysisDashboard: React.FC = () => {
    const gameState = useGameStore(s => s.gameState);
    const currentSliderTurn = useGameStore(s => s.currentSliderTurn);
    const setCurrentSliderTurn = useGameStore(s => s.setCurrentSliderTurn);
    const [selectedPlayer, setSelectedPlayer] = useState<number>(2); // Default BLUE
    const [xAxisMode, setXAxisMode] = useState<'move' | 'round'>('move');

    const gameHistory = gameState?.game_history || [];
    const totalTurns = gameHistory.length;

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (!totalTurns) return;
            if (e.key === 'ArrowLeft') {
                setCurrentSliderTurn(Math.max(1, (currentSliderTurn || totalTurns) - 1));
            } else if (e.key === 'ArrowRight') {
                setCurrentSliderTurn(Math.min(totalTurns, (currentSliderTurn || totalTurns) + 1));
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [currentSliderTurn, totalTurns, setCurrentSliderTurn]);

    const activeTurnData = useMemo(() => {
        if (totalTurns === 0) return null;
        const idx = Math.max(0, Math.min(totalTurns - 1, (currentSliderTurn || totalTurns) - 1));
        return gameHistory[idx];
    }, [gameHistory, currentSliderTurn, totalTurns]);

    const currentBoard = activeTurnData?.board_state || gameState?.board;

    const metrics = useMemo(() => {
        if (!currentBoard) return null;
        return calculateDashboardMetrics(currentBoard);
    }, [currentBoard]);

    const winProbs = useMemo(() => {
        if (!currentBoard || !metrics) return null;
        return calculateWinProbability(currentBoard, metrics);
    }, [currentBoard, metrics]);

    if (!gameState || !currentBoard || !metrics) {
        return <div className="p-4 text-gray-500">Connecting to Engine...</div>;
    }

    const currentPlayerStr = activeTurnData?.player_to_move || gameState.current_player;

    return (
        <div className="flex flex-col h-full bg-charcoal-900 text-gray-200 overflow-y-auto custom-scrollbar rounded-lg border border-charcoal-700">

            {/* Header / Slider */}
            <div className="shrink-0 p-4 border-b border-charcoal-700 bg-charcoal-800">
                <div className="flex items-center justify-between mb-2">
                    <h2 className="text-sm font-bold tracking-wider uppercase text-slate-300">Blokus Game Analysis & Prediction</h2>
                    <div className="flex items-center gap-4">
                        <div className="flex bg-charcoal-900 p-0.5 rounded border border-charcoal-600">
                            <button
                                onClick={() => setXAxisMode('move')}
                                className={`px-2 py-0.5 text-[10px] font-bold rounded transition-colors ${xAxisMode === 'move' ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}
                            >
                                MOVE
                            </button>
                            <button
                                onClick={() => setXAxisMode('round')}
                                className={`px-2 py-0.5 text-[10px] font-bold rounded transition-colors ${xAxisMode === 'round' ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}
                            >
                                ROUND
                            </button>
                        </div>
                        <span className="text-xs font-mono bg-blue-900/50 text-blue-400 px-2 py-1 rounded">Move {currentSliderTurn || totalTurns} / {totalTurns} ({currentPlayerStr} to move)</span>
                    </div>
                </div>
                <div className="flex items-center gap-4">
                    <span className="text-xs font-mono text-slate-500">1</span>
                    <input
                        type="range"
                        min={1}
                        max={totalTurns || 1}
                        value={currentSliderTurn || totalTurns}
                        onChange={(e) => setCurrentSliderTurn(parseInt(e.target.value, 10))}
                        className="flex-1 accent-neon-blue h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <span className="text-xs font-mono text-slate-500">{totalTurns}</span>
                </div>
            </div>

            {/* Overhauled Layout */}
            <div className="flex-1 p-3 flex flex-col gap-3 min-h-0 overflow-y-auto">

                {/* TOP ROW: Grids, Impact, Leaderboard */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 shrink-0">

                    {/* LEFT COLUMN: Spatial Visualizations (4) */}
                    <div className="lg:col-span-4 flex flex-col gap-3">
                        <div className="flex justify-between items-center bg-charcoal-800 border-l-[3px] border-neon-blue rounded-r px-3 py-1.5 shrink-0 shadow-sm">
                            <h3 className="text-xs font-bold text-gray-300 uppercase tracking-widest">Spatial Analysis</h3>
                            <div className="flex gap-1">
                                {[1, 2, 3, 4].map(p => (
                                    <button
                                        key={p}
                                        onClick={() => setSelectedPlayer(p)}
                                        className={`w-6 h-6 rounded flex items-center justify-center text-[10px] font-bold transition-all ${selectedPlayer === p ? 'ring-2 ring-white scale-110' : 'opacity-50 hover:opacity-100'}`}
                                        style={{ backgroundColor: PLAYER_COLORS[p] }}
                                    >
                                        {PLAYER_NAMES[p][0]}
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div className="bg-charcoal-800 border border-charcoal-700 rounded-lg p-2 flex flex-col h-[260px] hover:border-gray-600 transition-colors">
                            <h3 className="text-[10px] font-bold text-gray-400 uppercase text-center mb-2 tracking-wider">Frontier (Usable Corners)</h3>
                            <FrontierMap
                                frontiers={metrics.frontiers}
                                boardState={currentBoard}
                                selectedPlayer={selectedPlayer}
                                frontierMetrics={(activeTurnData?.metrics?.frontier_metrics?.[PLAYER_NAMES[selectedPlayer]])
                                    || gameState?.frontier_metrics?.[PLAYER_NAMES[selectedPlayer]]}
                                frontierClusters={(activeTurnData?.metrics?.frontier_clusters?.[PLAYER_NAMES[selectedPlayer]])
                                    || gameState?.frontier_clusters?.[PLAYER_NAMES[selectedPlayer]]}
                            />
                        </div>

                        <div className="bg-charcoal-800 border border-charcoal-700 rounded-lg p-2 flex flex-col h-[260px] hover:border-gray-600 transition-colors">
                            <h3 className="text-[10px] font-bold text-gray-400 uppercase text-center mb-2 tracking-wider">Endgame Dead-Zones</h3>
                            <DeadZoneMap deadZones={metrics.deadZones} boardState={currentBoard} selectedPlayer={selectedPlayer} />
                        </div>
                    </div>

                    {/* CENTER COLUMN: Move Impact Analysis (4) */}
                    <div className="lg:col-span-4 flex flex-col gap-3">
                        <MoveImpactAnalysis gameHistory={gameHistory} currentTurn={currentSliderTurn || totalTurns} />
                    </div>

                    {/* RIGHT COLUMN: Player Leaderboard (4) */}
                    <div className="lg:col-span-4 flex flex-col gap-3">
                        <PlayerLeaderboard gameState={gameState} metrics={metrics} winProbs={winProbs} remainingPieces={activeTurnData?.metrics?.remaining_pieces} />
                    </div>
                </div>

                {/* BOTTOM ROW: Tabbed Charts */}
                <div className="shrink-0 h-[300px] mb-4">
                    <TabbedCharts gameHistory={gameHistory} currentSliderTurn={currentSliderTurn || totalTurns} xAxisMode={xAxisMode} />
                </div>
            </div>
        </div>
    );
};


// --- Subcomponents for the dashboard ---

const FRONTIER_MODE_INFO = {
    Default: { label: 'Default', title: 'Show frontier (usable corner) points in your player color' },
    Flexibility: { label: 'Flex', title: 'Flexibility: how many legal moves pass through this corner. High = many placement options.' },
    Urgency: { label: 'Urgency', title: 'Urgency: Flexibility × Threat. Points with NO opponent pressure are grey (can wait).' },
    Cluster: { label: 'Cluster', title: 'Color by redundancy cluster — corners sharing similar move-sets are grouped by color.' },
};

const FrontierMap: React.FC<{
    frontiers: Record<number, { r: number, c: number }[]>,
    boardState: number[][],
    selectedPlayer: number,
    frontierMetrics?: any,
    frontierClusters?: any
}> = ({ frontiers, boardState, selectedPlayer, frontierMetrics, frontierClusters }) => {
    const [colorMode, setColorMode] = useState<'Default' | 'Flexibility' | 'Urgency' | 'Cluster'>('Default');
    const size = boardState.length;
    // Build a quick lookup for selected player's frontier
    const fMap = Array(size).fill(0).map(() => Array(size).fill(false));
    const cells = frontiers[selectedPlayer] || [];
    for (const { r, c } of cells) {
        fMap[r][c] = true;
    }

    const hasMetrics = !!(frontierMetrics?.urgency && Object.keys(frontierMetrics.urgency).length > 0);

    return (
        <div className="flex-1 flex flex-col min-h-0 relative">
            {/* Mode toggle buttons */}
            <div className="absolute top-[-26px] right-0 flex gap-1 z-10">
                {(['Default', 'Flexibility', 'Urgency', 'Cluster'] as const).map(mode => (
                    <button
                        key={mode}
                        onClick={() => setColorMode(mode)}
                        title={FRONTIER_MODE_INFO[mode].title}
                        className={`text-[9px] px-1.5 py-0.5 rounded transition-colors ${colorMode === mode ? 'bg-slate-500 text-white' : 'bg-charcoal-700 text-gray-400 hover:text-gray-200'}`}
                    >
                        {FRONTIER_MODE_INFO[mode].label}
                    </button>
                ))}
            </div>
            <div className="flex-1 flex items-center justify-center min-h-0 bg-slate-900 rounded p-[1px]">
                <div className="grid gap-[1px] w-full max-w-full max-h-full aspect-square" style={{ gridTemplateColumns: `repeat(${size}, minmax(0, 1fr))`, gridTemplateRows: `repeat(${size}, minmax(0, 1fr))` }}>
                    {boardState.map((row, r) => row.map((val, c) => {
                        let bg = 'bg-slate-800';
                        let isFrontier = fMap[r][c];
                        let title = undefined;

                        if (val > 0) {
                            bg = ['', 'bg-red-500', 'bg-blue-500', 'bg-yellow-400', 'bg-green-500'][val];
                            if (val !== selectedPlayer) bg += ' opacity-40';
                        } else if (isFrontier) {
                            const key = `${r},${c}`;
                            let flex = 0;
                            let bp = 0;
                            let urg = 0;
                            let cid = -1;

                            if (frontierMetrics || frontierClusters) {
                                flex = frontierMetrics?.utility?.[key] || 0;
                                bp = frontierMetrics?.block_pressure?.[key] || 0;
                                urg = frontierMetrics?.urgency?.[key] || 0;
                                cid = frontierClusters?.cluster_id?.[key] ?? -1;

                                if (flex !== undefined) {
                                    title = `[${r}, ${c}]\nFlexibility (Utility): ${flex}\nBlock Pressure: ${bp}\nUrgency: ${urg}\nCluster ID: ${cid}`;
                                }
                            }

                            if (colorMode === 'Default') {
                                bg = ['', 'bg-red-400', 'bg-blue-400', 'bg-yellow-300', 'bg-green-400'][selectedPlayer] + ' shadow-[0_0_8px_currentColor]';
                            } else if (colorMode === 'Flexibility') {
                                if (flex >= 30) bg = 'bg-red-500 shadow-[0_0_8px_currentColor]';
                                else if (flex >= 15) bg = 'bg-orange-400 shadow-[0_0_4px_currentColor]';
                                else if (flex >= 5) bg = 'bg-yellow-500';
                                else bg = 'bg-slate-500';
                            } else if (colorMode === 'Urgency') {
                                if (urg >= 10) bg = 'bg-red-500 shadow-[0_0_8px_currentColor]';
                                else if (urg >= 5) bg = 'bg-orange-400 shadow-[0_0_4px_currentColor]';
                                else if (urg >= 1) bg = 'bg-yellow-500';
                                else bg = 'bg-slate-500';
                            } else if (colorMode === 'Cluster') {
                                const clusterColors = ['bg-pink-500', 'bg-cyan-400', 'bg-purple-500', 'bg-amber-400', 'bg-lime-400', 'bg-indigo-400'];
                                if (cid === -1) {
                                    bg = 'bg-slate-600';
                                } else {
                                    bg = clusterColors[cid % clusterColors.length] + ' shadow-[0_0_6px_currentColor]';
                                }
                            }
                        }
                        return <div key={`${r}-${c}`} className={`${bg} ${isFrontier ? 'rounded-full scale-75 cursor-help' : 'rounded-sm'} transition-colors`} title={title} />;
                    }))}
                </div>
            </div>
            {/* Mode legend */}
            {colorMode !== 'Default' && (
                <div className="mt-1 text-[8px] text-slate-500 leading-tight text-center">
                    {!hasMetrics && (
                        <span className="text-amber-600">Metric data missing — play a move to populate</span>
                    )}
                    {hasMetrics && (
                        <>
                            {colorMode === 'Flexibility' && (
                                <span><span className="text-red-400">■</span> High (&ge;30) &nbsp;<span className="text-orange-400">■</span> Med (&ge;15) &nbsp;<span className="text-yellow-500">■</span> Low (&ge;5) &nbsp;<span className="text-slate-500">■</span> None</span>
                            )}
                            {colorMode === 'Urgency' && (
                                <span><span className="text-red-400">■</span> Critical &nbsp;<span className="text-orange-400">■</span> Contested &nbsp;<span className="text-yellow-500">■</span> Low &nbsp;<span className="text-slate-500">■</span> Safe (Can Wait)</span>
                            )}
                            {colorMode === 'Cluster' && (
                                <span>Each color = a redundancy cluster &nbsp;<span className="text-slate-500">■</span> = isolated / no cluster</span>
                            )}
                        </>
                    )}
                </div>
            )}
        </div>
    );
};


const DeadZoneMap: React.FC<{ deadZones: Record<number, boolean[][]>, boardState: number[][], selectedPlayer: number }> = ({ deadZones, boardState, selectedPlayer }) => {
    const size = boardState.length;
    const dz = deadZones[selectedPlayer];

    return (
        <div className="flex-1 flex flex-col min-h-0">
            <div className="flex-1 flex items-center justify-center min-h-0 bg-slate-900 rounded p-[1px]">
                <div className="grid gap-[1px] w-full max-w-full max-h-full aspect-square" style={{ gridTemplateColumns: `repeat(${size}, minmax(0, 1fr))`, gridTemplateRows: `repeat(${size}, minmax(0, 1fr))` }}>
                    {boardState.map((row, r) => row.map((val, c) => {
                        let bg = 'bg-slate-800';
                        if (val > 0) {
                            bg = ['', 'bg-red-500', 'bg-blue-500', 'bg-yellow-400', 'bg-green-500'][val];
                            // fade out opponents slightly to focus on deadzones
                            if (val !== selectedPlayer) bg += ' opacity-40';
                        } else if (dz && dz[r][c]) {
                            // Deadzone hatching effect using CSS
                            bg = 'bg-slate-950 relative overflow-hidden';
                        }

                        return (
                            <div key={`${r}-${c}`} className={`${bg} rounded-sm`}>
                                {(val === 0 && dz && dz[r][c]) && (
                                    <svg className="absolute inset-0 w-full h-full opacity-30 pointer-events-none" viewBox="0 0 10 10">
                                        <line x1="0" y1="10" x2="10" y2="0" stroke="#EF4444" strokeWidth="1" />
                                    </svg>
                                )}
                            </div>
                        )
                    }))}
                </div>
            </div>
        </div>
    );
};

const getPieceSize = (id: number) => {
    if (!id) return 0;
    if (id === 1) return 1;
    if (id === 2) return 2;
    if (id <= 4) return 3;
    if (id <= 9) return 4;
    return 5;
};

const MoveImpactAnalysis: React.FC<{
    gameHistory: any[],
    currentTurn: number
}> = ({ gameHistory, currentTurn }) => {
    const idx = Math.max(0, currentTurn - 1);
    const currEntry = gameHistory[idx];
    const prevEntry = idx > 0 ? gameHistory[idx - 1] : null;

    if (!currEntry) {
        return (
            <div className="flex flex-col h-full bg-charcoal-800 border border-charcoal-700 rounded-lg p-4 justify-center items-center text-slate-500 text-xs hover:border-gray-600 transition-colors">
                Waiting for moves...
            </div>
        );
    }

    const player = currEntry.player_to_move;
    const pieceId = currEntry.action?.piece_id;
    const pieceSize = getPieceSize(pieceId);

    // Compute Deltas
    let mobilityDeltaStr = '-';
    let blockStr = '-';
    let isPositiveMobility = false;
    let isNegativeMobility = false;

    if (prevEntry) {
        const currMobility = currEntry.metrics?.frontier_size?.[player] || 0;
        const prevMobility = prevEntry.metrics?.frontier_size?.[player] || 0;
        const diff = currMobility - prevMobility;
        if (diff > 0) {
            mobilityDeltaStr = `+${diff} Corners`;
            isPositiveMobility = true;
        } else if (diff < 0) {
            mobilityDeltaStr = `${diff} Corners`;
            isNegativeMobility = true;
        } else {
            mobilityDeltaStr = `No Change`;
        }

        let totalBlocked = 0;
        for (const p of ['RED', 'BLUE', 'YELLOW', 'GREEN']) {
            if (p !== player) {
                const c = currEntry.metrics?.frontier_size?.[p] || 0;
                const pr = prevEntry.metrics?.frontier_size?.[p] || 0;
                if (pr - c > 0) {
                    totalBlocked += (pr - c);
                }
            }
        }
        if (totalBlocked > 0) {
            blockStr = `Blocked ${totalBlocked} Opponent Corners`;
        } else {
            blockStr = `No Opponents Blocked`;
        }
    }

    return (
        <div className="flex flex-col h-full bg-charcoal-800 border border-charcoal-700 rounded-lg overflow-hidden hover:border-gray-600 transition-colors">
            <h3 className="text-[10px] font-bold text-gray-300 uppercase tracking-widest bg-charcoal-800 border-b border-charcoal-700 px-3 py-2 shrink-0 flex items-center gap-2">
                <svg className="w-3 h-3 text-neon-blue" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Move Impact (Turn {currentTurn})
            </h3>

            <div className="p-4 flex flex-col gap-5 flex-1 overflow-y-auto custom-scrollbar">
                <div>
                    <span className="text-[10px] text-gray-500 uppercase tracking-wider block mb-1 font-bold">Action Taken</span>
                    <div className="text-[15px] font-bold text-gray-200 bg-charcoal-900 border border-charcoal-700 rounded p-3 flex items-center justify-between shadow-inner">
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: PLAYER_COLORS[PLAYER_KEYS[player]] }}></div>
                            <span style={{ color: PLAYER_COLORS[PLAYER_KEYS[player]] }}>{player}</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span>Placed Piece {pieceId}</span>
                            <span className="text-[11px] font-normal text-gray-500 uppercase tracking-wider bg-charcoal-800 px-1.5 py-0.5 rounded">+{pieceSize} Score</span>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                    <div className="bg-charcoal-900 border border-charcoal-700 rounded p-3 shadow-sm flex flex-col justify-center">
                        <span className="text-[9px] text-slate-500 uppercase tracking-widest block mb-1 font-bold">Mobility Delta</span>
                        <div className={`text-[13px] font-bold ${isPositiveMobility ? 'text-green-400' : isNegativeMobility ? 'text-red-400' : 'text-gray-300'}`}>
                            {prevEntry ? (isPositiveMobility ? '▲ ' : isNegativeMobility ? '▼ ' : '') + mobilityDeltaStr : 'Baseline Set'}
                        </div>
                    </div>

                    <div className="bg-charcoal-900 border border-charcoal-700 rounded p-3 shadow-sm flex flex-col justify-center">
                        <span className="text-[9px] text-slate-500 uppercase tracking-widest block mb-1 font-bold">Defensive Impact</span>
                        <div className={`text-[12px] font-bold leading-tight ${blockStr.includes('Blocked ') ? 'text-blue-400' : 'text-gray-400'}`}>
                            {prevEntry ? blockStr : '-'}
                        </div>
                    </div>
                </div>

                {prevEntry && (
                    <div className="mt-auto text-[12px] text-gray-300 leading-relaxed bg-blue-900/10 border border-blue-900/40 p-3 rounded">
                        <strong className="text-neon-blue uppercase text-[10px] tracking-widest mb-1 block">Analysis</strong>
                        This move {isPositiveMobility ? 'successfully expanded' : 'reduced'} {player}'s future placement options{blockStr.includes('Blocked ') ? <><span className="text-blue-400 font-bold"> and successfully restricted opponent mobility</span> by eliminating {blockStr.replace(/\D/g, '')} legal corners.</> : ", but failed to exert defensive pressure on opponents."}
                    </div>
                )}
            </div>
        </div>
    );
};

const PlayerLeaderboard: React.FC<{
    gameState: any,
    metrics: DashboardMetrics,
    winProbs: Record<number, number> | null,
    remainingPieces?: Record<string, number[]>,
}> = ({ gameState, metrics, winProbs, remainingPieces }) => {
    const players = [1, 2, 3, 4]; // RED, BLUE, YELLOW, GREEN

    const getScore = (pName: string) => {
        const used = gameState?.pieces_used?.[pName] || [];
        return used.reduce((sum: number, id: number) => sum + getPieceSize(id), 0);
    };

    const getTerritory = (p: number) => {
        let count = 0;
        metrics.influenceMap.forEach(row => row.forEach(val => { if (val === p) count++; }));
        return count;
    };

    return (
        <div className="flex flex-col h-full bg-charcoal-800 border border-charcoal-700 rounded-lg overflow-hidden hover:border-gray-600 transition-colors shadow-sm">
            <h3 className="text-[10px] font-bold text-gray-300 uppercase tracking-widest bg-charcoal-800 border-b border-charcoal-700 px-3 py-2 shrink-0 flex justify-between items-center">
                <span>Player Leaderboard</span>
                {winProbs && <span className="text-[9px] text-neon-blue font-normal lowercase tracking-normal bg-charcoal-900 px-1.5 py-0.5 rounded border border-charcoal-600">heuristic</span>}
            </h3>
            <div className="flex-1 overflow-x-auto overflow-y-auto">
                <table className="w-full text-center border-collapse text-[11px] font-mono h-full">
                    <thead className="sticky top-0 bg-charcoal-900 z-10 shadow-sm border-b border-charcoal-700">
                        <tr className="text-gray-400 uppercase tracking-wider text-[9px]">
                            <th className="py-2.5 px-3 text-left font-bold">Player</th>
                            <th className="py-2.5 px-2 font-bold" title="Score (Squares Placed)">Score</th>
                            <th className="py-2.5 px-2 font-bold" title="Pieces Remaining in Hand">Pieces</th>
                            <th className="py-2.5 px-2 font-bold" title="Usable Frontier Corners (Mobility)">Frontier</th>
                            <th className="py-2.5 px-2 font-bold" title="Controlled Voronoi Territory">Territory</th>
                            <th className="py-2.5 px-3 font-bold text-right" title="Heuristic Win Probability">Win Prob</th>
                        </tr>
                    </thead>
                    <tbody className="bg-charcoal-800">
                        {players.map((p, i) => {
                            const pName = PLAYER_NAMES[p];
                            const score = getScore(pName);
                            const handSize = remainingPieces && Array.isArray(remainingPieces[pName]) ? remainingPieces[pName].length : 21;
                            const fSize = metrics.frontiers[p].length;
                            const territory = getTerritory(p);
                            const winProb = winProbs ? (winProbs[p] * 100).toFixed(1) + '%' : '-';

                            // Leader highlighting check
                            const isLeader = winProbs ? winProbs[p] === Math.max(...Object.values(winProbs)) : false;

                            return (
                                <tr key={p} className={`border-b ${i === 3 ? 'border-transparent' : 'border-charcoal-700/50'} hover:bg-white/5 transition-colors ${isLeader ? 'bg-blue-900/10' : ''}`}>
                                    <td className="py-3 px-3 text-left font-bold" style={{ color: PLAYER_COLORS[p] }}>
                                        <div className="flex items-center gap-1.5">
                                            {isLeader && <span className="text-yellow-400">★</span>}
                                            {pName}
                                        </div>
                                    </td>
                                    <td className="py-3 px-2 text-white font-bold">{score}</td>
                                    <td className="py-3 px-2 text-slate-400">{handSize}</td>
                                    <td className="py-3 px-2 text-blue-300">{fSize}</td>
                                    <td className="py-3 px-2 text-purple-300">{territory}</td>
                                    <td className={`py-3 px-3 text-right font-bold ${isLeader ? 'text-green-400' : 'text-slate-500'}`}>{winProb}</td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

const TabbedCharts: React.FC<{
    gameHistory: any[],
    currentSliderTurn: number,
    xAxisMode: 'move' | 'round'
}> = ({ gameHistory, currentSliderTurn, xAxisMode }) => {
    const [activeChart, setActiveChart] = useState<'mobility' | 'deadzone' | 'urgency'>('mobility');

    return (
        <div className="flex flex-col h-full bg-charcoal-800 border border-charcoal-700 rounded-lg overflow-hidden hover:border-gray-600 transition-colors shadow-sm">
            <div className="flex bg-charcoal-900 border-b border-charcoal-700 shrink-0">
                <button
                    onClick={() => setActiveChart('mobility')}
                    className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider transition-colors ${activeChart === 'mobility' ? 'bg-charcoal-800 text-neon-blue border-b-2 border-neon-blue shadow-[0_-2px_0_currentColor_inset]' : 'text-slate-500 hover:text-slate-300 hover:bg-charcoal-800'}`}
                >
                    Corner Differential (Mobility)
                </button>
                <button
                    onClick={() => setActiveChart('deadzone')}
                    className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider transition-colors ${activeChart === 'deadzone' ? 'bg-charcoal-800 text-neon-blue border-b-2 border-neon-blue shadow-[0_-2px_0_currentColor_inset]' : 'text-slate-500 hover:text-slate-300 hover:bg-charcoal-800'}`}
                >
                    Mobility (Frontier)
                </button>
                <button
                    onClick={() => setActiveChart('urgency')}
                    className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider transition-colors ${activeChart === 'urgency' ? 'bg-charcoal-800 text-neon-blue border-b-2 border-neon-blue shadow-[0_-2px_0_currentColor_inset]' : 'text-slate-500 hover:text-slate-300 hover:bg-charcoal-800'}`}
                >
                    Frontier Urgency & Threat
                </button>
            </div>
            <div className="flex-1 p-3 min-h-0 relative">
                <div className="absolute inset-0 p-3 pt-5">
                    {activeChart === 'mobility' && <ModuleC_CornerChart gameHistory={gameHistory} currentTurn={currentSliderTurn} mode={xAxisMode} />}
                    {activeChart === 'deadzone' && <ModuleE_FrontierChart gameHistory={gameHistory} currentTurn={currentSliderTurn} mode={xAxisMode} />}
                    {activeChart === 'urgency' && <ModuleF_UrgencyChart gameHistory={gameHistory} currentTurn={currentSliderTurn} mode={xAxisMode} />}
                </div>
            </div>
        </div>
    );
};

// --- Reused Charts from original file (Modified for new theme) ---

// --- Chart Round alignment helpers ---

const transformToRounds = (gameHistory: any[], extractor: (entry: any) => Record<string, number>) => {
    if (!gameHistory || gameHistory.length === 0) return [];

    const numRounds = Math.ceil(gameHistory.length / 4);
    const roundData = [];

    // We want to represent each round by the snapshot AFTER all (possible) moves in that round occurred.
    for (let r = 0; r < numRounds; r++) {
        const roundEntry: any = { round: r + 1 };
        const players = ['RED', 'BLUE', 'YELLOW', 'GREEN'];

        // Final state of round r is the state at move index (r * 4) + 3 (or latest available)
        const roundEndIdx = Math.min((r * 4) + 3, gameHistory.length - 1);

        for (const p of players) {
            // Find the last known value for this player at or before roundEndIdx
            // Since player order is fixed, we can be more efficient, but scan-back is robust.
            let val = 0;
            for (let i = roundEndIdx; i >= 0; i--) {
                const metrics = extractor(gameHistory[i]);
                if (metrics && metrics[p] !== undefined) {
                    val = metrics[p];
                    break;
                }
            }
            roundEntry[p] = val;
        }
        roundData.push(roundEntry);
    }
    return roundData;
};

// --- Reused Charts from original file (Modified for new theme) ---

export const ModuleC_CornerChart: React.FC<{ gameHistory: any[]; currentTurn: number; mode: 'move' | 'round'; }> = React.memo(({ gameHistory, currentTurn, mode }) => {
    const chartData = useMemo(() => {
        if (!gameHistory || gameHistory.length === 0) return [];

        const extractor = (entry: any) => entry.metrics?.corner_count || {};

        if (mode === 'round') {
            return transformToRounds(gameHistory, extractor);
        }

        return gameHistory.map((entry, idx) => {
            const turnNum = idx + 1;
            const metrics = extractor(entry);
            return {
                turn: turnNum,
                RED: metrics['RED'] || 0,
                BLUE: metrics['BLUE'] || 0,
                YELLOW: metrics['YELLOW'] || 0,
                GREEN: metrics['GREEN'] || 0,
            };
        });
    }, [gameHistory, mode]);

    const xKey = mode === 'move' ? 'turn' : 'round';
    const xRef = mode === 'move' ? currentTurn : Math.floor((currentTurn - 1) / 4) + 1;

    return (
        <div className="flex flex-col h-full w-full min-w-0">
            <h3 className="text-[10px] font-bold mb-2 text-slate-400 uppercase text-center shrink-0">
                Corner Differential (Mobility) vs {mode === 'move' ? 'Move' : 'Round'}
            </h3>
            <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                        <XAxis dataKey={xKey} stroke="#64748B" fontSize={8} tickMargin={5} minTickGap={10} />
                        <YAxis stroke="#64748B" fontSize={8} tickCount={5} />
                        <Tooltip contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '4px', fontSize: '10px' }} itemStyle={{ fontSize: '10px', padding: '2px 0' }} labelStyle={{ color: '#94A3B8', marginBottom: '4px' }} />
                        <ReferenceLine x={xRef} stroke="#94A3B8" strokeWidth={1} strokeDasharray="3 3" />
                        <Line type="monotone" dataKey="RED" stroke="#EF4444" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="BLUE" stroke="#3B82F6" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="YELLOW" stroke="#EAB308" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="GREEN" stroke="#22C55E" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
});

export const ModuleE_FrontierChart: React.FC<{ gameHistory: any[]; currentTurn: number; mode: 'move' | 'round'; }> = React.memo(({ gameHistory, currentTurn, mode }) => {
    const chartData = useMemo(() => {
        if (!gameHistory || gameHistory.length === 0) return [];

        const extractor = (entry: any) => entry.metrics?.frontier_size || {};

        if (mode === 'round') {
            return transformToRounds(gameHistory, extractor);
        }

        return gameHistory.map((entry, idx) => {
            const turnNum = idx + 1;
            const metrics = extractor(entry);
            return {
                turn: turnNum,
                RED: metrics['RED'] || 0,
                BLUE: metrics['BLUE'] || 0,
                YELLOW: metrics['YELLOW'] || 0,
                GREEN: metrics['GREEN'] || 0,
            };
        });
    }, [gameHistory, mode]);

    const xKey = mode === 'move' ? 'turn' : 'round';
    const xRef = mode === 'move' ? currentTurn : Math.floor((currentTurn - 1) / 4) + 1;

    return (
        <div className="flex flex-col h-full w-full min-w-0">
            <h3 className="text-[10px] font-bold mb-2 text-slate-400 uppercase text-center shrink-0">
                Mobility (Frontier) vs {mode === 'move' ? 'Move' : 'Round'}
            </h3>
            <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                        <XAxis dataKey={xKey} stroke="#64748B" fontSize={8} tickMargin={5} minTickGap={10} />
                        <YAxis stroke="#64748B" fontSize={8} tickCount={5} />
                        <Tooltip contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '4px', fontSize: '10px' }} itemStyle={{ fontSize: '10px', padding: '2px 0' }} labelStyle={{ color: '#94A3B8', marginBottom: '4px' }} />
                        <ReferenceLine x={xRef} stroke="#94A3B8" strokeWidth={1} strokeDasharray="3 3" />
                        <Line type="monotone" dataKey="RED" stroke="#EF4444" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="BLUE" stroke="#3B82F6" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="YELLOW" stroke="#EAB308" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="GREEN" stroke="#22C55E" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
});

export const ModuleF_UrgencyChart: React.FC<{ gameHistory: any[]; currentTurn: number; mode: 'move' | 'round'; }> = React.memo(({ gameHistory, currentTurn, mode }) => {
    const chartData = useMemo(() => {
        if (!gameHistory || gameHistory.length === 0) return [];

        const extractor = (entry: any) => {
            const fm = entry.metrics?.frontier_metrics;
            if (!fm) return {};
            const res: Record<string, number> = {};
            for (const p of ['RED', 'BLUE', 'YELLOW', 'GREEN']) {
                if (fm[p]?.urgency) {
                    const vals = Object.values(fm[p].urgency) as number[];
                    res[p] = vals.length > 0 ? Math.max(...vals) : 0;
                } else {
                    res[p] = 0;
                }
            }
            return res;
        };

        if (mode === 'round') {
            return transformToRounds(gameHistory, extractor);
        }

        return gameHistory.map((entry, idx) => {
            const turnNum = idx + 1;
            const metrics = extractor(entry);
            return {
                turn: turnNum,
                ...metrics
            };
        });
    }, [gameHistory, mode]);

    const xKey = mode === 'move' ? 'turn' : 'round';
    const xRef = mode === 'move' ? currentTurn : Math.floor((currentTurn - 1) / 4) + 1;

    return (
        <div className="flex flex-col h-full w-full min-w-0">
            <h3 className="text-[10px] font-bold mb-2 text-slate-400 uppercase text-center shrink-0">
                Frontier Urgency vs {mode === 'move' ? 'Move' : 'Round'}
            </h3>
            <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                        <XAxis dataKey={xKey} stroke="#64748B" fontSize={8} tickMargin={5} minTickGap={10} />
                        <YAxis stroke="#64748B" fontSize={8} tickCount={5} />
                        <Tooltip contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '4px', fontSize: '10px' }} itemStyle={{ fontSize: '10px', padding: '2px 0' }} labelStyle={{ color: '#94A3B8', marginBottom: '4px' }} />
                        <ReferenceLine x={xRef} stroke="#94A3B8" strokeWidth={1} strokeDasharray="3 3" />
                        <Line type="monotone" dataKey="RED" stroke="#EF4444" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="BLUE" stroke="#3B82F6" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="YELLOW" stroke="#EAB308" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="GREEN" stroke="#22C55E" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
});

