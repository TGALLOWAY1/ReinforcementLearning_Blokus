import React, { useEffect, useMemo, useState } from 'react';
import { useGameStore } from '../store/gameStore';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
    PieChart, Pie, Cell
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

            {/* 3-Column Layout */}
            <div className="flex-1 p-3 grid grid-cols-1 lg:grid-cols-12 gap-3 min-h-0 overflow-y-auto">

                {/* LEFT COLUMN: Predictive Line Charts */}
                <div className="lg:col-span-6 flex flex-col gap-3 min-h-0">
                    <SectionTitle title="Predictive Line Charts" />

                    <div className="flex-1 min-h-[160px] bg-charcoal-800 border border-charcoal-700 rounded-lg p-2 flex flex-col hover:border-gray-600 transition-colors">
                        <ModuleC_CornerChart gameHistory={gameHistory} currentTurn={currentSliderTurn || totalTurns} mode={xAxisMode} />
                    </div>

                    <div className="flex-1 min-h-[160px] bg-charcoal-800 border border-charcoal-700 rounded-lg p-2 flex flex-col hover:border-gray-600 transition-colors">
                        <ModuleE_FrontierChart gameHistory={gameHistory} currentTurn={currentSliderTurn || totalTurns} mode={xAxisMode} />
                    </div>

                    <div className="flex-1 min-h-[160px] bg-charcoal-800 border border-charcoal-700 rounded-lg p-2 flex flex-col hover:border-gray-600 transition-colors">
                        <ModuleF_UrgencyChart gameHistory={gameHistory} currentTurn={currentSliderTurn || totalTurns} mode={xAxisMode} />
                    </div>
                </div>

                {/* CENTER COLUMN: Spatial Visualizations */}
                <div className="lg:col-span-3 flex flex-col gap-3 min-h-0">
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

                    <div className="flex-1 flex flex-col gap-3 min-h-0 overflow-y-auto custom-scrollbar pr-1">
                        <div className="bg-charcoal-800 border border-charcoal-700 rounded-lg p-2 flex flex-col min-h-[220px] hover:border-gray-600 transition-colors">
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

                        <div className="bg-charcoal-800 border border-charcoal-700 rounded-lg p-2 flex flex-col min-h-[220px] hover:border-gray-600 transition-colors">
                            <h3 className="text-[10px] font-bold text-gray-400 uppercase text-center mb-2 tracking-wider">Endgame Dead-Zones</h3>
                            <DeadZoneMap deadZones={metrics.deadZones} boardState={currentBoard} selectedPlayer={selectedPlayer} />
                        </div>
                    </div>
                </div>

                {/* RIGHT COLUMN: Game State & Player Status */}
                <div className="lg:col-span-3 flex flex-col gap-3 min-h-0">
                    <SectionTitle title="Game State & Player Status" />

                    <div className="shrink-0 bg-charcoal-800 border border-charcoal-700 rounded-lg p-2 hover:border-gray-600 transition-colors">
                        <PiecesRemainingTable remainingPieces={activeTurnData?.metrics?.remaining_pieces} />
                    </div>

                    <div className="flex-1 bg-charcoal-800 border border-charcoal-700 rounded-lg p-2 min-h-[150px] hover:border-gray-600 transition-colors">
                        <TerritoryControlChart influenceMap={metrics.influenceMap} />
                    </div>

                    <div className="shrink-0 bg-charcoal-800 border border-charcoal-700 rounded-lg p-2 hover:border-gray-600 transition-colors">
                        <PlayerStatusSummary
                            gameState={gameState}
                            metrics={metrics}
                            remainingPieces={activeTurnData?.metrics?.remaining_pieces}
                            pieceLockRisk={activeTurnData?.metrics?.piece_lock_risk || gameState?.piece_lock_risk}
                        />
                    </div>

                    <div className="shrink-0 bg-charcoal-800 border border-charcoal-700 rounded-lg p-2 hover:border-gray-600 transition-colors">
                        <SelfBlockRiskCard selfBlockRisk={gameState?.self_block_risk} />
                    </div>
                </div>

            </div>

            {/* BOTTOM ROW: Win Probability */}
            {winProbs && (
                <div className="shrink-0 p-3 mt-auto mb-2 mx-3 bg-charcoal-800 border border-charcoal-700 rounded-lg flex items-center gap-4 hover:border-gray-600 transition-colors">
                    <div className="w-48 shrink-0">
                        <span className="text-xs font-bold text-gray-400 uppercase tracking-widest">Win Probability</span>
                    </div>
                    <div className="flex-1 h-6 flex rounded-full overflow-hidden bg-slate-800">
                        {[1, 2, 3, 4].map(p => {
                            const width = `${(winProbs[p] * 100).toFixed(1)}%`;
                            return winProbs[p] > 0 ? (
                                <div
                                    key={p}
                                    style={{ width, backgroundColor: PLAYER_COLORS[p] }}
                                    className="h-full flex items-center justify-center text-[10px] font-bold shadow-[inset_0_2px_4px_rgba(255,255,255,0.2)] transition-all duration-500"
                                    title={`${PLAYER_NAMES[p]}: ${width}`}
                                >
                                    {winProbs[p] > 0.1 ? width : ''}
                                </div>
                            ) : null;
                        })}
                    </div>
                </div>
            )}
        </div>
    );
};

const SectionTitle: React.FC<{ title: string }> = ({ title }) => (
    <div className="bg-charcoal-800 border-l-[3px] border-neon-blue rounded-r px-3 py-1.5 shrink-0 shadow-sm">
        <h3 className="text-xs font-bold text-gray-300 uppercase tracking-widest">{title}</h3>
    </div>
);

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

const TerritoryControlChart: React.FC<{ influenceMap: number[][] }> = ({ influenceMap }) => {
    const data = useMemo(() => {
        const counts = { 1: 0, 2: 0, 3: 0, 4: 0, 0: 0 };
        influenceMap.forEach(row => row.forEach(val => {
            if (val >= 0 && counts[val as keyof typeof counts] !== undefined) {
                counts[val as keyof typeof counts]++;
            }
        }));

        return [1, 2, 3, 4].map(p => ({
            name: PLAYER_NAMES[p],
            value: counts[p as keyof typeof counts],
            fill: PLAYER_COLORS[p]
        })).filter(d => d.value > 0);
    }, [influenceMap]);

    return (
        <div className="flex flex-col h-full items-center">
            <h3 className="text-[10px] font-bold text-slate-400 uppercase text-center shrink-0 mb-1">Territory Control (Influence)</h3>
            <div className="flex-1 w-full min-h-[100px]">
                <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                        <Pie
                            data={data}
                            innerRadius="60%"
                            outerRadius="90%"
                            paddingAngle={2}
                            dataKey="value"
                            stroke="none"
                        >
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.fill} />
                            ))}
                        </Pie>
                        <Tooltip
                            contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '4px', fontSize: '10px', padding: '4px 8px' }}
                            itemStyle={{ color: '#E2E8F0', fontSize: '12px', fontWeight: 'bold' }}
                            formatter={(value: any) => [`${value} cells`, '']}
                        />
                    </PieChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

const PiecesRemainingTable: React.FC<{ remainingPieces?: Record<string, number[]> }> = ({ remainingPieces }) => {
    // Blokus tile sizes from standard 21 pieces:
    // 1 square: piece 1
    // 2 squares: piece 2
    // 3 squares: pieces 3,4
    // 4 squares: pieces 5-9
    // 5 squares: pieces 10-21
    const getPieceSize = (id: number) => {
        if (id === 1) return 1;
        if (id === 2) return 2;
        if (id <= 4) return 3;
        if (id <= 9) return 4;
        return 5;
    };

    if (!remainingPieces) return <div className="text-[10px] text-slate-500 text-center">No data</div>;

    const players = ['BLUE', 'GREEN', 'YELLOW', 'RED'];
    const pKeys = players.map(p => PLAYER_KEYS[p]); // 2, 4, 3, 1

    const matrix = [1, 2, 3, 4, 5].map(size => {
        const rowData: Record<string, any> = { size };
        players.forEach(p => {
            const hand = remainingPieces[p] || [];
            const count = hand.filter(id => getPieceSize(id) === size).length;
            rowData[p] = count;
        });
        return rowData;
    });

    const totals = players.map(p => {
        const hand = remainingPieces[p] || [];
        return hand.reduce((sum, id) => sum + getPieceSize(id), 0);
    });

    return (
        <div>
            <h3 className="text-[10px] font-bold text-slate-400 uppercase text-center mb-2">Pieces Remaining (By Size)</h3>
            <div className="overflow-x-auto text-[10px] font-mono">
                <table className="w-full text-center border-collapse">
                    <thead>
                        <tr className="border-b border-charcoal-700">
                            <th className="py-1 text-gray-500 font-normal">SIZE</th>
                            {players.map((p, i) => (
                                <th key={p} className="py-1 tracking-widest" style={{ color: PLAYER_COLORS[pKeys[i]] }}>{p}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {matrix.map(row => (
                            <tr key={row.size} className="border-b border-charcoal-700/50 hover:bg-white/5 transition-colors">
                                <td className="py-1 text-gray-400">{row.size}</td>
                                {players.map(p => (
                                    <td key={p} className="py-1 text-slate-300">{row[p]}</td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                    <tfoot>
                        <tr className="bg-slate-800/50">
                            <td className="py-1.5 text-slate-400 text-[9px] uppercase tracking-wider">Total Sq.</td>
                            {totals.map((t, i) => (
                                <td key={i} className="py-1.5 font-bold text-white">{t}</td>
                            ))}
                        </tr>
                    </tfoot>
                </table>
            </div>
        </div>
    );
};

const PlayerStatusSummary: React.FC<{ gameState: any, metrics: DashboardMetrics, remainingPieces?: Record<string, number[]>, pieceLockRisk?: Record<string, number> }> = ({ gameState, metrics, remainingPieces, pieceLockRisk }) => {
    const players = [2, 4, 1, 3]; // BLUE, GREEN, RED, YELLOW

    return (
        <div>
            <h3 className="text-[10px] font-bold text-gray-400 uppercase text-center mb-2 tracking-wider">Player Status</h3>
            <div className="grid grid-cols-5 text-[9px] uppercase tracking-wider text-gray-500 mb-1 px-2 text-center">
                <div className="text-left py-1 col-span-2">Player / Agent</div>
                <div className="py-1">Frontier</div>
                <div className="py-1">Pieces</div>
                <div className="py-1">Lock Risk</div>
            </div>
            <div className="flex flex-col gap-1">
                {players.map(p => {
                    const pName = PLAYER_NAMES[p];
                    const fSize = metrics.frontiers[p].length;
                    const handSize = remainingPieces && Array.isArray(remainingPieces[pName]) ? remainingPieces[pName].length : 0;
                    const lockRisk = pieceLockRisk?.[pName] ?? 0;
                    const config = gameState?.players?.find((pc: any) => pc.player === pName);
                    const agentType = config?.agent_type || 'human';
                    const diff = config?.agent_config?.difficulty;

                    return (
                        <div key={p} className="grid grid-cols-5 text-[11px] font-mono bg-charcoal-900 rounded border border-charcoal-700 px-2 py-1.5 text-center items-center h-[32px]">
                            <div className="text-left flex flex-col justify-center leading-tight col-span-2">
                                <div className="font-bold tracking-widest text-[10px]" style={{ color: PLAYER_COLORS[p], textShadow: `0 0 10px ${PLAYER_COLORS[p]}40` }}>{pName}</div>
                                <div className="flex gap-1 mt-0.5">
                                    <span className={`text-[8px] px-1 rounded-sm font-bold uppercase ${agentType === 'human' ? 'bg-slate-700 text-slate-400' : 'bg-blue-900 text-blue-400'}`}>
                                        {agentType === 'mcts' ? (diff || 'MCTS') : agentType}
                                    </span>
                                    {config?.agent_config?.time_budget_ms && (
                                        <span className="text-[8px] text-slate-500">{config.agent_config.time_budget_ms}ms</span>
                                    )}
                                </div>
                            </div>
                            <div className="text-slate-300">{fSize}</div>
                            <div className="text-slate-300">{handSize}</div>
                            <div className={lockRisk > 0 ? 'text-red-400 font-bold' : 'text-slate-500'}>{lockRisk}</div>
                        </div>
                    );
                })}
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

const SelfBlockRiskCard: React.FC<{ selfBlockRisk?: { top_moves: Array<{ piece_id: number; risk: number; clusters_touched: number; frontier_points_used: number }> } }> = ({ selfBlockRisk }) => {
    const moves = selfBlockRisk?.top_moves?.slice(0, 5) || [];

    return (
        <div>
            <h3 className="text-[10px] font-bold text-gray-400 uppercase text-center mb-2 tracking-wider">Self-Block Risk (Current Player)</h3>
            {moves.length === 0 ? (
                <div className="text-[10px] text-slate-500 text-center py-2">No risky moves detected</div>
            ) : (
                <div className="overflow-x-auto text-[10px] font-mono">
                    <table className="w-full text-center border-collapse">
                        <thead>
                            <tr className="border-b border-charcoal-700">
                                <th className="py-1 text-gray-500 font-normal">Piece</th>
                                <th className="py-1 text-gray-500 font-normal">Risk</th>
                                <th className="py-1 text-gray-500 font-normal">Clusters</th>
                                <th className="py-1 text-gray-500 font-normal">Frontiers</th>
                            </tr>
                        </thead>
                        <tbody>
                            {moves.map((m, i) => (
                                <tr key={i} className="border-b border-charcoal-700/50 hover:bg-white/5 transition-colors">
                                    <td className="py-1 text-slate-300">P{m.piece_id}</td>
                                    <td className={`py-1 font-bold ${m.risk >= 6 ? 'text-red-400' : m.risk >= 3 ? 'text-orange-400' : 'text-yellow-400'}`}>{m.risk}</td>
                                    <td className="py-1 text-slate-400">{m.clusters_touched}</td>
                                    <td className="py-1 text-slate-400">{m.frontier_points_used}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};
