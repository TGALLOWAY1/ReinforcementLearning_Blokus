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
                    <span className="text-xs font-mono bg-blue-900/50 text-blue-400 px-2 py-1 rounded">Turn {currentSliderTurn || totalTurns} / {totalTurns} ({currentPlayerStr} to move)</span>
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
                        <ModuleC_CornerChart gameHistory={gameHistory} currentTurn={currentSliderTurn || totalTurns} />
                    </div>

                    <div className="flex-1 min-h-[160px] bg-charcoal-800 border border-charcoal-700 rounded-lg p-2 flex flex-col hover:border-gray-600 transition-colors">
                        <ModuleE_FrontierChart gameHistory={gameHistory} currentTurn={currentSliderTurn || totalTurns} />
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
                                frontierMetrics={gameState?.frontier_metrics}
                                frontierClusters={gameState?.frontier_clusters}
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
                        <PlayerStatusSummary metrics={metrics} remainingPieces={activeTurnData?.metrics?.remaining_pieces} />
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

const FrontierMap: React.FC<{
    frontiers: Record<number, { r: number, c: number }[]>,
    boardState: number[][],
    selectedPlayer: number,
    frontierMetrics?: any,
    frontierClusters?: any
}> = ({ frontiers, boardState, selectedPlayer, frontierMetrics, frontierClusters }) => {
    const [colorMode, setColorMode] = useState<'Default' | 'Urgency' | 'Cluster'>('Default');
    const size = boardState.length;
    // Build a quick lookup for selected player's frontier
    const fMap = Array(size).fill(0).map(() => Array(size).fill(false));
    const cells = frontiers[selectedPlayer] || [];
    for (const { r, c } of cells) {
        fMap[r][c] = true;
    }

    return (
        <div className="flex-1 flex flex-col min-h-0 relative">
            <div className="absolute top-[-26px] right-0 flex gap-1 z-10">
                <button onClick={() => setColorMode('Default')} className={`text-[9px] px-1.5 py-0.5 rounded ${colorMode === 'Default' ? 'bg-slate-500 text-white' : 'bg-charcoal-700 text-gray-400 hover:text-gray-200'}`}>Def</button>
                <button onClick={() => setColorMode('Urgency')} className={`text-[9px] px-1.5 py-0.5 rounded ${colorMode === 'Urgency' ? 'bg-slate-500 text-white' : 'bg-charcoal-700 text-gray-400 hover:text-gray-200'}`}>Urg</button>
                <button onClick={() => setColorMode('Cluster')} className={`text-[9px] px-1.5 py-0.5 rounded ${colorMode === 'Cluster' ? 'bg-slate-500 text-white' : 'bg-charcoal-700 text-gray-400 hover:text-gray-200'}`}>Cls</button>
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
                            let urg = 0;
                            let cid = -1;

                            if (frontierMetrics || frontierClusters) {
                                const u = frontierMetrics?.utility?.[key];
                                const bp = frontierMetrics?.block_pressure?.[key];
                                urg = frontierMetrics?.urgency?.[key] || 0;
                                cid = frontierClusters?.cluster_id?.[key] ?? -1;

                                if (u !== undefined) {
                                    title = `[${r}, ${c}]\nUtility: ${u}\nBlock Pressure: ${bp}\nUrgency: ${urg}\nCluster ID: ${cid}`;
                                }
                            }

                            if (colorMode === 'Default') {
                                bg = ['', 'bg-red-400', 'bg-blue-400', 'bg-yellow-300', 'bg-green-400'][selectedPlayer] + ' shadow-[0_0_8px_currentColor]';
                            } else if (colorMode === 'Urgency') {
                                if (urg >= 4) bg = 'bg-red-500 shadow-[0_0_8px_currentColor]';
                                else if (urg >= 2) bg = 'bg-orange-400 shadow-[0_0_4px_currentColor]';
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
                        return <div key={`${r}-${c}`} className={`${bg} ${isFrontier ? 'rounded-full scale-75 cursor-help' : 'rounded-sm'} transition-colors`} title={title} />
                    }))}
                </div>
            </div>
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

const PlayerStatusSummary: React.FC<{ metrics: DashboardMetrics, remainingPieces?: Record<string, number[]> }> = ({ metrics, remainingPieces }) => {
    const players = [2, 4, 1, 3]; // BLUE, GREEN, RED, YELLOW

    return (
        <div>
            <h3 className="text-[10px] font-bold text-gray-400 uppercase text-center mb-2 tracking-wider">Player Status</h3>
            <div className="grid grid-cols-3 text-[9px] uppercase tracking-wider text-gray-500 mb-1 px-2 text-center">
                <div className="text-left py-1">Player</div>
                <div className="py-1">Frontier Size</div>
                <div className="py-1">Pieces Left</div>
            </div>
            <div className="flex flex-col gap-1">
                {players.map(p => {
                    const pName = PLAYER_NAMES[p];
                    const fSize = metrics.frontiers[p].length;
                    const handSize = remainingPieces && Array.isArray(remainingPieces[pName]) ? remainingPieces[pName].length : 0;

                    return (
                        <div key={p} className="grid grid-cols-3 text-[11px] font-mono bg-charcoal-900 rounded border border-charcoal-700 px-2 py-1.5 text-center items-center h-[28px]">
                            <div className="font-bold text-left tracking-widest" style={{ color: PLAYER_COLORS[p], textShadow: `0 0 10px ${PLAYER_COLORS[p]}40` }}>{pName}</div>
                            <div className="text-slate-300">{fSize}</div>
                            <div className="text-slate-300">{handSize}</div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

// --- Reused Charts from original file (Modified for new theme) ---

export const ModuleC_CornerChart: React.FC<{ gameHistory: any[]; currentTurn: number; }> = React.memo(({ gameHistory, currentTurn }) => {
    const chartData = useMemo(() => {
        if (!gameHistory || gameHistory.length === 0) return [];
        return gameHistory.map((entry, idx) => {
            const turnNum = idx + 1;
            const metrics = entry.metrics;
            if (!metrics || !metrics.corner_count) return { turn: turnNum };
            return {
                turn: turnNum,
                RED: metrics.corner_count['RED'] || 0,
                BLUE: metrics.corner_count['BLUE'] || 0,
                YELLOW: metrics.corner_count['YELLOW'] || 0,
                GREEN: metrics.corner_count['GREEN'] || 0,
            };
        });
    }, [gameHistory]);

    return (
        <div className="flex flex-col h-full w-full min-w-0">
            <h3 className="text-[10px] font-bold mb-2 text-slate-400 uppercase text-center shrink-0">Corner Differential (Mobility)</h3>
            <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                        <XAxis dataKey="turn" stroke="#64748B" fontSize={8} tickMargin={5} minTickGap={10} />
                        <YAxis stroke="#64748B" fontSize={8} tickCount={5} />
                        <Tooltip contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '4px', fontSize: '10px' }} itemStyle={{ fontSize: '10px', padding: '2px 0' }} labelStyle={{ color: '#94A3B8', marginBottom: '4px' }} />
                        <ReferenceLine x={currentTurn} stroke="#94A3B8" strokeWidth={1} strokeDasharray="3 3" />
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

export const ModuleE_FrontierChart: React.FC<{ gameHistory: any[]; currentTurn: number; }> = React.memo(({ gameHistory, currentTurn }) => {
    const chartData = useMemo(() => {
        if (!gameHistory || gameHistory.length === 0) return [];
        return gameHistory.map((entry, idx) => {
            const turnNum = idx + 1;
            const metrics = entry.metrics;
            if (!metrics || !metrics.frontier_size) return { turn: turnNum };
            return {
                turn: turnNum,
                RED: metrics.frontier_size['RED'] || 0,
                BLUE: metrics.frontier_size['BLUE'] || 0,
                YELLOW: metrics.frontier_size['YELLOW'] || 0,
                GREEN: metrics.frontier_size['GREEN'] || 0,
            };
        });
    }, [gameHistory]);

    return (
        <div className="flex flex-col h-full w-full min-w-0">
            <h3 className="text-[10px] font-bold mb-2 text-slate-400 uppercase text-center shrink-0">Mobility (Frontier Size) vs Turn</h3>
            <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                        <XAxis dataKey="turn" stroke="#64748B" fontSize={8} tickMargin={5} minTickGap={10} />
                        <YAxis stroke="#64748B" fontSize={8} tickCount={5} />
                        <Tooltip contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '4px', fontSize: '10px' }} itemStyle={{ fontSize: '10px', padding: '2px 0' }} labelStyle={{ color: '#94A3B8', marginBottom: '4px' }} />
                        <ReferenceLine x={currentTurn} stroke="#94A3B8" strokeWidth={1} strokeDasharray="3 3" />
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


