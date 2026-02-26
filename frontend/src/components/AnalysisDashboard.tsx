import React, { useEffect, useMemo } from 'react';
import { useGameStore, useLegalMovesByPiece, useLegalMoves } from '../store/gameStore';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';

export const AnalysisDashboard: React.FC = () => {
    const gameState = useGameStore(s => s.gameState);
    const currentSliderTurn = useGameStore(s => s.currentSliderTurn);
    const setCurrentSliderTurn = useGameStore(s => s.setCurrentSliderTurn);

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

    if (!gameState) {
        return <div className="p-4 text-gray-500">Connecting to Engine...</div>;
    }

    const currentBoard = activeTurnData?.board_state || gameState.board;

    return (
        <div className="flex flex-col h-full bg-charcoal-900 border border-charcoal-700 rounded-lg overflow-hidden">

            {/* Top: The Board (Module A) */}
            <div className="flex-1 p-4 flex items-center justify-center min-h-[300px]">
                {currentBoard ? (
                    <BoardRenderer boardState={currentBoard} />
                ) : (
                    <div className="text-gray-500">Loading board...</div>
                )}
            </div>

            {/* Middle: Turn Slider */}
            <div className="shrink-0 p-4 border-y border-charcoal-700 bg-charcoal-800">
                <div className="flex items-center gap-4">
                    <span className="text-xs font-mono text-gray-400">Turn 1</span>
                    <input
                        type="range"
                        min={1}
                        max={totalTurns}
                        value={currentSliderTurn || totalTurns}
                        onChange={(e) => setCurrentSliderTurn(parseInt(e.target.value, 10))}
                        className="flex-1 accent-neon-blue"
                    />
                    <span className="text-xs font-mono text-neon-blue font-bold">Turn {currentSliderTurn || totalTurns}</span>
                </div>
            </div>

            {/* Bottom: Modules Grid */}
            <div className="shrink-0 h-[300px] grid grid-cols-2 lg:grid-cols-5 gap-2 bg-charcoal-900 p-2">
                <ModuleB_Voronoi boardState={currentBoard} />
                <ModuleC_CornerChart
                    gameHistory={gameHistory}
                    currentTurn={currentSliderTurn || totalTurns}
                />
                <ModuleE_FrontierChart
                    gameHistory={gameHistory}
                    currentTurn={currentSliderTurn || totalTurns}
                />
                <ModuleD_Pieces
                    player={activeTurnData?.player_to_move || gameState.current_player}
                    remainingPieces={activeTurnData?.metrics?.remaining_pieces}
                    penalty={activeTurnData?.metrics?.difficult_piece_penalty}
                />
                <ModuleF_LiveLegalMoves player={gameState.current_player} />
            </div>
        </div>
    );
};

// Module A: Lightweight Board Renderer
const BoardRenderer: React.FC<{ boardState: number[][] }> = React.memo(({ boardState }) => {
    if (!boardState || boardState.length === 0) return null;
    const size = boardState.length;

    const colors = [
        'bg-charcoal-800',    // 0 = Empty
        'bg-red-500',         // 1 = Red
        'bg-blue-500',        // 2 = Blue
        'bg-yellow-400',      // 3 = Yellow
        'bg-green-500'        // 4 = Green
    ];

    return (
        <div
            className="grid gap-[1px] bg-charcoal-600 p-[1px] rounded"
            style={{
                gridTemplateColumns: `repeat(${size}, minmax(0, 1fr))`,
                aspectRatio: '1 / 1',
                maxHeight: '100%',
                maxWidth: '100%'
            }}
        >
            {boardState.map((row, r) =>
                row.map((cell, c) => (
                    <div
                        key={`${r}-${c}`}
                        className={`${colors[cell]} w-full h-full min-w-[10px] min-h-[10px] rounded-sm`}
                    />
                ))
            )}
        </div>
    );
});

// Module B: Voronoi Influence Map
const ModuleB_Voronoi: React.FC<{ boardState: number[][] | undefined }> = React.memo(({ boardState }) => {
    const influenceMap = useMemo(() => {
        if (!boardState || boardState.length === 0) return null;
        const size = boardState.length;
        const map = Array(size).fill(0).map(() => Array(size).fill(0));
        const distances = Array(size).fill(0).map(() => Array(size).fill(Infinity));

        // Multi-source BFS queue: [row, col, playerId, distance]
        const queue: [number, number, number, number][] = [];

        // Add all existing pieces to the queue as starting points
        for (let r = 0; r < size; r++) {
            for (let c = 0; c < size; c++) {
                if (boardState[r][c] !== 0) {
                    queue.push([r, c, boardState[r][c], 0]);
                    map[r][c] = boardState[r][c];
                    distances[r][c] = 0;
                }
            }
        }

        let head = 0;
        const dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]];

        while (head < queue.length) {
            const [r, c, player, dist] = queue[head++];

            for (const [dr, dc] of dirs) {
                const nr = r + dr;
                const nc = c + dc;

                if (nr >= 0 && nr < size && nc >= 0 && nc < size) {
                    // Update if empty OR if we found a strictly shorter path to an already influenced cell
                    // (For Voronoi, we usually just record the *first* time we reach a cell, which is the shortest)
                    if (dist + 1 < distances[nr][nc]) {
                        distances[nr][nc] = dist + 1;
                        map[nr][nc] = player;
                        queue.push([nr, nc, player, dist + 1]);
                    }
                }
            }
        }
        return map;
    }, [boardState]);

    if (!influenceMap) {
        return (
            <div className="bg-charcoal-800 rounded border border-charcoal-700 p-2 flex flex-col items-center justify-center text-gray-500">
                <h3 className="text-xs font-semibold mb-2 text-gray-400">Voronoi Influence</h3>
                <p className="text-[10px]">No data</p>
            </div>
        );
    }

    const size = influenceMap.length;

    // Muted colors for influence, bright for actual pieces
    const getCellColor = (r: number, c: number) => {
        const owner = influenceMap[r][c];
        if (owner === 0) return 'bg-charcoal-800';

        if (boardState && boardState[r][c] !== 0) {
            // Actual piece
            return [
                '',
                'bg-red-500',
                'bg-blue-500',
                'bg-yellow-400',
                'bg-green-500'
            ][boardState[r][c]];
        }

        // Influence area
        return [
            '',
            'bg-red-900/40',
            'bg-blue-900/40',
            'bg-yellow-900/40',
            'bg-green-900/40'
        ][owner];
    };

    return (
        <div className="bg-charcoal-800 rounded border border-charcoal-700 p-2 flex flex-col h-full">
            <h3 className="text-[10px] font-semibold mb-2 text-gray-400 uppercase text-center shrink-0">Voronoi Influence</h3>
            <div className="flex-1 flex items-center justify-center min-h-0">
                <div
                    className="grid gap-[1px] bg-charcoal-900 p-[1px] rounded overflow-hidden max-h-full max-w-full"
                    style={{
                        gridTemplateColumns: `repeat(${size}, minmax(0, 1fr))`,
                        aspectRatio: '1 / 1'
                    }}
                >
                    {influenceMap.map((row, r) =>
                        row.map((_, c) => (
                            <div
                                key={`${r}-${c}`}
                                className={`${getCellColor(r, c)} w-full h-full min-w-[2px] min-h-[2px]`}
                            />
                        ))
                    )}
                </div>
            </div>
        </div>
    );
});

// Module D: Remaining Pieces & Penalty
const ModuleD_Pieces: React.FC<{
    player?: string;
    remainingPieces?: Record<string, number[]>;
    penalty?: Record<string, number>;
}> = React.memo(({ player, remainingPieces, penalty }) => {

    // We import PIECE_SHAPES at the top of the file ideally, but defining a tiny subset for the visualizer is okay
    // We will assume the user has 1..21 pieces

    if (!player || !remainingPieces || !remainingPieces[player]) {
        return (
            <div className="bg-charcoal-800 rounded border border-charcoal-700 p-2 flex flex-col items-center justify-center text-gray-500 min-h-0">
                <h3 className="text-[10px] font-semibold mb-2 text-gray-400 uppercase text-center shrink-0">Active Hand</h3>
                <p className="text-[10px]">No data</p>
            </div>
        );
    }

    const hand = remainingPieces[player];
    const difficultyScore = penalty ? penalty[player] : 0;

    // Simple color map for the player
    const playerColors: Record<string, string> = {
        'RED': 'bg-red-500',
        'BLUE': 'bg-blue-500',
        'YELLOW': 'bg-yellow-400',
        'GREEN': 'bg-green-500'
    };

    const colorClass = playerColors[player] || 'bg-gray-500';

    return (
        <div className="bg-charcoal-800 rounded border border-charcoal-700 p-2 flex flex-col h-full font-mono">
            <div className="flex justify-between items-center shrink-0 mb-2">
                <h3 className="text-[10px] font-semibold text-gray-400 uppercase">
                    Hand: <span className={playerColors[player] ? `text-${player.toLowerCase()}-400` : ''}>{player}</span>
                </h3>
                <div className="text-[10px] bg-charcoal-900 border border-charcoal-700 px-1.5 py-0.5 rounded text-gray-300">
                    Penalty: {difficultyScore.toFixed(1)}
                </div>
            </div>

            <div className="flex-1 overflow-y-auto min-h-0 bg-charcoal-900/50 rounded border border-charcoal-700 p-2">
                <div className="flex flex-wrap gap-1.5">
                    {hand.map(pieceId => (
                        <div key={pieceId} className="flex flex-col items-center gap-0.5" title={`Piece ${pieceId}`}>
                            <div className="w-4 h-4 text-[8px] flex items-center justify-center bg-charcoal-700 rounded-sm text-gray-400">
                                {pieceId}
                            </div>
                            <div className={`w-4 h-1 rounded-full ${colorClass}`} />
                        </div>
                    ))}
                    {hand.length === 0 && (
                        <div className="text-[10px] text-gray-500 italic w-full text-center mt-4">Empty Hand</div>
                    )}
                </div>
            </div>

            <div className="text-[9px] text-gray-500 mt-2 text-center shrink-0">
                {hand.length} pieces remaining
            </div>
        </div>
    );
});

// Module C: Corner Differential / Mobility Chart
export const ModuleC_CornerChart: React.FC<{
    gameHistory: any[];
    currentTurn: number;
}> = React.memo(({ gameHistory, currentTurn }) => {

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
                // Alternatively, could plot frontier_size or a combined mobility score
            };
        });
    }, [gameHistory]);



    return (
        <div className="bg-charcoal-800 rounded border border-charcoal-700 p-2 flex flex-col h-full col-span-2 lg:col-span-1 min-w-0">
            <h3 className="text-[10px] font-semibold mb-2 text-gray-400 uppercase text-center shrink-0">Corner Differential (Mobility)</h3>
            <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#3E3E42" vertical={false} />
                        <XAxis
                            dataKey="turn"
                            stroke="#9CA3AF"
                            fontSize={9}
                            tickMargin={5}
                            minTickGap={10}
                        />
                        <YAxis
                            stroke="#9CA3AF"
                            fontSize={9}
                            tickCount={5}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#18181B', borderColor: '#3E3E42', fontSize: '10px' }}
                            itemStyle={{ fontSize: '10px', padding: '2px 0' }}
                            labelStyle={{ color: '#9CA3AF', marginBottom: '4px' }}
                        />
                        <ReferenceLine
                            x={currentTurn}
                            stroke="#00F0FF"
                            strokeWidth={2}
                            strokeDasharray="3 3"
                        />
                        <Line type="monotone" dataKey="RED" stroke="#FF4D4D" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="BLUE" stroke="#00F0FF" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="YELLOW" stroke="#FFE600" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="GREEN" stroke="#00FF9D" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
});

// Module E: Frontier Size Chart
export const ModuleE_FrontierChart: React.FC<{
    gameHistory: any[];
    currentTurn: number;
}> = React.memo(({ gameHistory, currentTurn }) => {

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
        <div className="bg-charcoal-800 rounded border border-charcoal-700 p-2 flex flex-col h-full col-span-2 lg:col-span-1 min-w-0">
            <h3 className="text-[10px] font-semibold mb-2 text-gray-400 uppercase text-center shrink-0">Frontier Size</h3>
            <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#3E3E42" vertical={false} />
                        <XAxis
                            dataKey="turn"
                            stroke="#9CA3AF"
                            fontSize={9}
                            tickMargin={5}
                            minTickGap={10}
                        />
                        <YAxis
                            stroke="#9CA3AF"
                            fontSize={9}
                            tickCount={5}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#18181B', borderColor: '#3E3E42', fontSize: '10px' }}
                            itemStyle={{ fontSize: '10px', padding: '2px 0' }}
                            labelStyle={{ color: '#9CA3AF', marginBottom: '4px' }}
                        />
                        <ReferenceLine
                            x={currentTurn}
                            stroke="#00F0FF"
                            strokeWidth={2}
                            strokeDasharray="3 3"
                        />
                        <Line type="monotone" dataKey="RED" stroke="#FF4D4D" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="BLUE" stroke="#00F0FF" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="YELLOW" stroke="#FFE600" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="GREEN" stroke="#00FF9D" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
});

// Module F: Live Legal Moves Status
export const ModuleF_LiveLegalMoves: React.FC<{ player: string | undefined }> = ({ player }) => {
    const legalMoves = useLegalMoves();
    const legalByPiece = useLegalMovesByPiece();

    // Sort by count descending
    const sortedPieceCounts = [...legalByPiece].sort((a, b) => b.count - a.count);

    if (!player || legalMoves.length === 0) {
        return (
            <div className="bg-charcoal-800 rounded border border-charcoal-700 p-2 flex flex-col items-center justify-center text-gray-500 min-h-0">
                <h3 className="text-[10px] font-semibold mb-2 text-gray-400 uppercase text-center shrink-0">Live Legal Moves</h3>
                <p className="text-[10px]">Waiting for turn...</p>
            </div>
        );
    }

    // Simple color map for the player
    const playerColors: Record<string, string> = {
        'RED': 'bg-red-500',
        'BLUE': 'bg-blue-500',
        'YELLOW': 'bg-yellow-400',
        'GREEN': 'bg-green-500'
    };

    return (
        <div className="bg-charcoal-800 rounded border border-charcoal-700 p-2 flex flex-col h-full col-span-2 lg:col-span-1 min-w-0">
            <h3 className="text-[10px] font-semibold mb-2 text-gray-400 uppercase text-center shrink-0">
                <span className="text-neon-blue">Live: </span>{player} Moves
            </h3>

            <div className="text-[20px] font-bold text-center text-white mb-2 shrink-0">
                {legalMoves.length}
                <span className="text-[10px] font-normal text-gray-400 ml-1">total</span>
            </div>

            <div className="flex-1 overflow-y-auto min-h-0 pr-1 custom-scrollbar">
                <div className="grid grid-cols-2 gap-1.5">
                    {sortedPieceCounts.map(({ pieceId, count }) => {
                        if (count === 0) return null;
                        return (
                            <div key={pieceId} className="flex justify-between items-center bg-charcoal-900 border border-charcoal-700 rounded px-1.5 py-1">
                                <span className={`text-[8px] font-bold ${playerColors[player] ? `text-${player.toLowerCase()}-400` : 'text-gray-400'}`}>P{pieceId}</span>
                                <span className="text-[10px] text-gray-300 font-mono">{count}</span>
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
};
