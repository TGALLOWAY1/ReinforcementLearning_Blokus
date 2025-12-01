import React, { useState, useEffect } from 'react';
import { useGameStore } from '../store/gameStore';
import { BOARD_SIZE, CELL_SIZE, PIECE_NAMES, PIECE_SIZE } from '../constants/gameConstants';
import { getPieceShape } from '../utils/pieceUtils';

interface AgentVisualizationsProps {
  onPieceSelect?: (pieceId: number) => void;
  selectedPiece?: number | null;
  pieceOrientation?: number;
  setPieceOrientation?: (orientation: number) => void;
}

type TabType = 'policy' | 'value' | 'tree' | 'pieces';

export const AgentVisualizations: React.FC<AgentVisualizationsProps> = ({
  onPieceSelect,
  selectedPiece,
  pieceOrientation = 0,
  setPieceOrientation
}) => {
  const [activeTab, setActiveTab] = useState<TabType>('policy');
  const { gameState } = useGameStore();
  const [localOrientation, setLocalOrientation] = useState(0);
  
  const orientation = pieceOrientation !== undefined ? pieceOrientation : localOrientation;
  const setOrientation = setPieceOrientation || setLocalOrientation;

  // Generate random policy grid for demo if no data exists
  const generateRandomPolicy = (): number[][] => {
    return Array(BOARD_SIZE).fill(null).map(() =>
      Array(BOARD_SIZE).fill(null).map(() => Math.random())
    );
  };

  // Use real heatmap data from gameState if available, otherwise use random for demo
  const currentPolicyGrid = gameState?.heatmap || generateRandomPolicy();

  // Check if this is binary heatmap (legal moves) or continuous policy
  const isBinaryHeatmap = gameState?.heatmap !== undefined;
  
  // Get min/max for normalization (for continuous policy)
  const allValues = currentPolicyGrid.flat();
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const range = maxValue - minValue || 1;

  // Value function data (dummy for now)
  const winProbability = 0.68; // 68% win probability

  // Pieces data
  const currentPlayer = gameState?.current_player;
  const piecesUsed = gameState?.pieces_used?.[currentPlayer || ''] || [];
  const availablePieces = Array.from({length: 21}, (_, i) => i + 1).filter(pieceId => !piecesUsed.includes(pieceId));

  // Keyboard handlers for piece rotation/flip
  useEffect(() => {
    const handleGlobalKeyDown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }
      
      if (event.key === 'r' || event.key === 'R') {
        event.preventDefault();
        if (orientation < 4) {
          setOrientation((orientation + 1) % 4);
        } else {
          setOrientation(4 + ((orientation - 4 + 1) % 4));
        }
      } else if (event.key === 'f' || event.key === 'F') {
        event.preventDefault();
        if (orientation < 4) {
          setOrientation(orientation + 4);
        } else {
          setOrientation(orientation - 4);
        }
      }
    };

    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, [setOrientation, orientation]);

  const tabs: { id: TabType; label: string }[] = [
    { id: 'policy', label: 'Policy' },
    { id: 'value', label: 'Value' },
    { id: 'tree', label: 'Tree' },
    { id: 'pieces', label: 'Pieces' },
  ];

  return (
    <div className="h-full flex flex-col">
      {/* Tab Navigation */}
      <div className="flex border-b border-charcoal-700">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
              activeTab === tab.id
                ? 'bg-charcoal-800 text-neon-blue border-b-2 border-neon-blue'
                : 'bg-charcoal-900 text-gray-400 hover:text-gray-200 hover:bg-charcoal-800'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'policy' && (
          <div className="space-y-3">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Policy Heatmap
            </h3>
            <div className="overflow-auto max-h-96">
              <div className="relative inline-block" style={{ width: BOARD_SIZE * CELL_SIZE, height: BOARD_SIZE * CELL_SIZE }}>
                {/* Base board (read-only) */}
                <svg
                  width={BOARD_SIZE * CELL_SIZE}
                  height={BOARD_SIZE * CELL_SIZE}
                  style={{ background: 'transparent' }}
                >
                  {/* Grid lines */}
                  {Array.from({ length: BOARD_SIZE + 1 }).map((_, i) => (
                    <g key={i}>
                      <line
                        x1={i * CELL_SIZE}
                        y1={0}
                        x2={i * CELL_SIZE}
                        y2={BOARD_SIZE * CELL_SIZE}
                        stroke="#3E3E42"
                        strokeWidth={0.5}
                        opacity={0.3}
                      />
                      <line
                        x1={0}
                        y1={i * CELL_SIZE}
                        x2={BOARD_SIZE * CELL_SIZE}
                        y2={i * CELL_SIZE}
                        stroke="#3E3E42"
                        strokeWidth={0.5}
                        opacity={0.3}
                      />
                    </g>
                  ))}
                </svg>

                {/* Policy overlay grid */}
                <div
                  className="absolute inset-0 grid"
                  style={{
                    gridTemplateColumns: `repeat(${BOARD_SIZE}, ${CELL_SIZE}px)`,
                    gridTemplateRows: `repeat(${BOARD_SIZE}, ${CELL_SIZE}px)`,
                  }}
                >
                  {currentPolicyGrid.map((row, rowIndex) =>
                    row.map((value, colIndex) => {
                      if (isBinaryHeatmap) {
                        // Binary heatmap: 1.0 = legal (red), 0.0 = illegal (transparent/blue)
                        if (value === 1.0) {
                          return (
                            <div
                              key={`${rowIndex}-${colIndex}`}
                              className="border border-charcoal-700"
                              style={{
                                backgroundColor: 'rgba(255, 77, 77, 0.6)', // neon.red for legal positions
                              }}
                            />
                          );
                        } else {
                          return (
                            <div
                              key={`${rowIndex}-${colIndex}`}
                              className="border border-charcoal-700"
                              style={{
                                backgroundColor: 'rgba(0, 240, 255, 0.1)', // neon.blue with low opacity for illegal
                              }}
                            />
                          );
                        }
                      } else {
                        // Continuous policy: normalize and map to colors
                        const normalized = (value - minValue) / range;
                        const opacity = normalized;
                        const isHighProb = normalized > 0.5;
                        
                        return (
                          <div
                            key={`${rowIndex}-${colIndex}`}
                            className="border border-charcoal-700"
                            style={{
                              backgroundColor: isHighProb
                                ? `rgba(255, 77, 77, ${opacity})` // neon.red with opacity
                                : `rgba(0, 240, 255, ${1 - opacity})`, // neon.blue with inverse opacity
                            }}
                          />
                        );
                      }
                    })
                  )}
                </div>
              </div>
            </div>
            <div className="text-xs text-gray-400 space-y-1">
              {isBinaryHeatmap ? (
                <>
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-neon-red"></div>
                    <span>Legal Move Position</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-neon-blue opacity-30"></div>
                    <span>Illegal Position</span>
                  </div>
                </>
              ) : (
                <>
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-neon-red"></div>
                    <span>High Probability</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-neon-blue"></div>
                    <span>Low Probability</span>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {activeTab === 'value' && (
          <div className="space-y-3">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Value Function
            </h3>
            <div className="space-y-4">
              <div className="bg-charcoal-800 border border-charcoal-700 p-4">
                <div className="text-xs text-gray-400 mb-2">Win Probability</div>
                <div className="relative h-32 bg-charcoal-900 border border-charcoal-700">
                  <div
                    className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-neon-green to-neon-blue transition-all duration-500"
                    style={{
                      height: `${winProbability * 100}%`,
                    }}
                  >
                    <div className="absolute -top-6 left-0 right-0 text-center">
                      <span className="text-sm font-mono text-neon-green">
                        {(winProbability * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              <div className="text-xs text-gray-400">
                Estimated win probability for current game state
              </div>
            </div>
          </div>
        )}

        {activeTab === 'tree' && (
          <div className="space-y-3">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              MCTS Tree
            </h3>
            <div className="bg-charcoal-800 border border-charcoal-700 p-4 text-center">
              <div className="text-gray-400 text-sm py-8">
                Tree visualization coming soon
              </div>
            </div>
          </div>
        )}

        {activeTab === 'pieces' && (
          <div className="space-y-3">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Piece Selection
            </h3>
            {selectedPiece && (
              <div className="bg-charcoal-800 border border-charcoal-700 p-2 mb-2">
                <div className="text-xs text-gray-400">Selected</div>
                <div className="text-sm text-neon-blue font-mono">
                  {PIECE_NAMES[selectedPiece]} • {orientation}
                </div>
              </div>
            )}
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {availablePieces.map(pieceId => {
                const shape = getPieceShape(pieceId, orientation);
                const isSelected = selectedPiece === pieceId;
                
                const piecesUsed = gameState?.pieces_used || {};
                const currentPlayer = gameState?.current_player;
                const currentPlayerPiecesUsed = piecesUsed[currentPlayer] || [];
                const isPieceUsed = currentPlayerPiecesUsed.includes(pieceId);
                
                return (
                  <div
                    key={pieceId}
                    className={`
                      group relative p-2 border cursor-pointer transition-all
                      ${isSelected 
                        ? 'bg-charcoal-700 border-neon-blue' 
                        : isPieceUsed
                        ? 'bg-charcoal-900 border-charcoal-600 opacity-40 cursor-not-allowed'
                        : 'bg-charcoal-800 border-charcoal-700 hover:border-charcoal-600 hover:bg-charcoal-700'
                      }
                    `}
                    onClick={() => {
                      if (isPieceUsed || !onPieceSelect) return;
                      onPieceSelect(pieceId);
                    }}
                  >
                    <div className="flex items-center space-x-2">
                      {/* Piece visual */}
                      <div className="flex-shrink-0">
                        <svg
                          width={Math.min(shape[0]?.length * PIECE_SIZE, 32)}
                          height={Math.min(shape.length * PIECE_SIZE, 32)}
                          viewBox={`0 0 ${shape[0]?.length * PIECE_SIZE} ${shape.length * PIECE_SIZE}`}
                        >
                          {shape.map((row, rowIndex) =>
                            row.map((cell, colIndex) => (
                              cell === 1 && (
                                <rect
                                  key={`${rowIndex}-${colIndex}`}
                                  x={colIndex * PIECE_SIZE}
                                  y={rowIndex * PIECE_SIZE}
                                  width={PIECE_SIZE}
                                  height={PIECE_SIZE}
                                  fill={isSelected ? '#00F0FF' : '#64748B'}
                                  rx="1"
                                />
                              )
                            ))
                          )}
                        </svg>
                      </div>
                      
                      {/* Piece info */}
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-mono text-gray-300 truncate">
                          #{pieceId}
                        </div>
                        <div className="text-xs text-gray-400 truncate">
                          {PIECE_NAMES[pieceId]}
                        </div>
                      </div>
                      
                      {/* Used indicator */}
                      {isPieceUsed && (
                        <div className="text-xs text-neon-red">✕</div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
            {availablePieces.length === 0 && (
              <div className="text-center text-gray-400 text-sm py-8">
                All pieces used
              </div>
            )}
            <div className="text-xs text-gray-500 mt-2 space-y-1">
              <div>R: Rotate | F: Flip</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

