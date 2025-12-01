import React, { useState } from 'react';
import { useGameStore } from '../store/gameStore';
import { PIECE_NAMES, PIECE_SIZE } from '../constants/gameConstants';
import { getPieceShape } from '../utils/pieceUtils';

interface PieceTrayProps {
  onPieceSelect: (pieceId: number) => void;
  selectedPiece: number | null;
  pieceOrientation?: number;
  setPieceOrientation?: (orientation: number) => void;
  gameState?: any;
}

// Constants are now imported from shared constants file

export const PieceTray: React.FC<PieceTrayProps> = ({
  onPieceSelect,
  selectedPiece,
  pieceOrientation = 0,
  setPieceOrientation,
  gameState: propGameState
}) => {
  const { gameState: storeGameState } = useGameStore();
  const gameState = propGameState || storeGameState;
  const [localOrientation, setLocalOrientation] = useState(0);
  
  // Use prop orientation if available, otherwise use local state
  const orientation = pieceOrientation !== undefined ? pieceOrientation : localOrientation;
  const setOrientation = setPieceOrientation || setLocalOrientation;

  const currentPlayer = gameState?.current_player;
  const piecesUsed = gameState?.pieces_used?.[currentPlayer || ''] || [];
  const availablePieces = Array.from({length: 21}, (_, i) => i + 1).filter(pieceId => !piecesUsed.includes(pieceId));

  // Piece utility functions are now imported from shared utils

  React.useEffect(() => {
    const handleGlobalKeyDown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }
      
      if (event.key === 'r' || event.key === 'R') {
        event.preventDefault();
        // Rotate: 0->1->2->3->0 or 4->5->6->7->4
        if (orientation < 4) {
          setOrientation((orientation + 1) % 4);
        } else {
          setOrientation(4 + ((orientation - 4 + 1) % 4));
        }
      } else if (event.key === 'f' || event.key === 'F') {
        event.preventDefault();
        // Flip: 0-3 <-> 4-7
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

  return (
    <div className="h-full w-full flex flex-col">
      {/* Header */}
      <div className="px-3 py-2 border-b border-charcoal-700 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <h3 className="text-xs font-semibold text-gray-300 uppercase tracking-wider">
            Pieces
          </h3>
          <span className="text-xs text-gray-400 bg-charcoal-800 px-2 py-0.5 rounded">
            {availablePieces.length}
          </span>
        </div>
        {selectedPiece && (
          <div className="text-xs text-gray-400">
            {PIECE_NAMES[selectedPiece]} • {orientation}
          </div>
        )}
      </div>

      {/* Controls hint */}
      <div className="px-3 py-1.5 border-b border-charcoal-700 flex items-center justify-center space-x-4 text-xs text-gray-500">
        <div className="flex items-center space-x-1">
          <kbd className="px-1.5 py-0.5 bg-charcoal-800 border border-charcoal-700 rounded text-xs font-mono">R</kbd>
          <span>Rotate</span>
        </div>
        <div className="flex items-center space-x-1">
          <kbd className="px-1.5 py-0.5 bg-charcoal-800 border border-charcoal-700 rounded text-xs font-mono">F</kbd>
          <span>Flip</span>
        </div>
      </div>
      
      {/* Scrollable pieces grid */}
      <div className="flex-1 overflow-y-auto p-2">
        <div className="grid grid-cols-3 gap-2">
        {availablePieces.map(pieceId => {
          const shape = getPieceShape(pieceId, orientation);
          const isSelected = selectedPiece === pieceId;
          
          // Check if piece is already used
          const piecesUsed = gameState?.pieces_used || {};
          const currentPlayer = gameState?.current_player;
          const currentPlayerPiecesUsed = piecesUsed[currentPlayer] || [];
          const isPieceUsed = currentPlayerPiecesUsed.includes(pieceId);
          
          return (
            <div
              key={pieceId}
              className={`
                group relative bg-charcoal-800 border rounded-md p-2 cursor-pointer transition-all overflow-hidden
                ${isSelected 
                  ? 'border-neon-blue ring-2 ring-neon-blue ring-opacity-50' 
                  : isPieceUsed
                  ? 'border-charcoal-600 opacity-50 cursor-not-allowed'
                  : 'border-charcoal-700 hover:border-charcoal-600 hover:bg-charcoal-750'
                }
              `}
              onClick={() => {
                if (isPieceUsed) {
                  console.log('❌ Piece already used:', pieceId);
                  return;
                }
                console.log('🧩 Piece clicked:', pieceId, 'Current selected:', selectedPiece);
                onPieceSelect(pieceId);
              }}
            >
              {/* Used piece indicator */}
              {isPieceUsed && (
                <div className="absolute top-1 right-1 text-neon-red text-xs font-bold z-10">
                  ✕
                </div>
              )}
              
              {/* Piece visual */}
              <div className="flex justify-center items-center mb-1.5 min-h-[48px]">
                <svg
                  width={Math.min(shape[0]?.length * PIECE_SIZE, 40)}
                  height={Math.min(shape.length * PIECE_SIZE, 40)}
                  viewBox={`0 0 ${shape[0]?.length * PIECE_SIZE} ${shape.length * PIECE_SIZE}`}
                  className="max-w-full max-h-full"
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
                          fill={isSelected ? '#00F0FF' : isPieceUsed ? '#64748B' : '#94A3B8'}
                          rx="1"
                          className="transition-colors duration-200"
                        />
                      )
                    ))
                  )}
                </svg>
              </div>
              
              {/* Piece ID */}
              <div className="text-xs text-center text-gray-400 font-mono">
                #{pieceId}
              </div>
              
              {/* Selection indicator */}
              {isSelected && (
                <div className="absolute top-0 right-0 w-2 h-2 bg-neon-blue rounded-full border border-charcoal-900"></div>
              )}
            </div>
          );
        })}
        </div>
        
        {availablePieces.length === 0 && (
          <div className="text-center text-gray-500 py-16">
            <div className="text-4xl mb-4">🧩</div>
            <p className="text-sm font-medium text-gray-400">All pieces used</p>
          </div>
        )}
      </div>
    </div>
  );
};
