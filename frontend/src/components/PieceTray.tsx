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
    <div className="space-y-6">
      {/* Header with modern styling */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-bold text-slate-800 flex items-center space-x-2">
            <div className="w-2 h-2 bg-indigo-400 rounded-full"></div>
            <span>Your Pieces</span>
            <span className="bg-slate-100 text-slate-600 px-2 py-1 rounded-full text-sm font-medium">
              {availablePieces.length}
            </span>
          </h3>
          {selectedPiece && (
            <p className="text-sm text-slate-600 mt-2 bg-blue-50 px-3 py-2 rounded-lg border border-blue-200">
              <span className="font-medium">Selected:</span> {PIECE_NAMES[selectedPiece]} • 
              <span className="font-medium ml-1">Orientation:</span> {orientation}
            </p>
          )}
        </div>
        <div className="flex items-center space-x-4 text-xs text-slate-500">
          <div className="flex items-center space-x-2 bg-slate-100 px-3 py-2 rounded-lg">
            <kbd className="px-2 py-1 bg-white border border-slate-200 rounded text-xs font-mono shadow-sm">R</kbd>
            <span className="font-medium">Rotate</span>
          </div>
          <div className="flex items-center space-x-2 bg-slate-100 px-3 py-2 rounded-lg">
            <kbd className="px-2 py-1 bg-white border border-slate-200 rounded text-xs font-mono shadow-sm">F</kbd>
            <span className="font-medium">Flip</span>
          </div>
        </div>
      </div>
      
      {/* Pieces grid with modern tile styling */}
      <div className="grid grid-cols-6 gap-2 max-h-96 overflow-y-auto pr-2">
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
                group relative p-3 rounded-xl cursor-pointer transition-all duration-300 transform
                ${isSelected 
                  ? 'bg-gradient-to-br from-blue-50 to-blue-100 ring-2 ring-blue-400 shadow-lg shadow-blue-200/50 scale-105' 
                  : isPieceUsed
                  ? 'bg-gradient-to-br from-red-50 to-red-100 ring-2 ring-red-300 opacity-60 cursor-not-allowed'
                  : 'bg-white hover:bg-gradient-to-br hover:from-slate-50 hover:to-slate-100 hover:shadow-lg hover:shadow-slate-200/50 hover:scale-105'
                }
                ${!availablePieces.includes(pieceId) ? 'opacity-50 cursor-not-allowed' : ''}
                active:scale-95
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
              {/* Piece name tooltip */}
              <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 bg-slate-800 text-white text-xs px-3 py-2 rounded-lg opacity-0 group-hover:opacity-100 transition-all duration-200 pointer-events-none whitespace-nowrap z-10 shadow-lg">
                {PIECE_NAMES[pieceId]}
                {isPieceUsed && <span className="text-red-300 ml-1">(Used)</span>}
                <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-800"></div>
              </div>
              
              {/* Used piece indicator */}
              {isPieceUsed && (
                <div className="absolute top-1 right-1 bg-red-500 text-white text-xs px-1.5 py-0.5 rounded-full font-bold">
                  ✕
                </div>
              )}
              
              {/* Piece visual with enhanced styling */}
              <div className="flex justify-center items-center h-16 mb-2">
                <div className="p-2 bg-slate-50 rounded-lg">
                  <svg
                    width={Math.min(shape[0]?.length * PIECE_SIZE, 48)}
                    height={Math.min(shape.length * PIECE_SIZE, 48)}
                    viewBox={`0 0 ${shape[0]?.length * PIECE_SIZE} ${shape.length * PIECE_SIZE}`}
                    className="drop-shadow-sm"
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
                            fill={isSelected ? '#3B82F6' : '#64748B'}
                            rx="2"
                            className="transition-all duration-200"
                          />
                        )
                      ))
                    )}
                  </svg>
                </div>
              </div>
              
              {/* Piece ID indicator with modern styling */}
              <div className="text-xs text-center text-slate-500 font-mono bg-slate-100 px-2 py-1 rounded-full">
                #{pieceId}
              </div>
              
              {/* Selection indicator */}
              {isSelected && (
                <div className="absolute -top-1 -right-1 w-4 h-4 bg-blue-500 rounded-full border-2 border-white shadow-lg animate-pulse"></div>
              )}
            </div>
          );
        })}
      </div>
      
      {availablePieces.length === 0 && (
        <div className="text-center text-slate-500 py-16">
          <div className="text-6xl mb-4">🧩</div>
          <p className="text-lg font-medium">All pieces have been used!</p>
          <p className="text-sm mt-2">Great job completing the game!</p>
        </div>
      )}
    </div>
  );
};
