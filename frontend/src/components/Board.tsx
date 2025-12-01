import React, { useState, useCallback, useRef } from 'react';
import { useGameStore } from '../store/gameStore';
import { PLAYER_COLORS, BOARD_SIZE, CELL_SIZE } from '../constants/gameConstants';
import { calculatePiecePositions } from '../utils/pieceUtils';

interface BoardProps {
  onCellClick: (row: number, col: number) => void;
  onCellHover: (row: number, col: number) => void;
  selectedPiece: number | null;
  pieceOrientation: number;
}

// Constants are now imported from shared constants file

export const Board: React.FC<BoardProps> = ({
  onCellClick,
  onCellHover,
  selectedPiece,
  pieceOrientation
}) => {
  const { gameState } = useGameStore();
  const [hoveredCell, setHoveredCell] = useState<{row: number, col: number} | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  

  const handleMouseMove = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;
    
    const rect = svgRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    const col = Math.floor(x / CELL_SIZE);
    const row = Math.floor(y / CELL_SIZE);
    
    if (row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE) {
      setHoveredCell({ row, col });
      onCellHover(row, col);
    } else {
      setHoveredCell(null);
    }
  }, [onCellHover]);

  const handleClick = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) {
      return;
    }
    
    const rect = svgRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    const col = Math.floor(x / CELL_SIZE);
    const row = Math.floor(y / CELL_SIZE);
    
    if (row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE) {
      onCellClick(row, col);
    }
  }, [onCellClick]);

  const getCellColor = (row: number, col: number) => {
    if (!gameState?.board) return PLAYER_COLORS.empty;
    
    const cell = gameState.board[row]?.[col];
    if (!cell || cell === 0) return PLAYER_COLORS.empty;
    
    // Convert numeric values to color names
    const colorMap = {
      1: 'red',    // RED
      2: 'blue',   // BLUE  
      3: 'green',  // GREEN
      4: 'yellow'  // YELLOW
    };
    
    const colorName = colorMap[cell as keyof typeof colorMap];
    const color = PLAYER_COLORS[colorName as keyof typeof PLAYER_COLORS] || PLAYER_COLORS.empty;
    
    return color;
  };

  const isPreviewCell = (row: number, col: number) => {
    return piecePreview.some(pos => pos.row === row && pos.col === col);
  };

  const getCellFillColor = (row: number, col: number) => {
    // Check if this cell is part of the piece preview
    if (isPreviewCell(row, col)) {
      return PLAYER_COLORS.preview;
    }
    
    // Check if this cell is hovered
    if (hoveredCell && hoveredCell.row === row && hoveredCell.col === col) {
      return PLAYER_COLORS.hover;
    }
    
    // Return the normal cell color
    return getCellColor(row, col);
  };

  const getCellOpacity = (row: number, col: number) => {
    // Preview cells should be semi-transparent
    if (isPreviewCell(row, col)) {
      return 0.6;
    }
    
    // Hovered cells should be slightly transparent
    if (hoveredCell && hoveredCell.row === row && hoveredCell.col === col) {
      return 0.4;
    }
    
    // Normal cells are fully opaque
    return 1.0;
  };


  // Piece utilities are now imported from shared utils

  const getPiecePreview = () => {
    if (!selectedPiece || !hoveredCell) return [];
    
    // Calculate piece positions based on shape and orientation
    const positions = calculatePiecePositions(selectedPiece, pieceOrientation, hoveredCell.row, hoveredCell.col);
    
    // Filter out positions that are outside the board
    const validPositions = positions.filter(pos => 
      pos.row >= 0 && pos.row < BOARD_SIZE && 
      pos.col >= 0 && pos.col < BOARD_SIZE
    );
    
    return validPositions;
  };

  const piecePreview = getPiecePreview();

  return (
    <div className="flex flex-col items-center space-y-6">
      {/* Board container with modern styling */}
      <div className="relative">
        {/* Board shadow and glow effect */}
        <div className="absolute -inset-4 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-pink-500/20 rounded-2xl blur-xl opacity-30"></div>
        
        {/* Board card */}
        <div className="relative card p-6">
          <svg
            ref={svgRef}
            width={BOARD_SIZE * CELL_SIZE}
            height={BOARD_SIZE * CELL_SIZE}
            className="cursor-crosshair drop-shadow-lg"
            onMouseMove={handleMouseMove}
            onClick={handleClick}
          >
            {/* Grid lines with subtle styling */}
            {Array.from({ length: BOARD_SIZE + 1 }).map((_, i) => (
              <g key={i}>
                <line
                  x1={i * CELL_SIZE}
                  y1={0}
                  x2={i * CELL_SIZE}
                  y2={BOARD_SIZE * CELL_SIZE}
                  stroke={PLAYER_COLORS.grid}
                  strokeWidth={0.5}
                  opacity={0.6}
                />
                <line
                  x1={0}
                  y1={i * CELL_SIZE}
                  x2={BOARD_SIZE * CELL_SIZE}
                  y2={i * CELL_SIZE}
                  stroke={PLAYER_COLORS.grid}
                  strokeWidth={0.5}
                  opacity={0.6}
                />
              </g>
            ))}
            
            {/* Board cells with modern styling */}
            {Array.from({ length: BOARD_SIZE }).map((_, row) =>
              Array.from({ length: BOARD_SIZE }).map((_, col) => (
                <rect
                  key={`${row}-${col}`}
                  x={col * CELL_SIZE}
                  y={row * CELL_SIZE}
                  width={CELL_SIZE}
                  height={CELL_SIZE}
                  fill={getCellFillColor(row, col)}
                  opacity={getCellOpacity(row, col)}
                  stroke={PLAYER_COLORS.grid}
                  strokeWidth={0.5}
                  className="transition-all duration-200 hover:opacity-80"
                  rx="1"
                />
              ))
            )}
            
            {/* Hovered cell highlight with modern effect - only show if not part of piece preview */}
            {hoveredCell && !isPreviewCell(hoveredCell.row, hoveredCell.col) && (
              <rect
                x={hoveredCell.col * CELL_SIZE}
                y={hoveredCell.row * CELL_SIZE}
                width={CELL_SIZE}
                height={CELL_SIZE}
                fill={PLAYER_COLORS.hover}
                opacity={0.4}
                rx="2"
                className="pointer-events-none animate-pulse"
              />
            )}
            
          </svg>
        </div>
      </div>

      {/* Board info */}
      <div className="text-center space-y-2">
        <h3 className="text-lg font-semibold text-slate-700">Game Board</h3>
        <p className="text-sm text-slate-500">
          Click on a cell to place your selected piece
        </p>
      </div>
    </div>
  );
};
