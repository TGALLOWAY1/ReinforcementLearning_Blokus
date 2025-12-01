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
    // Check if this cell is part of the piece preview (ghost piece)
    if (isPreviewCell(row, col)) {
      // Get the color for the selected piece based on current player
      const currentPlayer = gameState?.current_player;
      const colorMap: { [key: string]: string } = {
        'RED': PLAYER_COLORS.red,
        'BLUE': PLAYER_COLORS.blue,
        'GREEN': PLAYER_COLORS.green,
        'YELLOW': PLAYER_COLORS.yellow,
      };
      const playerColor = currentPlayer ? colorMap[currentPlayer] || PLAYER_COLORS.blue : PLAYER_COLORS.blue;
      return playerColor;
    }
    
    // Check if this cell is hovered
    if (hoveredCell && hoveredCell.row === row && hoveredCell.col === col) {
      return PLAYER_COLORS.hover;
    }
    
    // Return the normal cell color
    return getCellColor(row, col);
  };

  const getCellOpacity = (row: number, col: number) => {
    // Preview cells (ghost piece) should have 20% opacity fill
    if (isPreviewCell(row, col)) {
      return 0.2;
    }
    
    // Hovered cells should be slightly visible
    if (hoveredCell && hoveredCell.row === row && hoveredCell.col === col) {
      return 0.3;
    }
    
    // Normal cells are fully opaque
    return 1.0;
  };

  const getCellStroke = (row: number, col: number) => {
    // Ghost piece cells get neon border
    if (isPreviewCell(row, col)) {
      const currentPlayer = gameState?.current_player;
      const colorMap: { [key: string]: string } = {
        'RED': PLAYER_COLORS.red,
        'BLUE': PLAYER_COLORS.blue,
        'GREEN': PLAYER_COLORS.green,
        'YELLOW': PLAYER_COLORS.yellow,
      };
      const playerColor = currentPlayer ? colorMap[currentPlayer] || PLAYER_COLORS.blue : PLAYER_COLORS.blue;
      return playerColor;
    }
    return 'none';
  };

  const getCellStrokeWidth = (row: number, col: number) => {
    // Ghost piece cells get 2px border
    if (isPreviewCell(row, col)) {
      return 2;
    }
    return 0;
  };

  const hasPlacedPiece = (row: number, col: number) => {
    // Check if this cell has a placed piece (not empty, not preview, not hover)
    const cellColor = getCellColor(row, col);
    return cellColor !== PLAYER_COLORS.empty && !isPreviewCell(row, col);
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
    <div className="flex flex-col items-center justify-center h-full">
      <svg
        ref={svgRef}
        width={BOARD_SIZE * CELL_SIZE}
        height={BOARD_SIZE * CELL_SIZE}
        className="cursor-crosshair"
        style={{ background: 'transparent' }}
        onMouseMove={handleMouseMove}
        onClick={handleClick}
      >
        {/* Grid lines with charcoal styling */}
        {Array.from({ length: BOARD_SIZE + 1 }).map((_, i) => (
          <g key={i}>
            <line
              x1={i * CELL_SIZE}
              y1={0}
              x2={i * CELL_SIZE}
              y2={BOARD_SIZE * CELL_SIZE}
              stroke="#3E3E42"
              strokeWidth={0.5}
              opacity={0.5}
            />
            <line
              x1={0}
              y1={i * CELL_SIZE}
              x2={BOARD_SIZE * CELL_SIZE}
              y2={i * CELL_SIZE}
              stroke="#3E3E42"
              strokeWidth={0.5}
              opacity={0.5}
            />
          </g>
        ))}
        
        {/* Board cells */}
        {Array.from({ length: BOARD_SIZE }).map((_, row) =>
          Array.from({ length: BOARD_SIZE }).map((_, col) => {
            const cellColor = getCellFillColor(row, col);
            const opacity = getCellOpacity(row, col);
            const stroke = getCellStroke(row, col);
            const strokeWidth = getCellStrokeWidth(row, col);
            const hasPiece = hasPlacedPiece(row, col);
            
            return (
              <g key={`${row}-${col}`}>
                <rect
                  x={col * CELL_SIZE}
                  y={row * CELL_SIZE}
                  width={CELL_SIZE}
                  height={CELL_SIZE}
                  fill={cellColor}
                  opacity={opacity}
                  stroke={stroke}
                  strokeWidth={strokeWidth}
                  className="transition-all duration-200"
                  style={{
                    filter: hasPiece ? `drop-shadow(0 0 2px ${cellColor})` : 'none'
                  }}
                />
              </g>
            );
          })
        )}
        
        {/* Hovered cell highlight - only show if not part of piece preview */}
        {hoveredCell && !isPreviewCell(hoveredCell.row, hoveredCell.col) && (
          <rect
            x={hoveredCell.col * CELL_SIZE}
            y={hoveredCell.row * CELL_SIZE}
            width={CELL_SIZE}
            height={CELL_SIZE}
            fill={PLAYER_COLORS.hover}
            opacity={0.3}
            className="pointer-events-none"
          />
        )}
      </svg>
    </div>
  );
};
