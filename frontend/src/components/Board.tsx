import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react';
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

// Memoized cell component to prevent unnecessary re-renders
const CellRect = React.memo<{
  row: number;
  col: number;
  cellColor: string;
  opacity: number;
  stroke: string;
  strokeWidth: number;
  hasPiece: boolean;
}>(({ row, col, cellColor, opacity, stroke, strokeWidth, hasPiece }) => (
  <g>
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
));
CellRect.displayName = 'CellRect';

export const Board: React.FC<BoardProps> = ({
  onCellClick,
  onCellHover,
  selectedPiece,
  pieceOrientation
}) => {
  const { gameState, previewMove, setPreviewMove } = useGameStore();
  const [hoveredCell, setHoveredCell] = useState<{row: number, col: number} | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  
  // Performance instrumentation - measure render time
  useEffect(() => {
    console.time('[UI] Board render');
    return () => {
      console.timeEnd('[UI] Board render');
    };
  });
  
  // Performance instrumentation - measure effect time when gameState changes
  useEffect(() => {
    if (gameState) {
      console.time('[UI] Board effect (gameState change)');
      // Any async operations or computations would go here
      console.timeEnd('[UI] Board effect (gameState change)');
    }
  }, [gameState]);
  

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
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const col = Math.floor((event.clientX - rect.left) / CELL_SIZE);
    const row = Math.floor((event.clientY - rect.top) / CELL_SIZE);
    if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) return;

    if (previewMove) setPreviewMove(null);

    // Anchor = top-left of piece. Preview uses hoveredCell as anchor, so when user
    // clicks a preview cell, pass hoveredCell (the anchor), not the clicked cell.
    if (selectedPiece && hoveredCell) {
      const previewPositions = calculatePiecePositions(
        selectedPiece,
        pieceOrientation,
        hoveredCell.row,
        hoveredCell.col
      );
      const isOnPreview = previewPositions.some((p) => p.row === row && p.col === col);
      if (isOnPreview) {
        onCellClick(hoveredCell.row, hoveredCell.col);
        return;
      }
    }
    onCellClick(row, col);
  }, [onCellClick, selectedPiece, pieceOrientation, hoveredCell, previewMove, setPreviewMove]);

  // Memoize cell color calculation to avoid recomputation
  const cellColors = useMemo(() => {
    if (!gameState?.board) return null;
    const colors: (string | null)[][] = [];
    for (let row = 0; row < BOARD_SIZE; row++) {
      colors[row] = [];
      for (let col = 0; col < BOARD_SIZE; col++) {
        const cell = gameState.board[row]?.[col];
        if (!cell || cell === 0) {
          colors[row][col] = PLAYER_COLORS.empty;
        } else {
          const colorMap = {
            1: 'red',    // RED -> red
            2: 'blue',   // BLUE -> blue
            3: 'yellow', // YELLOW -> yellow
            4: 'green'   // GREEN -> green
          };
          const colorName = colorMap[cell as keyof typeof colorMap];
          colors[row][col] = PLAYER_COLORS[colorName as keyof typeof PLAYER_COLORS] || PLAYER_COLORS.empty;
        }
      }
    }
    return colors;
  }, [gameState?.board]);

  const getCellColor = useCallback((row: number, col: number) => {
    if (!cellColors) return PLAYER_COLORS.empty;
    return cellColors[row]?.[col] || PLAYER_COLORS.empty;
  }, [cellColors]);

  const isPreviewCell = (row: number, col: number) => {
    return piecePreview.some(pos => pos.row === row && pos.col === col);
  };

  const getCellFillColor = (row: number, col: number) => {
    // Preview cells (ghost piece) should be transparent (hollow)
    if (isPreviewCell(row, col)) {
      return 'transparent';
    }
    
    // Return the normal cell color
    return getCellColor(row, col);
  };

  const getCellOpacity = (row: number, col: number) => {
    // Preview cells are transparent (hollow), so opacity doesn't matter
    if (isPreviewCell(row, col)) {
      return 1.0;
    }
    
    // Normal cells are fully opaque
    return 1.0;
  };

  const getCellStroke = (row: number, col: number) => {
    // Ghost piece cells get preview color stroke only (hollow rectangle)
    if (isPreviewCell(row, col)) {
      return PLAYER_COLORS.preview;
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
    // MCTS table click preview takes precedence
    if (previewMove) {
      const positions = calculatePiecePositions(
        previewMove.piece_id,
        previewMove.orientation,
        previewMove.anchor_row,
        previewMove.anchor_col
      );
      return positions.filter(
        (pos) =>
          pos.row >= 0 &&
          pos.row < BOARD_SIZE &&
          pos.col >= 0 &&
          pos.col < BOARD_SIZE
      );
    }
    if (!selectedPiece || !hoveredCell) return [];

    // Calculate piece positions based on shape and orientation
    const positions = calculatePiecePositions(selectedPiece, pieceOrientation, hoveredCell.row, hoveredCell.col);

    // Filter out positions that are outside the board
    const validPositions = positions.filter(
      (pos) =>
        pos.row >= 0 &&
        pos.row < BOARD_SIZE &&
        pos.col >= 0 &&
        pos.col < BOARD_SIZE
    );

    return validPositions;
  };

  const piecePreview = getPiecePreview();

  return (
    <div className="flex flex-col items-center justify-center h-full bg-transparent">
      <svg
        ref={svgRef}
        width={BOARD_SIZE * CELL_SIZE}
        height={BOARD_SIZE * CELL_SIZE}
        className="cursor-crosshair bg-charcoal-900/50"
        onMouseMove={handleMouseMove}
        onClick={handleClick}
      >
        {/* Grid lines - higher contrast against dark background */}
        {Array.from({ length: BOARD_SIZE + 1 }).map((_, i) => (
          <g key={i}>
            <line
              x1={i * CELL_SIZE}
              y1={0}
              x2={i * CELL_SIZE}
              y2={BOARD_SIZE * CELL_SIZE}
              stroke="#525252"
              strokeWidth={1}
            />
            <line
              x1={0}
              y1={i * CELL_SIZE}
              x2={BOARD_SIZE * CELL_SIZE}
              y2={i * CELL_SIZE}
              stroke="#525252"
              strokeWidth={1}
            />
          </g>
        ))}
        
        {/* Board cells - memoized to prevent unnecessary re-renders */}
        {Array.from({ length: BOARD_SIZE }).map((_, row) =>
          Array.from({ length: BOARD_SIZE }).map((_, col) => {
            const cellColor = getCellFillColor(row, col);
            const opacity = getCellOpacity(row, col);
            const stroke = getCellStroke(row, col);
            const strokeWidth = getCellStrokeWidth(row, col);
            const hasPiece = hasPlacedPiece(row, col);
            
            return (
              <CellRect
                key={`${row}-${col}`}
                row={row}
                col={col}
                cellColor={cellColor}
                opacity={opacity}
                stroke={stroke}
                strokeWidth={strokeWidth}
                hasPiece={hasPiece}
              />
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
            fill={PLAYER_COLORS.grid}
            opacity={0.3}
            className="pointer-events-none"
          />
        )}
      </svg>
    </div>
  );
};
