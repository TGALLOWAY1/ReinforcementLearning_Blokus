import React, { useState, useCallback } from 'react';
import { useGameStore, useIsMyTurn } from '../store/gameStore';
import { Board } from '../components/Board';
import { ResearchSidebar } from '../components/ResearchSidebar';
import { AgentVisualizations } from '../components/AgentVisualizations';
import { LogConsole } from '../components/LogConsole';
import { useNavigate } from 'react-router-dom';

export const Play: React.FC = () => {
  const navigate = useNavigate();
  const { 
    gameState, 
    selectedPiece, 
    pieceOrientation,
    selectPiece, 
    setPieceOrientation,
    makeMove,
    setError,
    error
  } = useGameStore();
  
  const [isMakingMove, setIsMakingMove] = useState(false);
  
  // Call hooks at the top level
  const isMyTurn = useIsMyTurn();

  const handleCellClick = useCallback(async (row: number, col: number) => {
    if (!selectedPiece) {
      return;
    }
    
    if (isMakingMove) {
      return;
    }
    
    // Reset any previous error state
    setError(null);
    
    // Check if it's a human player's turn (not an agent)
    const currentPlayer = gameState?.current_player;
    const playerConfig = gameState?.players?.find((p: any) => p.player === currentPlayer);
    const isHumanPlayer = playerConfig?.agent_type === 'human';
    
    // Check if the selected piece is already used
    const piecesUsed = gameState?.pieces_used || {};
    const currentPlayerPiecesUsed = currentPlayer ? piecesUsed[currentPlayer] || [] : [];
    const isPieceAlreadyUsed = currentPlayerPiecesUsed.includes(selectedPiece);
    
    if (isPieceAlreadyUsed) {
      setError(`Piece ${selectedPiece} has already been used by ${currentPlayer}`);
      return;
    }
    
    // If we don't have player info, assume it's a human player for now
    if (gameState?.players && !isHumanPlayer) {
      setError('Only human players can make manual moves');
      return;
    }

    // Legal move validation is now handled by the backend
    
    // Check WebSocket connection
    const { websocket, connectionStatus } = useGameStore.getState();
    if (!websocket || connectionStatus !== 'connected') {
      setError('Not connected to game. Please refresh the page.');
      return;
    }
    
    setIsMakingMove(true);
    setError(null);


    try {
      const currentPlayer = gameState?.current_player || '';
      
      const moveRequest = {
        player: currentPlayer.toUpperCase(),
        piece_id: selectedPiece,
        orientation: pieceOrientation,
        anchor_row: row,
        anchor_col: col
      };
      
      // Send the move to the backend
      console.log('🎯 Sending move to backend:', moveRequest);
      const response = await makeMove(moveRequest);
      
      if (response && response.success) {
        console.log('✅ Move successful');
        // Clear the selected piece only after successful move
        selectPiece(null);
        setPieceOrientation(0);
      } else {
        // Move failed
        console.log('❌ Move failed');
        const errorMessage = response?.message || 'Unknown error occurred';
        setError(typeof errorMessage === 'string' ? errorMessage : JSON.stringify(errorMessage));
      }
    } catch (err) {
      console.error('Move error:', err);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsMakingMove(false);
    }
  }, [selectedPiece, isMakingMove, gameState, pieceOrientation, makeMove, selectPiece, setPieceOrientation, setError]);

  const handleCellHover = useCallback((_row: number, _col: number) => {
    // Could add hover effects here
  }, []);

  const handleNewGame = useCallback(() => {
    navigate('/');
  }, [navigate]);

  if (!gameState) {
    return (
      <div className="h-screen bg-charcoal-900 flex items-center justify-center">
        <div className="text-center">
          <div className="text-xl text-gray-200 mb-4">No game loaded</div>
          <div className="text-sm text-gray-400 mb-4">
            Connection Status: {useGameStore.getState().connectionStatus}
          </div>
          <button
            onClick={() => navigate('/')}
            className="bg-charcoal-800 border border-neon-blue hover:border-neon-blue/80 text-neon-blue px-6 py-2"
          >
            Go to Home
          </button>
        </div>
      </div>
    );
  }

  // Check if WebSocket is connected
  const connectionStatus = useGameStore.getState().connectionStatus;
  if (connectionStatus !== 'connected') {
    return (
      <div className="h-screen bg-charcoal-900 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-200 mb-4">
            {connectionStatus === 'connecting' ? 'Connecting to game...' : 'Connection lost'}
          </h1>
          <p className="text-gray-400 mb-8">
            {connectionStatus === 'connecting' 
              ? 'Please wait while we establish the connection.' 
              : 'Please refresh the page to reconnect.'}
          </p>
          {connectionStatus === 'connecting' && (
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-neon-blue mx-auto"></div>
          )}
          {connectionStatus === 'disconnected' && (
            <button
              onClick={() => window.location.reload()}
              className="bg-charcoal-800 border border-neon-blue hover:border-neon-blue/80 text-neon-blue px-6 py-2"
            >
              Refresh Page
            </button>
          )}
          
        </div>
      </div>
    );
  }

  return (
    <div 
      className="h-screen overflow-hidden grid"
      style={{
        gridTemplateColumns: '20rem 1fr 24rem',
        gridTemplateRows: '1fr 12rem',
        gridTemplateAreas: `
          "left-sidebar center right-sidebar"
          "bottom-bar bottom-bar bottom-bar"
        `
      }}
    >
      {/* Left Sidebar - Controls & Training Params */}
      <div className="panel h-full" style={{ gridArea: 'left-sidebar' }}>
        <div className="h-full overflow-y-auto">
          <ResearchSidebar onNewGame={handleNewGame} />
        </div>
      </div>

      {/* Center - Game Board */}
      <div className="h-full overflow-hidden" style={{ gridArea: 'center' }}>
        <div className="h-full flex flex-col items-center justify-center p-4">
          {/* Error Display */}
          {error && (
            <div className="w-full max-w-4xl mb-4 bg-charcoal-800 border border-neon-red p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-neon-red rounded-full"></div>
                  <p className="text-sm text-gray-200">{typeof error === 'string' ? error : JSON.stringify(error)}</p>
                </div>
                <button
                  onClick={() => setError(null)}
                  className="text-neon-red hover:text-neon-red/80"
                >
                  <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
            </div>
          )}

          {/* Game Board - Centered */}
          <div className="flex-1 flex items-center justify-center">
            <Board
              onCellClick={handleCellClick}
              onCellHover={handleCellHover}
              selectedPiece={selectedPiece}
              pieceOrientation={pieceOrientation}
            />
          </div>

          {/* Status Strip */}
          <div className="w-full max-w-4xl mt-4 bg-charcoal-800 border border-charcoal-700 p-3">
            <div className="flex items-center justify-between text-sm font-mono">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <span className="text-gray-400">Current Player:</span>
                  <span className="text-neon-blue font-semibold">
                    {gameState?.current_player?.toUpperCase() || 'N/A'}
                  </span>
                </div>
                <div className="w-px h-4 bg-charcoal-600"></div>
                <div className="flex items-center space-x-2">
                  <span className="text-gray-400">Move #:</span>
                  <span className="text-neon-green font-semibold">
                    {gameState?.move_count ?? 0}
                  </span>
                </div>
                <div className="w-px h-4 bg-charcoal-600"></div>
                <div className="flex items-center space-x-2">
                  <span className="text-gray-400">Confidence:</span>
                  <span className="text-neon-yellow font-semibold">
                    {gameState?.legal_moves && gameState.legal_moves.length > 0 
                      ? `${Math.min(100, Math.round((gameState.legal_moves.length / 100) * 100))}%`
                      : 'N/A'}
                  </span>
                </div>
              </div>
              {isMyTurn && (
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-neon-green rounded-full animate-pulse"></div>
                  <span className="text-neon-green text-xs">Your Turn</span>
                </div>
              )}
            </div>
          </div>

        </div>
      </div>

      {/* Right Sidebar - Visualizations (Heatmaps/Trees) */}
      <div className="panel h-full" style={{ gridArea: 'right-sidebar' }}>
        <AgentVisualizations
          onPieceSelect={selectPiece}
          selectedPiece={selectedPiece}
          pieceOrientation={pieceOrientation}
          setPieceOrientation={setPieceOrientation}
        />
      </div>

      {/* Bottom Bar - Log Console */}
      <div className="panel h-full" style={{ gridArea: 'bottom-bar' }}>
        <LogConsole />
      </div>
    </div>
  );
};
