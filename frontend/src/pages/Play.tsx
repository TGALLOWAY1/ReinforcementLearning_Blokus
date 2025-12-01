import React, { useState, useCallback } from 'react';
import { useGameStore, useIsMyTurn } from '../store/gameStore';
import { Board } from '../components/Board';
import { PieceTray } from '../components/PieceTray';
import { Sidebar } from '../components/Sidebar';
import { ControlsPanel } from '../components/ControlsPanel';
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

  const handleAgentChange = useCallback((playerId: string, agentType: string) => {
    // This would require API support for changing agents mid-game
    console.log(`Change ${playerId} to ${agentType}`);
  }, []);

  if (!gameState) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="text-xl text-gray-600 mb-4">No game loaded</div>
          <div className="text-sm text-gray-500 mb-4">
            Connection Status: {useGameStore.getState().connectionStatus}
          </div>
          <button
            onClick={() => navigate('/')}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg"
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
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">
            {connectionStatus === 'connecting' ? 'Connecting to game...' : 'Connection lost'}
          </h1>
          <p className="text-gray-600 mb-8">
            {connectionStatus === 'connecting' 
              ? 'Please wait while we establish the connection.' 
              : 'Please refresh the page to reconnect.'}
          </p>
          {connectionStatus === 'connecting' && (
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
          )}
          {connectionStatus === 'disconnected' && (
            <button
              onClick={() => window.location.reload()}
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg"
            >
              Refresh Page
            </button>
          )}
          
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Modern Header */}
      <div className="glass border-b border-slate-200/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                <span className="text-white font-bold text-lg">B</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-800">Blokus Game</h1>
                <p className="text-sm text-slate-600 font-mono">
                  {gameState?.game_id ? gameState.game_id.slice(0, 8) + '...' : 'Unknown'}
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => navigate('/')}
                className="btn-secondary"
              >
                ← Back to Home
              </button>
            </div>
          </div>
        </div>
      </div>


      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border-l-4 border-red-400 p-4">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">{typeof error === 'string' ? error : JSON.stringify(error)}</p>
              </div>
              <div className="ml-auto pl-3">
                <button
                  onClick={() => setError(null)}
                  className="text-red-400 hover:text-red-600"
                >
                  <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Game Area */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Top Controls Panel */}
        <div className="mb-8">
          <ControlsPanel
            onNewGame={handleNewGame}
            onAgentChange={handleAgentChange}
            gameState={gameState}
          />
        </div>

        <div className="flex flex-col xl:flex-row gap-8">
          {/* Main Game Area - Board and Pieces */}
          <div className="flex-1 space-y-8">
            {/* Game Board */}
            <div className="card">
              <div className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-bold text-slate-800 flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    <span>Game Board</span>
                  </h2>
                  {isMyTurn ? (
                    <div className="bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm font-medium flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                      <span>Your turn!</span>
                    </div>
                  ) : (
                    <div className="bg-slate-100 text-slate-600 px-4 py-2 rounded-full text-sm font-medium">
                      Waiting for {gameState?.current_player?.toLowerCase() || 'player'} to play...
                    </div>
                  )}
                </div>
                <div className="flex justify-center">
                  <Board
                    onCellClick={handleCellClick}
                    onCellHover={handleCellHover}
                    selectedPiece={selectedPiece}
                    pieceOrientation={pieceOrientation}
                  />
                </div>
              </div>
            </div>

            {/* Piece Tray */}
            <div className="card">
              <div className="p-6">
                <PieceTray
                  onPieceSelect={selectPiece}
                  selectedPiece={selectedPiece}
                  pieceOrientation={pieceOrientation}
                  setPieceOrientation={setPieceOrientation}
                  gameState={gameState}
                />
              </div>
            </div>
          </div>

          {/* Sidebar - Game Status and Players */}
          <div className="xl:w-80">
            <Sidebar />
          </div>
        </div>
      </div>
    </div>
  );
};
