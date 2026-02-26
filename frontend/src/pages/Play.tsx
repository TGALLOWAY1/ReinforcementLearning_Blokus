import React, { useState, useCallback } from 'react';
import { useGameStore } from '../store/gameStore';
import { Board } from '../components/Board';
import { RightPanel } from '../components/RightPanel';
import { PieceTray } from '../components/PieceTray';
import { LogConsole } from '../components/LogConsole';
import { GameConfigModal } from '../components/GameConfigModal';
import { useNavigate } from 'react-router-dom';
import { IS_DEPLOY_PROFILE } from '../constants/gameConstants';

export const Play: React.FC = () => {
  const navigate = useNavigate();
  const {
    gameState,
    selectedPiece,
    pieceOrientation,
    selectPiece,
    setPieceOrientation,
    makeMove,
    passTurn,
    setError,
    error
  } = useGameStore();

  const [isMakingMove, setIsMakingMove] = useState(false);
  const [isPassing, setIsPassing] = useState(false);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [showInfoModal, setShowInfoModal] = useState(false);
  const [showLogConsole, setShowLogConsole] = useState(false);
  const isTelemetryOpen = useGameStore(s => s.activeRightTab === 'telemetry');

  // Call hooks at the top level

  // Determine if current player is human
  const currentPlayer = gameState?.current_player;
  const playerConfig = gameState?.players?.find((p: any) => p.player === currentPlayer);
  const isHumanPlayer = playerConfig?.agent_type === 'human' || !gameState?.players; // Default to human if no player config
  const legalMovesCount = gameState?.legal_moves?.length || 0;

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
    const { connectionStatus } = useGameStore.getState();
    if (connectionStatus !== 'connected') {
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
      console.log('[UI] Sending move', {
        gameId: gameState?.game_id,
        player: moveRequest.player,
        pieceId: moveRequest.piece_id,
        orientation: moveRequest.orientation,
        row: moveRequest.anchor_row,
        col: moveRequest.anchor_col,
      });
      const response = await makeMove(moveRequest);

      if (response && response.success) {
        console.log('✅ Move successful');
        // Clear the selected piece only after successful move
        selectPiece(null);
        setPieceOrientation(0);
      } else {
        // Move failed - display the specific error message from the backend
        console.log('❌ Move failed, response:', response);
        const errorMessage = response?.message || 'Move failed';
        console.log('📝 Error message from backend:', errorMessage);
        setError(errorMessage);
      }
    } catch (err) {
      console.error('Move error:', err);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsMakingMove(false);
    }
  }, [selectedPiece, isMakingMove, gameState, pieceOrientation, makeMove, selectPiece, setPieceOrientation, setError]);

  const handleCellHover = useCallback(() => {
    // Could add hover effects here
  }, []);



  const handleGameCreated = useCallback(() => {
    setShowConfigModal(false);
  }, []);

  const handlePassTurn = useCallback(async () => {
    if (isPassing || !currentPlayer) {
      return;
    }

    setIsPassing(true);
    setError(null);

    try {
      const response = await passTurn();
      if (response && response.success) {
        console.log('✅ Pass successful');
      } else {
        const errorMessage = response?.message || 'Pass failed';
        setError(errorMessage);
      }
    } catch (err) {
      console.error('Pass error:', err);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsPassing(false);
    }
  }, [currentPlayer, isPassing, passTurn, setError]);

  // Show game config modal if no game is loaded
  if (!gameState) {
    return (
      <div className="h-screen bg-charcoal-900 flex items-center justify-center">
        <GameConfigModal
          isOpen={showConfigModal || true}
          onClose={() => setShowConfigModal(false)}
          onGameCreated={handleGameCreated}
          canClose={false}
        />
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
    <div className="fixed h-screen w-screen bg-charcoal-900 flex overflow-hidden pr-4">
      {/* Left Column - PieceTray */}
      <aside className="w-80 border-r border-charcoal-700 bg-charcoal-900 flex flex-col overflow-y-auto">
        <PieceTray
          onPieceSelect={selectPiece}
          selectedPiece={selectedPiece}
          pieceOrientation={pieceOrientation}
          setPieceOrientation={setPieceOrientation}
          gameState={gameState}
        />
      </aside>

      {/* Center Column - Board and Log Console */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {gameState?.game_over && (
          <div className="w-full mb-4 bg-charcoal-800 border border-charcoal-700 p-4 flex items-center justify-between">
            <div className="text-gray-200">Game finished. Winner: <span className="font-semibold">{gameState.winner || 'None'}</span></div>
            {!IS_DEPLOY_PROFILE && (
              <div className="space-x-2">
                <button
                  onClick={() => navigate('/history')}
                  className="bg-charcoal-700 text-gray-200 px-4 py-2 rounded"
                >
                  History
                </button>
                <button
                  onClick={() => navigate(`/analysis/${gameState.game_id}`)}
                  className="bg-neon-blue text-black px-4 py-2 rounded"
                >
                  View Analysis
                </button>
              </div>
            )}
          </div>
        )}
        {/* Turn Indicator and Pass Button */}
        {gameState && !gameState.game_over && (
          <div className="w-full mb-4 bg-charcoal-800 border border-charcoal-700 p-4 flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${currentPlayer === 'RED' ? 'bg-red-500' :
                  currentPlayer === 'BLUE' ? 'bg-blue-500' :
                    currentPlayer === 'GREEN' ? 'bg-green-500' :
                      currentPlayer === 'YELLOW' ? 'bg-yellow-500' : 'bg-gray-500'
                  }`}></div>
                <span className="text-sm text-gray-200">
                  Current Player: <span className="font-semibold">{currentPlayer}</span>
                </span>
                {legalMovesCount > 0 && (
                  <span className="text-xs text-gray-400">
                    ({legalMovesCount} legal moves)
                  </span>
                )}
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setShowLogConsole(true)}
                className="p-2 text-gray-400 hover:text-gray-200 transition-colors"
                title="View Logs"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </button>
              <button
                onClick={() => setShowInfoModal(true)}
                className="p-2 text-gray-400 hover:text-gray-200 transition-colors"
                title="Game Information"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </button>
              {!IS_DEPLOY_PROFILE && (
                <button
                  onClick={() => setShowConfigModal(true)}
                  className="p-2 text-gray-400 hover:text-gray-200 transition-colors"
                  title="Game Settings"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </button>
              )}
              {isHumanPlayer && (
                <button
                  onClick={handlePassTurn}
                  disabled={isPassing || isMakingMove}
                  className={`px-4 py-2 text-sm font-medium border transition-colors ${isPassing || isMakingMove
                    ? 'bg-charcoal-900 border-charcoal-700 text-gray-500 cursor-not-allowed'
                    : legalMovesCount === 0
                      ? 'bg-charcoal-800 border-neon-yellow text-neon-yellow hover:bg-charcoal-700 hover:border-neon-yellow/80'
                      : 'bg-charcoal-800 border-charcoal-700 text-gray-200 hover:bg-charcoal-700 hover:border-charcoal-600'
                    }`}
                  title={legalMovesCount === 0 ? 'Pass turn (no moves available)' : 'Pass turn'}
                >
                  {isPassing ? 'Passing...' : legalMovesCount === 0 ? 'Pass Turn (No Moves)' : 'Pass Turn'}
                </button>
              )}
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="w-full mb-4 bg-charcoal-800 border border-neon-red p-4">
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

        {/* Board Container */}
        <div className="relative flex-1 flex items-center justify-center overflow-auto">
          <Board
            onCellClick={handleCellClick}
            onCellHover={handleCellHover}
            selectedPiece={selectedPiece}
            pieceOrientation={pieceOrientation}
          />
        </div>
      </main>

      {/* Right Column - Controls and Visualizations */}
      <aside className={`border-l border-charcoal-700 bg-charcoal-900 flex flex-col overflow-hidden transition-all duration-300 ${isTelemetryOpen ? 'w-[800px]' : 'w-96'}`}>
        <RightPanel onNewGame={() => setShowConfigModal(true)} />
      </aside>

      {/* Game Config Modal */}
      <GameConfigModal
        isOpen={showConfigModal}
        onClose={() => setShowConfigModal(false)}
        onGameCreated={handleGameCreated}
      />

      {/* Log Console Modal/Panel */}
      {showLogConsole && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-end justify-center z-50 p-4">
          <div className="bg-charcoal-800 border border-charcoal-700 rounded-t-lg w-full max-w-4xl h-[60vh] flex flex-col">
            <div className="flex items-center justify-between p-4 border-b border-charcoal-700">
              <h2 className="text-xl font-bold text-gray-200">Game Logs</h2>
              <button
                onClick={() => setShowLogConsole(false)}
                className="text-gray-400 hover:text-gray-200 transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="flex-1 overflow-hidden">
              <LogConsole />
            </div>
          </div>
        </div>
      )}

      {/* Info Modal */}
      {showInfoModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-charcoal-800 border border-charcoal-700 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-gray-200">How to Play Blokus</h2>
                <button
                  onClick={() => setShowInfoModal(false)}
                  className="text-gray-400 hover:text-gray-200 transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="space-y-4 text-gray-300">
                <div>
                  <h3 className="text-lg font-semibold text-neon-blue mb-2">Objective</h3>
                  <p>Place all your pieces on the board to score the most points.</p>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-neon-blue mb-2">Rules</h3>
                  <ul className="list-disc list-inside space-y-2 ml-2">
                    <li>Each player starts in their corner:
                      <ul className="list-circle list-inside ml-4 mt-1 space-y-1 text-sm text-gray-400">
                        <li>Red: top-left</li>
                        <li>Blue: top-right</li>
                        <li>Green: bottom-right</li>
                        <li>Yellow: bottom-left</li>
                      </ul>
                    </li>
                    <li>Your first piece must cover your starting corner</li>
                    <li>After that, pieces of the same color can only touch at corners (not edges)</li>
                    <li>You can place pieces adjacent to any opponent's pieces</li>
                    <li>Each square on a piece is worth 1 point, plus 15 bonus points if you place all 21 pieces</li>
                  </ul>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-neon-blue mb-2">Controls</h3>
                  <ul className="list-disc list-inside space-y-2 ml-2">
                    <li>Click a piece in the piece tray to select it</li>
                    <li>Click on the board to place the piece</li>
                    <li>Press <kbd className="px-2 py-1 bg-charcoal-900 border border-charcoal-700 rounded text-xs">R</kbd> to rotate the selected piece</li>
                    <li>Press <kbd className="px-2 py-1 bg-charcoal-900 border border-charcoal-700 rounded text-xs">F</kbd> to flip the selected piece</li>
                    <li>Click "Pass Turn" if you cannot make a move</li>
                  </ul>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-neon-blue mb-2">Scoring</h3>
                  <ul className="list-disc list-inside space-y-2 ml-2">
                    <li>1 point per square covered by your pieces</li>
                    <li>15 bonus points for using all 21 pieces</li>
                    <li>Additional bonuses for corner control and center control</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
