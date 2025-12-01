import React, { useState } from 'react';
import { useGameStore } from '../store/gameStore';
import { API_BASE } from '../constants/gameConstants';
import { useNavigate } from 'react-router-dom';

export const Home: React.FC = () => {
  const navigate = useNavigate();
  const { createGame, connect } = useGameStore();
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [gameConfig, setGameConfig] = useState({
    players: [
      { player: 'RED', agent_type: 'human', agent_config: {} },
      { player: 'BLUE', agent_type: 'random', agent_config: {} },
      { player: 'GREEN', agent_type: 'heuristic', agent_config: {} },
      { player: 'YELLOW', agent_type: 'mcts', agent_config: {} }
    ],
    auto_start: true
  });

  const handleCreateGame = async () => {
    console.log('🎮 Starting game creation...');
    setIsCreating(true);
    setError(null);

    try {
      console.log('📋 Creating game with config:', gameConfig);
      const gameId = await createGame(gameConfig);
      console.log('🎯 Game created with ID:', gameId);
      
      console.log('Connecting to WebSocket...');
      await connect(gameId);
      console.log('WebSocket connected, navigating to play page');
      
      // Check store state before navigation
      const storeState = useGameStore.getState();
      console.log('Store state after connection:', {
        gameState: storeState.gameState,
        connectionStatus: storeState.connectionStatus,
        error: storeState.error
      });
      
      // If no game state, try to fetch it via REST API as fallback
      if (!storeState.gameState) {
        console.log('No game state from WebSocket, fetching via REST API...');
        try {
          const response = await fetch(`${API_BASE}/api/games/${gameId}`);
          if (response.ok) {
            const gameState = await response.json();
            console.log('Game state from REST API:', gameState);
            useGameStore.getState().setGameState(gameState);
          }
        } catch (err) {
          console.error('Failed to fetch game state via REST API:', err);
        }
      }
      
      navigate('/play');
    } catch (err) {
      console.error('Error creating game:', err);
      setError(err instanceof Error ? err.message : 'Failed to create game');
    } finally {
      setIsCreating(false);
    }
  };

  const updatePlayer = (index: number, field: string, value: string) => {
    setGameConfig(prev => ({
      ...prev,
      players: prev.players.map((player, i) => 
        i === index ? { ...player, [field]: value } : player
      )
    }));
  };

  const addPlayer = () => {
    if (gameConfig.players.length < 4) {
      const playerColors = ['RED', 'BLUE', 'GREEN', 'YELLOW'];
      const nextColor = playerColors[gameConfig.players.length];
      setGameConfig(prev => ({
        ...prev,
        players: [...prev.players, { player: nextColor, agent_type: 'random', agent_config: {} }]
      }));
    }
  };

  const removePlayer = (index: number) => {
    if (gameConfig.players.length > 2) {
      setGameConfig(prev => ({
        ...prev,
        players: prev.players.filter((_, i) => i !== index)
      }));
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        <div className="bg-white rounded-xl shadow-xl p-8">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-800 mb-2">
              🎮 Blokus RL
            </h1>
            <p className="text-gray-600 mb-6">
              Play against AI agents in this strategic board game
            </p>
            
            {/* Game Description */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-8">
              <h2 className="text-xl font-semibold text-blue-800 mb-4">🎯 How to Play Blokus</h2>
              <div className="text-sm text-blue-700 space-y-3">
                <div>
                  <strong>Objective:</strong> Place all your pieces on the board to score the most points.
                </div>
                <div>
                  <strong>Rules:</strong>
                  <ul className="list-disc list-inside mt-2 space-y-1">
                    <li>Each player starts in their corner (Red: top-left, Blue: top-right, Green: bottom-right, Yellow: bottom-left)</li>
                    <li>Your first piece must cover your starting corner</li>
                    <li>After that, pieces of the same color can only touch at corners (not edges)</li>
                    <li>You can place pieces adjacent to any opponent's pieces</li>
                    <li>Each square on a piece is worth 1 point, plus 15 bonus points if you place all 21 pieces</li>
                  </ul>
                </div>
                <div>
                  <strong>Controls:</strong> Click a piece to select it, then click on the board to place it. Use R to rotate and F to flip pieces.
                </div>
              </div>
            </div>
            <div className="flex justify-center space-x-4">
              <button
                onClick={() => navigate('/train')}
                className="text-blue-600 hover:text-blue-800 text-sm"
              >
                📊 Training & Evaluation
              </button>
            </div>
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
              <div className="text-red-800">{error}</div>
            </div>
          )}

          <div className="space-y-6">
            {/* Game Configuration */}
            <div>
              <h2 className="text-xl font-semibold text-gray-800 mb-4">
                Game Configuration
              </h2>
              
              <div className="space-y-4">
                {/* Players */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Players ({gameConfig.players.length}/4)
                  </label>
                  <div className="space-y-3">
                    {gameConfig.players.map((player, index) => (
                      <div key={index} className="flex items-center space-x-3">
                        <div className="w-20">
                          <select
                            value={player.player}
                            onChange={(e) => updatePlayer(index, 'player', e.target.value)}
                            className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                          >
                            <option value="RED">Red</option>
                            <option value="BLUE">Blue</option>
                            <option value="GREEN">Green</option>
                            <option value="YELLOW">Yellow</option>
                          </select>
                        </div>
                        <div className="flex-1">
                          <select
                            value={player.agent_type}
                            onChange={(e) => updatePlayer(index, 'agent_type', e.target.value)}
                            className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                          >
                            <option value="human">Human</option>
                            <option value="random">Random Agent</option>
                            <option value="heuristic">Heuristic Agent</option>
                            <option value="mcts">MCTS Agent</option>
                          </select>
                        </div>
                        {gameConfig.players.length > 2 && (
                          <button
                            onClick={() => removePlayer(index)}
                            className="text-red-600 hover:text-red-800 text-sm"
                          >
                            Remove
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                  
                  {gameConfig.players.length < 4 && (
                    <button
                      onClick={addPlayer}
                      className="mt-2 text-blue-600 hover:text-blue-800 text-sm"
                    >
                      + Add Player
                    </button>
                  )}
                </div>

                {/* Game Settings */}
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="auto_start"
                    checked={gameConfig.auto_start}
                    onChange={(e) => setGameConfig(prev => ({
                      ...prev,
                      auto_start: e.target.checked
                    }))}
                    className="mr-2"
                  />
                  <label htmlFor="auto_start" className="text-sm text-gray-700">
                    Auto-start game
                  </label>
                </div>
              </div>
            </div>

            {/* Create Game Button */}
            <button
              onClick={handleCreateGame}
              disabled={isCreating}
              className={`
                w-full py-3 px-6 rounded-lg font-medium text-white transition-colors duration-200
                ${isCreating 
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : 'bg-blue-600 hover:bg-blue-700'
                }
              `}
            >
              {isCreating ? 'Creating Game...' : 'Start New Game'}
            </button>

            {/* Quick Start Options */}
            <div className="border-t pt-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">
                Quick Start
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <button
                  onClick={() => {
                    setGameConfig({
                      players: [
                        { player: 'RED', agent_type: 'human', agent_config: {} },
                        { player: 'BLUE', agent_type: 'random', agent_config: {} },
                        { player: 'GREEN', agent_type: 'heuristic', agent_config: {} },
                        { player: 'YELLOW', agent_type: 'mcts', agent_config: {} }
                      ],
                      auto_start: true
                    });
                  }}
                  className="p-4 border border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-colors duration-200"
                >
                  <div className="text-sm font-medium text-gray-800">4 Players</div>
                  <div className="text-xs text-gray-600 mt-1">Human vs All Agents</div>
                </button>
                
                <button
                  onClick={() => {
                    setGameConfig({
                      players: [
                        { player: 'RED', agent_type: 'human', agent_config: {} },
                        { player: 'BLUE', agent_type: 'random', agent_config: {} }
                      ],
                      auto_start: true
                    });
                  }}
                  className="p-4 border border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-colors duration-200"
                >
                  <div className="text-sm font-medium text-gray-800">vs Random</div>
                  <div className="text-xs text-gray-600 mt-1">Easy opponent</div>
                </button>
                
                <button
                  onClick={() => {
                    setGameConfig({
                      players: [
                        { player: 'RED', agent_type: 'human', agent_config: {} },
                        { player: 'BLUE', agent_type: 'heuristic', agent_config: {} }
                      ],
                      auto_start: true
                    });
                  }}
                  className="p-4 border border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-colors duration-200"
                >
                  <div className="text-sm font-medium text-gray-800">vs Heuristic</div>
                  <div className="text-xs text-gray-600 mt-1">Medium opponent</div>
                </button>
                
                <button
                  onClick={() => {
                    setGameConfig({
                      players: [
                        { player: 'RED', agent_type: 'human', agent_config: {} },
                        { player: 'BLUE', agent_type: 'mcts', agent_config: {} }
                      ],
                      auto_start: true
                    });
                  }}
                  className="p-4 border border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-colors duration-200"
                >
                  <div className="text-sm font-medium text-gray-800">vs MCTS</div>
                  <div className="text-xs text-gray-600 mt-1">Hard opponent</div>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
