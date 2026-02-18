import React, { useState } from 'react';
import { useGameStore } from '../store/gameStore';
import { API_BASE } from '../constants/gameConstants';

interface GameConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  onGameCreated: () => void;
}

export const GameConfigModal: React.FC<GameConfigModalProps> = ({
  isOpen,
  onClose,
  onGameCreated
}) => {
  const { createGame, connect } = useGameStore();
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [gameConfig, setGameConfig] = useState({
    players: [
      { player: 'RED', agent_type: 'human', agent_config: {} },
      { player: 'BLUE', agent_type: 'mcts', agent_config: { time_budget_ms: 1000 } },
      { player: 'GREEN', agent_type: 'mcts', agent_config: { time_budget_ms: 3000 } },
      { player: 'YELLOW', agent_type: 'mcts', agent_config: { time_budget_ms: 5000 } }
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
      console.log('WebSocket connected');
      
      // Check store state
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
      
      onGameCreated();
      onClose();
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

  const quickStartPresets = [
    {
      name: '4 Players',
      description: 'Human vs MCTS (1s/3s/5s)',
      config: {
        players: [
          { player: 'RED', agent_type: 'human', agent_config: {} },
          { player: 'BLUE', agent_type: 'mcts', agent_config: { time_budget_ms: 1000 } },
          { player: 'GREEN', agent_type: 'mcts', agent_config: { time_budget_ms: 3000 } },
          { player: 'YELLOW', agent_type: 'mcts', agent_config: { time_budget_ms: 5000 } }
        ],
        auto_start: true
      }
    },
    {
      name: 'vs Random',
      description: 'Easy opponent',
      config: {
        players: [
          { player: 'RED', agent_type: 'human', agent_config: {} },
          { player: 'BLUE', agent_type: 'random', agent_config: {} }
        ],
        auto_start: true
      }
    },
    {
      name: 'vs Heuristic',
      description: 'Medium opponent',
      config: {
        players: [
          { player: 'RED', agent_type: 'human', agent_config: {} },
          { player: 'BLUE', agent_type: 'heuristic', agent_config: {} }
        ],
        auto_start: true
      }
    },
    {
      name: 'vs MCTS 1s',
      description: 'Fast opponent',
      config: {
        players: [
          { player: 'RED', agent_type: 'human', agent_config: {} },
          { player: 'BLUE', agent_type: 'mcts', agent_config: { time_budget_ms: 1000 } }
        ],
        auto_start: true
      }
    },
    {
      name: 'vs MCTS 3s',
      description: 'Balanced opponent',
      config: {
        players: [
          { player: 'RED', agent_type: 'human', agent_config: {} },
          { player: 'BLUE', agent_type: 'mcts', agent_config: { time_budget_ms: 3000 } }
        ],
        auto_start: true
      }
    },
    {
      name: 'vs MCTS 5s',
      description: 'Deep-thinking opponent',
      config: {
        players: [
          { player: 'RED', agent_type: 'human', agent_config: {} },
          { player: 'BLUE', agent_type: 'mcts', agent_config: { time_budget_ms: 5000 } }
        ],
        auto_start: true
      }
    }
  ];

  const applyQuickStart = async (preset: typeof quickStartPresets[0]) => {
    setGameConfig(preset.config);
    setIsCreating(true);
    setError(null);

    try {
      const gameId = await createGame(preset.config);
      await connect(gameId);
      
      const storeState = useGameStore.getState();
      if (!storeState.gameState) {
        try {
          const response = await fetch(`${API_BASE}/api/games/${gameId}`);
          if (response.ok) {
            const gameState = await response.json();
            useGameStore.getState().setGameState(gameState);
          }
        } catch (err) {
          console.error('Failed to fetch game state via REST API:', err);
        }
      }
      
      onGameCreated();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create game');
    } finally {
      setIsCreating(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-charcoal-800 border border-charcoal-700 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-200">Game Configuration</h2>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-200 transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {error && (
            <div className="bg-red-900 border border-red-700 rounded-lg p-4 mb-6">
              <div className="text-red-200">{error}</div>
            </div>
          )}

          {/* Quick Start Section */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-gray-200 mb-3">Quick Start</h3>
            <div className="grid grid-cols-2 gap-3">
              {quickStartPresets.map((preset, idx) => (
                <button
                  key={idx}
                  onClick={() => applyQuickStart(preset)}
                  disabled={isCreating}
                  className="p-4 border border-charcoal-700 rounded-lg hover:border-neon-blue hover:bg-charcoal-700 transition-colors text-left disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <div className="text-sm font-medium text-gray-200">{preset.name}</div>
                  <div className="text-xs text-gray-400 mt-1">{preset.description}</div>
                </button>
              ))}
            </div>
          </div>

          <div className="border-t border-charcoal-700 pt-6">
            <h3 className="text-lg font-semibold text-gray-200 mb-4">Custom Configuration</h3>

            {/* Players */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Players ({gameConfig.players.length}/4)
              </label>
              <div className="space-y-3">
                {gameConfig.players.map((player, index) => (
                  <div key={index} className="flex items-center space-x-3">
                    <div className="w-24">
                      <select
                        value={player.player}
                        onChange={(e) => updatePlayer(index, 'player', e.target.value)}
                        className="w-full bg-charcoal-900 border border-charcoal-700 text-gray-200 rounded-md px-3 py-2 text-sm focus:outline-none focus:border-neon-blue"
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
                        className="w-full bg-charcoal-900 border border-charcoal-700 text-gray-200 rounded-md px-3 py-2 text-sm focus:outline-none focus:border-neon-blue"
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
                        className="text-red-400 hover:text-red-300 text-sm px-2"
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
                  className="mt-2 text-neon-blue hover:text-neon-blue/80 text-sm"
                >
                  + Add Player
                </button>
              )}
            </div>

            {/* Game Settings */}
            <div className="flex items-center mb-6">
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
              <label htmlFor="auto_start" className="text-sm text-gray-300">
                Auto-start game
              </label>
            </div>

            {/* Create Game Button */}
            <button
              onClick={handleCreateGame}
              disabled={isCreating}
              className={`
                w-full py-3 px-6 rounded-lg font-medium text-white transition-colors duration-200
                ${isCreating 
                  ? 'bg-gray-600 cursor-not-allowed' 
                  : 'bg-neon-blue hover:bg-neon-blue/80'
                }
              `}
            >
              {isCreating ? 'Creating Game...' : 'Start New Game'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

