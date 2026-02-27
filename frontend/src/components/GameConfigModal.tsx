import React, { useState } from 'react';
import { useGameStore } from '../store/gameStore';
import { API_BASE, DEPLOY_MCTS_PRESETS, IS_DEPLOY_PROFILE } from '../constants/gameConstants';

interface GameConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  onGameCreated: () => void;
  /** When false (e.g. initial view), hide close button - user must start a game */
  canClose?: boolean;
}

export const GameConfigModal: React.FC<GameConfigModalProps> = ({
  isOpen,
  onClose,
  onGameCreated,
  canClose = true
}) => {
  const deployConfig = {
    players: [
      { player: 'RED', agent_type: 'human', agent_config: {} },
      { player: 'BLUE', agent_type: 'mcts', agent_config: { difficulty: 'easy', time_budget_ms: DEPLOY_MCTS_PRESETS.easy } },
      { player: 'GREEN', agent_type: 'mcts', agent_config: { difficulty: 'medium', time_budget_ms: DEPLOY_MCTS_PRESETS.medium } },
      { player: 'YELLOW', agent_type: 'mcts', agent_config: { difficulty: 'hard', time_budget_ms: DEPLOY_MCTS_PRESETS.hard } }
    ],
    auto_start: true
  };

  const researchDefaultConfig = {
    players: [
      { player: 'RED', agent_type: 'human', agent_config: {} },
      { player: 'BLUE', agent_type: 'mcts', agent_config: { difficulty: 'easy', time_budget_ms: DEPLOY_MCTS_PRESETS.easy } },
      { player: 'GREEN', agent_type: 'mcts', agent_config: { difficulty: 'medium', time_budget_ms: DEPLOY_MCTS_PRESETS.medium } },
      { player: 'YELLOW', agent_type: 'mcts', agent_config: { difficulty: 'hard', time_budget_ms: DEPLOY_MCTS_PRESETS.hard } }
    ],
    auto_start: true
  };

  const { createGame, connect, loadGame } = useGameStore();
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [maxTime, setMaxTime] = useState<number>(DEPLOY_MCTS_PRESETS.hard);

  const [gameConfig, setGameConfig] = useState<any>(IS_DEPLOY_PROFILE ? deployConfig : researchDefaultConfig);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

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
    if (IS_DEPLOY_PROFILE) {
      return;
    }
    setGameConfig((prev: any) => ({
      ...prev,
      players: prev.players.map((player: any, i: number) =>
        i === index ? { ...player, [field]: value } : player
      )
    }));
  };

  const addPlayer = () => {
    if (IS_DEPLOY_PROFILE) {
      return;
    }
    if (gameConfig.players.length < 4) {
      const playerColors = ['RED', 'BLUE', 'GREEN', 'YELLOW'];
      const nextColor = playerColors[gameConfig.players.length];
      setGameConfig((prev: any) => ({
        ...prev,
        players: [...prev.players, { player: nextColor, agent_type: 'random', agent_config: {} }]
      }));
    }
  };

  const removePlayer = (index: number) => {
    if (IS_DEPLOY_PROFILE) {
      return;
    }
    if (gameConfig.players.length > 2) {
      setGameConfig((prev: any) => ({
        ...prev,
        players: prev.players.filter((_: any, i: number) => i !== index)
      }));
    }
  };

  const researchQuickStartPresets = [
    {
      name: '4 Players',
      description: `Human vs MCTS Easy/Medium/Hard (${DEPLOY_MCTS_PRESETS.easy}/${DEPLOY_MCTS_PRESETS.medium}/${DEPLOY_MCTS_PRESETS.hard}ms)`,
      config: {
        players: [
          { player: 'RED', agent_type: 'human', agent_config: {} },
          { player: 'BLUE', agent_type: 'mcts', agent_config: { difficulty: 'easy', time_budget_ms: DEPLOY_MCTS_PRESETS.easy } },
          { player: 'GREEN', agent_type: 'mcts', agent_config: { difficulty: 'medium', time_budget_ms: DEPLOY_MCTS_PRESETS.medium } },
          { player: 'YELLOW', agent_type: 'mcts', agent_config: { difficulty: 'hard', time_budget_ms: DEPLOY_MCTS_PRESETS.hard } }
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

  const deployQuickStartPresets = [
    {
      name: 'Deploy Preset',
      description: `Human vs MCTS Easy/Medium/Hard (${DEPLOY_MCTS_PRESETS.easy}/${DEPLOY_MCTS_PRESETS.medium}/${DEPLOY_MCTS_PRESETS.hard}ms)`,
      config: deployConfig
    }
  ];

  const quickStartPresets = IS_DEPLOY_PROFILE ? deployQuickStartPresets : researchQuickStartPresets;

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

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsCreating(true);
    setError(null);

    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        const content = e.target?.result as string;
        const history = JSON.parse(content);

        if (!Array.isArray(history)) {
          throw new Error("Invalid save file format (expected array)");
        }

        await loadGame(history);
        onGameCreated();
        onClose();
      } catch (err) {
        console.error("Failed to load game:", err);
        setError(err instanceof Error ? err.message : "Failed to load game file");
      } finally {
        setIsCreating(false);
        if (fileInputRef.current) {
          fileInputRef.current.value = ''; // Reset input
        }
      }
    };
    reader.onerror = () => {
      setError("Failed to read the file");
      setIsCreating(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    };
    reader.readAsText(file);
  };

  if (!isOpen) return null;

  // Deploy profile: minimal first-page UI — Human vs MCTS (easy/medium/hard) only, no config
  if (IS_DEPLOY_PROFILE) {
    return (
      <div className="fixed inset-0 bg-charcoal-900 flex items-center justify-center z-50 p-4">
        <div className="bg-charcoal-800 border border-charcoal-700 rounded-lg max-w-md w-full p-8 text-center relative">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="absolute top-4 left-4 text-gray-400 hover:text-neon-blue transition-colors"
            title="Advanced Settings"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
          {canClose && (
            <button
              onClick={onClose}
              className="absolute top-4 right-4 text-gray-400 hover:text-gray-200 transition-colors"
              aria-label="Close"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
          <h1 className="text-3xl font-bold text-gray-200 mb-2">Blokus</h1>
          <p className="text-gray-400 mb-6">
            You (Red) vs 3 AI opponents at Easy, Medium, and Hard
          </p>

          {showSettings && (
            <div className="mb-6 bg-charcoal-900 p-4 rounded-lg border border-charcoal-700 text-left">
              <label className="block text-sm text-gray-300 font-medium mb-2">
                Max AI Thinking Time: {maxTime / 1000}s
              </label>
              <input
                type="range"
                min="400"
                max="9000"
                step="100"
                value={maxTime}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                  const val = parseInt(e.target.value);
                  setMaxTime(val);
                  setGameConfig((prev: any) => ({
                    ...prev,
                    players: prev.players.map((p: any) => {
                      if (p.agent_type === 'mcts') {
                        let budget = val;
                        if (p.agent_config.difficulty === 'easy') budget = Math.floor(val / 4.5);
                        else if (p.agent_config.difficulty === 'medium') budget = Math.floor(val / 2);
                        return {
                          ...p,
                          agent_config: { ...p.agent_config, time_budget_ms: budget }
                        };
                      }
                      return p;
                    })
                  }));
                }}
                className="w-full accent-neon-blue h-2 bg-charcoal-700 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-xs text-gray-500 mt-2">
                Adjusts the maximum time the Hard AI will think. Easy and Medium scale proportionally.
              </p>
            </div>
          )}

          {error && (
            <div className="bg-red-900 border border-red-700 rounded-lg p-4 mb-6 text-red-200 text-sm">
              {error}
            </div>
          )}

          <div className="flex space-x-3">
            <button
              onClick={handleCreateGame}
              disabled={isCreating}
              className={`
                flex-1 py-4 px-6 rounded-lg font-medium text-white transition-colors duration-200
                ${isCreating
                  ? 'bg-gray-600 cursor-not-allowed'
                  : 'bg-neon-blue hover:bg-neon-blue/80 text-black'
                }
              `}
            >
              {isCreating ? 'Starting...' : 'Start Game'}
            </button>
            <input
              type="file"
              accept=".json"
              ref={fileInputRef}
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isCreating}
              className={`
                py-4 px-6 rounded-lg font-medium border transition-colors duration-200
                ${isCreating
                  ? 'bg-charcoal-700 border-charcoal-600 text-gray-500 cursor-not-allowed'
                  : 'bg-charcoal-800 border-charcoal-600 text-gray-300 hover:bg-charcoal-700 hover:text-white'
                }
              `}
            >
              Load
            </button>
          </div>
        </div>
      </div>
    );
  }

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
                {gameConfig.players.map((player: any, index: number) => (
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
                onChange={(e) => setGameConfig((prev: any) => ({
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
            <div className="flex space-x-3">
              <button
                onClick={handleCreateGame}
                disabled={isCreating}
                className={`
                  flex-1 py-3 px-6 rounded-lg font-medium text-white transition-colors duration-200
                  ${isCreating
                    ? 'bg-gray-600 cursor-not-allowed'
                    : 'bg-neon-blue hover:bg-neon-blue/80'
                  }
                `}
              >
                {isCreating ? 'Creating Game...' : 'Start New Game'}
              </button>

              <input
                type="file"
                accept=".json"
                ref={fileInputRef}
                onChange={handleFileUpload}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isCreating}
                className={`
                  py-3 px-6 rounded-lg font-medium border transition-colors duration-200
                  ${isCreating
                    ? 'bg-charcoal-700 border-charcoal-600 text-gray-500 cursor-not-allowed'
                    : 'bg-charcoal-800 border-charcoal-600 text-gray-300 hover:bg-charcoal-700 hover:text-white'
                  }
                `}
              >
                Load From File
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
