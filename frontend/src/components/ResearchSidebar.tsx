import React, { useState } from 'react';
import { useGameStore } from '../store/gameStore';

interface ResearchSidebarProps {
  onNewGame?: () => void;
}

// Environment Controls Section Component
export const EnvironmentControlsSection: React.FC<{ onNewGame?: () => void }> = ({ onNewGame }) => {
  const [autoStep, setAutoStep] = useState(false);

  const handleStartEpisode = () => {
    console.log('Start Episode clicked');
    // TODO: Connect to actual episode start logic
  };

  const handleReset = () => {
    console.log('Reset clicked');
    if (onNewGame) {
      onNewGame();
    }
    // TODO: Connect to actual reset logic
  };

  return (
    <div className="flex flex-col space-y-2">
      <button
        onClick={handleStartEpisode}
        className="bg-charcoal-800 border border-charcoal-700 text-gray-200 px-4 py-2 text-sm font-medium hover:bg-charcoal-700 hover:border-charcoal-600 transition-colors"
      >
        Start Episode
      </button>
      <button
        onClick={handleReset}
        className="bg-charcoal-800 border border-charcoal-700 text-gray-200 px-4 py-2 text-sm font-medium hover:bg-charcoal-700 hover:border-charcoal-600 transition-colors"
      >
        Reset
      </button>
      <button
        onClick={() => setAutoStep(!autoStep)}
        className={`border px-4 py-2 text-sm font-medium transition-colors ${autoStep
          ? 'bg-charcoal-700 border-neon-blue text-neon-blue'
          : 'bg-charcoal-800 border-charcoal-700 text-gray-200 hover:bg-charcoal-700 hover:border-charcoal-600'
          }`}
      >
        Auto-Step {autoStep ? 'ON' : 'OFF'}
      </button>
      <button
        onClick={() => useGameStore.getState().saveGame()}
        className="bg-charcoal-800 border border-charcoal-700 text-gray-200 px-4 py-2 text-sm font-medium hover:bg-charcoal-700 hover:border-charcoal-600 transition-colors"
      >
        Save Game
      </button>
      <div className="relative">
        <input
          type="file"
          accept=".json"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (event) => {
              try {
                const history = JSON.parse(event.target?.result as string);
                useGameStore.getState().loadGame(history);
              } catch (err) {
                console.error("Failed to parse game log", err);
              }
            };
            reader.readAsText(file);
          }}
          className="absolute inset-0 opacity-0 cursor-pointer"
        />
        <button
          className="w-full bg-charcoal-800 border border-charcoal-700 text-gray-200 px-4 py-2 text-sm font-medium hover:bg-charcoal-700 hover:border-charcoal-600 transition-colors"
        >
          Load Game
        </button>
      </div>
    </div>
  );
};

// Training Parameters Section Component
export const TrainingParametersSection: React.FC = () => {
  const [learningRate, setLearningRate] = useState('0.001');
  const [exploration, setExploration] = useState('0.1');
  const [discountFactor, setDiscountFactor] = useState('0.99');
  const [agentType, setAgentType] = useState('DQN');

  return (
    <div className="space-y-3">
      {/* Learning Rate */}
      <div>
        <label className="block text-xs text-gray-400 mb-1">Learning Rate</label>
        <input
          type="text"
          value={learningRate}
          onChange={(e) => setLearningRate(e.target.value)}
          className="w-full bg-charcoal-900 border border-charcoal-700 text-gray-200 px-3 py-2 text-sm focus:outline-none focus:border-neon-blue"
        />
      </div>

      {/* Exploration (epsilon) */}
      <div>
        <label className="block text-xs text-gray-400 mb-1">Exploration (ε)</label>
        <input
          type="text"
          value={exploration}
          onChange={(e) => setExploration(e.target.value)}
          className="w-full bg-charcoal-900 border border-charcoal-700 text-gray-200 px-3 py-2 text-sm focus:outline-none focus:border-neon-blue"
        />
      </div>

      {/* Discount Factor */}
      <div>
        <label className="block text-xs text-gray-400 mb-1">Discount Factor (γ)</label>
        <input
          type="text"
          value={discountFactor}
          onChange={(e) => setDiscountFactor(e.target.value)}
          className="w-full bg-charcoal-900 border border-charcoal-700 text-gray-200 px-3 py-2 text-sm focus:outline-none focus:border-neon-blue"
        />
      </div>

      {/* Agent Type */}
      <div>
        <label className="block text-xs text-gray-400 mb-1">Agent Type</label>
        <select
          value={agentType}
          onChange={(e) => setAgentType(e.target.value)}
          className="w-full bg-charcoal-900 border border-charcoal-700 text-gray-200 px-3 py-2 text-sm focus:outline-none focus:border-neon-blue"
        >
          <option value="DQN">DQN</option>
          <option value="PPO">PPO</option>
          <option value="MCTS">MCTS</option>
        </select>
      </div>
    </div>
  );
};

// Model Status Section Component
export const ModelStatusSection: React.FC = () => {
  // Dummy state for model status
  const episode = 142;
  const winRate = 68.5;

  // Dummy data for sparkline (reward trend going up)
  const sparklineData = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90];
  const sparklineWidth = 200;
  const sparklineHeight = 40;
  const padding = 4;
  const maxValue = Math.max(...sparklineData);
  const minValue = Math.min(...sparklineData);
  const range = maxValue - minValue || 1;

  // Generate SVG path for sparkline
  const pathData = sparklineData.map((value, index) => {
    const x = padding + (index / (sparklineData.length - 1)) * (sparklineWidth - 2 * padding);
    const y = sparklineHeight - padding - ((value - minValue) / range) * (sparklineHeight - 2 * padding);
    return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
  }).join(' ');

  return (
    <div className="space-y-4">
      {/* Episode and Win Rate Stats */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-400">Episode:</span>
          <span className="text-sm font-mono text-neon-blue">{episode}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-400">Win Rate:</span>
          <span className="text-sm font-mono text-neon-green">{winRate.toFixed(1)}%</span>
        </div>
      </div>

      {/* Sparkline Chart */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-400">Reward Trend</span>
        </div>
        <div className="bg-charcoal-900 border border-charcoal-700 p-2">
          <svg
            width={sparklineWidth}
            height={sparklineHeight}
            className="w-full"
            viewBox={`0 0 ${sparklineWidth} ${sparklineHeight}`}
            preserveAspectRatio="none"
          >
            <path
              d={pathData}
              fill="none"
              stroke="#00F0FF"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
      </div>
    </div>
  );
};

// Legacy ResearchSidebar component (kept for backward compatibility if needed)
export const ResearchSidebar: React.FC<ResearchSidebarProps> = ({ onNewGame }) => {
  return (
    <div className="h-full flex flex-col space-y-6 p-4">
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
          Environment Controls
        </h3>
        <EnvironmentControlsSection onNewGame={onNewGame} />
      </div>

      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
          Training Parameters
        </h3>
        <TrainingParametersSection />
      </div>

      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
          Model Status
        </h3>
        <ModelStatusSection />
      </div>
    </div>
  );
};

