import React, { useState, useEffect } from 'react';
import { useGameStore } from '../store/gameStore';
import { BOARD_SIZE } from '../constants/gameConstants';

interface AgentVisualizationsProps {
  selectedPiece?: number | null;
  pieceOrientation?: number;
  setPieceOrientation?: (orientation: number) => void;
}

type TabType = 'policy' | 'value' | 'tree';

// Policy View Component (extracted for standalone use)
export const PolicyView: React.FC<{ overrideHeatmap?: number[][] }> = ({ overrideHeatmap }) => {
  const { gameState } = useGameStore();

  // Use override when provided (e.g. freeze mode in Telemetry)
  const heatmapSource = overrideHeatmap ?? gameState?.heatmap;

  // Generate random policy grid for demo if no data exists
  const generateRandomPolicy = (): number[][] => {
    return Array(BOARD_SIZE).fill(null).map(() =>
      Array(BOARD_SIZE).fill(null).map(() => Math.random())
    );
  };

  // Use real heatmap data from gameState if available, otherwise use random for demo
  const currentPolicyGrid = heatmapSource || generateRandomPolicy();

  // Check if this is binary heatmap (legal moves) or continuous policy
  const isBinaryHeatmap = heatmapSource !== undefined;

  // Get min/max for normalization (for continuous policy)
  const allValues = currentPolicyGrid.flat();
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const range = maxValue - minValue || 1;

  // Ensure we always have exactly 20×20 cells
  const rows = Array.from({ length: BOARD_SIZE }, (_, r) => r);
  const cols = Array.from({ length: BOARD_SIZE }, (_, c) => c);

  const getPolicyValueForCell = (row: number, col: number): number => {
    // Ensure we have data for this cell
    if (currentPolicyGrid[row] && currentPolicyGrid[row][col] !== undefined) {
      return currentPolicyGrid[row][col];
    }
    return 0;
  };

  return (
    <div className="space-y-3">
      <div className="w-full max-w-full">
        <div
          className="grid gap-[1px] bg-charcoal-800"
          style={{
            gridTemplateColumns: `repeat(${BOARD_SIZE}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${BOARD_SIZE}, minmax(0, 1fr))`,
            aspectRatio: '1 / 1', // Ensure square grid
          }}
        >
          {rows.map((r) =>
            cols.map((c) => {
              const value = getPolicyValueForCell(r, c);

              if (isBinaryHeatmap) {
                // Binary heatmap: 1.0 = legal (red), 0.0 = illegal (transparent/blue)
                if (value === 1.0) {
                  return (
                    <div
                      key={`${r}-${c}`}
                      className="aspect-square bg-neon-red bg-opacity-60 border border-charcoal-700"
                    />
                  );
                } else {
                  return (
                    <div
                      key={`${r}-${c}`}
                      className="aspect-square bg-neon-blue bg-opacity-10 border border-charcoal-700"
                    />
                  );
                }
              } else {
                // Continuous policy: normalize and map to colors
                const normalized = (value - minValue) / range;
                const opacity = normalized;
                const isHighProb = normalized > 0.5;

                return (
                  <div
                    key={`${r}-${c}`}
                    className="aspect-square border border-charcoal-700"
                    style={{
                      backgroundColor: isHighProb
                        ? `rgba(255, 77, 77, ${opacity})` // neon.red with opacity
                        : `rgba(0, 240, 255, ${1 - opacity})`, // neon.blue with inverse opacity
                    }}
                  />
                );
              }
            })
          )}
        </div>
      </div>
      <div className="text-xs text-gray-400 space-y-1">
        {isBinaryHeatmap ? (
          <>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-neon-red"></div>
              <span>Legal Move Position</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-neon-blue opacity-30"></div>
              <span>Illegal Position</span>
            </div>
          </>
        ) : (
          <>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-neon-red"></div>
              <span>High Probability</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-neon-blue"></div>
              <span>Low Probability</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

// Value View Component (extracted for standalone use)
export const ValueView: React.FC = () => {
  // Value function data (dummy for now)
  const winProbability = 0.68; // 68% win probability

  return (
    <div className="space-y-3">
      <div className="bg-charcoal-800 border border-charcoal-700 p-4">
        <div className="text-xs text-gray-400 mb-2">Win Probability</div>
        <div className="relative h-32 bg-charcoal-900 border border-charcoal-700">
          <div
            className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-neon-green to-neon-blue transition-all duration-500"
            style={{
              height: `${winProbability * 100}%`,
            }}
          >
            <div className="absolute -top-6 left-0 right-0 text-center">
              <span className="text-sm font-mono text-neon-green">
                {(winProbability * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>
      <div className="text-xs text-gray-400">
        Estimated win probability for current game state
      </div>
    </div>
  );
};

export const AgentVisualizations: React.FC<AgentVisualizationsProps> = ({
  pieceOrientation = 0,
  setPieceOrientation
}) => {
  const [activeTab, setActiveTab] = useState<TabType>('policy');
  const [localOrientation, setLocalOrientation] = useState(0);

  const orientation = pieceOrientation !== undefined ? pieceOrientation : localOrientation;
  const setOrientation = setPieceOrientation || setLocalOrientation;

  // Keyboard handlers for piece rotation/flip
  useEffect(() => {
    const handleGlobalKeyDown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }

      if (event.key === 'r' || event.key === 'R') {
        event.preventDefault();
        if (orientation < 4) {
          setOrientation((orientation + 1) % 4);
        } else {
          setOrientation(4 + ((orientation - 4 + 1) % 4));
        }
      } else if (event.key === 'f' || event.key === 'F') {
        event.preventDefault();
        if (orientation < 4) {
          setOrientation(orientation + 4);
        } else {
          setOrientation(orientation - 4);
        }
      }
    };

    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, [setOrientation, orientation]);

  const tabs: { id: TabType; label: string }[] = [
    { id: 'policy', label: 'Policy' },
    { id: 'value', label: 'Value' },
    { id: 'tree', label: 'Tree' },
  ];

  return (
    <div className="h-full flex flex-col">
      {/* Tab Navigation */}
      <div className="flex border-b border-charcoal-700">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${activeTab === tab.id
                ? 'bg-charcoal-800 text-neon-blue border-b-2 border-neon-blue'
                : 'bg-charcoal-900 text-gray-400 hover:text-gray-200 hover:bg-charcoal-800'
              }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'policy' && (
          <PolicyView />
        )}

        {activeTab === 'value' && (
          <ValueView />
        )}

        {activeTab === 'tree' && (
          <div className="space-y-3">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              MCTS Tree
            </h3>
            <div className="bg-charcoal-800 border border-charcoal-700 p-4 text-center">
              <div className="text-gray-400 text-sm py-8">
                Tree visualization coming soon
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

