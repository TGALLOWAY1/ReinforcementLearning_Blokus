import React from 'react';

interface ControlsPanelProps {
  onNewGame: () => void;
  onAgentChange: (playerId: string, agentType: string) => void;
  gameState: any;
}

export const ControlsPanel: React.FC<ControlsPanelProps> = ({
  onNewGame,
  onAgentChange,
  gameState
}) => {
  return (
    <div className="card card-hover">
      <div className="p-6 flex flex-col lg:flex-row items-center justify-between space-y-4 lg:space-y-0 lg:space-x-6">
        {/* Game Controls */}
        <div className="flex items-center space-x-6">
          <button
            onClick={onNewGame}
            className="btn-primary"
          >
            New Game
          </button>
          
          <div className="text-sm text-slate-600 flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              <kbd className="px-2 py-1 bg-slate-100 border border-slate-200 rounded text-xs font-mono">R</kbd>
              <span className="font-medium">Rotate piece</span>
            </div>
            <div className="flex items-center space-x-2">
              <kbd className="px-2 py-1 bg-slate-100 border border-slate-200 rounded text-xs font-mono">F</kbd>
              <span className="font-medium">Flip piece</span>
            </div>
            <div className="flex items-center space-x-2">
              <kbd className="px-2 py-1 bg-slate-100 border border-slate-200 rounded text-xs font-mono">Click</kbd>
              <span className="font-medium">Place piece</span>
            </div>
          </div>
        </div>

        {/* Agent Settings */}
        <div className="flex items-center space-x-4">
          <span className="text-sm font-medium text-slate-700">Agents:</span>
          {gameState?.players?.map((player: any) => (
            <div key={player.player} className="flex items-center space-x-2">
              <label className="text-sm font-medium text-slate-700 capitalize">
                {player.player}:
              </label>
              <select
                className="input-modern text-sm"
                onChange={(e) => onAgentChange(player.player, e.target.value)}
                defaultValue="human"
              >
                <option value="human">Human</option>
                <option value="random">Random</option>
                <option value="heuristic">Heuristic</option>
                <option value="mcts">MCTS</option>
              </select>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
