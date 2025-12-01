import React, { useState } from 'react';
import { useGameStore } from '../store/gameStore';

interface SidebarProps {
  onNewGame?: () => void;
  onAgentChange?: (playerId: string, agentType: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = () => {
  const { gameState, connectionStatus } = useGameStore();
  const [isStatusExpanded, setIsStatusExpanded] = useState(false);
  const [isPlayersExpanded, setIsPlayersExpanded] = useState(true);

  const getPlayerColor = (player: string) => {
    const colors = {
      RED: 'bg-red-500',
      BLUE: 'bg-blue-500', 
      GREEN: 'bg-green-500',
      YELLOW: 'bg-yellow-500',
      red: 'bg-red-500',
      blue: 'bg-blue-500', 
      green: 'bg-green-500',
      yellow: 'bg-yellow-500'
    };
    return colors[player as keyof typeof colors] || 'bg-gray-500';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'text-green-600';
      case 'connecting': return 'text-yellow-600';
      case 'disconnected': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div className="w-80 space-y-4">
      {/* Game Status Card */}
      <div className="card card-hover">
        <button
          onClick={() => setIsStatusExpanded(!isStatusExpanded)}
          className="flex items-center justify-between w-full text-left p-4 rounded-xl hover:bg-slate-50/50 transition-all duration-200"
        >
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <h2 className="text-lg font-semibold text-slate-800">Game Status</h2>
          </div>
          <div className="flex items-center space-x-2">
            <div className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(connectionStatus)}`}>
              {connectionStatus.toUpperCase()}
            </div>
            <svg
              className={`w-5 h-5 text-slate-400 transition-transform duration-200 ${isStatusExpanded ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </button>
        
        {/* Collapsible Status Details */}
        {isStatusExpanded && gameState && (
          <div className="px-4 pb-4">
            <div className="bg-gradient-to-r from-slate-50 to-slate-100 rounded-lg p-4 space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-slate-600">Game ID</span>
                <span className="text-sm font-mono text-slate-800 bg-slate-200 px-2 py-1 rounded">
                  {gameState?.game_id?.slice(0, 8)}...
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-slate-600">Moves</span>
                <span className="text-sm font-semibold text-slate-800">{gameState?.move_count || 0}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-slate-600">Status</span>
                <span className={`text-sm font-semibold px-2 py-1 rounded-full ${
                  gameState?.game_over 
                    ? 'bg-red-100 text-red-700' 
                    : 'bg-green-100 text-green-700'
                }`}>
                  {gameState?.game_over ? 'Finished' : 'Active'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Players Card */}
      <div className="card card-hover">
        <button
          onClick={() => setIsPlayersExpanded(!isPlayersExpanded)}
          className="flex items-center justify-between w-full text-left p-4 rounded-xl hover:bg-slate-50/50 transition-all duration-200"
        >
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
            <h3 className="text-lg font-semibold text-slate-800">Players</h3>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-sm text-slate-500 bg-slate-100 px-2 py-1 rounded-full">
              {gameState?.scores ? Object.keys(gameState.scores).length : 0}
            </span>
            <svg
              className={`w-5 h-5 text-slate-400 transition-transform duration-200 ${isPlayersExpanded ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </button>
        
        {isPlayersExpanded && gameState?.scores && (
          <div className="px-4 pb-4 space-y-3">
            {Object.keys(gameState.scores).map((playerKey) => {
              const isActive = gameState?.current_player === playerKey;
              const score = gameState?.scores?.[playerKey];
              const piecesUsed = gameState?.pieces_used?.[playerKey] || [];
              const piecesRemaining = 21 - piecesUsed.length;
              
              return (
                <div key={playerKey} className="group relative">
                  <div className={`
                    p-4 rounded-xl border-2 transition-all duration-300 hover:shadow-lg hover:scale-[1.02]
                    ${isActive 
                      ? 'border-blue-400 bg-gradient-to-br from-blue-50 to-blue-100 shadow-lg shadow-blue-200/50' 
                      : 'border-slate-200 bg-gradient-to-br from-slate-50 to-slate-100 hover:border-slate-300 hover:shadow-md'
                    }
                  `}>
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <div className={`w-8 h-8 rounded-full ${getPlayerColor(playerKey)} shadow-lg ring-2 ring-white`}></div>
                        <div>
                          <span className="font-bold text-slate-800 capitalize text-lg">
                            {playerKey.toLowerCase()}
                          </span>
                          {isActive && (
                            <div className="text-xs bg-blue-200 text-blue-800 px-2 py-1 rounded-full font-medium mt-1 animate-pulse">
                              Current Turn
                            </div>
                          )}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-bold text-slate-800">{score}</div>
                        <div className="text-xs text-slate-500">points</div>
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-600">Pieces Left:</span>
                        <span className="font-semibold text-slate-800">{piecesRemaining}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-600">Pieces Used:</span>
                        <span className="font-semibold text-slate-800">{piecesUsed.length}/21</span>
                      </div>
                    </div>
                    
                    {/* Progress bar for pieces */}
                    <div className="mt-3">
                      <div className="w-full bg-slate-200 rounded-full h-2.5">
                        <div
                          className={`h-2.5 rounded-full transition-all duration-500 ${
                            isActive ? 'bg-gradient-to-r from-blue-500 to-blue-600' : 'bg-gradient-to-r from-slate-400 to-slate-500'
                          }`}
                          style={{
                            width: `${(piecesUsed.length / 21) * 100}%`
                          }}
                        ></div>
                      </div>
                      <div className="text-xs text-slate-500 mt-1 text-center">
                        {Math.round((piecesUsed.length / 21) * 100)}% complete
                      </div>
                    </div>
                  </div>
                  
                  {/* Active player indicator */}
                  {isActive && (
                    <div className="absolute -top-1 -right-1">
                      <div className="w-4 h-4 bg-green-400 rounded-full border-2 border-white shadow-lg animate-pulse"></div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>



      {/* Game Rules Card */}
      <div className="card card-hover">
        <div className="p-4">
          <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center space-x-2">
            <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
            <span>Game Rules</span>
          </h3>
          <div className="text-sm text-slate-600 space-y-3">
            <div className="flex items-start space-x-2">
              <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
              <span>First move must cover your corner</span>
            </div>
            <div className="flex items-start space-x-2">
              <div className="w-1.5 h-1.5 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
              <span>Pieces touch only at corners</span>
            </div>
            <div className="flex items-start space-x-2">
              <div className="w-1.5 h-1.5 bg-orange-500 rounded-full mt-2 flex-shrink-0"></div>
              <span>No edge-to-edge contact</span>
            </div>
            <div className="flex items-start space-x-2">
              <div className="w-1.5 h-1.5 bg-purple-500 rounded-full mt-2 flex-shrink-0"></div>
              <span>Score = piece size + bonuses</span>
            </div>
          </div>
        </div>
      </div>

      {/* Winner Display */}
      {gameState?.game_over && gameState?.winner && (
        <div className="card">
          <div className="bg-gradient-to-r from-yellow-50 to-amber-50 border border-yellow-200 rounded-xl p-6">
            <div className="text-center">
              <div className="text-4xl mb-3">🏆</div>
              <div className="text-xl font-bold text-yellow-800 mb-2">
                Game Over!
              </div>
              <div className="text-yellow-700 capitalize font-medium">
                Winner: {gameState?.winner}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
