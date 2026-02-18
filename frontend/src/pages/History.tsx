import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { API_BASE } from '../constants/gameConstants';

export const History: React.FC = () => {
  const [games, setGames] = useState<any[]>([]);

  useEffect(() => {
    const load = async () => {
      const resp = await fetch(`${API_BASE}/api/history?limit=50`);
      if (!resp.ok) return;
      const data = await resp.json();
      setGames(data.games || []);
    };
    load();
  }, []);

  return (
    <div className="min-h-screen bg-charcoal-900 text-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold">Game History</h1>
        <Link to="/play" className="text-neon-blue">Back to Play</Link>
      </div>
      <div className="bg-charcoal-800 border border-charcoal-700 rounded p-4">
        {games.length === 0 ? (
          <div className="text-gray-400">No finished games yet.</div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left border-b border-charcoal-700">
                <th className="py-2">Game</th>
                <th>Winner</th>
                <th>Moves</th>
                <th>Duration (ms)</th>
                <th>AI Nodes</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {games.map((g) => (
                <tr key={g.game_id} className="border-b border-charcoal-700/40">
                  <td className="py-2 font-mono">{g.game_id.slice(0, 8)}</td>
                  <td>{g.winner || 'NONE'}</td>
                  <td>{g.move_count}</td>
                  <td>{g.gameDurationMs}</td>
                  <td>{g.totalAiNodesEvaluated}</td>
                  <td>
                    <Link className="text-neon-blue" to={`/analysis/${g.game_id}`}>View analysis</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};
