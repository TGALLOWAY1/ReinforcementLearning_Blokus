import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { API_BASE } from '../constants/gameConstants';

export const Analysis: React.FC = () => {
  const { gameId } = useParams();
  const [analysis, setAnalysis] = useState<any>(null);
  const [trends, setTrends] = useState<any>(null);

  useEffect(() => {
    const load = async () => {
      if (!gameId) return;
      const [analysisResp, trendsResp] = await Promise.all([
        fetch(`${API_BASE}/api/analysis/${gameId}`),
        fetch(`${API_BASE}/api/trends`),
      ]);
      if (analysisResp.ok) setAnalysis(await analysisResp.json());
      if (trendsResp.ok) setTrends(await trendsResp.json());
    };
    load();
  }, [gameId]);

  return (
    <div className="min-h-screen bg-charcoal-900 text-gray-200 p-6">
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-2xl font-bold">Game Analysis</h1>
        <div className="space-x-4">
          <Link to="/history" className="text-neon-blue">History</Link>
          <Link to="/play" className="text-neon-blue">Back to Play</Link>
        </div>
      </div>

      {analysis && (
        <div className="space-y-4">
          <div className="bg-charcoal-800 p-4 rounded border border-charcoal-700">
            <h2 className="font-semibold mb-2">Aggregates</h2>
            <pre className="text-sm whitespace-pre-wrap">{JSON.stringify(analysis.aggregates, null, 2)}</pre>
          </div>
          <div className="bg-charcoal-800 p-4 rounded border border-charcoal-700">
            <h2 className="font-semibold mb-2">Per-Move Stats</h2>
            <pre className="text-sm whitespace-pre-wrap max-h-96 overflow-auto">{JSON.stringify(analysis.moves, null, 2)}</pre>
          </div>
        </div>
      )}

      {trends && (
        <div className="bg-charcoal-800 p-4 rounded border border-charcoal-700 mt-6">
          <h2 className="font-semibold mb-2">Cross-Game Trends</h2>
          <pre className="text-sm whitespace-pre-wrap">{JSON.stringify(trends, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};
