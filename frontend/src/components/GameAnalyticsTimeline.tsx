/**
 * Game Analytics Timeline panel.
 * Loads backend analytics via /api/analysis/{gameId}/steps and /summary.
 * Toggle: Live (gameStore) vs Logged (backend) for debugging.
 */

import React, { useEffect, useState, useMemo } from 'react';
import { useGameStore, type LegalMovesHistoryEntry } from '../store/gameStore';
import { LegalMovesPerTurnPlot } from './LegalMovesPerTurnPlot';
import { MobilityBucketsChart } from './MobilityBucketsChart';
import { API_BASE } from '../constants/gameConstants';
import { PLAYER_COLORS } from '../constants/gameConstants';
import { PIECE_NAMES } from '../constants/gameConstants';

const PLAYER_ID_TO_NAME: Record<number, string> = {
  1: 'RED',
  2: 'BLUE',
  3: 'YELLOW',
  4: 'GREEN',
};

interface StepLogEntry {
  game_id: string;
  turn_index: number;
  player_id: number;
  action: { piece_id: number; orientation: number; anchor_row: number; anchor_col: number };
  legal_moves_before: number;
  legal_moves_after: number;
  metrics: Record<string, number>;
}

interface SummaryResponse {
  game_id: string;
  mobilityCurve: Array<{
    turn_index: number;
    player_id: number;
    legal_moves_before: number;
    legal_moves_after: number;
  }>;
  deltas: Array<{ turn_index: number; player_id: number; delta: number }>;
  totalSteps: number;
}

export interface GameAnalyticsTimelineProps {
  gameId: string;
}

export const GameAnalyticsTimeline: React.FC<GameAnalyticsTimelineProps> = ({ gameId }) => {
  const [dataSource, setDataSource] = useState<'live' | 'logged'>('live');
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [steps, setSteps] = useState<StepLogEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const gameState = useGameStore((s) => s.gameState);
  const legalMovesHistory = useGameStore((s) => s.legalMovesHistory);

  const isLiveMatch = gameState?.game_id === gameId;

  useEffect(() => {
    if (!gameId) return;
    setLoading(true);
    setError(null);
    Promise.all([
      fetch(`${API_BASE}/api/analysis/${gameId}/summary`),
      fetch(`${API_BASE}/api/analysis/${gameId}/steps?limit=500&offset=0`),
    ])
      .then(async ([sumResp, stepsResp]) => {
        if (sumResp.ok) setSummary(await sumResp.json());
        if (stepsResp.ok) {
          const d = await stepsResp.json();
          setSteps(d.steps ?? []);
        }
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [gameId]);

  const loggedHistory: LegalMovesHistoryEntry[] = useMemo(() => {
    if (!summary?.mobilityCurve?.length) return [];
    const entries: LegalMovesHistoryEntry[] = [];
    const maxTurn = Math.max(...summary.mobilityCurve.map((c) => c.turn_index), 0);
    for (let t = 0; t <= maxTurn; t++) {
      const curve = summary.mobilityCurve.find((c) => c.turn_index === t);
      const byPlayer: Record<string, { totalCellWeighted: number; buckets?: Record<number, number> }> = {};
      if (curve) {
        const name = PLAYER_ID_TO_NAME[curve.player_id] ?? `P${curve.player_id}`;
        byPlayer[name] = { totalCellWeighted: curve.legal_moves_after };
      }
      entries.push({ turn: t, byPlayer } as LegalMovesHistoryEntry);
    }
    return entries;
  }, [summary]);

  return (
    <div className="bg-charcoal-800 rounded-lg border border-charcoal-700 p-4">
      <h2 className="text-lg font-semibold mb-3">Game Analytics Timeline</h2>

      <div className="flex items-center gap-4 mb-4">
        <span className="text-sm text-gray-400">Data source:</span>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="radio"
            name="dataSource"
            checked={dataSource === 'live'}
            onChange={() => setDataSource('live')}
            className="text-neon-blue"
          />
          <span className="text-sm">Live (store)</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="radio"
            name="dataSource"
            checked={dataSource === 'logged'}
            onChange={() => setDataSource('logged')}
            className="text-neon-blue"
          />
          <span className="text-sm">Logged (backend)</span>
        </label>
        {dataSource === 'live' && !isLiveMatch && (
          <span className="text-xs text-amber-500">Store has different game; use Logged for this game</span>
        )}
      </div>

      {loading && <p className="text-sm text-gray-500 mb-3">Loading backend logs…</p>}
      {error && <p className="text-sm text-red-400 mb-3">Error: {error}</p>}

      {dataSource === 'live' ? (
        <>
          <LegalMovesPerTurnPlot overrideHistory={isLiveMatch ? undefined : []} />
          <MobilityBucketsChart overrideHistory={isLiveMatch ? undefined : []} />
          {!isLiveMatch && legalMovesHistory.length === 0 && (
            <p className="text-sm text-gray-500 py-4">No live data for this game. Switch to Logged to view backend analytics.</p>
          )}
        </>
      ) : (
        <>
          {summary && summary.totalSteps > 0 ? (
            <>
              <div className="mb-4">
                <LegalMovesPerTurnPlot overrideHistory={loggedHistory} />
              </div>
              <div className="mb-4">
                <p className="text-xs text-gray-500 mb-2">
                  Mobility by Piece Size: Not available in backend logs (requires cell-weighted buckets).
                </p>
                <div className="h-20 flex items-center justify-center bg-charcoal-800/50 rounded text-gray-500 text-sm">
                  Use Live data for bucket charts
                </div>
              </div>
            </>
          ) : (
            <p className="text-sm text-gray-500 py-4">
              No backend logs for this game. Ensure ENABLE_STRATEGY_LOGGER=true and STRATEGY_LOG_DIR points to logs.
            </p>
          )}

          {steps.length > 0 && (
            <div className="mt-4">
              <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">
                Chosen Move Impact
              </h3>
              <div className="space-y-2 max-h-80 overflow-y-auto">
                {steps.map((s, i) => (
                  <MoveImpactCard key={i} step={s} />
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

const MoveImpactCard: React.FC<{ step: StepLogEntry }> = ({ step }) => {
  const playerName = PLAYER_ID_TO_NAME[step.player_id] ?? `P${step.player_id}`;
  const color = (PLAYER_COLORS as Record<string, string>)[playerName.toLowerCase()] ?? '#94a3b8';
  const m = step.metrics ?? {};
  const delta = m.mobility_me_delta ?? step.legal_moves_after - step.legal_moves_before;
  const blocking = m.blocking ?? 0;
  const blockEff = m.block_eff ?? 0;
  const areaGain = m.area_gain ?? 0;
  const cornersDelta = m.corners_me_delta ?? 0;
  const pieceName = PIECE_NAMES[step.action?.piece_id] ?? `Piece ${step.action?.piece_id}`;

  return (
    <div
      className="p-3 rounded border border-charcoal-600 bg-charcoal-900/50"
      style={{ borderLeftWidth: 3, borderLeftColor: color }}
    >
      <div className="flex justify-between items-start mb-1">
        <span className="text-sm font-medium" style={{ color }}>
          Turn {step.turn_index} · {playerName}
        </span>
        <span className="text-xs text-gray-500">
          {pieceName} @ ({step.action?.anchor_row},{step.action?.anchor_col})
        </span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
        <div>
          <span className="text-gray-500">Mobility Δ</span>
          <span className={delta >= 0 ? 'text-green-400' : 'text-red-400'}> {delta >= 0 ? '+' : ''}{delta}</span>
        </div>
        <div>
          <span className="text-gray-500">Blocking</span>
          <span className={blocking > 0 ? 'text-amber-400' : 'text-gray-400'}> {blocking}</span>
        </div>
        <div>
          <span className="text-gray-500">Area gain</span>
          <span className="text-gray-300"> {areaGain}</span>
        </div>
        <div>
          <span className="text-gray-500">Corners Δ</span>
          <span className={cornersDelta >= 0 ? 'text-green-400' : 'text-red-400'}>
            {cornersDelta >= 0 ? '+' : ''}{cornersDelta}
          </span>
        </div>
      </div>
      <div className="text-[10px] text-gray-500 mt-1">
        Moves: {step.legal_moves_before} → {step.legal_moves_after}
        {blockEff > 0 && ` · Block eff: ${blockEff}`}
      </div>
    </div>
  );
};
