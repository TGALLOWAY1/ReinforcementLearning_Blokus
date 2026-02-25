/**
 * Telemetry / Charts dashboard for in-game visual debugging.
 * Live-updating charts from gameStore; supports Freeze and view last N turns.
 */

import React, { useState, useCallback, useMemo } from 'react';
import { useGameStore, computeLegalMovesByPiece, type GameState, type LegalMovesHistoryEntry } from '../store/gameStore';
import { LegalMovesBarChart } from './LegalMovesBarChart';
import { LegalMovesPerTurnPlot } from './LegalMovesPerTurnPlot';
import { MobilityBucketsChart } from './MobilityBucketsChart';
import { PolicyView } from './AgentVisualizations';
import { DebugLogsPanel } from './DebugLogsPanel';
import { FrontierSizePlot } from './FrontierSizePlot';
import { CornerDiffPlot } from './CornerDiffPlot';
import { ENABLE_DEBUG_UI } from '../constants/gameConstants';

type TelemetrySubTab = 'charts' | 'events';
type TurnsView = 20 | 50 | 'all';

function formatTs(): string {
  const d = new Date();
  return d.toTimeString().slice(0, 12);
}

export const TelemetryPanel: React.FC = () => {
  const gameState = useGameStore((s) => s.gameState);
  const legalMovesHistory = useGameStore((s) => s.legalMovesHistory);

  const [frozen, setFrozen] = useState(false);
  const [frozenSnapshot, setFrozenSnapshot] = useState<{
    gameState: GameState | null;
    legalMovesHistory: LegalMovesHistoryEntry[];
    legalMovesByPiece: { pieceId: number; count: number }[];
    timestamp: string;
  } | null>(null);
  const [turnsView, setTurnsView] = useState<TurnsView>('all');
  const [subTab, setSubTab] = useState<TelemetrySubTab>('charts');

  const captureSnapshot = useCallback(() => {
    const gs = useGameStore.getState().gameState;
    const hist = useGameStore.getState().legalMovesHistory;
    setFrozenSnapshot({
      gameState: gs,
      legalMovesHistory: hist,
      legalMovesByPiece: computeLegalMovesByPiece(gs),
      timestamp: formatTs(),
    });
  }, []);

  const handleFreezeToggle = useCallback(() => {
    if (!frozen) captureSnapshot();
    setFrozen((f) => !f);
  }, [frozen, captureSnapshot]);

  // When switching from frozen to live, clear snapshot
  const handleLive = useCallback(() => {
    setFrozen(false);
    setFrozenSnapshot(null);
  }, []);

  const effectiveGameState = frozen && frozenSnapshot ? frozenSnapshot.gameState : gameState;
  const effectiveHistory = frozen && frozenSnapshot ? frozenSnapshot.legalMovesHistory : legalMovesHistory;
  const effectiveLegalMovesByPiece = frozen && frozenSnapshot ? frozenSnapshot.legalMovesByPiece : null;

  const slicedHistory = useMemo(() => {
    if (turnsView === 'all') return effectiveHistory;
    const n = typeof turnsView === 'number' ? turnsView : 50;
    return effectiveHistory.slice(-n);
  }, [effectiveHistory, turnsView]);

  const currentPlayer = effectiveGameState?.current_player;
  const metrics = effectiveGameState?.mobility_metrics ?? (currentPlayer && effectiveHistory.length > 0
    ? effectiveHistory[effectiveHistory.length - 1]?.byPlayer?.[currentPlayer]
    : undefined);
  const totalCellWeighted = typeof metrics === 'object' && metrics !== null
    ? (metrics as { totalCellWeighted?: number }).totalCellWeighted
    : undefined;
  const legalMovesTotal = effectiveGameState?.legal_moves?.length ?? 0;

  const advMetrics = effectiveGameState?.current_player && effectiveGameState?.advanced_metrics
    ? effectiveGameState.advanced_metrics[effectiveGameState.current_player]
    : undefined;

  if (!ENABLE_DEBUG_UI) return null;

  if (subTab === 'events') {
    return (
      <div className="flex flex-col h-full">
        <div className="flex gap-1 p-2 border-b border-charcoal-700 shrink-0">
          <button
            type="button"
            onClick={() => setSubTab('charts')}
            className="flex-1 py-1.5 text-xs rounded bg-charcoal-800 text-gray-400"
          >
            Charts
          </button>
          <button
            type="button"
            onClick={() => setSubTab('events')}
            className="flex-1 py-1.5 text-xs rounded bg-charcoal-600 text-white"
          >
            Events
          </button>
        </div>
        <div className="flex-1 min-h-0">
          <DebugLogsPanel />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* KPI strip */}
      <div className="shrink-0 p-2 border-b border-charcoal-700">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px] font-mono">
          <div className="bg-charcoal-800 rounded px-2 py-1">
            <span className="text-gray-500">game_id</span>
            <div className="text-gray-300 truncate">{effectiveGameState?.game_id ?? '—'}</div>
          </div>
          <div className="bg-charcoal-800 rounded px-2 py-1">
            <span className="text-gray-500">plyIndex</span>
            <div className="text-gray-300">{effectiveGameState?.move_count ?? '—'}</div>
          </div>
          <div className="bg-charcoal-800 rounded px-2 py-1">
            <span className="text-gray-500">activePlayer</span>
            <div className="text-gray-300">{effectiveGameState?.current_player ?? '—'}</div>
          </div>
          <div className="bg-charcoal-800 rounded px-2 py-1">
            <span className="text-gray-500">legal_moves_total</span>
            <div className="text-gray-300">{legalMovesTotal}</div>
          </div>
          <div className="bg-charcoal-800 rounded px-2 py-1">
            <span className="text-gray-500">totalCellWeighted</span>
            <div className="text-gray-300">{totalCellWeighted != null ? totalCellWeighted.toFixed(2) : '—'}</div>
          </div>
          <div className="bg-charcoal-800 rounded px-2 py-1">
            <span className="text-gray-500">center_control</span>
            <div className="text-gray-300">{metrics?.centerControl ?? '—'}</div>
          </div>
          <div className="bg-charcoal-800 rounded px-2 py-1">
            <span className="text-gray-500">frontier_size</span>
            <div className="text-gray-300">{metrics?.frontierSize ?? '—'}</div>
          </div>
          <div className="bg-charcoal-800 rounded px-2 py-1">
            <span className="text-gray-500">corner_diff</span>
            <div className="text-gray-300">{advMetrics?.corner_differential ?? '—'}</div>
          </div>
          <div className="bg-charcoal-800 rounded px-2 py-1">
            <span className="text-gray-500">territory_%</span>
            <div className="text-gray-300">{advMetrics?.territory_ratio != null ? (advMetrics.territory_ratio * 100).toFixed(1) + '%' : '—'}</div>
          </div>
          <div className="bg-charcoal-800 rounded px-2 py-1">
            <span className="text-gray-500">piece_penalty</span>
            <div className="text-gray-300">{advMetrics?.piece_penalty ?? '—'}</div>
          </div>
          <div className="bg-charcoal-800 rounded px-2 py-1">
            <span className="text-gray-500">center_prox</span>
            <div className="text-gray-300">{advMetrics?.center_proximity ?? '—'}</div>
          </div>
          <div className="bg-charcoal-800 rounded px-2 py-1">
            <span className="text-gray-500">opp_adj</span>
            <div className="text-gray-300">{advMetrics?.opponent_adjacency ?? '—'}</div>
          </div>
          <div className="bg-charcoal-800 rounded px-2 py-1">
            <span className="text-gray-500">last_update</span>
            <div className="text-gray-300">{frozen && frozenSnapshot ? frozenSnapshot.timestamp : formatTs()}</div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap gap-2 mt-2 items-center">
          <button
            type="button"
            onClick={frozen ? handleLive : handleFreezeToggle}
            className={`px-2 py-1 text-xs rounded ${frozen ? 'bg-neon-yellow text-black' : 'bg-charcoal-700 text-gray-300'}`}
          >
            {frozen ? 'Live' : 'Freeze'}
          </button>
          <select
            value={turnsView}
            onChange={(e) => setTurnsView(e.target.value as TurnsView)}
            className="text-xs bg-charcoal-800 border border-charcoal-600 text-gray-200 rounded px-2 py-1"
          >
            <option value={20}>Last 20 turns</option>
            <option value={50}>Last 50 turns</option>
            <option value="all">All turns</option>
          </select>
          <div className="flex gap-1">
            <button
              type="button"
              onClick={() => setSubTab('charts')}
              className="px-2 py-1 text-xs rounded bg-charcoal-600 text-white"
            >
              Charts
            </button>
            <button
              type="button"
              onClick={() => setSubTab('events')}
              className="px-2 py-1 text-xs rounded bg-charcoal-800 text-gray-400"
            >
              Events
            </button>
          </div>
        </div>
      </div>

      {/* Chart wall */}
      <div className="flex-1 p-2 space-y-4 min-h-0">
        {/* Row A: LegalMovesBarChart + PolicyView */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="bg-charcoal-800/50 rounded-lg border border-charcoal-700 p-3">
            <h4 className="text-xs font-semibold text-gray-400 uppercase mb-0.5">Legal moves by piece</h4>
            <p className="text-[10px] text-gray-500 mb-2">Count per piece for active player (available pieces only)</p>
            <LegalMovesBarChart overrideData={effectiveLegalMovesByPiece ?? undefined} />
          </div>
          <div className="bg-charcoal-800/50 rounded-lg border border-charcoal-700 p-3">
            <h4 className="text-xs font-semibold text-gray-400 uppercase mb-0.5">Legal positions heatmap</h4>
            <p className="text-[10px] text-gray-500 mb-2">1.0 = legal, 0.0 = illegal (20×20)</p>
            <PolicyView overrideHeatmap={effectiveGameState?.heatmap} />
          </div>
        </div>

        {/* Row B: LegalMovesPerTurnPlot + MobilityBucketsChart */}
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-charcoal-800/50 rounded-lg border border-charcoal-700 p-3">
            <h4 className="text-xs font-semibold text-gray-400 uppercase mb-0.5">Mobility over time</h4>
            <p className="text-[10px] text-gray-500 mb-2">Cell-weighted mobility = Σ(P×S/O) per player per turn</p>
            <LegalMovesPerTurnPlot overrideHistory={slicedHistory} />
          </div>
          <div className="bg-charcoal-800/50 rounded-lg border border-charcoal-700 p-3">
            <h4 className="text-xs font-semibold text-gray-400 uppercase mb-0.5">Mobility by piece size</h4>
            <p className="text-[10px] text-gray-500 mb-2">Bucket counts (1=mono … 5=pento) per player</p>
            <MobilityBucketsChart overrideHistory={slicedHistory} />
          </div>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="bg-charcoal-800/50 rounded-lg border border-charcoal-700 p-3">
            <h4 className="text-xs font-semibold text-gray-400 uppercase mb-0.5">Frontier Size over time</h4>
            <p className="text-[10px] text-gray-500 mb-2">Number of frontier cells available to play</p>
            <FrontierSizePlot overrideHistory={slicedHistory} />
          </div>
          <div className="bg-charcoal-800/50 rounded-lg border border-charcoal-700 p-3">
            <h4 className="text-xs font-semibold text-gray-400 uppercase mb-0.5">Mobility over time</h4>
            <p className="text-[10px] text-gray-500 mb-2">Corner Differential (Own - Opponent)</p>
            <CornerDiffPlot overrideHistory={slicedHistory} />
          </div>
        </div>
      </div>
    </div>
  );
};
