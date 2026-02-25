/**
 * Debug / Logs panel for in-game telemetry.
 * Only rendered when ENABLE_DEBUG_UI is true (VITE_ENABLE_DEBUG_UI=true).
 */

import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useGameStore } from '../store/gameStore';
import {
  useDebugLogStore,
  DEBUG_EVENT_TYPES,
  type DebugEvent,
} from '../store/debugLogStore';
import { ENABLE_DEBUG_UI } from '../constants/gameConstants';
import { PIECE_SHAPES } from '../constants/gameConstants';

function piecesRemainingSummary(piecesUsed: Record<string, number[]> | undefined): string {
  if (!piecesUsed) return '—';
  const players = ['RED', 'BLUE', 'YELLOW', 'GREEN'] as const;
  return players
    .map((p) => {
      const used = piecesUsed[p]?.length ?? 0;
      const total = 21;
      const squares = (piecesUsed[p] || []).reduce((sum, pid) => {
        const shape = PIECE_SHAPES[pid];
        return sum + (shape ? shape.flat().filter((c) => c === 1).length : 0);
      }, 0);
      return `${p}: ${total - used}p (${squares}s)`;
    })
    .join(' | ');
}

function formatTs(ms: number): string {
  const d = new Date(ms);
  return d.toTimeString().slice(0, 12);
}

export const DebugLogsPanel: React.FC = () => {
  const gameState = useGameStore((s) => s.gameState);
  const legalMovesHistory = useGameStore((s) => s.legalMovesHistory);

  const events = useDebugLogStore((s) => s.events);
  const lastWsTimestamp = useDebugLogStore((s) => s.lastWsTimestamp);
  const lastStateDiff = useDebugLogStore((s) => s.lastStateDiff);
  const paused = useDebugLogStore((s) => s.paused);
  const autoScroll = useDebugLogStore((s) => s.autoScroll);
  const typeFilters = useDebugLogStore((s) => s.typeFilters);
  const searchText = useDebugLogStore((s) => s.searchText);

  const clear = useDebugLogStore((s) => s.clear);
  const setPaused = useDebugLogStore((s) => s.setPaused);
  const setAutoScroll = useDebugLogStore((s) => s.setAutoScroll);
  const toggleTypeFilter = useDebugLogStore((s) => s.toggleTypeFilter);
  const setSearchText = useDebugLogStore((s) => s.setSearchText);

  const [expandedId, setExpandedId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const filteredEvents = useMemo(() => {
    return events.filter((e) => {
      if (!typeFilters.has(e.type)) return false;
      if (searchText) {
        const text = JSON.stringify(e.payload).toLowerCase();
        if (!text.includes(searchText.toLowerCase())) return false;
      }
      return true;
    });
  }, [events, typeFilters, searchText]);

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [events, autoScroll]);

  const handleCopyLast = () => {
    const last = events[0];
    if (last) {
      navigator.clipboard.writeText(JSON.stringify(last, null, 2));
    }
  };

  if (!ENABLE_DEBUG_UI) return null;

  const lastHistoryEntry = legalMovesHistory[legalMovesHistory.length - 1];
  const metrics = lastHistoryEntry?.byPlayer?.[gameState?.current_player || ''] ?? gameState?.mobility_metrics;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="shrink-0 p-2 border-b border-charcoal-700">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Debug / Logs</h3>

        {/* Live game header */}
        <div className="text-[10px] font-mono text-gray-400 space-y-0.5 mb-2">
          <div>game_id: {gameState?.game_id ?? '—'}</div>
          <div>plyIndex / move_count: {gameState?.move_count ?? '—'}</div>
          <div>activePlayer: {gameState?.current_player ?? '—'}</div>
          <div>remaining: {piecesRemainingSummary(gameState?.pieces_used)}</div>
          <div>last WS: {lastWsTimestamp ? formatTs(lastWsTimestamp) : '—'}</div>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap gap-1 mb-2">
          <button
            type="button"
            onClick={() => setPaused(!paused)}
            className={`px-2 py-0.5 text-[10px] rounded ${paused ? 'bg-neon-yellow text-black' : 'bg-charcoal-700 text-gray-300'}`}
          >
            {paused ? 'Resume' : 'Pause'}
          </button>
          <button
            type="button"
            onClick={clear}
            className="px-2 py-0.5 text-[10px] rounded bg-charcoal-700 text-gray-300 hover:bg-charcoal-600"
          >
            Clear
          </button>
          <button
            type="button"
            onClick={handleCopyLast}
            className="px-2 py-0.5 text-[10px] rounded bg-charcoal-700 text-gray-300 hover:bg-charcoal-600"
          >
            Copy last JSON
          </button>
          <label className="flex items-center gap-1 text-[10px] text-gray-400">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
              className="rounded"
            />
            Auto-scroll
          </label>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-1 mb-1">
          {DEBUG_EVENT_TYPES.map((t) => (
            <label key={t} className="flex items-center gap-0.5 text-[10px]">
              <input
                type="checkbox"
                checked={typeFilters.has(t)}
                onChange={() => toggleTypeFilter(t)}
                className="rounded"
              />
              <span className="text-gray-500">{t.replace(/_/g, ' ')}</span>
            </label>
          ))}
        </div>
        <input
          type="text"
          placeholder="Search events..."
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          className="w-full px-2 py-0.5 text-[10px] bg-charcoal-800 border border-charcoal-600 rounded text-gray-300 placeholder-gray-500"
        />
      </div>

      {/* State diff */}
      {lastStateDiff && (
        <div className="shrink-0 p-2 border-b border-charcoal-700 text-[10px] font-mono">
          <h4 className="text-gray-500 mb-1">State diff (last step)</h4>
          <pre className="text-gray-400 overflow-x-auto max-h-24 overflow-y-auto">
            {JSON.stringify(lastStateDiff, null, 2)}
          </pre>
        </div>
      )}

      {/* Metrics snapshot */}
      {metrics && (
        <div className="shrink-0 p-2 border-b border-charcoal-700 text-[10px] font-mono">
          <h4 className="text-gray-500 mb-1">Mobility metrics</h4>
          <div className="text-gray-400">
            totalPlacements: {metrics.totalPlacements} | totalOrientationNormalized: {metrics.totalOrientationNormalized?.toFixed(2)} |
            totalCellWeighted: {metrics.totalCellWeighted?.toFixed(2)}
          </div>
          <div>buckets: {JSON.stringify(metrics.buckets)}</div>
        </div>
      )}

      {/* Event timeline */}
      <div className="flex-1 min-h-0 flex flex-col">
        <h4 className="text-[10px] text-gray-500 px-2 py-1 shrink-0">Event timeline ({filteredEvents.length})</h4>
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto overflow-x-hidden font-mono text-[10px]"
        >
          {filteredEvents.map((e) => (
            <EventRow
              key={e.id}
              event={e}
              expanded={expandedId === e.id}
              onToggle={() => setExpandedId((x) => (x === e.id ? null : e.id))}
            />
          ))}
          {filteredEvents.length === 0 && (
            <div className="p-2 text-gray-500 text-center">No events (or filters hide all)</div>
          )}
        </div>
      </div>
    </div>
  );
};

function EventRow({
  event,
  expanded,
  onToggle,
}: {
  event: DebugEvent;
  expanded: boolean;
  onToggle: () => void;
}) {
  const payloadStr = JSON.stringify(event.payload);
  const preview = payloadStr.length > 80 ? payloadStr.slice(0, 80) + '…' : payloadStr;

  return (
    <div
      className="border-b border-charcoal-800 hover:bg-charcoal-800/50 cursor-pointer"
      onClick={onToggle}
    >
      <div className="flex items-start gap-1 p-1">
        <span className="text-gray-500 shrink-0">{formatTs(event.timestamp)}</span>
        <span className="text-neon-blue shrink-0">{event.type}</span>
        <span className="text-gray-400 truncate">{preview}</span>
      </div>
      {expanded && (
        <pre className="p-2 bg-charcoal-900 text-gray-400 text-[9px] overflow-x-auto whitespace-pre-wrap break-all">
          {JSON.stringify(event.payload, null, 2)}
        </pre>
      )}
    </div>
  );
}
