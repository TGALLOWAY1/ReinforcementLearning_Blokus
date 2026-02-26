import React, { useState } from 'react';
import { DebugLogsPanel } from './DebugLogsPanel';
import { AnalysisDashboard } from './AnalysisDashboard';
import { ENABLE_DEBUG_UI } from '../constants/gameConstants';

type TelemetrySubTab = 'charts' | 'events';

export const TelemetryPanel: React.FC = () => {
  const [subTab, setSubTab] = useState<TelemetrySubTab>('charts');

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
            Charts / Dashboard
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
    <div className="flex flex-col h-full overflow-hidden">
      {/* Controls */}
      <div className="shrink-0 p-2 border-b border-charcoal-700">
        <div className="flex flex-wrap gap-2 items-center">
          <div className="flex gap-1">
            <button
              type="button"
              onClick={() => setSubTab('charts')}
              className="px-2 py-1 text-xs rounded bg-charcoal-600 text-white"
            >
              Charts / Dashboard
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

      <div className="flex-1 p-2 min-h-0 overflow-y-auto">
        <AnalysisDashboard />
      </div>
    </div>
  );
};
