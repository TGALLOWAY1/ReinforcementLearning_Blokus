import React from 'react';
import { AnalysisDashboard } from './AnalysisDashboard';
import { ENABLE_DEBUG_UI } from '../constants/gameConstants';

export const TelemetryPanel: React.FC = () => {
  if (!ENABLE_DEBUG_UI) return null;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-1 min-h-0 overflow-y-auto">
        <AnalysisDashboard />
      </div>
    </div>
  );
};
