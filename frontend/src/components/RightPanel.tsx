import React, { useState } from 'react';
import {
  EnvironmentControlsSection,
  TrainingParametersSection,
  ModelStatusSection,
} from './ResearchSidebar';
import { PolicyView, ValueView } from './AgentVisualizations';
import { LegalMovesBarChart } from './LegalMovesBarChart';
import { LegalMovesPerTurnPlot } from './LegalMovesPerTurnPlot';
import { MobilityBucketsChart } from './MobilityBucketsChart';
import { MctsTopMovesTable } from './MctsTopMovesTable';
import { TelemetryPanel } from './TelemetryPanel';
import { IS_DEPLOY_PROFILE, ENABLE_DEBUG_UI } from '../constants/gameConstants';
import { useGameStore } from '../store/gameStore';

interface CollapsibleSectionProps {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
  title,
  children,
  defaultOpen = true,
}) => {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <section className="border-b border-charcoal-700">
      <button
        type="button"
        className="w-full flex items-center justify-between px-3 py-2 text-sm font-semibold text-gray-300 hover:bg-charcoal-800 transition-colors"
        onClick={() => setOpen((o) => !o)}
      >
        <span>{title}</span>
        <span className="text-neon-blue text-lg">{open ? '−' : '+'}</span>
      </button>
      {open && <div className="px-3 pb-3">{children}</div>}
    </section>
  );
};

interface RightPanelProps {
  onNewGame?: () => void;
}


export const RightPanel: React.FC<RightPanelProps> = ({ onNewGame }) => {
  const activeTab = useGameStore((s) => s.activeRightTab);
  const setActiveTab = useGameStore((s) => s.setActiveRightTab);

  // Deploy: New Game + Legal Moves chart + Legal Positions grid
  if (IS_DEPLOY_PROFILE) {
    return (
      <div className="h-full flex flex-col overflow-hidden">
        <div className="p-4 border-b border-charcoal-700 shrink-0 flex flex-col gap-2">
          <button
            onClick={onNewGame}
            className="w-full py-3 px-4 rounded-lg font-medium bg-neon-blue text-black hover:bg-neon-blue/80 transition-colors"
          >
            New Game
          </button>
          {ENABLE_DEBUG_UI && (
            <div className="flex gap-1">
              <button
                type="button"
                onClick={() => setActiveTab('main')}
                className={`flex-1 py-1.5 text-xs rounded ${activeTab === 'main' ? 'bg-charcoal-600 text-white' : 'bg-charcoal-800 text-gray-400 hover:text-gray-200'}`}
              >
                Game
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('telemetry')}
                className={`flex-1 py-1.5 text-xs rounded ${activeTab === 'telemetry' ? 'bg-charcoal-600 text-white' : 'bg-charcoal-800 text-gray-400 hover:text-gray-200'}`}
              >
                Telemetry
              </button>
            </div>
          )}
        </div>
        <div className="flex-1 overflow-hidden">
          {activeTab === 'telemetry' ? (
            <TelemetryPanel />
          ) : (
            <div className="h-full overflow-y-auto">
              <section className="p-3 border-b border-charcoal-700">
                <LegalMovesBarChart />
              </section>
              <section className="p-3 border-b border-charcoal-700">
                <LegalMovesPerTurnPlot />
              </section>
              <section className="p-3 border-b border-charcoal-700">
                <MobilityBucketsChart />
              </section>
              <MctsTopMovesTable />
              <section className="p-3 border-b border-charcoal-700">
                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
                  Legal Positions
                </h3>
                <PolicyView />
              </section>
            </div>
          )}
        </div>
      </div>
    );
  }

  const researchMainContent = (
    <div className="h-full overflow-y-auto">
      {/* Collapsible controls */}
      <CollapsibleSection title="Environment Controls" defaultOpen>
        <EnvironmentControlsSection onNewGame={onNewGame} />
      </CollapsibleSection>

      <CollapsibleSection title="Training Parameters" defaultOpen>
        <TrainingParametersSection />
      </CollapsibleSection>

      <CollapsibleSection title="Model Status" defaultOpen>
        <ModelStatusSection />
      </CollapsibleSection>

      {/* Legal Moves Over Time */}
      <section className="p-3 border-b border-charcoal-700">
        <LegalMovesPerTurnPlot />
      </section>

      {/* Mobility by Piece Size */}
      <section className="p-3 border-b border-charcoal-700">
        <MobilityBucketsChart />
      </section>

      <MctsTopMovesTable />

      {/* Always-visible Policy & Value */}
      <section className="p-3 border-b border-charcoal-700">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Policy
        </h3>
        <PolicyView />
      </section>

      <section className="p-3 border-b border-charcoal-700">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Value
        </h3>
        <ValueView />
      </section>

      {/* Optional Tree - can be added later as a collapsible section if needed */}
    </div>
  );

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {ENABLE_DEBUG_UI && (
        <div className="flex gap-1 p-2 border-b border-charcoal-700 shrink-0">
          <button
            type="button"
            onClick={() => setActiveTab('main')}
            className={`flex-1 py-1.5 text-xs rounded ${activeTab === 'main' ? 'bg-charcoal-600 text-white' : 'bg-charcoal-800 text-gray-400 hover:text-gray-200'}`}
          >
            Game
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('telemetry')}
            className={`flex-1 py-1.5 text-xs rounded ${activeTab === 'telemetry' ? 'bg-charcoal-600 text-white' : 'bg-charcoal-800 text-gray-400 hover:text-gray-200'}`}
          >
            Telemetry
          </button>
        </div>
      )}
      <div className="flex-1 overflow-hidden">
        {ENABLE_DEBUG_UI && activeTab === 'telemetry' ? <TelemetryPanel /> : researchMainContent}
      </div>
    </div>
  );
};

