import React, { useState } from 'react';
import {
  EnvironmentControlsSection,
  TrainingParametersSection,
  ModelStatusSection,
} from './ResearchSidebar';
import { PolicyView, ValueView } from './AgentVisualizations';
import { TelemetryPanel } from './TelemetryPanel';
import { LegalMovesBarChart } from './LegalMovesBarChart';
import {
  ModuleC_CornerChart,
  ModuleE_FrontierChart,
} from './AnalysisDashboard';
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
  const gameState = useGameStore((s) => s.gameState);
  const gameHistory = gameState?.game_history || [];
  const liveTurn = gameHistory.length;

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
                Dashboard
              </button>
            </div>
          )}
        </div>
        <div className="flex-1 overflow-hidden">
          {activeTab === 'telemetry' ? (
            <TelemetryPanel />
          ) : (
            <div className="h-full overflow-y-auto">
              {/* Restored Live Charts for the Main Tab */}
              <section className="p-3 border-b border-charcoal-700">
                <LegalMovesBarChart />
              </section>

              <section className="p-3 border-b border-charcoal-700 h-[220px]">
                <ModuleE_FrontierChart gameHistory={gameHistory} currentTurn={liveTurn} />
              </section>

              <section className="p-3 border-b border-charcoal-700 h-[220px]">
                <ModuleC_CornerChart gameHistory={gameHistory} currentTurn={liveTurn} />
              </section>

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

      <CollapsibleSection title="Training Parameters" defaultOpen={false}>
        <TrainingParametersSection />
      </CollapsibleSection>

      <CollapsibleSection title="Model Status" defaultOpen={false}>
        <ModelStatusSection />
      </CollapsibleSection>

      {/* Restored Live Charts for the Main Tab */}
      <section className="p-3 border-b border-charcoal-700">
        <LegalMovesBarChart />
      </section>

      <section className="p-3 border-b border-charcoal-700 h-[220px]">
        <ModuleE_FrontierChart gameHistory={gameHistory} currentTurn={liveTurn} />
      </section>

      <section className="p-3 border-b border-charcoal-700 h-[220px]">
        <ModuleC_CornerChart gameHistory={gameHistory} currentTurn={liveTurn} />
      </section>

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
            Dashboard
          </button>
        </div>
      )}
      <div className="flex-1 overflow-hidden">
        {ENABLE_DEBUG_UI && activeTab === 'telemetry' ? <TelemetryPanel /> : researchMainContent}
      </div>
    </div>
  );
};

