import React, { useState } from 'react';
import {
  EnvironmentControlsSection,
  TrainingParametersSection,
  ModelStatusSection,
} from './ResearchSidebar';
import { PolicyView } from './AgentVisualizations';
import { TelemetryPanel } from './TelemetryPanel';
import { LegalMovesBarChart } from './LegalMovesBarChart';
import { ExplainMovePanel } from './ExplainMovePanel';
import { IS_DEPLOY_PROFILE, ENABLE_DEBUG_UI } from '../constants/gameConstants';
import { useGameStore } from '../store/gameStore';
import { MergedAnalysisPanel } from './telemetry/MergedAnalysisPanel';

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

/** Hint modal: shows Legal Moves by Piece and Legal Positions grids. */
const HintModal: React.FC<{ onClose: () => void }> = ({ onClose }) => (
  <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4" onClick={onClose}>
    <div
      className="bg-charcoal-800 border border-charcoal-700 rounded-xl max-w-xl w-full max-h-[90vh] overflow-y-auto"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-center justify-between p-4 border-b border-charcoal-700">
        <h2 className="text-lg font-bold text-gray-200">💡 Move Hints</h2>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-gray-200 transition-colors"
          title="Close"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      <div className="p-4 space-y-4">
        <section>
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Legal Moves by Piece
          </h3>
          <LegalMovesBarChart />
        </section>
        <section>
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Legal Positions
          </h3>
          <PolicyView />
        </section>
      </div>
    </div>
  </div>
);

interface RightPanelProps {
  onNewGame?: () => void;
}

export const RightPanel: React.FC<RightPanelProps> = ({ onNewGame }) => {
  const activeTab = useGameStore((s) => s.activeRightTab);
  const setActiveTab = useGameStore((s) => s.setActiveRightTab);
  const gameState = useGameStore((s) => s.gameState);
  const [showHint, setShowHint] = useState(false);

  const isDemoMode =
    (gameState?.players as any)?.demo_mode ||
    window.location.search.includes('demo=1') ||
    gameState?.players?.length === 4;

  // Deploy / Demo mode: simplified nav
  if (IS_DEPLOY_PROFILE || isDemoMode) {
    return (
      <div className="h-full flex flex-col overflow-hidden">
        <div className="p-4 border-b border-charcoal-700 shrink-0 flex flex-col gap-2">
          <button
            onClick={onNewGame}
            className="w-full py-3 px-4 rounded-lg font-medium bg-neon-blue text-black hover:bg-neon-blue/80 transition-colors"
          >
            New Game
          </button>

          <div className="flex gap-1">
            <button
              type="button"
              onClick={() => setActiveTab('explanation')}
              className={`flex-1 py-1.5 text-xs rounded ${activeTab === 'explanation' ? 'bg-charcoal-600 text-white' : 'bg-charcoal-800 text-gray-400 hover:text-gray-200'}`}
            >
              Explanation
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('analysis')}
              className={`flex-1 py-1.5 text-xs rounded ${activeTab === 'analysis' ? 'bg-charcoal-600 text-white' : 'bg-charcoal-800 text-gray-400 hover:text-gray-200'}`}
            >
              Analysis
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-hidden">
          {activeTab === 'explanation' ? (
            <ExplainMovePanel />
          ) : activeTab === 'analysis' ? (
            <MergedAnalysisPanel />
          ) : activeTab === 'telemetry' ? (
            <TelemetryPanel />
          ) : null}
        </div>

        {showHint && <HintModal onClose={() => setShowHint(false)} />}
      </div>
    );
  }

  // Research mode
  const researchMainContent = (
    <div className="h-full overflow-y-auto">
      <CollapsibleSection title="Environment Controls" defaultOpen>
        <EnvironmentControlsSection onNewGame={onNewGame} />
      </CollapsibleSection>
      <CollapsibleSection title="Training Parameters" defaultOpen={false}>
        <TrainingParametersSection />
      </CollapsibleSection>
      <CollapsibleSection title="Model Status" defaultOpen={false}>
        <ModelStatusSection />
      </CollapsibleSection>
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
            onClick={() => setActiveTab('analysis')}
            className={`flex-1 py-1.5 text-xs rounded ${activeTab === 'analysis' ? 'bg-charcoal-600 text-white' : 'bg-charcoal-800 text-gray-400 hover:text-gray-200'}`}
          >
            Analysis
          </button>
        </div>
      )}
      <div className="flex-1 overflow-hidden">
        {ENABLE_DEBUG_UI && activeTab === 'analysis' ? <MergedAnalysisPanel /> :
          ENABLE_DEBUG_UI && activeTab === 'telemetry' ? <TelemetryPanel /> : researchMainContent}
      </div>

      {showHint && <HintModal onClose={() => setShowHint(false)} />}
    </div>
  );
};

/** Exported for use in Play.tsx near the board */
export { HintModal };
