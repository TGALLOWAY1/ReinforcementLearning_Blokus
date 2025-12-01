import React, { useState } from 'react';
import {
  EnvironmentControlsSection,
  TrainingParametersSection,
  ModelStatusSection,
} from './ResearchSidebar';
import { PolicyView, ValueView } from './AgentVisualizations';

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
  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto">
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
    </div>
  );
};

