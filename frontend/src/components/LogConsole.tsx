import React, { useEffect, useRef } from 'react';
import { useGameStore } from '../store/gameStore';

export const LogConsole: React.FC = () => {
  const logs = useGameStore((state) => state.logs);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const formatLogMessage = (log: { timestamp: string; message: string; level?: 'INFO' | 'WARN' | 'ERROR' }) => {
    const level = log.level || 'INFO';
    return `[${log.timestamp}] ${level}: ${log.message}`;
  };

  return (
    <div className="h-full bg-black text-gray-200 font-mono text-xs overflow-y-auto" ref={scrollRef}>
      <div className="p-2 space-y-0.5">
        {logs.length === 0 ? (
          <div className="text-gray-600">No logs yet...</div>
        ) : (
          logs.map((log, index) => (
            <div
              key={index}
              className="text-neon-green"
            >
              {formatLogMessage(log)}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

