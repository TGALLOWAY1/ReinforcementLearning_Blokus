import React, { useState } from 'react';
import { usePentobiAvailability } from '../hooks/usePentobi';
import {
  PENTOBI_DEFAULT_LEVEL,
  PENTOBI_LEVEL_NOTES,
  PENTOBI_MOVE_CAP_MS,
  buildPentobiGameConfig,
  seatNumber,
  type PentobiGameConfig,
  type PlayerColor,
} from '../utils/pentobiConfig';

interface PentobiCardProps {
  isCreating: boolean;
  onPlay: (config: PentobiGameConfig) => void;
}

const COLORS: PlayerColor[] = ['RED', 'BLUE', 'GREEN', 'YELLOW'];

/**
 * "Play Pentobi" entry point for the human protocol (M5): the human takes one
 * seat, three Pentobi seats are served natively by the backend, every game is
 * logged server-side. Degrades to a clear message when the backend or the
 * engine binary is unavailable.
 */
export const PentobiCard: React.FC<PentobiCardProps> = ({ isCreating, onPlay }) => {
  const availability = usePentobiAvailability();
  const [level, setLevel] = useState<number>(PENTOBI_DEFAULT_LEVEL);
  const [humanPlayer, setHumanPlayer] = useState<PlayerColor>('RED');
  const { levelRange } = availability;
  const levels = Array.from({ length: levelRange.max - levelRange.min + 1 }, (_, i) => levelRange.min + i);
  const capSeconds = (availability.capMs ?? PENTOBI_MOVE_CAP_MS) / 1000;

  return (
    <div className="rounded-lg border border-neon-blue/40 bg-neon-blue/5 p-4 text-left" data-testid="pentobi-card">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-lg" aria-hidden>♟</span>
        <span className="text-sm font-bold text-neon-blue">Play Pentobi (served by the backend)</span>
      </div>
      <p className="text-xs text-gray-400 mb-3">
        You take one colour; three Pentobi seats play the others. Standard rules and scoring, {capSeconds} s cap per AI
        move, every game logged to <code className="text-gray-300">data/human_games/</code> for the 20-game protocol.
      </p>

      {availability.isLoading ? (
        <div className="text-xs text-gray-400 py-2">Checking the backend…</div>
      ) : !availability.available ? (
        <div className="text-xs text-gray-400 py-2">{availability.error}</div>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 mb-2">
            <label className="text-xs text-gray-300">
              <span className="block mb-1 font-medium">Pentobi level</span>
              <select
                value={level}
                onChange={(e) => setLevel(parseInt(e.target.value, 10))}
                className="w-full bg-charcoal-900 border border-charcoal-700 text-gray-200 rounded-md px-3 py-2 text-sm focus:outline-none focus:border-neon-blue"
                data-testid="pentobi-level"
              >
                {levels.map((lv) => (
                  <option key={lv} value={lv}>
                    Level {lv}
                  </option>
                ))}
              </select>
            </label>
            <label className="text-xs text-gray-300">
              <span className="block mb-1 font-medium">Your colour</span>
              <select
                value={humanPlayer}
                onChange={(e) => setHumanPlayer(e.target.value as PlayerColor)}
                className="w-full bg-charcoal-900 border border-charcoal-700 text-gray-200 rounded-md px-3 py-2 text-sm focus:outline-none focus:border-neon-blue"
                data-testid="pentobi-colour"
              >
                {COLORS.map((c) => (
                  <option key={c} value={c}>
                    {c.charAt(0) + c.slice(1).toLowerCase()} (moves {seatNumber(c)}{['st', 'nd', 'rd', 'th'][seatNumber(c) - 1]})
                  </option>
                ))}
              </select>
            </label>
          </div>
          <p className="text-[11px] text-gray-500 mb-3">{PENTOBI_LEVEL_NOTES[level]}</p>
          <button
            onClick={() => onPlay(buildPentobiGameConfig({ level, humanPlayer }))}
            disabled={isCreating}
            className={`w-full rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
              isCreating ? 'bg-charcoal-700 text-gray-500 cursor-not-allowed' : 'bg-neon-blue text-black hover:bg-neon-blue/80'
            }`}
            data-testid="pentobi-start"
          >
            {isCreating ? 'Starting...' : `Play Pentobi level ${level}`}
          </button>
        </>
      )}
    </div>
  );
};
