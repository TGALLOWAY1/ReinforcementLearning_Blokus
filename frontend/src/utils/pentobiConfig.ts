import { BACKEND_TRANSPORT } from '../store/backendTransport';

/**
 * Game config for the human protocol (M5): one human seat, three Pentobi
 * seats served natively by the backend (D-024). Pentobi's level is the
 * strength knob; the per-move cap is the D-019 hard limit of 10 s, enforced
 * server-side with a deterministic fallback move.
 */

export const PENTOBI_MOVE_CAP_MS = 10000;
export const PENTOBI_MIN_LEVEL = 1;
export const PENTOBI_MAX_LEVEL = 9;
export const PENTOBI_DEFAULT_LEVEL = 3;

const PLAYER_COLORS_ORDER = ['RED', 'BLUE', 'GREEN', 'YELLOW'] as const;
export type PlayerColor = (typeof PLAYER_COLORS_ORDER)[number];

/** Engine turn order (engine/board.py Player enum): RED 1, BLUE 2, YELLOW 3, GREEN 4. */
export const ENGINE_TURN_ORDER: readonly PlayerColor[] = ['RED', 'BLUE', 'YELLOW', 'GREEN'];
export const seatNumber = (color: PlayerColor): number => ENGINE_TURN_ORDER.indexOf(color) + 1;

/**
 * Measured on this project's Mac, one thread, per move: levels 1-3 and 7 from
 * the protocol-v3 runs EXP-016/016b (2,100-2,500 moves each); levels 4-6, 8, 9
 * from one self-play game each (training/reports/m4_serving_gate/level_timings.json).
 */
export const PENTOBI_LEVEL_NOTES: Record<number, string> = {
  1: '~3 simulations, 8 ms/move. Ties the repo\'s best 8-second search (EXP-016b).',
  2: '~30 simulations, 8 ms/move. Beats the repo\'s best search by 4 points (EXP-016b).',
  3: '~90 simulations, 10 ms/move (max 0.09 s). Beats the repo\'s best search by 8 points; M3 target anchor.',
  4: '~180 simulations, 5 ms/move (max 0.02 s).',
  5: '~670 simulations, 10 ms/move (max 0.03 s).',
  6: '~5,000 simulations, 0.06 s/move (max 0.24 s).',
  7: '~70,000 simulations, 1.2 s/move (max 4.9 s). Beats the repo\'s best search by 28 points (EXP-016).',
  8: '~350,000 simulations, 1.9 s/move (max 8.6 s in one game): above the 8 s M4 latency gate; not gated at this level.',
  9: '~1.7 million simulations, 9.4 s/move mean, 90th percentile 27 s, max 54 s: many moves hit the 10 s cap and fall back to the greedy move.',
};

export interface BuildPentobiGameConfigOptions {
  level?: number;
  humanPlayer?: PlayerColor;
  /** Game seed; per-seat engine seeds are derived from it so seats differ. */
  seed?: number;
  scoringMode?: 'standard' | 'house';
}

export interface PentobiSeatConfig {
  level: number;
  seed: number;
  time_budget_ms: number;
}

export interface PentobiGameSeat {
  player: PlayerColor;
  agent_type: 'human' | 'pentobi';
  agent_config: PentobiSeatConfig | Record<string, never>;
}

export interface PentobiGameConfig {
  transport: typeof BACKEND_TRANSPORT;
  orientation_space: 'frontend';
  players: PentobiGameSeat[];
  auto_start: true;
  scoring_mode: 'standard' | 'house';
}

const freshSeed = (): number => Math.floor(Math.random() * 2_000_000_000);

export function buildPentobiGameConfig(options: BuildPentobiGameConfigOptions = {}): PentobiGameConfig {
  const level = options.level ?? PENTOBI_DEFAULT_LEVEL;
  if (!Number.isInteger(level) || level < PENTOBI_MIN_LEVEL || level > PENTOBI_MAX_LEVEL) {
    throw new Error(`Pentobi level must be an integer in ${PENTOBI_MIN_LEVEL}-${PENTOBI_MAX_LEVEL}`);
  }
  const humanPlayer = options.humanPlayer ?? 'RED';
  const seed = options.seed ?? freshSeed();
  const players: PentobiGameSeat[] = PLAYER_COLORS_ORDER.map((color, index): PentobiGameSeat => {
    if (color === humanPlayer) {
      return { player: color, agent_type: 'human', agent_config: {} };
    }
    return {
      player: color,
      agent_type: 'pentobi',
      agent_config: { level, seed: (seed + 1000 * (index + 1)) % 2_147_483_647, time_budget_ms: PENTOBI_MOVE_CAP_MS },
    };
  });
  return {
    transport: BACKEND_TRANSPORT,
    orientation_space: 'frontend',
    players,
    auto_start: true,
    scoring_mode: options.scoringMode ?? 'standard',
  };
}
