import { describe, it, expect } from 'vitest';
import {
  PENTOBI_LEVEL_NOTES,
  PENTOBI_MOVE_CAP_MS,
  buildPentobiGameConfig,
} from '../pentobiConfig';

describe('buildPentobiGameConfig', () => {
  it('seats the human where asked and Pentobi everywhere else', () => {
    const cfg = buildPentobiGameConfig({ level: 7, humanPlayer: 'BLUE', seed: 123 });
    expect(cfg.players.map((p) => p.player)).toEqual(['RED', 'BLUE', 'GREEN', 'YELLOW']);
    expect(cfg.players.map((p) => p.agent_type)).toEqual(['pentobi', 'human', 'pentobi', 'pentobi']);
    expect(cfg.transport).toBe('backend');
    expect(cfg.orientation_space).toBe('frontend');
    expect(cfg.scoring_mode).toBe('standard');
    expect(cfg.auto_start).toBe(true);
    for (const seat of cfg.players.filter((p) => p.agent_type === 'pentobi')) {
      expect(seat.agent_config.level).toBe(7);
      expect(seat.agent_config.time_budget_ms).toBe(PENTOBI_MOVE_CAP_MS);
    }
  });

  it('derives distinct per-seat seeds from the game seed so seats do not mirror each other', () => {
    const cfg = buildPentobiGameConfig({ level: 3, humanPlayer: 'RED', seed: 42 });
    const seeds = cfg.players.filter((p) => p.agent_type === 'pentobi').map((p) => p.agent_config.seed);
    expect(new Set(seeds).size).toBe(3);
    expect(buildPentobiGameConfig({ level: 3, humanPlayer: 'RED', seed: 42 })).toEqual(cfg);
  });

  it('defaults to level 3, the human on RED and a fresh seed', () => {
    const a = buildPentobiGameConfig({});
    const b = buildPentobiGameConfig({});
    expect(a.players[0]).toEqual({ player: 'RED', agent_type: 'human', agent_config: {} });
    expect(a.players[1].agent_config.level).toBe(3);
    expect(a.players[1].agent_config.seed).not.toBe(b.players[1].agent_config.seed);
  });

  it('rejects levels outside 1-9', () => {
    expect(() => buildPentobiGameConfig({ level: 0 })).toThrow(/level/);
    expect(() => buildPentobiGameConfig({ level: 10 })).toThrow(/level/);
  });

  it('describes every level for the picker', () => {
    for (let level = 1; level <= 9; level += 1) {
      expect(PENTOBI_LEVEL_NOTES[level]).toMatch(/\S/);
    }
    expect(PENTOBI_MOVE_CAP_MS).toBe(10000);
  });
});

describe('seat numbering', () => {
  it('follows the engine turn order RED, BLUE, YELLOW, GREEN', async () => {
    const { seatNumber } = await import('../pentobiConfig');
    expect(['RED', 'BLUE', 'YELLOW', 'GREEN'].map((c) => seatNumber(c as any))).toEqual([1, 2, 3, 4]);
  });
});
