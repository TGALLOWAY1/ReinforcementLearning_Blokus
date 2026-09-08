import { describe, it, expect, vi } from 'vitest';
import {
  createBackendGame,
  isBackendConfig,
  postAdvanceTurn,
  postMove,
  postPass,
  toBackendPayload,
} from '../backendTransport';

const okJson = (body: unknown) =>
  ({ ok: true, status: 200, json: async () => body }) as unknown as Response;

const badJson = (status: number, body: unknown) =>
  ({ ok: false, status, json: async () => body }) as unknown as Response;

const config = {
  transport: 'backend' as const,
  players: [
    { player: 'RED', agent_type: 'human', agent_config: {} },
    { player: 'BLUE', agent_type: 'pentobi', agent_config: { level: 3, seed: 1, time_budget_ms: 10000 } },
  ],
  auto_start: true,
  scoring_mode: 'standard',
};

describe('backend transport payload', () => {
  it('strips the transport marker and asks for frontend orientation indices', () => {
    const payload = toBackendPayload(config) as Record<string, unknown>;
    expect(payload.transport).toBeUndefined();
    expect(payload.orientation_space).toBe('frontend');
    expect(payload.players).toEqual(config.players);
    expect(payload.scoring_mode).toBe('standard');
  });

  it('recognises backend-served configs', () => {
    expect(isBackendConfig(config)).toBe(true);
    expect(isBackendConfig({ players: [] })).toBe(false);
    expect(isBackendConfig(null)).toBe(false);
  });
});

describe('backend transport requests', () => {
  it('creates a game with POST /api/games', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(okJson({ game_id: 'g1', game_state: { game_id: 'g1' } }));
    const result = await createBackendGame(config, { fetchImpl, apiBase: 'http://api' });
    expect(result.game_id).toBe('g1');
    expect(result.game_state).toEqual({ game_id: 'g1' });
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe('http://api/api/games');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body).orientation_space).toBe('frontend');
    expect(JSON.parse(init.body).transport).toBeUndefined();
  });

  it('posts a human move with the nested move shape the API expects', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(okJson({ success: true, message: 'ok', game_state: { game_id: 'g1' } }));
    const resp = await postMove('g1', { player: 'RED', piece_id: 5, orientation: 6, anchor_row: 1, anchor_col: 2 },
      { fetchImpl, apiBase: '' });
    expect(resp.success).toBe(true);
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe('/api/games/g1/move');
    expect(JSON.parse(init.body)).toEqual({
      move: { piece_id: 5, orientation: 6, anchor_row: 1, anchor_col: 2 },
      player: 'RED',
    });
  });

  it('posts a pass and an agent turn', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(okJson({ success: true, message: 'ok', game_state: {} }));
    await postPass('g1', 'RED', { fetchImpl, apiBase: '' });
    await postAdvanceTurn('g1', { fetchImpl, apiBase: '' });
    expect(fetchImpl.mock.calls[0][0]).toBe('/api/games/g1/pass');
    expect(JSON.parse(fetchImpl.mock.calls[0][1].body)).toEqual({ player: 'RED' });
    expect(fetchImpl.mock.calls[1][0]).toBe('/api/games/g1/advance_turn');
    expect(fetchImpl.mock.calls[1][1].method).toBe('POST');
  });

  it('surfaces the backend detail message on errors', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(badJson(400, { detail: 'pentobi-gtp is not installed on this backend' }));
    await expect(createBackendGame(config, { fetchImpl, apiBase: '' })).rejects.toThrow(/pentobi-gtp is not installed/);
  });
});
