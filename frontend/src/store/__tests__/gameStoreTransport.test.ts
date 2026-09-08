import { beforeEach, describe, expect, it, vi } from 'vitest';

/** Minimal stand-in for the Pyodide worker: records posted messages, lets the test push replies. */
class FakeWorker {
  static instances: FakeWorker[] = [];
  posted: any[] = [];
  onmessage: ((e: { data: any }) => void) | null = null;
  constructor() {
    FakeWorker.instances.push(this);
  }
  postMessage(msg: any) {
    this.posted.push(msg);
  }
  reply(data: any) {
    this.onmessage?.({ data });
  }
}

vi.mock('../blokusWorker?worker', () => ({ default: FakeWorker }));

const state = (game_id: string, current_player: string, agent_type: string) => ({
  game_id,
  status: 'in_progress',
  current_player,
  board: Array.from({ length: 20 }, () => Array(20).fill(0)),
  scores: { RED: 0, BLUE: 0, GREEN: 0, YELLOW: 0 },
  pieces_used: { RED: [], BLUE: [], GREEN: [], YELLOW: [] },
  move_count: 0,
  game_over: false,
  winner: null,
  legal_moves: [],
  created_at: '',
  updated_at: '',
  players: [
    { player: 'RED', agent_type, agent_config: {} },
    { player: 'BLUE', agent_type: 'human', agent_config: {} },
  ],
});

const okJson = (body: unknown) => ({ ok: true, status: 200, json: async () => body });

const flush = () => new Promise((r) => setTimeout(r, 0));

/** The store polls worker readiness every 100 ms; wait until it has posted `type`. */
const untilPosted = async (worker: FakeWorker, type: string) => {
  for (let i = 0; i < 50; i += 1) {
    if (worker.posted.some((m) => m.type === type)) return;
    await new Promise((r) => setTimeout(r, 20));
  }
  throw new Error(`worker never received ${type}`);
};

describe('game store transports', () => {
  let store: typeof import('../gameStore').useGameStore;

  beforeEach(async () => {
    vi.resetModules();
    FakeWorker.instances = [];
    store = (await import('../gameStore')).useGameStore;
  });

  it('ignores a late worker reply once a backend game has replaced the worker game', async () => {
    // Worker game: first seat is an MCTS agent, so the store asks the worker to advance.
    const created = store.getState().createGame({ players: [], auto_start: true });
    await flush();
    const worker = FakeWorker.instances[0];
    worker.reply({ type: 'ready' });
    await untilPosted(worker, 'init_game');
    worker.reply({ type: 'state_update', state: state('local-1', 'RED', 'mcts') });
    await created;
    expect(worker.posted.some((m) => m.type === 'advance_turn')).toBe(true);

    // Backend game starts while the worker is still "thinking".
    const fetchImpl = vi.fn().mockResolvedValue(okJson({ game_id: 'b1', game_state: state('b1', 'BLUE', 'pentobi') }));
    vi.stubGlobal('fetch', fetchImpl);
    await store.getState().createGame({ transport: 'backend', players: [], auto_start: true });
    expect(store.getState().transport).toBe('backend');
    expect(store.getState().gameState?.game_id).toBe('b1');

    // The worker's late move_response must not clobber the backend game.
    worker.reply({ type: 'move_response', response: { success: true, message: 'ok', game_state: state('local-1', 'BLUE', 'mcts') } });
    await flush();
    expect(store.getState().gameState?.game_id).toBe('b1');
    expect(store.getState().transport).toBe('backend');
  });

  it('ignores a late backend reply once another game has started', async () => {
    let resolveAdvance: (v: any) => void = () => {};
    const fetchImpl = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith('/advance_turn')) return new Promise((r) => (resolveAdvance = r));
      return Promise.resolve(okJson({ game_id: 'b1', game_state: state('b1', 'RED', 'pentobi') }));
    });
    vi.stubGlobal('fetch', fetchImpl);
    await store.getState().createGame({ transport: 'backend', players: [], auto_start: true });
    await store.getState().connect('b1');
    expect(store.getState().isAdvancingTurn).toBe(true);

    // A new worker game replaces it before the backend answers.
    const created = store.getState().createGame({ players: [], auto_start: true });
    await flush();
    const worker = FakeWorker.instances[0];
    worker.reply({ type: 'ready' });
    await untilPosted(worker, 'init_game');
    worker.reply({ type: 'state_update', state: state('local-2', 'BLUE', 'human') });
    await created;
    expect(store.getState().transport).toBe('worker');

    resolveAdvance(okJson({ success: true, message: 'Agent moved', game_state: state('b1', 'BLUE', 'pentobi') }));
    await flush();
    await flush();
    expect(store.getState().gameState?.game_id).toBe('local-2');
    expect(store.getState().isAdvancingTurn).toBe(false);
  });

  it('loadGame after a backend game goes back to the worker transport', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(okJson({ game_id: 'b1', game_state: state('b1', 'BLUE', 'pentobi') })));
    await store.getState().createGame({ transport: 'backend', players: [], auto_start: true });
    expect(store.getState().transport).toBe('backend');
    const loaded = store.getState().loadGame([]);
    await flush();
    const worker = FakeWorker.instances[0];
    worker.reply({ type: 'ready' });
    await untilPosted(worker, 'load_game');
    worker.reply({ type: 'state_update', state: state('local-webworker', 'BLUE', 'human') });
    await loaded;
    expect(store.getState().transport).toBe('worker');
    expect(worker.posted.some((m) => m.type === 'load_game')).toBe(true);
  });
});
