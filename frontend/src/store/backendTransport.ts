import { API_BASE } from '../constants/gameConstants';

/**
 * HTTP transport for games served by the FastAPI backend (M4).
 *
 * The default Play page runs the engine in a Pyodide web worker. Games that
 * need a natively served agent (Pentobi, D-024) are created with
 * `transport: 'backend'` and driven through the REST gameplay routes instead:
 * POST /api/games, /move, /pass and /advance_turn. The backend is asked to
 * speak the frontend's orientation indices (`orientation_space: 'frontend'`),
 * so the UI sends and receives the same indices it uses with the worker.
 */

export interface TransportOptions {
  fetchImpl?: typeof fetch;
  apiBase?: string;
}

export interface BackendMoveRequest {
  player: string;
  piece_id: number;
  orientation: number;
  anchor_row: number;
  anchor_col: number;
}

export interface BackendMoveResponse<S = unknown> {
  success: boolean;
  message: string;
  game_state?: S;
}

export interface BackendCreateResponse<S = unknown> {
  game_id: string;
  game_state: S;
  message?: string;
}

export const BACKEND_TRANSPORT = 'backend';

export function isBackendConfig(config: unknown): boolean {
  return Boolean(config && typeof config === 'object' && (config as { transport?: string }).transport === BACKEND_TRANSPORT);
}

/** Payload for POST /api/games: the transport marker is client-only. */
export function toBackendPayload(config: Record<string, unknown>): Record<string, unknown> {
  const { transport: _transport, ...rest } = config;
  return { ...rest, orientation_space: (rest.orientation_space as string | undefined) ?? 'frontend' };
}

/** FastAPI validation errors carry `detail` as a list of {loc, msg}; render them readably. */
function describeDetail(detail: unknown): string | null {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((d) => (d && typeof d === 'object' && 'msg' in d
        ? `${Array.isArray((d as any).loc) ? (d as any).loc.slice(1).join('.') + ': ' : ''}${(d as any).msg}`
        : String(d)))
      .filter(Boolean);
    if (parts.length) return parts.join('; ');
  }
  return null;
}

async function request<T>(path: string, body: unknown, options: TransportOptions, method = 'POST'): Promise<T> {
  const fetchImpl = options.fetchImpl ?? fetch;
  const apiBase = options.apiBase ?? API_BASE;
  const response = await fetchImpl(`${apiBase}${path}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: method === 'DELETE' ? undefined : JSON.stringify(body ?? {}),
  });
  if (!response.ok) {
    let detail = `Backend request failed (${response.status})`;
    try {
      const err = (await response.json()) as { detail?: unknown; message?: unknown };
      detail = describeDetail(err?.detail) ?? describeDetail(err?.message) ?? detail;
    } catch {
      /* keep the status message */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

/** Forget a backend game and release its engine processes. */
export function deleteBackendGame(gameId: string, options: TransportOptions = {}): Promise<{ success: boolean }> {
  return request<{ success: boolean }>(`/api/games/${encodeURIComponent(gameId)}`, undefined, options, 'DELETE');
}

export function createBackendGame<S = unknown>(
  config: Record<string, unknown>,
  options: TransportOptions = {},
): Promise<BackendCreateResponse<S>> {
  return request<BackendCreateResponse<S>>('/api/games', toBackendPayload(config), options);
}

export function postMove<S = unknown>(
  gameId: string,
  move: BackendMoveRequest,
  options: TransportOptions = {},
): Promise<BackendMoveResponse<S>> {
  const { player, ...rest } = move;
  return request<BackendMoveResponse<S>>(`/api/games/${encodeURIComponent(gameId)}/move`, { move: rest, player }, options);
}

export function postPass<S = unknown>(
  gameId: string,
  player: string,
  options: TransportOptions = {},
): Promise<BackendMoveResponse<S>> {
  return request<BackendMoveResponse<S>>(`/api/games/${encodeURIComponent(gameId)}/pass`, { player }, options);
}

export function postAdvanceTurn<S = unknown>(
  gameId: string,
  options: TransportOptions = {},
): Promise<BackendMoveResponse<S>> {
  return request<BackendMoveResponse<S>>(`/api/games/${encodeURIComponent(gameId)}/advance_turn`, {}, options);
}
