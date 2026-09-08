import { useEffect, useState } from 'react';
import { API_BASE } from '../constants/gameConstants';

export interface PentobiAvailability {
  isLoading: boolean;
  /** True when the backend is reachable and lists a natively served Pentobi agent. */
  available: boolean;
  /** Why it is unavailable (backend down, binary missing). */
  error: string | null;
  levelRange: { min: number; max: number; default: number };
  capMs: number | null;
}

/**
 * Asks GET /api/agents whether the backend can serve Pentobi. The backend lists
 * the agent only when the pentobi-gtp binary is installed, so a missing entry
 * with a healthy backend means "engine not built on this machine".
 */
export const usePentobiAvailability = (): PentobiAvailability => {
  const [state, setState] = useState<PentobiAvailability>({
    isLoading: true,
    available: false,
    error: null,
    levelRange: { min: 1, max: 9, default: 3 },
    capMs: null,
  });

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const resp = await fetch(`${API_BASE}/api/agents`);
        if (!resp.ok) throw new Error(`agents endpoint returned ${resp.status}`);
        const agents = (await resp.json()) as Array<{ type: string; config_schema?: Record<string, any> }>;
        const entry = agents.find((a) => a.type === 'pentobi');
        if (cancelled) return;
        if (!entry) {
          setState((prev) => ({
            ...prev,
            isLoading: false,
            available: false,
            error: 'The backend is running but pentobi-gtp is not installed there (see docs/agent-strength-rebuild/PENTOBI.md).',
          }));
          return;
        }
        const level = entry.config_schema?.level ?? {};
        setState({
          isLoading: false,
          available: true,
          error: null,
          levelRange: { min: level.min ?? 1, max: level.max ?? 9, default: level.default ?? 3 },
          capMs: entry.config_schema?.time_budget_ms?.max ?? null,
        });
      } catch (err) {
        if (cancelled) return;
        setState((prev) => ({
          ...prev,
          isLoading: false,
          available: false,
          error: `Backend not reachable (${err instanceof Error ? err.message : String(err)}). Start it with: python run_server.py`,
        }));
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
};
