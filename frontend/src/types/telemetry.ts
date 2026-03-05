/**
 * Move Delta Telemetry Types
 * Schema definition for capturing game metrics per move, storing them, and delivering them to the frontend.
 */

export type MetricKey = 'frontierSize' | 'mobility' | 'deadSpace' | 'frontierUtility' | string;

export interface PlayerMetricSnapshot {
    playerId: string; // e.g., 'RED', 'BLUE'
    metrics: Record<MetricKey, number>;
}

export interface MoveTelemetrySnapshot {
    ply: number;
    moverId: string;
    moveId: string; // e.g., '14-0-3-4'
    before: PlayerMetricSnapshot[];
    after: PlayerMetricSnapshot[];
}

export interface MoveTelemetryDelta {
    ply: number;
    moverId: string;
    moveId: string;
    deltaSelf: Record<MetricKey, number>;
    deltaOppTotal: Record<MetricKey, number>;
    deltaOppByPlayer: Record<string, Record<MetricKey, number>>;
    impactScore?: number;
}

export interface GameTelemetry {
    gameId: string;
    players: string[];
    moves: MoveTelemetryDelta[];
    normalization: 'z-score' | 'min-max' | 'none';
    weights?: Record<MetricKey, number>;
}

// Minimal runtime guards

export function isRecordOfNumbers(obj: unknown): obj is Record<string, number> {
    if (typeof obj !== 'object' || obj === null) return false;
    return Object.values(obj).every(val => typeof val === 'number');
}

export function isPlayerMetricSnapshot(obj: unknown): obj is PlayerMetricSnapshot {
    if (typeof obj !== 'object' || obj === null) return false;
    const o = obj as any;
    if (typeof o.playerId !== 'string') return false;
    if (!isRecordOfNumbers(o.metrics)) return false;
    return true;
}

export function isMoveTelemetrySnapshot(obj: unknown): obj is MoveTelemetrySnapshot {
    if (typeof obj !== 'object' || obj === null) return false;
    const o = obj as any;
    if (typeof o.ply !== 'number') return false;
    if (typeof o.moverId !== 'string') return false;
    if (typeof o.moveId !== 'string') return false;
    if (!Array.isArray(o.before) || !o.before.every(isPlayerMetricSnapshot)) return false;
    if (!Array.isArray(o.after) || !o.after.every(isPlayerMetricSnapshot)) return false;
    return true;
}

export function isMoveTelemetryDelta(obj: unknown): obj is MoveTelemetryDelta {
    if (typeof obj !== 'object' || obj === null) return false;
    const o = obj as any;
    if (typeof o.ply !== 'number') return false;
    if (typeof o.moverId !== 'string') return false;
    if (typeof o.moveId !== 'string') return false;

    if (!isRecordOfNumbers(o.deltaSelf)) return false;
    if (!isRecordOfNumbers(o.deltaOppTotal)) return false;

    if (typeof o.deltaOppByPlayer !== 'object' || o.deltaOppByPlayer === null) return false;
    for (const key in o.deltaOppByPlayer) {
        if (!isRecordOfNumbers(o.deltaOppByPlayer[key])) return false;
    }

    // impactScore is optional
    if (o.impactScore !== undefined && typeof o.impactScore !== 'number') return false;

    return true;
}

export function isGameTelemetry(obj: unknown): obj is GameTelemetry {
    if (typeof obj !== 'object' || obj === null) return false;
    const o = obj as any;
    if (typeof o.gameId !== 'string') return false;
    if (!Array.isArray(o.players) || !o.players.every((p: any) => typeof p === 'string')) return false;
    if (!Array.isArray(o.moves) || !o.moves.every(isMoveTelemetryDelta)) return false;
    if (!['z-score', 'min-max', 'none'].includes(o.normalization)) return false;

    if (o.weights !== undefined && !isRecordOfNumbers(o.weights)) return false;

    return true;
}
