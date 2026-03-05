/**
 * Phase Segmentation + Strategy Mix
 *
 * Divides a game into Opening / Mid-game / End-game phases based on
 * remaining piece count (approximated from ply number), then computes
 * per-phase stats for a given player:
 *   - avg impact per move
 *   - dominant contributing metric
 *   - biggest turning move (highest impact)
 */

import { MoveTelemetryDelta } from '../types/telemetry';
import {
    computeGameImpactScores,
    WeightPreset,
    NormalizationMethod,
    MoveImpactScore,
} from './moveImpactScore';

export type Phase = 'opening' | 'mid' | 'endgame';

export interface PhaseStats {
    phase: Phase;
    label: string;
    moves: { delta: MoveTelemetryDelta; score: MoveImpactScore }[];
    avgImpact: number;
    dominantMetric: string;
    biggestTurningMove: { delta: MoveTelemetryDelta; score: MoveImpactScore } | null;
}

export interface StrategyMixResult {
    playerId: string;
    totalImpact: number;
    metricContributions: Record<string, number>; // metric -> cumulative contribution %
    phases: PhaseStats[];
}

/** Very lightweight heuristic: use ply thresholds (4 players, ~21 pieces each → ~84 moves total) */
function classifyPhase(ply: number, totalPlies: number): Phase {
    const pct = ply / (totalPlies || 1);
    if (pct < 0.33) return 'opening';
    if (pct < 0.67) return 'mid';
    return 'endgame';
}

const PHASE_LABELS: Record<Phase, string> = {
    opening: 'Opening',
    mid: 'Mid-game',
    endgame: 'End-game',
};

export function computeStrategyMix(
    allMoves: MoveTelemetryDelta[],
    playerId: string,
    preset: WeightPreset = 'balanced',
    normalization: NormalizationMethod = 'z-score'
): StrategyMixResult {
    const scores = computeGameImpactScores(allMoves, preset, normalization);

    // Filter to the given player's moves
    const playerMoves = allMoves
        .map((m, i) => ({ delta: m, score: scores[i] }))
        .filter(({ delta }) => delta.moverId === playerId);

    const totalPlies = allMoves.length > 0 ? allMoves[allMoves.length - 1].ply : 1;

    // Bucket into phases
    const phaseBuckets: Record<Phase, { delta: MoveTelemetryDelta; score: MoveImpactScore }[]> = {
        opening: [],
        mid: [],
        endgame: [],
    };

    playerMoves.forEach(entry => {
        const phase = classifyPhase(entry.delta.ply, totalPlies);
        phaseBuckets[phase].push(entry);
    });

    // Compute per-phase stats
    const phases: PhaseStats[] = (['opening', 'mid', 'endgame'] as Phase[]).map(phase => {
        const moves = phaseBuckets[phase];
        const avgImpact = moves.length > 0
            ? moves.reduce((a, m) => a + m.score.total, 0) / moves.length
            : 0;

        // Dominant metric: highest absolute cumulative contribution sum
        const metricSums: Record<string, number> = {};
        moves.forEach(({ score }) => {
            score.contributions.forEach(c => {
                metricSums[c.metric] = (metricSums[c.metric] || 0) + Math.abs(c.contribution);
            });
        });
        const dominantMetric = Object.entries(metricSums).sort((a, b) => b[1] - a[1])[0]?.[0] ?? '-';

        const biggestTurningMove = moves.length > 0
            ? moves.reduce((best, m) => m.score.total > best.score.total ? m : best, moves[0])
            : null;

        return {
            phase,
            label: PHASE_LABELS[phase],
            moves,
            avgImpact,
            dominantMetric,
            biggestTurningMove,
        };
    });

    // Overall metric contribution breakdown
    const metricContributions: Record<string, number> = {};
    let contribTotal = 0;
    playerMoves.forEach(({ score }) => {
        score.contributions.forEach(c => {
            metricContributions[c.metric] = (metricContributions[c.metric] || 0) + Math.abs(c.contribution);
            contribTotal += Math.abs(c.contribution);
        });
    });
    // Normalize to percentages
    Object.keys(metricContributions).forEach(k => {
        metricContributions[k] = contribTotal > 0 ? (metricContributions[k] / contribTotal) * 100 : 0;
    });

    const totalImpact = playerMoves.reduce((a, m) => a + m.score.total, 0);

    return { playerId, totalImpact, metricContributions, phases };
}
