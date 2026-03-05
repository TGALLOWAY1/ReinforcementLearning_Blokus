/**
 * Move Impact Score computation
 * 
 * Computes a composite score from move delta metrics using weight presets.
 * Supports z-score and min-max normalization across all moves in a game.
 */

import { MoveTelemetryDelta } from '../types/telemetry';

export type WeightPreset = 'balanced' | 'blocking' | 'expansion' | 'late-game';
export type NormalizationMethod = 'z-score' | 'min-max' | 'none';

export interface WeightConfig {
    frontierSize: number;
    mobility: number;
    deadSpace: number;
    centerControl: number;
    pieceLockRisk: number;
    [key: string]: number;
}

export const WEIGHT_PRESETS: Record<WeightPreset, WeightConfig> = {
    balanced: {
        frontierSize: 1.0,
        mobility: 1.0,
        deadSpace: -0.5,
        centerControl: 0.5,
        pieceLockRisk: -0.3,
    },
    blocking: {
        frontierSize: 0.5,
        mobility: 0.5,
        deadSpace: -1.5,
        centerControl: 0.3,
        pieceLockRisk: -0.5,
    },
    expansion: {
        frontierSize: 2.0,
        mobility: 1.5,
        deadSpace: -0.3,
        centerControl: 1.0,
        pieceLockRisk: -0.1,
    },
    'late-game': {
        frontierSize: 0.3,
        mobility: 2.0,
        deadSpace: -1.0,
        centerControl: 0.2,
        pieceLockRisk: -1.5,
    },
};

export interface MoveImpactContribution {
    metric: string;
    selfDelta: number;
    oppDelta: number;
    normalizedSelf: number;
    weight: number;
    contribution: number;
}

export interface MoveImpactScore {
    total: number;
    contributions: MoveImpactContribution[];
    preset: WeightPreset;
    normalization: NormalizationMethod;
}

/**
 * Raw score for a single move, before normalization.
 */
export function computeRawScore(
    delta: MoveTelemetryDelta,
    weights: WeightConfig
): number {
    let score = 0;
    for (const [metric, weight] of Object.entries(weights)) {
        const selfVal = delta.deltaSelf[metric] ?? 0;
        const oppVal = delta.deltaOppTotal?.[metric] ?? 0;
        // Self improvement counts positively, opponent suppression (negative delta for them) counts positively
        score += weight * (selfVal - oppVal);
    }
    return score;
}

/**
 * z-score normalise an array of numbers in-place, returning the normalized array.
 */
function zScore(values: number[]): number[] {
    const mean = values.reduce((a, b) => a + b, 0) / (values.length || 1);
    const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / (values.length || 1);
    const std = Math.sqrt(variance) || 1;
    return values.map(v => (v - mean) / std);
}

/**
 * min-max normalise an array of numbers, returning the normalized array.
 */
function minMax(values: number[]): number[] {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    return values.map(v => (v - min) / range);
}

/**
 * Compute impact scores for all moves in a game.
 * Returns an array of MoveImpactScore, one per telemetry move.
 */
export function computeGameImpactScores(
    moves: MoveTelemetryDelta[],
    preset: WeightPreset = 'balanced',
    normalization: NormalizationMethod = 'z-score'
): MoveImpactScore[] {
    const weights = WEIGHT_PRESETS[preset];

    // First pass: compute raw scores per metric per move
    const rawScores = moves.map(m => computeRawScore(m, weights));

    // Normalize raw scores
    let normalized: number[];
    if (normalization === 'z-score') {
        normalized = zScore(rawScores);
    } else if (normalization === 'min-max') {
        normalized = minMax(rawScores);
    } else {
        normalized = rawScores;
    }

    // Per-metric raw values for contribution breakdown
    return moves.map((delta, i) => {
        const contributions: MoveImpactContribution[] = Object.entries(weights).map(([metric, weight]) => {
            const selfDelta = delta.deltaSelf[metric] ?? 0;
            const oppDelta = delta.deltaOppTotal?.[metric] ?? 0;
            const combined = selfDelta - oppDelta;
            return {
                metric,
                selfDelta,
                oppDelta,
                normalizedSelf: combined,
                weight,
                contribution: weight * combined,
            };
        });

        return {
            total: normalized[i],
            contributions,
            preset,
            normalization,
        };
    });
}
