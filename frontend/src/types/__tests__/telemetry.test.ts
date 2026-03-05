import { describe, it, expect } from 'vitest';
import {
    isMoveTelemetryDelta,
    isGameTelemetry
} from '../telemetry';

describe('Telemetry Type Guards', () => {
    it('validates a correct MoveTelemetryDelta payload', () => {
        const validDelta: any = {
            ply: 14,
            moverId: 'RED',
            moveId: '14-0-3-4',
            deltaSelf: {
                frontierSize: 2,
                mobility: -1,
                deadSpace: 0
            },
            deltaOppTotal: {
                frontierSize: -3,
                mobility: -4,
                deadSpace: 1
            },
            deltaOppByPlayer: {
                BLUE: { frontierSize: -2, mobility: -2, deadSpace: 1 },
                GREEN: { frontierSize: -1, mobility: -2, deadSpace: 0 },
                YELLOW: { frontierSize: 0, mobility: 0, deadSpace: 0 }
            },
            impactScore: 1.5
        };

        expect(isMoveTelemetryDelta(validDelta)).toBe(true);
    });

    it('preserves unknown metrics in MoveTelemetryDelta', () => {
        const validDeltaWithUnknown: any = {
            ply: 15,
            moverId: 'BLUE',
            moveId: '15-1-5-5',
            deltaSelf: {
                frontierSize: 1,
                newExperimentalMetric: 42 // Unknown metric
            },
            deltaOppTotal: {
                newExperimentalMetric: -10
            },
            deltaOppByPlayer: {
                RED: { newExperimentalMetric: -10 }
            }
        };

        expect(isMoveTelemetryDelta(validDeltaWithUnknown)).toBe(true);
    });

    it('rejects an invalid MoveTelemetryDelta payload', () => {
        const invalidDelta: any = {
            ply: 'not-a-number', // INVALID
            moverId: 'RED',
            moveId: '14-0-3-4',
            deltaSelf: { frontierSize: 2 },
            deltaOppTotal: { frontierSize: -3 },
            deltaOppByPlayer: {}
        };

        expect(isMoveTelemetryDelta(invalidDelta)).toBe(false);

        const invalidDelta2: any = {
            ply: 14,
            moverId: 'RED',
            moveId: '14-0-3-4',
            deltaSelf: {
                frontierSize: 'invalid-string' // INVALID
            },
            deltaOppTotal: {},
            deltaOppByPlayer: {}
        };

        expect(isMoveTelemetryDelta(invalidDelta2)).toBe(false);
    });

    it('validates a correct GameTelemetry payload', () => {
        const validGame: any = {
            gameId: 'game-123',
            players: ['RED', 'BLUE', 'GREEN', 'YELLOW'],
            moves: [
                {
                    ply: 1,
                    moverId: 'RED',
                    moveId: '1-0-0-0',
                    deltaSelf: { frontierSize: 4 },
                    deltaOppTotal: { frontierSize: 0 },
                    deltaOppByPlayer: { BLUE: { frontierSize: 0 } }
                }
            ],
            normalization: 'z-score',
            weights: {
                frontierSize: 1.0,
                mobility: 1.5
            }
        };

        expect(isGameTelemetry(validGame)).toBe(true);
    });

    it('rejects an invalid GameTelemetry payload', () => {
        const invalidGame: any = {
            gameId: 'game-123',
            players: ['RED', 'BLUE'],
            moves: [],
            normalization: 'unsupported-norm' // INVALID
        };

        expect(isGameTelemetry(invalidGame)).toBe(false);
    });
});
