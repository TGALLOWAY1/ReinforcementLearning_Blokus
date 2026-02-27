export interface DashboardMetrics {
    influenceMap: number[][]; // 1-4 for player, 0 for contested, -1 for dead to all
    deadZones: Record<number, boolean[][]>; // true if deadzone for player
    distances: Record<number, number[][]>; // raw distance matrices
    frontiers: Record<number, { r: number; c: number }[]>; // usable corners
}

export function calculateDashboardMetrics(board: number[][]): DashboardMetrics {
    const size = board.length;

    const metrics: DashboardMetrics = {
        influenceMap: Array(size).fill(0).map(() => Array(size).fill(0)),
        deadZones: {
            1: Array(size).fill(0).map(() => Array(size).fill(false)),
            2: Array(size).fill(0).map(() => Array(size).fill(false)),
            3: Array(size).fill(0).map(() => Array(size).fill(false)),
            4: Array(size).fill(0).map(() => Array(size).fill(false)),
        },
        distances: {
            1: Array(size).fill(0).map(() => Array(size).fill(Infinity)),
            2: Array(size).fill(0).map(() => Array(size).fill(Infinity)),
            3: Array(size).fill(0).map(() => Array(size).fill(Infinity)),
            4: Array(size).fill(0).map(() => Array(size).fill(Infinity)),
        },
        frontiers: { 1: [], 2: [], 3: [], 4: [] }
    };

    const players = [1, 2, 3, 4];
    const orthDirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    const diagDirs = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
    const allDirs = [...orthDirs, ...diagDirs];

    // For each player, find Aura (orthogonally adjacent to their pieces)
    // and Frontier (diagonally adjacent, not in Aura, and empty)
    for (const p of players) {
        const aura = new Set<string>();
        const rawFrontier = new Set<string>();

        for (let r = 0; r < size; r++) {
            for (let c = 0; c < size; c++) {
                if (board[r][c] === p) {
                    // Add orthogonal neighbors to aura
                    for (const [dr, dc] of orthDirs) {
                        const nr = r + dr, nc = c + dc;
                        if (nr >= 0 && nr < size && nc >= 0 && nc < size && board[nr][nc] === 0) {
                            aura.add(`${nr},${nc}`);
                        }
                    }
                    // Add diagonal neighbors to raw frontier
                    for (const [dr, dc] of diagDirs) {
                        const nr = r + dr, nc = c + dc;
                        if (nr >= 0 && nr < size && nc >= 0 && nc < size && board[nr][nc] === 0) {
                            rawFrontier.add(`${nr},${nc}`);
                        }
                    }
                }
            }
        }

        // Filter frontier (must not be in aura)
        const frontierCells: { r: number; c: number }[] = [];
        rawFrontier.forEach(key => {
            if (!aura.has(key)) {
                const [r, c] = key.split(',').map(Number);
                frontierCells.push({ r, c });
            }
        });

        metrics.frontiers[p] = frontierCells;

        // Run BFS to find distances
        const dist = metrics.distances[p];
        const q: { r: number; c: number; d: number; }[] = [];

        // Add start corners for this player if board is empty for them
        // (If they haven't placed their first piece yet)
        if (rawFrontier.size === 0 && aura.size === 0) {
            // Find corner corresponding to player
            // 1: top-left, 2: bottom-right, 3: bottom-left, 4: top-right
            const startCorners = {
                1: { r: 0, c: 0 },
                2: { r: size - 1, c: size - 1 },
                3: { r: size - 1, c: 0 },
                4: { r: 0, c: size - 1 }
            };
            const sc = startCorners[p as keyof typeof startCorners];
            if (sc && board[sc.r][sc.c] === 0) {
                dist[sc.r][sc.c] = 0;
                q.push({ r: sc.r, c: sc.c, d: 0 });
            }
        } else {
            for (const cell of frontierCells) {
                dist[cell.r][cell.c] = 1;
                q.push({ r: cell.r, c: cell.c, d: 1 });
            }
        }

        let head = 0;
        while (head < q.length) {
            const curr = q[head++];
            for (const [dr, dc] of allDirs) {
                const nr = curr.r + dr;
                const nc = curr.c + dc;
                if (nr >= 0 && nr < size && nc >= 0 && nc < size) {
                    if (board[nr][nc] === 0 && !aura.has(`${nr},${nc}`)) {
                        if (dist[nr][nc] > curr.d + 1) {
                            dist[nr][nc] = curr.d + 1;
                            q.push({ r: nr, c: nc, d: curr.d + 1 });
                        }
                    }
                }
            }
        }

        // Populate dead zones for player p
        for (let r = 0; r < size; r++) {
            for (let c = 0; c < size; c++) {
                if (board[r][c] === 0) {
                    if (dist[r][c] === Infinity) {
                        metrics.deadZones[p][r][c] = true;
                    }
                }
            }
        }
    }

    // Populate influence map (Voronoi)
    for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
            if (board[r][c] !== 0) {
                metrics.influenceMap[r][c] = board[r][c]; // Occupied by piece
            } else {
                // Find minimum distance among all players
                let minDist = Infinity;
                let owner = -1; // -1 means dead to all
                let isContested = false;

                for (const p of players) {
                    const d = metrics.distances[p][r][c];
                    if (d < minDist) {
                        minDist = d;
                        owner = p;
                        isContested = false;
                    } else if (d === minDist && d !== Infinity) {
                        isContested = true;
                    }
                }

                if (minDist === Infinity) {
                    metrics.influenceMap[r][c] = -1; // Dead to all
                } else if (isContested) {
                    metrics.influenceMap[r][c] = 0; // Contested
                } else {
                    metrics.influenceMap[r][c] = owner;
                }
            }
        }
    }

    return metrics;
}

// Helper to determine heuristic win probability
export function calculateWinProbability(
    board: number[][],
    metrics: DashboardMetrics
): Record<number, number> {
    // A heuristic based on:
    // 1. Squares placed (score)
    // 2. Controlled Voronoi territory
    // 3. Active frontier cells
    const scores = { 1: 0, 2: 0, 3: 0, 4: 0 };
    const territories = { 1: 0, 2: 0, 3: 0, 4: 0 };

    const size = board.length;
    for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
            if (board[r][c] > 0) scores[board[r][c] as keyof typeof scores]++;
            const inf = metrics.influenceMap[r][c];
            if (inf > 0) territories[inf as keyof typeof territories]++;
        }
    }

    const scoresWeights = 0.5;
    const territoryWeights = 0.3;
    const mobilityWeights = 0.2;

    const totalScore = Math.max(1, Object.values(scores).reduce((a, b) => a + b, 0));
    const totalTerritory = Math.max(1, Object.values(territories).reduce((a, b) => a + b, 0));
    const totalFrontier = Math.max(1, [1, 2, 3, 4].reduce((a, b) => a + metrics.frontiers[b].length, 0));

    const probs: Record<number, number> = {};
    for (const p of [1, 2, 3, 4]) {
        const s = scores[p as keyof typeof scores] / totalScore;
        const t = territories[p as keyof typeof territories] / totalTerritory;
        const f = metrics.frontiers[p].length / totalFrontier;

        probs[p] = (s * scoresWeights) + (t * territoryWeights) + (f * mobilityWeights);
    }

    // Normalize to 100%
    const totalProb = Object.values(probs).reduce((a, b) => a + b, 0);
    for (const p of [1, 2, 3, 4]) {
        probs[p] = totalProb > 0 ? probs[p] / totalProb : 0.25;
    }

    return probs;
}
