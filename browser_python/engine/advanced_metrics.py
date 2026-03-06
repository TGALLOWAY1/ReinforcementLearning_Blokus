import math
from typing import Dict, List, Set, Tuple

from .board import Board, Player
from .metrics_config import TelemetryConfig


def compute_corner_differential(board: Board, player: Player) -> int:
    """M = C_own - sum(C_opp)"""
    own_corners = len(board.get_frontier(player))
    opp_corners = 0
    for p in Player:
        if p != player:
            opp_corners += len(board.get_frontier(p))
    return own_corners - opp_corners


def compute_territory_control(board: Board) -> Tuple[List[List[int]], Dict[str, float]]:
    influence_map = [[0 for _ in range(board.SIZE)] for _ in range(board.SIZE)]
    territory_counts = {p.value: 0 for p in Player}

    empty_cells = []
    for r in range(board.SIZE):
        for c in range(board.SIZE):
            if board.grid[r][c] == 0:
                empty_cells.append((r, c))

    if not empty_cells:
        return influence_map, {p.name: 0.0 for p in Player}

    for r, c in empty_cells:
        min_dist = float('inf')
        closest_players = []

        for p in Player:
            frontier = board.get_frontier(p)
            for fr, fc in frontier:
                dist = abs(fr - r) + abs(fc - c)
                if dist < min_dist:
                    min_dist = dist
                    closest_players = [p.value]
                elif dist == min_dist:
                    if p.value not in closest_players:
                        closest_players.append(p.value)

        if len(closest_players) == 1:
            owner = closest_players[0]
            influence_map[r][c] = owner
            territory_counts[owner] += 1

    total_empty = len(empty_cells)
    ratios = {p.name: territory_counts[p.value] / total_empty for p in Player}
    return influence_map, ratios


def compute_piece_penalty(pieces_used: Set[int]) -> int:
    penalty = 0
    for pid in range(1, 22):
        if pid not in pieces_used:
            if pid in [17, 19, 20]:
                penalty += 5
            elif pid >= 11:
                penalty += 2
            elif pid >= 5 and pid <= 10:
                penalty += 1
    return penalty


def compute_center_proximity(board: Board, player: Player) -> int:
    min_dist = 9
    found = False

    for r in range(board.SIZE):
        for c in range(board.SIZE):
            if board.grid[r][c] == player.value:
                dist_r = max(0, abs(r - 9.5) - 0.5)
                dist_c = max(0, abs(c - 9.5) - 0.5)
                dist = int(dist_r + dist_c)
                if dist < min_dist:
                    min_dist = dist
                    found = True

    return min_dist if found else 9


def compute_opponent_adjacency(board: Board, player: Player) -> int:
    count = 0
    for r in range(board.SIZE):
        for c in range(board.SIZE):
            if board.grid[r][c] == player.value:
                is_adj = False
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < board.SIZE and 0 <= nc < board.SIZE:
                        val = board.grid[nr][nc]
                        if val != 0 and val != player.value:
                            is_adj = True
                            break
                if is_adj:
                    count += 1
    return count


def compute_dead_space_split(board: Board, player: Player) -> Tuple[float, float]:
    """
    Splits dead space into two causal concepts (using descriptive naming):
    - deadSpaceNearSelf: Unreachable squares near your pieces (often wasted cavities).
    - deadSpaceNearOpponents: Unreachable squares near opponents' pieces (often sealed off by you, i.e., denial).
    Returns: (deadSpaceNearSelf, deadSpaceNearOpponents)
    """
    dead_zones = [[False for _ in range(board.SIZE)] for _ in range(board.SIZE)]
    visited = set()

    all_frontiers = set()
    for p in Player:
        all_frontiers.update(board.get_frontier(p))

    ds_near_self = 0
    ds_near_opp = 0

    for r in range(board.SIZE):
        for c in range(board.SIZE):
            if board.grid[r][c] == 0 and (r, c) not in visited:
                component = []
                queue = [(r, c)]
                visited.add((r, c))
                head = 0

                while head < len(queue):
                    curr_r, curr_c = queue[head]
                    head += 1
                    component.append((curr_r, curr_c))

                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = curr_r+dr, curr_c+dc
                        if 0 <= nr < board.SIZE and 0 <= nc < board.SIZE:
                            if board.grid[nr][nc] == 0 and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                queue.append((nr, nc))

                has_frontier = any(cell in all_frontiers for cell in component)
                if not has_frontier:
                    # It's a dead pocket
                    # See who borders it
                    borders_self = False
                    borders_opp = False
                    for cr, cc in component:
                        dead_zones[cr][cc] = True
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0 <= nr < board.SIZE and 0 <= nc < board.SIZE:
                                val = board.grid[nr][nc]
                                if val != 0:
                                    if val == player.value:
                                        borders_self = True
                                    else:
                                        borders_opp = True

                    if borders_self:
                        ds_near_self += len(component)
                    elif borders_opp:
                        ds_near_opp += len(component)

    return float(ds_near_self), float(ds_near_opp)

# Keep the original method for UI backwards compatibility while replacing usages where needed
def compute_dead_zones(board: Board) -> List[List[bool]]:
    dead_zones = [[False for _ in range(board.SIZE)] for _ in range(board.SIZE)]
    visited = set()

    all_frontiers = set()
    for p in Player:
        all_frontiers.update(board.get_frontier(p))

    for r in range(board.SIZE):
        for c in range(board.SIZE):
            if board.grid[r][c] == 0 and (r, c) not in visited:
                component = []
                queue = [(r, c)]
                visited.add((r, c))
                head = 0

                while head < len(queue):
                    curr_r, curr_c = queue[head]
                    head += 1
                    component.append((curr_r, curr_c))

                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = curr_r+dr, curr_c+dc
                        if 0 <= nr < board.SIZE and 0 <= nc < board.SIZE:
                            if board.grid[nr][nc] == 0 and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                queue.append((nr, nc))

                has_frontier = any(cell in all_frontiers for cell in component)
                if not has_frontier:
                    for cr, cc in component:
                        dead_zones[cr][cc] = True

    return dead_zones


def compute_effective_frontier(board: Board, player: Player) -> float:
    """
    Computes an effective frontier score by weighting each raw frontier corner.
    w = space_factor * (1 - vulnerability_est)
    Where space_factor looks at nearby empty cells, and vulnerability looks at nearby enemy frontiers.
    """
    frontier = board.get_frontier(player)
    if not frontier:
        return 0.0

    opp_frontiers = set()
    for p in Player:
        if p != player:
            opp_frontiers.update(board.get_frontier(p))

    effective_score = 0.0
    R = TelemetryConfig.ANCHOR_RADIUS

    for r, c in frontier:
        # Space factor: count empty cells in Radius R that are NOT orthogonally adjacent to player
        space_score = 0
        vuln_score = 0

        # We'll just scan a bounding box of radius R
        r_min, r_max = max(0, r - R), min(board.SIZE - 1, r + R)
        c_min, c_max = max(0, c - R), min(board.SIZE - 1, c + R)

        for br in range(r_min, r_max + 1):
            for bc in range(c_min, c_max + 1):
                if board.grid[br][bc] == 0:
                    # check orth adj to player
                    is_orth = False
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = br+dr, bc+dc
                        if 0 <= nr < board.SIZE and 0 <= nc < board.SIZE:
                            if board.grid[nr][nc] == player.value:
                                is_orth = True
                                break
                    if not is_orth:
                        space_score += 1

                # Vulnerability factor: opponent frontiers nearby
                if (br, bc) in opp_frontiers:
                    # Distance closer means higher vulnerability
                    dist = max(abs(br - r), abs(bc - c))
                    if dist <= R:
                        # 0.2 vuln per opponent frontier within radius R, maxes at 0.8
                        vuln_score += 0.2

        # Cap vuln penalty
        vuln_factor = min(0.8, vuln_score)

        # log scale space factor so massive open areas don't artificially blow out the score
        # add 1 so score doesn't 0 out entirely if space_score=0 (but space shouldn't be 0 for a frontier cell)
        weight = math.log1p(space_score) * (1.0 - vuln_factor)
        effective_score += weight

    return effective_score


def compute_frontier_spread(board: Board, player: Player) -> Tuple[int, int]:
    """
    Returns (component_count, quadrant_coverage).
    Clusters frontier corners using distance <= 2.
    """
    frontier = list(board.get_frontier(player))
    if not frontier:
        return 0, 0

    parent = {f: f for f in frontier}

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Connect components if dist <= 2
    for i in range(len(frontier)):
        for j in range(i + 1, len(frontier)):
            r1, c1 = frontier[i]
            r2, c2 = frontier[j]
            dist = max(abs(r1 - r2), abs(c1 - c2))
            if dist <= 2:
                union(frontier[i], frontier[j])

    components = set(find(f) for f in frontier)

    # Quadrants
    quads = set()
    mid = board.SIZE // 2
    for r, c in frontier:
        q = (r >= mid, c >= mid)
        quads.add(q)

    return len(components), len(quads)
