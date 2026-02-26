from typing import Dict, List, Set, Tuple
from .board import Board, Player

def compute_corner_differential(board: Board, player: Player) -> int:
    """M = C_own - sum(C_opp)"""
    own_corners = len(board.get_frontier(player))
    opp_corners = 0
    for p in Player:
        if p != player:
            opp_corners += len(board.get_frontier(p))
    return own_corners - opp_corners

def compute_territory_control(board: Board) -> Tuple[List[List[int]], Dict[str, float]]:
    """
    Returns influence_map (20x20 of player.value or 0)
    and territory_ratios (dict of player.name -> % of empty squares)
    """
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
    """Weighted penalty sum for remaining pieces. Heaviest for 5-square pieces, esp Cross (20), W (19), U (17)."""
    # Piece sizes: 1..21. Size 5: 11..21
    penalty = 0
    for pid in range(1, 22):
        if pid not in pieces_used:
            if pid in [17, 19, 20]:
                penalty += 5
            elif pid >= 11: # Size 5
                penalty += 2
            elif pid >= 5 and pid <= 10: # Size 4
                penalty += 1
    return penalty

def compute_center_proximity(board: Board, player: Player) -> int:
    """Min distance to center (9,9 to 10,10)"""
    min_dist = 9 # Default max distance
    found = False
    
    for r in range(board.SIZE):
        for c in range(board.SIZE):
            if board.grid[r][c] == player.value:
                # distance to closest of the 4 center squares
                # which is 9 or 10.
                dist_r = max(0, abs(r - 9.5) - 0.5)
                dist_c = max(0, abs(c - 9.5) - 0.5)
                dist = int(dist_r + dist_c)
                if dist < min_dist:
                    min_dist = dist
                    found = True
                    
    return min_dist if found else 9

def compute_opponent_adjacency(board: Board, player: Player) -> int:
    """Number of player's pieces sharing edge with opponent"""
    count = 0
    for r in range(board.SIZE):
        for c in range(board.SIZE):
            if board.grid[r][c] == player.value:
                # Check 4 orth neighbors
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

def compute_dead_zones(board: Board) -> List[List[bool]]:
    """Returns 20x20 boolean matrix of dead zones (unreachable empty cells)."""
    dead_zones = [[False for _ in range(board.SIZE)] for _ in range(board.SIZE)]
    visited = set()
    
    # Collect all frontiers
    all_frontiers = set()
    for p in Player:
        all_frontiers.update(board.get_frontier(p))
        
    for r in range(board.SIZE):
        for c in range(board.SIZE):
            if board.grid[r][c] == 0 and (r, c) not in visited:
                # BFS to find component of empty cells
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
                                
                # Check if component has any frontiers
                has_frontier = any(cell in all_frontiers for cell in component)
                if not has_frontier:
                    for cr, cc in component:
                        dead_zones[cr][cc] = True
                        
    return dead_zones
