
from typing import Dict, Optional

import numpy as np
import pandas as pd


class MatchupMatrix:
    def __init__(self):
        self.wins = {} # (agent_a, agent_b) -> count a beat b
        self.games = {} # (agent_a, agent_b) -> count games
        self.draws = {} # (agent_a, agent_b) -> count draws (if desired)
        self.agents = set()

    def add_game(self, agent_scores: Dict[str, int]):
        """
        Record a game result.
        Decomposes 4-player game into pairwise matchups.
        """
        agent_ids = list(agent_scores.keys())
        for ag in agent_ids:
            self.agents.add(ag)
            
        n = len(agent_ids)
        for i in range(n):
            for j in range(i+1, n):
                a = agent_ids[i]
                b = agent_ids[j]
                
                # Canonical key: tuple sorted? 
                # No, we want directional wins.
                # games lookup: use sorted for total games?
                # Let's simple store: games[a][b] += 1 and games[b][a] += 1
                
                self._record_pair(a, b, agent_scores[a], agent_scores[b])
                self._record_pair(b, a, agent_scores[b], agent_scores[a])

    def _record_pair(self, me, opp, my_score, opp_score):
        if me not in self.games: self.games[me] = {}
        if me not in self.wins: self.wins[me] = {}
        
        self.games[me][opp] = self.games[me].get(opp, 0) + 1
        
        if my_score > opp_score:
            self.wins[me][opp] = self.wins[me].get(opp, 0) + 1
            
    def get_win_rate(self, agent_a, agent_b) -> Optional[float]:
        g = self.games.get(agent_a, {}).get(agent_b, 0)
        if g == 0:
            return None
        w = self.wins.get(agent_a, {}).get(agent_b, 0)
        return w / g

    def to_dataframe(self) -> pd.DataFrame:
        agents = sorted(list(self.agents))
        matrix = pd.DataFrame(index=agents, columns=agents, dtype=float)
        
        for a in agents:
            for b in agents:
                if a == b:
                    matrix.loc[a, b] = np.nan
                else:
                    wr = self.get_win_rate(a, b)
                    matrix.loc[a, b] = wr
                    
        return matrix

def compute_matchup_matrix(results_path: str) -> pd.DataFrame:
    import json
    tracker = MatchupMatrix()
    
    with open(results_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            
            # scores is dict str->int
            scores = rec['final_scores']
            
            # map player_id -> agent_id
            agent_map = rec['agent_ids']
            
            agent_scores = {}
            for pid, score in scores.items():
                aid = agent_map.get(pid, pid) # fallback to pid
                agent_scores[aid] = score
                
            tracker.add_game(agent_scores)
            
    return tracker.to_dataframe()
