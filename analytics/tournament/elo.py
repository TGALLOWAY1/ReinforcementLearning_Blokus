
from typing import Dict, List, Tuple
import math

def get_expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(ratings: Dict[str, float], 
               results: List[Tuple[str, str, float]], 
               k_factor: int = 32) -> Dict[str, float]:
    """
    Updates Elo ratings based on a list of match results.
    
    Args:
        ratings: Dict of agent_id -> current rating
        results: List of (agent_a_id, agent_b_id, score_a). 
                 score_a is 1.0 for win, 0.5 for draw, 0.0 for loss.
        k_factor: K-factor for updates.
        
    Returns:
        Updated ratings dictionary (new copy).
    """
    new_ratings = ratings.copy()
    
    for id_a, id_b, score_a in results:
        ra = new_ratings.get(id_a, 1200.0)
        rb = new_ratings.get(id_b, 1200.0)
        
        expected_a = get_expected_score(ra, rb)
        
        # Standard update
        # new_ra = ra + k * (actual - expected)
        # new_rb = rb + k * ((1-actual) - (1-expected))
        
        delta = k_factor * (score_a - expected_a)
        
        new_ratings[id_a] = ra + delta
        new_ratings[id_b] = rb - delta
        
    return new_ratings

class EloTracker:
    def __init__(self, k_factor=32, default_rating=1200.0):
        self.ratings = {}
        self.k_factor = k_factor
        self.default_rating = default_rating
        self.games_played = {}
        
    def get_rating(self, agent_id: str) -> float:
        return self.ratings.get(agent_id, self.default_rating)
        
    def update_game(self, agent_id_scores: Dict[str, int]):
        """
        Update ratings based on a multiplayer game result.
        Treats a multiplayer game as a set of pairwise matchups.
        
        Args:
            agent_id_scores: Dict[agent_id, score]
        """
        agents = list(agent_id_scores.keys())
        n = len(agents)
        if n < 2:
            return
            
        results = []
        # Generate pairwise results
        for i in range(n):
            for j in range(i + 1, n):
                a = agents[i]
                b = agents[j]
                
                s_a = agent_id_scores[a]
                s_b = agent_id_scores[b]
                
                if s_a > s_b:
                    res = 1.0
                elif s_a < s_b:
                    res = 0.0
                else:
                    res = 0.5
                    
                results.append((a, b, res))
                
        # To avoid inflating K for multiplayer, we might scale K?
        # Standard approach: treat as independent pairwise games.
        # Logic matches update_elo.
        
        self.ratings = update_elo(self.ratings, results, self.k_factor)
        
        for ag in agents:
            self.games_played[ag] = self.games_played.get(ag, 0) + 1
