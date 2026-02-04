
from typing import Dict, Any
from . import MetricInput
from .mobility import get_mobility_counts

def compute_blocking_metrics(inp: MetricInput) -> Dict[str, Any]:
    """
    Computes blocking metrics based on mobility reduction of opponents.
    
    Outputs:
    - blocking: sum of max(0, opp_mobility_before - opp_mobility_after)
    - block_eff: blocking / area_gain
    - blocking_target_id: opponent with max loss
    - blocking_target_loss
    """
    counts_before, counts_after = get_mobility_counts(inp)
    
    opponents = inp.opponents
    blocking_sum = 0
    max_loss = -1
    target_id = None
    
    for opp_id in opponents:
        before = counts_before.get(opp_id, 0)
        after = counts_after.get(opp_id, 0)
        loss = max(0, before - after)
        blocking_sum += loss
        
        if loss > max_loss:
            max_loss = loss
            target_id = opp_id
            
    # Area gain for efficiency metric
    placed = inp.get_placed_squares()
    area_gain = len(placed) if placed else 0
    
    block_eff = blocking_sum / max(1, area_gain)
    
    return {
        "blocking": blocking_sum,
        "block_eff": block_eff,
        "blocking_target_id": target_id,
        "blocking_target_loss": max_loss if target_id is not None else 0
    }
