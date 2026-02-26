
import argparse
import os

import numpy as np
import pandas as pd


def aggregate_agents(game_summary_path: str, output_path: str):
    if not os.path.exists(game_summary_path):
        print(f"Game summary not found at {game_summary_path}")
        return
        
    print(f"Loading game summary from {game_summary_path}...")
    df = pd.read_parquet(game_summary_path)
    
    # helper for z-scoring
    def zscore(series):
        if series.std() == 0:
            return 0
        return (series - series.mean()) / series.std()
    
    # 1. Group by agent_id
    # Select numeric columns for mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude game_id, player_id if they are numeric (unlikely for strings/IDs)
    
    agent_stats = df.groupby('agent_id')[numeric_cols].mean()
    agent_stats['games_played'] = df.groupby('agent_id').size()
    
    # Check if we have enough data for z-scores
    if len(agent_stats) < 2:
        print("Warning: Need at least 2 agents for meaningful Z-scores. Indices will be 0.")
    
    # 2. Compute indices
    # Map user friendly names to columns
    # We use .get to be safe against missing cols
    def get_col(name):
        # try exact match
        if name in agent_stats.columns: return agent_stats[name]
        return pd.Series(0, index=agent_stats.index)

    # Calculate Z-scores globally across agents
    # Columns requested:
    # m_blocking_mean, m_corner_block_mean, m_dist_to_opp_min_mean
    # m_center_distance_mean, m_center_gain_Opening_mean
    # m_mobility_me_delta_mean, m_corners_me_delta_mean
    # m_area_gain_mean, m_delta_perimeter_mean
    # avg_turn_big5
    
    z_blocking = zscore(get_col('m_blocking_mean'))
    z_corner_block = zscore(get_col('m_corner_block_mean'))
    z_dist_opp = zscore(get_col('m_dist_to_opp_min_mean'))
    
    z_center_dist = zscore(get_col('m_center_distance_mean'))
    z_center_gain_open = zscore(get_col('m_center_gain_Opening_mean'))
    
    z_mob_delta = zscore(get_col('m_mobility_me_delta_mean'))
    z_corners_delta = zscore(get_col('m_corners_me_delta_mean'))
    
    z_area_gain = zscore(get_col('m_area_gain_mean'))
    z_delta_perim = zscore(get_col('m_delta_perimeter_mean'))
    
    z_turn_big5 = zscore(get_col('avg_turn_big5'))
    
    # Composite Indices
    agent_stats['AggressionIndex'] = z_blocking + z_corner_block - z_dist_opp
    agent_stats['CenterFocus'] = -z_center_dist + z_center_gain_open
    agent_stats['MobilityCare'] = z_mob_delta + z_corners_delta
    agent_stats['ExpansionIndex'] = z_area_gain + z_delta_perim
    agent_stats['PieceTempo'] = -z_turn_big5
    
    print(f"Aggregated {len(agent_stats)} agents.")
    print(agent_stats[['games_played', 'AggressionIndex', 'CenterFocus', 'MobilityCare', 'PieceTempo']].head())
    
    agent_stats.to_parquet(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="logs/analytics/game_summary.parquet")
    parser.add_argument("--output", default="logs/analytics/agent_summary.parquet")
    args = parser.parse_args()
    
    aggregate_agents(args.input, args.output)
