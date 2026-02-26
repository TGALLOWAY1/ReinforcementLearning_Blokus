
import argparse
import json
import os

import pandas as pd

from .phase_split import get_phase_label


def load_jsonl(path):
    data = []
    if not os.path.exists(path):
        return data
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def aggregate_games(log_dir: str, output_path: str):
    steps_path = os.path.join(log_dir, "steps.jsonl")
    results_path = os.path.join(log_dir, "results.jsonl")
    
    print(f"Loading logs from {log_dir}...")
    steps_data = load_jsonl(steps_path)
    results_data = load_jsonl(results_path)
    
    if not steps_data:
        print("No steps data found.")
        return
        
    # Convert to DF
    steps_df = pd.DataFrame(steps_data)
    
    # Flatten metrics
    # metrics is a dict column. We want to promote keys to columns.
    metrics_df = pd.json_normalize(steps_df['metrics'])
    metrics_df.columns = [f"m_{c}" for c in metrics_df.columns] # Prefix to avoid collision
    
    # Combine
    df = pd.concat([steps_df.drop(columns=['metrics', 'action', 'pieces_remaining'], errors='ignore'), metrics_df], axis=1)
    
    # Enrich with game info (agent_ids)
    # Results: game_id -> agent_map
    game_agent_map = {} # (game_id, player_id) -> agent_id
    game_winners = {}
    game_scores = {}
    
    for res in results_data:
        gid = res['game_id']
        game_winners[gid] = res['winner_id']
        # Scores is dict str->int
        for pid_str, score in res['final_scores'].items():
            game_scores[(gid, int(pid_str))] = score
            
        # Agent map
        for pid_str, agent_id in res['agent_ids'].items():
            game_agent_map[(gid, int(pid_str))] = agent_id
            
    # Add agent_id col
    df['agent_id'] = df.apply(lambda r: game_agent_map.get((r['game_id'], r['player_id']), "Unknown"), axis=1)
    
    # Compute phases
    # We need total moves per (game, player)
    move_counts = df.groupby(['game_id', 'player_id'])['turn_index'].count().reset_index(name='total_moves')
    df = df.merge(move_counts, on=['game_id', 'player_id'], how='left')
    
    df['phase'] = df.apply(lambda r: get_phase_label(r['turn_index'], r['total_moves']), axis=1)
    
    # Calculate Piece Tempo (Big 5)
    # Need piece_size. It's in metrics: m_piece_size
    # Big 5 pieces have size=5 (IDs 11-21)
    # We want avg turn index for moves where piece size >= 5
    # We'll compute this during aggregation
    
    # Aggregate per game/player
    # Group by game_id, player_id, agent_id
    # We want generic means, plus phase-specific means
    
    # 1. Overall means
    # Identify metric columns (start with m_)
    metric_cols = [c for c in df.columns if c.startswith('m_')]
    
    overall_stats = df.groupby(['game_id', 'player_id', 'agent_id'])[metric_cols].mean().add_suffix('_mean')
    
    # 2. Phase-specific means
    # Pivot phases?
    # Group by game, player, phase -> mean -> pivot
    phase_stats = df.groupby(['game_id', 'player_id', 'phase'])[metric_cols].mean().unstack(level='phase')
    # Flatten columns: 'm_center_distance' + 'Opening' -> 'm_center_distance_Opening_mean'
    phase_stats.columns = [f"{c[0]}_{c[1]}_mean" for c in phase_stats.columns]
    
    # 3. Piece Tempo
    # Filter for size >= 5
    if 'm_piece_size' in df.columns:
        big_pieces = df[df['m_piece_size'] >= 5]
        tempo_stats = big_pieces.groupby(['game_id', 'player_id'])['turn_index'].mean().to_frame(name='avg_turn_big5')
    else:
        tempo_stats = pd.DataFrame()
        
    # Merge all
    summary = overall_stats.join(phase_stats).join(tempo_stats)
    
    # Add scalar results info (score, win)
    summary = summary.reset_index()
    summary['final_score'] = summary.apply(lambda r: game_scores.get((r['game_id'], r['player_id']), 0), axis=1)
    summary['is_winner'] = summary.apply(lambda r: 1 if game_winners.get(r['game_id']) == r['player_id'] else 0, axis=1)
    
    print(f"Aggregated {len(summary)} player-games.")
    
    # Save
    summary.to_parquet(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="logs/analytics")
    parser.add_argument("--output", default="logs/analytics/game_summary.parquet")
    args = parser.parse_args()
    
    aggregate_games(args.log_dir, args.output)
