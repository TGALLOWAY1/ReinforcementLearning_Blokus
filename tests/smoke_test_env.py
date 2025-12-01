"""
Smoke test for Blokus RL environment.
Verifies basic functionality of the Gymnasium-compatible environment.
"""

import numpy as np
from envs.blokus_v0 import make_gymnasium_env


def run_smoke_test():
    """Run smoke test on the Blokus RL environment."""
    print("=" * 60)
    print("Blokus RL Environment Smoke Test")
    print("=" * 60)
    
    # Instantiate the environment
    print("\n1. Creating environment...")
    env = make_gymnasium_env(render_mode=None, max_episode_steps=100)
    
    # Print observation and action space information
    print("\n2. Environment Information:")
    print(f"   Observation space shape: {env.observation_space.shape}")
    print(f"   Action space size: {env.action_space.n}")
    
    # Expected observation shape
    expected_shape = (30, 20, 20)
    if env.observation_space.shape != expected_shape:
        print(f"   ⚠️  WARNING: Expected observation shape {expected_shape}, got {env.observation_space.shape}")
    else:
        print(f"   ✓ Observation shape matches expected {expected_shape}")
    
    # Run 5 episodes
    print("\n3. Running 5 episodes...")
    print("-" * 60)
    
    for episode in range(5):
        print(f"\nEpisode {episode + 1}:")
        
        # Reset environment
        obs, info = env.reset()
        
        # Verify observation shape
        if obs.shape != expected_shape:
            print(f"   ❌ ERROR: Observation shape is {obs.shape}, expected {expected_shape}")
        else:
            print(f"   ✓ Observation shape: {obs.shape}")
        
        # Episode tracking
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done:
            # Get legal action mask from info
            if 'legal_action_mask' not in info:
                print(f"   ❌ ERROR: 'legal_action_mask' not found in info dict")
                print(f"   Available keys: {list(info.keys())}")
                break
            
            legal_action_mask = info['legal_action_mask']
            
            # Select a random valid action based on the mask
            valid_actions = np.where(legal_action_mask)[0]
            
            if len(valid_actions) == 0:
                print(f"   ⚠️  No valid actions available at step {steps}")
                break
            
            action = np.random.choice(valid_actions)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update tracking
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Verify observation shape at each step
            if obs.shape != expected_shape:
                print(f"   ❌ ERROR at step {steps}: Observation shape is {obs.shape}, expected {expected_shape}")
        
        # Log episode results
        print(f"   Steps: {steps}")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Done: {done}")
    
    print("\n" + "=" * 60)
    print("Smoke test completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()

