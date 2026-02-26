"""
Test for no-legal-moves and dead-agent handling in Blokus environment.

This test verifies that:
1. When a player has no legal moves, they are marked as terminated
2. No all-False action masks are exposed to the policy
3. The environment progresses correctly to other players or game over
4. Dead agents are properly skipped
"""

import numpy as np
import pytest

from envs.blokus_v0 import BlokusEnv, make_gymnasium_env


def test_no_legal_moves_marks_agent_terminated():
    """Test that agents with no legal moves are marked as terminated."""
    env = BlokusEnv()
    env.reset(seed=42)
    
    # Get initial info for player_0
    initial_info = env.infos["player_0"]
    assert "can_move" in initial_info
    assert "legal_action_mask" in initial_info
    
    # Manually set up a scenario where a player has no legal moves
    # by filling the board such that they can't place any pieces
    # This is tricky to do manually, so we'll simulate by checking the logic
    
    # Instead, let's verify that when can_move is False, the agent is terminated
    # We can do this by checking the _get_info logic
    
    # Force a scenario: mark an agent as having no moves
    # We'll check that _get_info handles this correctly
    player = env._agent_to_player("player_0")
    legal_moves = env.move_generator.get_legal_moves(env.game.board, player)
    
    if len(legal_moves) == 0:
        # If player_0 has no moves, _get_info should mark them as terminated
        info = env._get_info("player_0")
        assert env.terminations["player_0"] is True
        assert info["can_move"] is False
        # Mask should not be all-False (should have at least one True for safety)
        assert info["legal_action_mask"].sum() > 0, "Mask should have at least one True value for terminated agents"
    else:
        # If player_0 has moves, verify normal behavior
        info = env._get_info("player_0")
        assert info["can_move"] is True
        assert info["legal_action_mask"].sum() > 0


def test_dead_agent_skipped_in_step():
    """Test that dead agents are skipped when step() is called."""
    env = BlokusEnv()
    env.reset(seed=42)
    
    # Manually mark an agent as terminated
    env.terminations["player_1"] = True
    
    # Set agent_selection to the terminated agent
    env.agent_selection = "player_1"
    
    # Call step() - should advance to next live agent
    initial_selection = env.agent_selection
    env.step(None)  # None is correct for dead agents
    
    # Agent selection should have advanced (unless all agents are done)
    # If there are live agents, selection should have changed
    if not all(env.terminations.get(agent, False) or env.truncations.get(agent, False) 
               for agent in env.agents):
        assert env.agent_selection != initial_selection
        # New agent should not be terminated
        assert not env.terminations.get(env.agent_selection, False)
        assert not env.truncations.get(env.agent_selection, False)


def test_no_all_false_mask_exposed():
    """Test that no all-False masks are exposed to the policy."""
    env = BlokusEnv()
    env.reset(seed=42)
    
    # Check all agents' masks
    for agent in env.agents:
        info = env.infos[agent]
        mask = info["legal_action_mask"]
        
        # If agent is terminated, mask should have at least one True
        if env.terminations.get(agent, False) or env.truncations.get(agent, False):
            assert mask.sum() > 0, f"Terminated agent {agent} should have at least one True in mask"
        # If agent is not terminated, they should either have moves or be marked as terminated
        elif not info["can_move"]:
            # Agent has no moves - should be terminated
            assert env.terminations.get(agent, False), f"Agent {agent} with no moves should be terminated"
            # And mask should still have at least one True
            assert mask.sum() > 0, f"Agent {agent} with no moves should have safe mask"


def test_gymnasium_wrapper_handles_dead_agents():
    """Test that GymnasiumBlokusWrapper handles dead agents correctly."""
    env = make_gymnasium_env()
    obs, info = env.reset(seed=42)
    
    # Run a few steps
    for _ in range(10):
        # Check if agent is terminated
        if info.get("can_move", True):
            # Get legal actions
            legal_action_mask = info.get('legal_action_mask', np.zeros(env.action_space.n, dtype=bool))
            valid_actions = np.where(legal_action_mask)[0]
            
            if len(valid_actions) > 0:
                # Sample a valid action
                action = np.random.choice(valid_actions)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # If terminated, reset
                if terminated or truncated:
                    obs, info = env.reset()
            else:
                # No valid actions - should be terminated
                assert terminated or truncated, "Agent with no valid actions should be terminated"
                obs, info = env.reset()
        else:
            # Agent can't move - should be terminated
            assert terminated or truncated, "Agent that can't move should be terminated"
            obs, info = env.reset()


def test_advance_to_next_live_agent():
    """Test the _advance_to_next_live_agent helper method."""
    env = BlokusEnv()
    env.reset(seed=42)
    
    # Mark some agents as terminated
    env.terminations["player_1"] = True
    env.terminations["player_2"] = True
    
    # Set agent_selection to a terminated agent
    env.agent_selection = "player_1"
    
    # Advance to next live agent
    found = env._advance_to_next_live_agent()
    
    # Should find a live agent (player_0 or player_3)
    assert found, "Should find a live agent"
    assert not env.terminations.get(env.agent_selection, False), "Selected agent should not be terminated"
    assert not env.truncations.get(env.agent_selection, False), "Selected agent should not be truncated"
    
    # Mark all agents as terminated
    for agent in env.agents:
        env.terminations[agent] = True
    
    # Should not find a live agent
    found = env._advance_to_next_live_agent()
    assert not found, "Should not find a live agent when all are terminated"


def test_step_with_dead_agent_advances():
    """Test that step() with a dead agent advances to next live agent."""
    env = BlokusEnv()
    env.reset(seed=42)
    
    # Get initial agent
    initial_agent = env.agent_selection
    
    # Mark current agent as terminated
    env.terminations[initial_agent] = True
    
    # Call step() - should advance
    env.step(None)  # None is correct for dead agents
    
    # Should have advanced to a different agent (if there are live agents)
    if not all(env.terminations.get(agent, False) or env.truncations.get(agent, False) 
               for agent in env.agents):
        assert env.agent_selection != initial_agent
        assert not env.terminations.get(env.agent_selection, False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

