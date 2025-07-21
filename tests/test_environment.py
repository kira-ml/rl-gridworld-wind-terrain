"""
Unit tests for the GridWorld environment.
"""
import pytest
import numpy as np
from src.envs.gridworld import GridWorldEnv
from src.config.schema import GridWorldConfig

def test_env_initialization():
    """Test environment initialization."""
    config = GridWorldConfig(
        grid_size=(4, 4),
        start_pos=(0, 0),
        goal_pos=(3, 3)
    )
    env = GridWorldEnv(config)
    assert env.grid_height == 4
    assert env.grid_width == 4
    assert tuple(env.agent_pos) == (0, 0)

def test_env_reset():
    """Test environment reset."""
    config = GridWorldConfig(
        grid_size=(4, 4),
        start_pos=(0, 0),
        goal_pos=(3, 3)
    )
    env = GridWorldEnv(config)
    env.agent_pos = np.array([2, 2])
    state = env.reset()
    assert tuple(env.agent_pos) == (0, 0)
    assert tuple(state) == (0, 0)

def test_invalid_action():
    """Test invalid action handling."""
    config = GridWorldConfig(
        grid_size=(4, 4),
        start_pos=(0, 0),
        goal_pos=(3, 3)
    )
    env = GridWorldEnv(config)
    # Try to move up from (0,0)
    state, reward, done, _ = env.step(0)  # Up
    assert tuple(state) == (0, 0)  # Should not move
    assert reward < 0  # Should get negative reward

def test_goal_reached():
    """Test goal state detection."""
    config = GridWorldConfig(
        grid_size=(2, 2),
        start_pos=(0, 0),
        goal_pos=(1, 1)
    )
    env = GridWorldEnv(config)
    # Move to goal: right then down
    _, _, done, _ = env.step(1)  # Right
    state, reward, done, _ = env.step(2)  # Down
    assert done
    assert reward > 0
    assert tuple(state) == (1, 1)
