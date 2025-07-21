"""
Configuration for pytest with common fixtures.
"""
import pytest
import numpy as np
import torch
from pathlib import Path

from src.config.schema import GridWorldConfig
from src.envs.gridworld import GridWorldEnv
from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent
from src.utils.experiment_tracker import ExperimentTracker

@pytest.fixture
def test_env():
    """Create a small test environment."""
    config = GridWorldConfig(
        grid_size=(4, 4),
        start_pos=(0, 0),
        goal_pos=(3, 3),
        max_steps=100
    )
    return GridWorldEnv(config)

@pytest.fixture
def test_q_learning_agent(test_env):
    """Create a Q-learning agent for testing."""
    config = {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995
    }
    return QLearningAgent(test_env, config)

@pytest.fixture
def test_dqn_agent(test_env):
    """Create a DQN agent for testing."""
    config = {
        "hidden_dim": 64,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "buffer_size": 1000,
        "batch_size": 32
    }
    return DQNAgent(test_env, config)

@pytest.fixture
def experiment_tracker(tmp_path):
    """Create an experiment tracker using a temporary directory."""
    return ExperimentTracker(base_dir=str(tmp_path / "experiments"))

def set_random_seed(seed=42):
    """Set random seed for reproducibility in tests."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
