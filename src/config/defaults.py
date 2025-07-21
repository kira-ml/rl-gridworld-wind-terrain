"""
Default configuration settings for all components.
These can be overridden by environment variables or command-line arguments.
"""

from .schema import (
    TabularAgentConfig,
    DQNConfig,
    PolicyIterationConfig,
    GridWorldConfig,
    load_config_with_env_overrides
)

# Environment Configuration
DEFAULT_ENV_CONFIG = load_config_with_env_overrides(
    GridWorldConfig,
    {
        "grid_size": (7, 7),
        "start_pos": (0, 0),
        "goal_pos": (6, 6),
        "max_steps": 200,
        "rewards": {
            "default_step": -1.0,
            "collision": -2.0,
            "goal": 100.0
        }
    }
)

# Agent Configurations
QL_AGENT_CONFIG = load_config_with_env_overrides(
    TabularAgentConfig,
    {
        "num_episodes": 400,
        "max_steps": 200,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995
    }
)

SARSA_AGENT_CONFIG = load_config_with_env_overrides(
    TabularAgentConfig,
    {
        "num_episodes": 500,
        "max_steps": 200,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995
    }
)

DQN_AGENT_CONFIG = load_config_with_env_overrides(
    DQNConfig,
    {
        "num_episodes": 1200,
        "max_steps": 200,
        "hidden_dim": 128,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "buffer_size": 10000,
        "batch_size": 64,
        "target_update_freq": 10
    }
)

PI_AGENT_CONFIG = load_config_with_env_overrides(
    PolicyIterationConfig,
    {
        "num_episodes": 150,
        "max_steps": 200,
        "gamma": 0.99,
        "theta": 0.001,
        "max_iterations": 100
    }
)
