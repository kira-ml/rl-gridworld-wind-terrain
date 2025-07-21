"""
Configuration schema validation using Pydantic.
Defines and validates all configuration objects used in the project.
"""

from typing import Dict, List, Tuple, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
import os

class BaseAgentConfig(BaseModel):
    """Base configuration for all agents."""
    num_episodes: int = Field(default=100, ge=1)
    max_steps: int = Field(default=200, ge=1)
    seed: Optional[int] = None
    
    class Config:
        extra = "forbid"  # Prevent typos in config keys

class TabularAgentConfig(BaseAgentConfig):
    """Configuration for tabular agents (Q-Learning, SARSA)."""
    alpha: float = Field(default=0.1, gt=0, lt=1)
    gamma: float = Field(default=0.99, gt=0, lt=1)
    epsilon_start: float = Field(default=1.0, gt=0, lt=1)
    epsilon_min: float = Field(default=0.01, gt=0, lt=1)
    epsilon_decay: float = Field(default=0.995, gt=0, lt=1)

class DQNConfig(BaseAgentConfig):
    """Configuration for DQN agent."""
    hidden_dim: int = Field(default=128, gt=0)
    learning_rate: float = Field(default=0.001, gt=0)
    gamma: float = Field(default=0.99, gt=0, lt=1)
    epsilon_start: float = Field(default=1.0, gt=0, lt=1)
    epsilon_min: float = Field(default=0.01, gt=0, lt=1)
    epsilon_decay: float = Field(default=0.995, gt=0, lt=1)
    buffer_size: int = Field(default=10000, gt=0)
    batch_size: int = Field(default=64, gt=0)
    target_update_freq: int = Field(default=10, gt=0)

class PolicyIterationConfig(BaseAgentConfig):
    """Configuration for Policy Iteration agent."""
    gamma: float = Field(default=0.99, gt=0, lt=1)
    theta: float = Field(default=0.001, gt=0)
    max_iterations: int = Field(default=100, gt=0)

class GridWorldConfig(BaseModel):
    """Configuration for GridWorld environment."""
    grid_size: Tuple[int, int] = Field(default=(7, 7))
    start_pos: Tuple[int, int] = Field(default=(0, 0))
    goal_pos: Tuple[int, int] = Field(default=(6, 6))
    max_steps: int = Field(default=200, gt=0)
    rewards: Dict[str, float] = Field(
        default={
            "default_step": -1.0,
            "collision": -2.0,
            "goal": 100.0
        }
    )
    
    @validator("grid_size")
    def validate_grid_size(cls, v):
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError("Grid dimensions must be positive")
        return v
    
    @validator("start_pos", "goal_pos")
    def validate_positions(cls, v, values):
        if "grid_size" in values:
            if v[0] >= values["grid_size"][0] or v[1] >= values["grid_size"][1]:
                raise ValueError("Position out of grid bounds")
        return v

def load_config_with_env_overrides(config_class: BaseModel, config_dict: dict) -> BaseModel:
    """
    Load config with environment variable overrides.
    Environment variables should be prefixed with 'RL_' and use underscore format.
    Example: RL_LEARNING_RATE=0.001 will override learning_rate
    """
    # Get all environment variables with RL_ prefix
    env_vars = {
        k[3:].lower(): v
        for k, v in os.environ.items()
        if k.startswith("RL_")
    }
    
    # Convert types based on schema
    converted_vars = {}
    for key, value in env_vars.items():
        if key in config_dict:
            field_type = config_class.__annotations__.get(key)
            if field_type == float:
                converted_vars[key] = float(value)
            elif field_type == int:
                converted_vars[key] = int(value)
            elif field_type == bool:
                converted_vars[key] = value.lower() in ("true", "1", "yes")
            else:
                converted_vars[key] = value
    
    # Merge with priority: env vars > config dict
    merged_config = {**config_dict, **converted_vars}
    return config_class(**merged_config)
