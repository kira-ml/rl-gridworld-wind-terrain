from typing import Dict, Tuple, Any, Literal, List

# ====== Type Aliases ======
AlgorithmOption = Literal["q_learning", "sarsa", "dqn", "policy_iteration"]
UpdateStrategy = Literal["max", "expected", "softmax"]

# ====== Stochastic Terrain Presets ======
STOCHASTIC_TERRAIN_WINDY: Dict[str, float] = {
    "ice": 0.1,   # 10% chance of slipping
    "swamp": 0.3   # 30% chance of slow-down
}

# ====== Environment Configuration ======
DEFAULT_ENV_CONFIG: Dict[str, Any] = {
    "grid_size": (7, 7),                # (rows, cols)
    "start_pos": (0, 0),                # Starting position (row, col)
    "goal_pos": (6, 6),                 # Goal position (row, col)
    
    # Wind zones configuration
    "wind_zones": [
        {
            "area": [(1, 2), (1, 3), (1, 4)],  # List of affected (row, col) positions
            "direction": (0, 1),               # Wind direction (dy, dx)
            "strength": 0.8,                   # Probability of wind effect
            "push_distance": 1                 # How many cells the wind pushes
        },
        {
            "area": [(3, 3), (3, 4), (4, 3), (4, 4)],
            "direction": (-1, 0),             # Upward wind
            "strength": 0.6,
            "push_distance": 1
        }
    ],
    
    # Terrain types and their effects
    "terrain": {
        "ice": {
            "positions": [(2, 1), (2, 2), (2, 3)],
            "slip_prob": 0.3,                # Probability of slipping
            "effect": "slip"                 # Agent might slip in a random direction
        },
        "mud": {
            "positions": [(5, 1), (5, 2)],
            "slow_prob": 0.4,                # Probability of getting slowed
            "step_cost": -2.0                # Additional penalty for stepping on mud
        },
        "quicksand": {
            "positions": [(4, 5), (5, 5)],
            "trap_prob": 0.2,                # Probability of getting trapped
            "escape_cost": -5.0              # Penalty for escaping quicksand
        }
    },
    
    # Rewards configuration
    "rewards": {
        "default_step": -1.0,    # Default cost per step
        "goal": 100.0,           # Reward for reaching goal
        "collision": -2.0        # Penalty for hitting walls
    },
    
    "max_steps": 200            # Maximum steps per episode
}

# ====== Multi-Environment Configurations ======
ENVIRONMENTS: Dict[str, Dict[str, Any]] = {
    "5x5_basic": {**DEFAULT_ENV_CONFIG},
    "10x10_windy": {
        **DEFAULT_ENV_CONFIG,
        "grid_size": (10, 10),
        "goal_pos": (9, 9),
        "wind": [0, 1, 1, 2, 2, 1, 0, 0, 1, 0],
        "stochastic_terrain": STOCHASTIC_TERRAIN_WINDY,
    },
    "7x7_dense": {
        **DEFAULT_ENV_CONFIG,
        "grid_size": (7, 7),
        "goal_pos": (6, 6),
        "wind": [0, 1, 1, 1, 1, 1, 0],
    }
}

# ====== Agent (Q-learning) Configuration ======
QL_AGENT_CONFIG: Dict[str, Any] = {
    "algorithm": "q_learning",      # Could be "q_learning", "sarsa", "dqn", etc.
    "alpha": 0.1,                   # Learning rate
    "gamma": 0.99,                  # Discount factor
    "epsilon": 0.1,                 # Exploration rate
    "min_epsilon": 0.01,            # Floor value for epsilon
    "decay_rate": 0.995,            # Epsilon decay per episode
    "init_q": 0.0,                  # Initial Q-value
    "adaptive_lr": False,           # Enable learning rate annealing
    "update_strategy": "max",      # Can switch to "expected" or "softmax"
}

# ====== Agent (DQN) Configuration ======
DQN_AGENT_CONFIG: Dict[str, Any] = {
    "algorithm": "dqn",
    "hidden_dim": 128,
    "buffer_size": 10000,
    "batch_size": 64,
    "sync_frequency": 5,
    "gamma": 0.99,
    "learning_rate": 1e-3,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "reward_step_penalty": -1.0,
    "max_grad_norm": 1.0,
    "num_episodes": 500,
    "max_steps": 200,
    # Add more DQN-specific options as needed
}

# ====== Agent (SARSA) Configuration ======
SARSA_AGENT_CONFIG: Dict[str, Any] = {
    "algorithm": "sarsa",
    "alpha": 0.1,            # Learning rate
    "gamma": 0.99,           # Discount factor
    "epsilon": 1.0,          # Initial exploration rate
    "min_epsilon": 0.01,     # Minimum exploration rate
    "decay_rate": 0.995,     # Epsilon decay rate
    "num_episodes": 1000     # Number of training episodes
}

# ====== Agent (Policy Iteration) Configuration ======
PI_AGENT_CONFIG: Dict[str, Any] = {
    "algorithm": "policy_iteration",
    "gamma": 0.99,                  # Discount factor
    "theta": 0.01,                  # Convergence threshold
    "max_iterations": 100,          # Maximum iterations for convergence
    "num_episodes": 50,             # Number of episodes to run
}

# ====== Path Configuration ======
PATHS: Dict[str, str] = {
    "data": "data/processed/gridworld_v1.pkl",
    "models": "models/trained/q_learning_agent.pkl",
    "logs": "logs/training_log.csv",
    "plots": "results/plots/",
}

# ====== Example for curriculum training loop ======
# for level_name, config in ENVIRONMENTS.items():
#     env = GridWorldEnv(config=config)
#     agent = QLearningAgent(env, QL_AGENT_CONFIG)
#     agent.train(num_episodes=500)
