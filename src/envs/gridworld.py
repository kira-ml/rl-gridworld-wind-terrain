from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from typing import List, Tuple, Dict, Any
import random

class GridWorldEnv(Env):
    """
    Enhanced GridWorld environment with sophisticated wind zones and stochastic terrain.
    
    Features:
    - Directional wind zones that can push the agent in any direction
    - Multiple terrain types:
        - Ice: Random slipping in any direction
        - Mud: Movement penalties and slowdown
        - Quicksand: Possible trapping with escape penalties
    - Configurable parameters for all features
    """

    def __init__(self, config: Dict[str, Any] = None):
        super(GridWorldEnv, self).__init__()
        
        # Use default config if none provided
        self.config = config or {}
        
        # Grid dimensions
        self.grid_height, self.grid_width = self.config.get("grid_size", (7, 7))
        
        # Action space: 0=Up, 1=Right, 2=Down, 3=Left
        self._action_space = Discrete(4)
        
        # Observation space: (row, col)
        self._observation_space = Box(
            low=0,
            high=max(self.grid_height, self.grid_width) - 1,
            shape=(2,),
            dtype=np.int32
        )
        
    @property
    def action_space(self):
        return self._action_space
        
    @property
    def observation_space(self):
        return self._observation_space
        
        # Action to direction mapping
        self.action_to_dir = {
            0: (-1, 0),   # Up
            1: (0, 1),    # Right
            2: (1, 0),    # Down
            3: (0, -1)    # Left
        }
        
        # Initialize state
        self.agent_pos = None
        self.steps_taken = 0
        self.reset()

    def reset(self):
        """Reset the environment state."""
        self.agent_pos = np.array(self.config.get("start_pos", (0, 0)))
        self.steps_taken = 0
        return self.agent_pos.copy()

    def _check_wind_effect(self, position: np.ndarray) -> np.ndarray:
        """Apply wind effects if agent is in a wind zone."""
        pos = position.copy()
        
        # Check each wind zone
        for wind_zone in self.config.get("wind_zones", []):
            # Check if current position is in wind zone
            if tuple(pos) in wind_zone["area"]:
                # Apply wind effect based on probability
                if random.random() < wind_zone["strength"]:
                    dy, dx = wind_zone["direction"]
                    push = wind_zone["push_distance"]
                    new_pos = pos + np.array([dy * push, dx * push])
                    
                    # Ensure wind doesn't push agent out of bounds
                    if self._is_valid_position(new_pos):
                        pos = new_pos
        
        return pos

    def _check_terrain_effect(self, position: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Apply terrain effects based on current position."""
        pos = position.copy()
        additional_reward = 0
        done = False
        pos_tuple = tuple(pos)
        
        terrain = self.config.get("terrain", {})
        
        # Check ice effect (random slipping)
        if "ice" in terrain and pos_tuple in terrain["ice"]["positions"]:
            if random.random() < terrain["ice"]["slip_prob"]:
                # Random slip direction
                slip_dir = random.choice(list(self.action_to_dir.values()))
                new_pos = pos + np.array(slip_dir)
                if self._is_valid_position(new_pos):
                    pos = new_pos
        
        # Check mud effect (movement penalty)
        if "mud" in terrain and pos_tuple in terrain["mud"]["positions"]:
            if random.random() < terrain["mud"]["slow_prob"]:
                additional_reward += terrain["mud"]["step_cost"]
        
        # Check quicksand effect (possible trapping)
        if "quicksand" in terrain and pos_tuple in terrain["quicksand"]["positions"]:
            if random.random() < terrain["quicksand"]["trap_prob"]:
                additional_reward += terrain["quicksand"]["escape_cost"]
        
        return pos, additional_reward, done

    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if a position is within grid bounds."""
        return (0 <= pos[0] < self.grid_height and 
                0 <= pos[1] < self.grid_width)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        self.steps_taken += 1
        
        # Get move direction
        dy, dx = self.action_to_dir[action]
        new_pos = self.agent_pos + np.array([dy, dx])
        
        # Check if move is valid
        if not self._is_valid_position(new_pos):
            new_pos = self.agent_pos
            reward = self.config.get("rewards", {}).get("collision", -2.0)
        else:
            reward = self.config.get("rewards", {}).get("default_step", -1.0)
        
        # Apply wind effects
        new_pos = self._check_wind_effect(new_pos)
        
        # Apply terrain effects
        new_pos, terrain_reward, terrain_done = self._check_terrain_effect(new_pos)
        reward += terrain_reward
        
        # Update position
        self.agent_pos = new_pos
        
        # Check if goal reached
        done = (
            terrain_done or
            np.array_equal(self.agent_pos, self.config.get("goal_pos", (self.grid_height-1, self.grid_width-1))) or
            self.steps_taken >= self.config.get("max_steps", 200)
        )
        
        # Add goal reward if applicable
        if np.array_equal(self.agent_pos, self.config.get("goal_pos", (self.grid_height-1, self.grid_width-1))):
            reward += self.config.get("rewards", {}).get("goal", 100.0)
        
        info = {
            "position": self.agent_pos.tolist(),
            "steps": self.steps_taken,
            "wind_active": tuple(self.agent_pos) in [pos for zone in self.config.get("wind_zones", []) for pos in zone["area"]],
            "terrain": self._get_terrain_at_position(self.agent_pos)
        }
        
        return self.agent_pos.copy(), reward, done, info

    def _get_terrain_at_position(self, pos: np.ndarray) -> str:
        """Get the type of terrain at a given position."""
        pos_tuple = tuple(pos)
        terrain = self.config.get("terrain", {})
        
        for terrain_type, data in terrain.items():
            if pos_tuple in data["positions"]:
                return terrain_type
        return "normal"

    def render(self, mode='human'):
        """Render the current state of the environment."""
        if mode != 'human':
            return
            
        grid = np.full(self.config["grid_size"], ".", dtype=str)
        
        # Draw wind zones
        for wind_zone in self.config.get("wind_zones", []):
            for pos in wind_zone["area"]:
                grid[pos] = "W"
        
        # Draw terrain
        for terrain_type, data in self.config.get("terrain", {}).items():
            symbol = terrain_type[0].upper()
            for pos in data["positions"]:
                grid[pos] = symbol
        
        # Draw agent and goal
        grid[tuple(self.agent_pos)] = "A"
        grid[tuple(self.config["goal_pos"])] = "G"
        
        # Print the grid
        print("\n" + "=" * (self.grid_width * 2 + 1))
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("=" * (self.grid_width * 2 + 1))
        print(f"Steps: {self.steps_taken}")
        print(f"Position: {tuple(self.agent_pos)}")
        print(f"Terrain: {self._get_terrain_at_position(self.agent_pos)}")
