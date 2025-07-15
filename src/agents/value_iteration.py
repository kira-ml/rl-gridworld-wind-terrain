import numpy as np
from numpy import ndindex
from envs.gridworld import GridWorldEnv
from config import QL_AGENT_CONFIG
from typing import Any, Dict, Tuple
from tqdm import trange


class ValueIterationAgent:
    def __init__(self, env: GridWorldEnv, config: Dict[str, Any]) -> None:
        required_keys = {"gamma", "theta", "max_iterations"}
        if not required_keys.issubset(config.keys()):
            missing = required_keys - config.keys()
            raise ValueError(f"Missing required config keys: {missing}")
        self.env = env
        self.config = config
        self.V = np.zeros((env.grid_height, env.grid_width))
        self.policy: np.ndarray = np.zeros((env.grid_height, env.grid_width), dtype=int)


    def run_value_iteration(self):
        for _ in trange(self.config["max_iterations"], desc="Value Iteration"):
            delta = 0
            for state in ndindex(self.V.shape):
                if self.env.is_terminal(state):
                    self.V[state] = 0
                    continue
                v = self.V[state]
                max_value = -float('inf')
                
                for action in range(self.env.action_space.n):
                    next_state, reward, _, _ = self.env.steps(state, action)
                    max_value = max(max_value, reward + self.config["gamma"] * self.V[next_state])
                self.V[state] = max_value
                delta = max(delta, abs(v - self.V[state]))
            
            if delta < self.config["theta"]:
                print(f"Converged after {_} iterations")
                break
            elif _ == self.config["max_iterations"] - 1:
                print("Warning: Maximum iteration reached without convergence")
                
                 

    def extract_policy(self) -> None:
        self.policy = np.zeros_like(self.V, dtype=int)
        for state in np.ndindex(self.V.shape):
            if self.env.is_terminal(state):
                continue
            best_action = 0
            best_value = -float('inf')
            for action in range(self.env.action_space.n):
                next_state, reward, _, _ = self.env.step(state, action)
                value = reward + self.config["gamma"] * self.V[next_state]
            if value > best_value:
                best_value = value
                best_action = action

            if best_action is None:
                raise RuntimeError(f"No valid action found for state {state}")
                
            self.policy[state] = best_action

    def _calculate_state_value(self, state: Tuple[int, int]) -> float:

        values = []
        for action in range(self.env.action_space.n):
            next_state, reward, _, _ = self.env.step(state, action)
            values.append(reward + self.config["gamma"] * self.V[next_state])
        return max(values) if values else 0
    def act(self, state: Tuple[int, int]) -> int:
        if not hasattr(self, 'policy') or self.policy is None:
            raise ValueError("Policy not initialized. Run value iteration first.")
        return self.policy[state]

    def save(self, filepath: str) -> None:
        np.savez(filepath, V=self.V, policy=self.policy)


    def load(self, filepath: str) -> None:
        data = np.load(filepath)
        self.V = data['V']
        self.policy = data['policy']