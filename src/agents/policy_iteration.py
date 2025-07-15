import numpy as np
from numpy import ndindex
from envs.gridworld import GridWorldEnv
from config import QL_AGENT_CONFIG
from tqdm import trange


class PolicyIterationAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        height, width = env.grid_height, env.grid_width
        self.V = np.zeros((height, width))
        self.policy = np.random.randint(0, 4, size=(height, width))
        self.transitions = np.zeros((height, width, 4, 2), dtype=int)
        self.rewards = np.zeros((height, width, 4))

        # Initialize the transition and reward models
        self._initialize_transition_model()

    def _initialize_transition_model(self):
        """Initialize the transition and reward models by sampling from the environment."""
        height, width = self.env.grid_height, self.env.grid_width
        
        for i in range(height):
            for j in range(width):
                state = (i, j)
                for action in range(4):  # 0=Up, 1=Right, 2=Down, 3=Left
                    # Save current agent position
                    original_pos = self.env.agent_pos.copy() if hasattr(self.env, 'agent_pos') else None
                    
                    # Set agent to current state
                    self.env.agent_pos = np.array(state)
                    
                    # Take action and observe result
                    next_state, reward, _, _ = self.env.step(action)
                    
                    # Store transition and reward
                    self.transitions[i, j, action] = next_state
                    self.rewards[i, j, action] = reward
                    
                    # Restore agent position
                    if original_pos is not None:
                        self.env.agent_pos = original_pos
    def policy_evaluation(self):
        """Evaluate the current policy until convergence."""
        gamma = self.config["gamma"]
        theta = self.config["theta"]

        while True:
            delta = 0
            for i, j in ndindex(self.V.shape):
                v = self.V[i, j]
                action = self.policy[i, j]
                next_state = self.transitions[i, j, action]
                reward = self.rewards[i, j, action]
                self.V[i, j] = reward + gamma * self.V[next_state[0], next_state[1]]
                delta = max(delta, abs(v - self.V[i, j]))
            if delta < theta:
                break

    def policy_improvement(self):
        """Update the policy greedily based on the value function."""
        policy_stable = True
        for i, j in ndindex(self.V.shape):
            old_action = self.policy[i, j]
            q_values = []
            for a in range(4):
                next_state = self.transitions[i, j, a]
                reward = self.rewards[i, j, a]
                q_values.append(reward + self.config["gamma"] * self.V[next_state[0], next_state[1]])
            best_action = np.argmax(q_values)
            self.policy[i, j] = best_action
            if best_action != old_action:
                policy_stable = False
        return policy_stable

    def run_policy_iteration(self):
        """Perform the full policy iteration."""
        for _ in trange(self.config["max_iterations"]):
            self.policy_evaluation()
            if self.policy_improvement():
                break

    def act(self, state):
        """Return the best action for a given state."""
        # Convert numpy array to tuple if needed
        if isinstance(state, np.ndarray):
            state = tuple(state)
        # Handle tuple of numpy arrays
        if isinstance(state, tuple) and isinstance(state[0], np.ndarray):
            state = tuple(map(int, state[0]))
        return self.policy[state]


#  STEP 5: Main Execution
if __name__ == "__main__":
    # Instantiate the environment
    from envs.gridworld import GridWorldEnv
    from config import QL_AGENT_CONFIG

    env = GridWorldEnv()
    agent = PolicyIterationAgent(env, QL_AGENT_CONFIG)

    agent.run_policy_iteration()

    # Inspect final results
    print("V shape:", agent.V.shape)
    print("Policy shape:", agent.policy.shape)
    print("Transitions shape:", agent.transitions.shape)
    print("Reward shape:", agent.rewards.shape)
