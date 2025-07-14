import numpy as np
from numpy import ndindex
from envs.gridworld import GridWorldEnv
from config import QL_AGENT_CONFIG
from tqdm import trange


class PolicyIterationAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        if not hasattr(env, 'observation_space') or not hasattr(env.observation_space, 'n'):
            raise ValueError("Environment must have discrete observation_space with attribute 'n'")


        if not hasattr(env, 'action_space') or not hasattr(env.action_space, 'n'):
            raise ValueError("Environment must have a discrete action_space with attribute 'n'")
        

        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        
        self.V = np.zeros(env.num_states)
        self.policy = np.random.randint(0, self.num_actions, size=self.num_states)
        self.transitions = np.zeros((self.num_states, self.num_actions), dtype=int)
        self.rewards = np.zeros((self.num_states, self.num_actions), dtype=float)

        # Fix: Remove call to non-existent get_transition_reward, use a placeholder or implement logic here
        for state in range(self.num_states):
            for action in range(self.num_actions):
                self.transitions[state, action] = state
                self.rewards[state, action] = 0.0


    def _initialize_transition_model(self):


        try:
            for state in range(self.num_states):
                for action in range(self.num_actions):
                    next_state, reward, done, _ = self.env.unwrapped.P[state][action][0]

                    self.transitions[state, action] = next_state
                    self.rewards[state, action] = reward

        except AttributeError:
            print("Warning: env does not expose .P transition model; using default self-transitions.")
            

            for state in range(self.env.observation_space.n):
                for action in range(self.env.action_space.n):
                    self.transitions[state, action] = state
                    self.rewards[state, action] = 0
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
