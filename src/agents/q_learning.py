import numpy as np
from envs.gridworld import GridWorldEnv
from config import QL_AGENT_CONFIG as AGENT_CONFIG

class QLearningAgent:
    """
    Q-Learning Agent for GridWorld Environment
    
    This implementation uses a tabular approach to learn state-action values
    through experience. The agent follows an epsilon-greedy policy for exploration
    and updates Q-values using the Q-learning update rule.
    
    Key Features:
    - Tabular Q-function representation
    - Epsilon-greedy exploration policy with decay
    - Experience-based value updates
    
    Attributes:
        env: The environment the agent interacts with
        config: Configuration dictionary containing hyperparameters
        n_actions: Number of possible actions
        Q: Q-value table of shape (height, width, actions)
        alpha: Learning rate - controls how much new information overrides old
        gamma: Discount factor for future rewards
        epsilon: Exploration rate - probability of taking random action
        epsilon_decay: Multiplicative decay factor for epsilon
        epsilon_min: Minimum exploration rate
    """

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.initial_alpha = config["alpha"]
        self.current_alpha = config["alpha"]
        self.n_steps = config.get("n_steps", 3)
        self.replay_buffer = []
        
        # Initialize metrics tracking
        self.metrics = {
            "cumulative_rewards": [],
            "episode_lengths": [],
            "success_rates": [],
            "epsilon_values": [],
            "learning_rates": [],
            "average_q_values": [],
            "episode_max_q_values": []
        }
        
        if not hasattr(env, "grid_height"):
            raise ValueError("Environment must have grid_height attribute")
            
        self.n_actions = env.action_space.n
        self.state_shape = (env.grid_height, env.grid_width)
        self.Q = np.zeros(self.state_shape + (self.n_actions,))
        self.alpha = config["alpha"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        
        # Episode tracking
        self._episode_count = 0
        self._total_steps = 0



    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            self.replay_buffer = []
            
            # Episode tracking variables
            episode_reward = 0
            episode_steps = 0
            episode_max_q = float('-inf')
            
            while not done:
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                # Track maximum Q-value
                current_max_q = np.max(self.Q[state[0], state[1]])
                episode_max_q = max(episode_max_q, current_max_q)
                
                self.replay_buffer.append((state, action, reward, next_state, done))

                if len(self.replay_buffer) >= self.n_steps or done:
                    self._n_step_update()
                    self.replay_buffer.pop(0)
                self._update_q_value(state, action, reward, next_state, done)

                state = next_state
                self._total_steps += 1

            # Update metrics
            success = np.array_equal(state, self.config.get("goal_pos", (self.env.grid_height-1, self.env.grid_width-1)))
            self._update_metrics(episode_reward, episode_steps, success, episode_max_q)
            
            # Decay parameters
            self.decay_epsilon()



    def _n_step_update(self):
        state, action, _, _, _ = self.replay_buffer[0]
        total_reward = 0
        gamma_power = 1
        
        # Define reward clipping if not already in config
        reward_clip = self.config.get("reward_clip", 100)
        
        for i in range(len(self.replay_buffer)):
            _, _, reward, _, done = self.replay_buffer[i]
            clipped_reward = np.clip(reward, -reward_clip, reward_clip)
            total_reward += gamma_power * clipped_reward
            gamma_power *= self.gamma
            if done:
                break
        
        last_state = self.replay_buffer[-1][3]
        last_done = self.replay_buffer[-1][4]

        if not last_done:
            total_reward += gamma_power * np.max(self.Q[last_state[0], last_state[1]])

        self._update_q_value(state, action, total_reward, last_state, last_done)



    def act(self, state):
        row, col = state
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state[0], state[1]])
    

    def _update_q_value(self, state, action, reward, next_state, done):
        row, col = state
        next_row, next_col = next_state
        current_q = self.Q[row, col, action]
        
        target_q = reward
        if not done:
            target_q += self.gamma * np.max(self.Q[next_row, next_col])
        
        self.Q[row, col, action] = (
            (1 - self.current_alpha) * current_q + self.current_alpha * target_q
        )


    def _update_metrics(self, episode_reward, episode_steps, success, max_q):
        """Update all training metrics after each episode."""
        self._episode_count += 1
        
        # Update main metrics
        self.metrics["cumulative_rewards"].append(episode_reward)
        self.metrics["episode_lengths"].append(episode_steps)
        self.metrics["success_rates"].append(float(success))
        self.metrics["epsilon_values"].append(self.epsilon)
        self.metrics["learning_rates"].append(self.current_alpha)
        self.metrics["episode_max_q_values"].append(max_q)
        
        # Calculate and store average Q-value
        avg_q = np.mean(self.Q)
        self.metrics["average_q_values"].append(avg_q)
        
    def get_metrics(self):
        """Return current metrics with computed statistics."""
        return {
            "total_episodes": self._episode_count,
            "total_steps": self._total_steps,
            "current_metrics": self.metrics,
            "summary_stats": {
                "avg_reward_last_10": np.mean(self.metrics["cumulative_rewards"][-10:]),
                "avg_steps_last_10": np.mean(self.metrics["episode_lengths"][-10:]),
                "success_rate_last_10": np.mean(self.metrics["success_rates"][-10:]) * 100,
                "current_epsilon": self.epsilon,
                "current_learning_rate": self.current_alpha,
                "average_q_value": np.mean(self.metrics["average_q_values"][-10:])
            }
        }

    def decay_epsilon(self):
        """Decay exploration rate and learning rate according to schedule."""
        # Decay epsilon
        if self.epsilon > self.config.get("min_epsilon", 0.01):
            self.epsilon *= self.config.get("decay_rate", 0.995)
            
        # Decay learning rate if adaptive learning is enabled
        if self.config.get("adaptive_lr", False) and "total_episodes" in self.config:
            self.current_alpha = max(
                self.initial_alpha * (1 - self._episode_count/self.config["total_episodes"]),
                self.config.get("min_alpha", 0.01)
            )
    
    def state_dict(self):
        """
        Get a serializable representation of the agent's state.
        
        Returns:
            dict: Dictionary containing agent state for saving to checkpoint
        """
        return {
            "q_table": self.Q,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "metrics": self.metrics,
            "episode_count": self._episode_count
        }
    
    def load_state_dict(self, state_dict):
        """
        Load agent state from a dictionary.
        
        Args:
            state_dict: Dictionary containing saved agent state
        """
        self.Q = state_dict["q_table"]
        self.epsilon = state_dict.get("epsilon", 0.01)  # Default to low exploration for loading
        self.alpha = state_dict.get("alpha", self.alpha)
        self.gamma = state_dict.get("gamma", self.gamma)
        
        # Optionally restore metrics if available
        if "metrics" in state_dict:
            self.metrics = state_dict["metrics"]
        
        if "episode_count" in state_dict:
            self._episode_count = state_dict["episode_count"]
     
    def update(self, state, action, reward, next_state):

        self._update_q_value(state, action, reward, next_state, done=False)

    
    def get_policy(self):

        policy = {}
        for row in range(self.state_shape[0]):
            for col in range(self.state_shape[1]):
                state = (row, col)
                policy[state] = np.argmax(self.Q[row, col])
        
        return policy

