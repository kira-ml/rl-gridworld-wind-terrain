import numpy as np
from envs.gridworld import GridWorldEnv
from config import QL_AGENT_CONFIG as AGENT_CONFIG

class QLearningAgent:

    """
    Q-Learning agent for reinforcement learning in discrete state-action spaces.
    
    Attributes:
        env: The environment the agent interacts with
        config: Configuration dictionary containing hyperparameters
        n_actions: Number of possible actions
        Q: Q-value table of shape (height, width, actions)
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
    """

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.initial_alpha = config["alpha"]
        self.current_alpha = config["alpha"]
        self.n_steps = config.get("n_steps", 3)
        self.replay_buffer = []




        if not hasattr(env, "grid_height"):
            raise ValueError("Environment must have grid_height attribute")
            
        self.n_actions = env.action_space.n
        self.state_shape = (env.grid_height, env.grid_width)
        self.Q = np.zeros(self.state_shape + (self.n_actions,))
        self.alpha = config["alpha"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]



    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            self.replay_buffer = []
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.append((state, action, reward, next_state, done))

                if len(self.replay_buffer) >= self.n_steps or done:
                    self._n_step_update()
                    self.replay_buffer.pop(0)
                self._update_q_value(state, action, reward, next_state, done)

                state = next_state

            
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


    def decay_epsilon(self):
        # Decay epsilon
        if self.epsilon > self.config.get("min_epsilon", 0.01):
            self.epsilon *= self.config.get("decay_rate", 0.995)
            
        # Decay learning rate if adaptive learning is enabled
        if self.config.get("adaptive_lr", False) and "total_episodes" in self.config:
            current_episode = getattr(self, "_episode_count", 0) + 1
            self._episode_count = current_episode
            self.current_alpha = max(
                self.initial_alpha * (1 - current_episode/self.config["total_episodes"]),
                self.config.get("min_alpha", 0.01)
            )

     
    def update(self, state, action, reward, next_state):

        self._update_q_value(state, action, reward, next_state, done=False)

    
    def get_policy(self):

        policy = {}
        for row in range(self.state_shape[0]):
            for col in range(self.state_shape[1]):
                state = (row, col)
                policy[state] = np.argmax(self.Q[row, col])
        
        return policy

