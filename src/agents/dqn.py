import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
from collections import deque
from envs.gridworld import GridWorldEnv
from config import DQN_AGENT_CONFIG

# Set up logger
agent_logger = logging.getLogger('gridworld_rl.agent')

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dueling DQN architecture
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )
    def forward(self, x):
        x = self.feature(x)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


def unwrap_state(state):
    """Unwraps nested tuple states to get the actual state array."""
    while isinstance(state, tuple):
        state = state[0]
    return state


class DQNAgent:
    """
    Deep Q-Network (DQN) agent for reinforcement learning with neural networks.
    
    Implements Double DQN with Dueling architecture for improved stability and performance.
    """
    def __init__(self, env, config):
        """
        Initialize the DQN agent.
        
        Args:
            env: The environment to interact with
            config: Configuration dictionary containing hyperparameters
        """
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for reproducibility
        seed = config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # Get action space size
        self.action_size = env.action_space.n
        
        # Initialize state size
        state = env.reset()
        state = unwrap_state(state)
        self.state_size = np.array(state).size
        
        # Set hyperparameters from config
        self.buffer_size = config.get("buffer_size", 10000)
        self.batch_size = config.get("batch_size", 64)
        self.sync_frequency = config.get("sync_frequency", 5)
        self.gamma = config.get("gamma", 0.99)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.epsilon_min = config.get("epsilon_min", 0.05)
        self.reward_step_penalty = config.get("reward_step_penalty", -1.0)
        self.max_steps = config.get("max_steps", 200)
        
        # Initialize models
        self.online_model, self.target_model = self._build_models()
        self.online_model.train()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=self.buffer_size)





    def _build_models(self):
        """Build online and target networks."""
        hidden_dim = self.config.get("hidden_dim", 128)
        online = DuelingDQN(self.state_size, self.action_size, hidden_dim).to(self.device)
        target = DuelingDQN(self.state_size, self.action_size, hidden_dim).to(self.device)
        target.load_state_dict(online.state_dict())
        target.eval()
        return online, target
    
    def act(self, state):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            int: Selected action
        """
        state = unwrap_state(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.online_model(state_tensor)
                return q_values.argmax().item()
    
    def _sample_minibatch(self):
        """Sample a minibatch from replay buffer."""
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        return states, actions, rewards, next_states, dones
    
    def train(self, num_episodes=None):
        """
        Train the agent for the specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to train for. Defaults to config value.
        """
        if num_episodes is None:
            num_episodes = self.config.get("num_episodes", 500)
            
        agent_logger.info(f"Starting DQN training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            state = unwrap_state(state)
            total_reward = 0
            
            for t in range(self.max_steps):
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = unwrap_state(next_state)
                
                # Reward shaping: add step penalty
                shaped_reward = reward + self.reward_step_penalty
                
                # Store transition in replay buffer
                self.replay_buffer.append((state, action, shaped_reward, next_state, float(done)))
                
                # Update state and total reward
                state = next_state
                total_reward += shaped_reward
                
                # Train on minibatch if buffer is large enough
                if len(self.replay_buffer) >= self.batch_size:
                    self._update_network()
                
                if done:
                    break
            
            # Sync target network periodically
            if episode % self.sync_frequency == 0:
                self.target_model.load_state_dict(self.online_model.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Log progress
            if episode % 10 == 0:
                agent_logger.info(f"DQN Episode {episode}/{num_episodes} | "
                                 f"Total Reward: {total_reward:.2f} | Epsilon: {self.epsilon:.3f}")
                
        agent_logger.info("DQN Training completed successfully")
    
    def _update_network(self):
        """Update the network parameters using a sampled minibatch."""
        states, actions, rewards, next_states, dones = self._sample_minibatch()
        
        # Double DQN target
        with torch.no_grad():
            next_actions = self.online_model(next_states).argmax(1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).view(-1)
            target = rewards + (1 - dones) * self.gamma * next_q
            
        # Compute Q-values and loss
        q_values = self.online_model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).view(-1)
        loss = nn.SmoothL1Loss()(q_value, target)
        
        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), 
                                       self.config.get("max_grad_norm", 1.0))
        self.optimizer.step()
    
    def get_policy(self):
        """Return the current greedy policy."""
        policy = {}
        for i in range(self.env.grid_height):
            for j in range(self.env.grid_width):
                state = (i, j)
                # Convert state to tensor format
                state_np = unwrap_state(state)
                state_tensor = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    q_values = self.online_model(state_tensor)
                    policy[state] = q_values.argmax().item()
        
        return policy

# For backward compatibility with main.py
DQN = DuelingDQN

__all__ = ["DQNAgent", "DQN"]