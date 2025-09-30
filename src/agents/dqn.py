import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
import time
import multiprocessing
from collections import deque
from functools import partial
# --- Import your GridWorldEnv and config ---
# from envs.gridworld import GridWorldEnv
# from config import DQN_AGENT_CONFIG

# Set up logger
agent_logger = logging.getLogger('gridworld_rl.agent')

# --- Set device for computation ---
# Force CPU as requested for optimization
device = torch.device("cpu")
torch.set_num_threads(multiprocessing.cpu_count())
torch.set_num_interop_threads(multiprocessing.cpu_count())

# --- Dueling DQN architecture ---
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.feature_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.advantage_stream = nn.Linear(hidden_dim, output_dim)
        self.value_stream = nn.Linear(hidden_dim, 1)
        self._initialize_weights() # Initialize weights upon creation

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.activation(self.feature_layer(x))
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

def unwrap_state(state):
    """Unwraps nested tuple states to get the actual state array."""
    while isinstance(state, tuple):
        state = state[0]
    # Ensure output is a numpy array for consistent handling
    if not isinstance(state, np.ndarray):
        return np.array(state)
    return state

# --- Prioritized Replay Buffer ---
# Simplified version focusing on core functionality
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.full = False

    def push(self, state, action, reward, next_state, done):
        # Ensure states are numpy arrays for consistent storage
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, float(done))
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size, beta=0.4):
        if self.full:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]

        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples)) # Unzips list of tuples
        states = np.array(batch[0], dtype=np.float32)
        actions = np.array(batch[1], dtype=np.int64)
        rewards = np.array(batch[2], dtype=np.float32)
        next_states = np.array(batch[3], dtype=np.float32)
        dones = np.array(batch[4], dtype=np.float32)

        return (states, actions, rewards, next_states, dones), indices, weights

    def update_priorities(self, batch_indices, errors):
        for idx, error in zip(batch_indices, errors):
            self.priorities[idx] = error + 1e-5 # Small constant to avoid zero priority

    def __len__(self):
        return len(self.buffer)


# --- Main DQNAgent Class ---
class DQNAgent:
    """
    Deep Q-Network (DQN) agent for reinforcement learning with neural networks.
    Implements Double DQN with Dueling architecture and prioritized experience replay.
    """
    def __init__(self, env, config):
        self.env = env
        self.config = config
        # --- Fix device mismatch ---
        self.device = device # Use the globally defined CPU device

        # --- Teacher Policy ---
        self.teacher_policy = None
        self.teacher_loaded = False
        self.teacher_success_threshold = config.get("teacher_success_threshold", 0.8)
        self.teacher_use_episodes = config.get("teacher_use_episodes", 100) # Use teacher for first N episodes

        # --- Seeds ---
        seed = config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # --- Environment Info ---
        self.action_size = env.action_space.n
        state = env.reset()
        state = unwrap_state(state)
        self.state_size = state.size # Use .size for flat array

        # --- Hyperparameters ---
        self.buffer_size = config.get("buffer_size", 10000)
        self.batch_size = config.get("batch_size", 64)
        # Sync every N steps, not episodes, for more consistent updates
        self.sync_target_steps = config.get("sync_target_steps", 1000)
        # Add sync_frequency as an alias for sync_target_steps for compatibility
        self.sync_frequency = config.get("sync_frequency", 10)  # Default to every 10 episodes
        self.gamma = config.get("gamma", 0.99)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.reward_step_penalty = config.get("reward_step_penalty", 0.0) # Default to no penalty
        self.max_steps = config.get("max_steps", 200)

        # --- N-step Learning ---
        self.n_steps = config.get("n_steps", 1) # Default to 1-step (standard DQN)
        self.n_step_buffer = deque(maxlen=self.n_steps)
        self.n_step_gamma = self.gamma ** self.n_steps

        # --- Model Setup ---
        self.hidden_dim = config.get("hidden_dim", 128)
        self.online_model = DuelingDQN(self.state_size, self.action_size, self.hidden_dim).to(self.device)
        self.target_model = DuelingDQN(self.state_size, self.action_size, self.hidden_dim).to(self.device)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()

        # --- Optimizer & Scheduler ---
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.learning_rate)
        # Simplified scheduler or remove if not needed initially
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(...)

        # --- Replay Buffer ---
        self.alpha = config.get("priority_alpha", 0.6)
        self.beta = config.get("priority_beta", 0.4)
        self.beta_increment = config.get("beta_increment", 0.001)
        self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, self.alpha)

        # --- Tracking ---
        self.episode_count = 0
        self.total_steps = 0 # Track total steps for target sync

    def _build_models(self):
        # Not used anymore, models built in __init__
        pass

    def load_teacher_policy(self):
        """Load a teacher policy from a successful Q-learning agent."""
        import os
        import glob
        import pickle # Assuming Q-table was saved with pickle

        # Adjust path as needed
        checkpoint_dir = os.path.join("logs", "checkpoints", "q_learning")
        if not os.path.exists(checkpoint_dir):
            agent_logger.warning("No Q-learning checkpoints found for policy distillation")
            return

        # Find the most recent .pkl file (assuming Q-table saved this way)
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pkl")) # Or *.pt if saved with torch.save
        if not checkpoints:
             # Fallback to .pt if .pkl not found
             checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
             if not checkpoints:
                 agent_logger.warning("No Q-learning checkpoints (.pkl or .pt) found for policy distillation")
                 return

        checkpoints.sort(key=os.path.getmtime, reverse=True) # Sort by modification time

        try:
            # Try loading with pickle first
            if checkpoints[0].endswith('.pkl'):
                with open(checkpoints[0], 'rb') as f:
                    q_table = pickle.load(f)
            else: # Assume it's a torch file containing the q_table
                 # Load the checkpoint with torch.load
                 checkpoint_data = torch.load(checkpoints[0], weights_only=False) # Adjust weights_only if needed
                 # Extract the policy if available - adjust key if different
                 if "q_table" in checkpoint_data:
                     q_table = checkpoint_data["q_table"]
                 elif "Q" in checkpoint_data: # Common alternative key
                     q_table = checkpoint_data["Q"]
                 else:
                     raise KeyError("Checkpoint does not contain 'q_table' or 'Q' key")

            # Convert Q-table to policy - Ensure state keys match GridWorldEnv states
            policy = {}
            for state_tuple, actions in q_table.items():
                # Assuming state_tuple from Q-table is the (row, col) tuple directly usable
                # This is critical: state_tuple must match the key format used by unwrap_state and the env
                if isinstance(actions, (list, np.ndarray)):
                    policy[state_tuple] = np.argmax(actions)
                elif isinstance(actions, dict):
                    policy[state_tuple] = max(actions.items(), key=lambda x: x[1])[0]
                else:
                    agent_logger.warning(f"Unknown Q-table action format for state {state_tuple}")
                    continue

            self.teacher_policy = policy
            self.teacher_loaded = True
            agent_logger.info(f"Loaded teacher policy from {checkpoints[0]} with {len(policy)} states")
        except Exception as e:
            agent_logger.warning(f"Failed to load teacher policy from {checkpoints[0]}: {e}")


    def act(self, state, use_teacher=True):
        """
        Select an action using epsilon-greedy policy.
        Args:
            state: Current state (raw from env)
            use_teacher: Whether to attempt using teacher policy (e.g., disable during eval)
        Returns:
            int: Selected action
        """
        state = unwrap_state(state)

        # --- Teacher Distillation (Simplified) ---
        if (use_teacher and self.teacher_policy and self.episode_count < self.teacher_use_episodes and
            hasattr(self.env, 'agent_pos')): # Check if env exposes agent position
            try:
                # --- Critical: Ensure state_key format matches teacher_policy keys ---
                # Example: If teacher uses (row, col) and env.agent_pos gives (row, col)
                state_key = tuple(self.env.agent_pos) # Use env's exposed position
                if state_key in self.teacher_policy:
                     # Simple threshold or episode-based decay
                     teacher_threshold = max(0.5, 1.0 - (self.episode_count / self.teacher_use_episodes))
                     if np.random.rand() < teacher_threshold:
                         return self.teacher_policy[state_key]
            except (TypeError, ValueError, AttributeError) as e:
                # Safely handle conversion errors or missing attributes
                agent_logger.debug(f"Teacher distillation error: {e}")

        # --- Epsilon-Greedy ---
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)

        # --- Exploitation ---
        self.online_model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_model(state_tensor)
            action = q_values.argmax().item()
        self.online_model.train() # Switch back to train mode
        return action

    def remember(self, state, action, reward, next_state, done):
        """Store a single-step transition."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def _compute_n_step_return(self, reward, next_state, done):
        """Computes the n-step return and final next state."""
        self.n_step_buffer.append((reward, next_state, done))
        if len(self.n_step_buffer) < self.n_steps:
            return None # Not enough steps yet

        # Calculate n-step return G_t:t+n
        R = 0
        for i, (r, _, d) in enumerate(self.n_step_buffer):
            R += (self.gamma ** i) * r
            if d: # If any step in the n-step sequence is terminal
                # Truncate the buffer as subsequent rewards don't matter
                while len(self.n_step_buffer) > i + 1:
                     self.n_step_buffer.popleft()
                break

        # The state and action are from n steps ago (the start of the sequence)
        # The next_state and done are from the most recent step
        n_step_reward = R
        final_next_state = self.n_step_buffer[-1][1]
        final_done = self.n_step_buffer[-1][2]

        # Remove the oldest element to make space for the next one
        self.n_step_buffer.popleft()

        return n_step_reward, final_next_state, final_done

    def train(self, num_episodes=None):
        if num_episodes is None:
            num_episodes = self.config.get("num_episodes", 500)

        agent_logger.info(f"Starting DQN training for {num_episodes} episodes")
        rewards_history = []

        for episode in range(1, num_episodes + 1):
            self.episode_count = episode
            state = self.env.reset()
            state = unwrap_state(state)
            total_reward = 0
            self.n_step_buffer.clear() # Clear buffer at start of each episode

            for step in range(self.max_steps):
                self.total_steps += 1
                action = self.act(state)

                next_state, reward, done, _ = self.env.step(action)
                next_state = unwrap_state(next_state)

                # --- Reward Shaping ---
                shaped_reward = reward
                if self.reward_step_penalty != 0:
                    shaped_reward += self.reward_step_penalty

                # --- N-step Handling ---
                if self.n_steps > 1:
                    n_step_info = self._compute_n_step_return(shaped_reward, next_state, done)
                    if n_step_info is not None:
                        n_step_reward, n_step_next_state, n_step_done = n_step_info
                        # Store the n-step transition
                        self.remember(state, action, n_step_reward, n_step_next_state, n_step_done)
                else:
                    # Standard 1-step DQN
                    self.remember(state, action, shaped_reward, next_state, done)

                state = next_state
                total_reward += shaped_reward

                # --- Training Step ---
                if len(self.replay_buffer) >= self.batch_size:
                    self._update_network()

                # --- Target Network Sync ---
                if self.total_steps % self.sync_target_steps == 0:
                    self.target_model.load_state_dict(self.online_model.state_dict())
                    agent_logger.debug(f"Target network synced at step {self.total_steps}")

                if done:
                    break

            # --- End of Episode ---
            rewards_history.append(total_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            # Anneal beta for prioritized replay
            self.beta = min(1.0, self.beta + self.beta_increment)

            # --- Logging ---
            if episode % 100 == 0 or episode == 1:
                avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
                success_rate = np.mean([1 if r > 0 else 0 for r in rewards_history[-100:]]) if len(rewards_history) >= 100 else np.mean([1 if r > 0 else 0 for r in rewards_history])
                agent_logger.info(f"Episode {episode}/{num_episodes} | "
                                 f"Avg Reward (100ep): {avg_reward:.2f} | "
                                 f"Success Rate (100ep): {success_rate:.2f} | "
                                 f"Epsilon: {self.epsilon:.3f}")

        agent_logger.info("DQN Training completed.")

    def _update_network(self):
        """Update the network using Double DQN with prioritized replay."""
        self.online_model.train()
        self.target_model.eval()

        (states, actions, rewards, next_states, dones), indices, weights = self.replay_buffer.sample(self.batch_size, self.beta)
        weights_tensor = torch.FloatTensor(weights).to(self.device)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.BoolTensor(dones.astype(bool)).to(self.device) # Use BoolTensor

        with torch.no_grad():
            # --- Double DQN ---
            next_actions = self.online_model(next_states_tensor).argmax(1)
            next_q_values = self.target_model(next_states_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # --- N-step Target (already incorporated via stored reward/discount) ---
            target_q_values = rewards_tensor + ( (~dones_tensor).float() * self.n_step_gamma * next_q_values )

        current_q_values = self.online_model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # --- Loss with IS Weights ---
        elementwise_loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
        loss = (weights_tensor * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), self.config.get("max_grad_norm", 1.0))
        self.optimizer.step()

        # --- Update Priorities ---
        with torch.no_grad():
            td_errors = torch.abs(current_q_values - target_q_values).cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        self.online_model.eval() # Switch back to eval mode after training step

    def _evaluate(self, num_episodes=10):
        """Evaluate the agent's performance."""
        self.online_model.eval()
        curr_epsilon = self.epsilon
        self.epsilon = 0.0 # No exploration during evaluation

        rewards = []
        successes = []
        steps_list = []

        for _ in range(num_episodes):
            state = self.env.reset()
            state = unwrap_state(state)
            total_reward = 0
            steps = 0
            for _ in range(self.max_steps):
                steps += 1
                action = self.act(state, use_teacher=False) # Disable teacher during eval
                next_state, reward, done, _ = self.env.step(action)
                next_state = unwrap_state(next_state)
                total_reward += reward
                state = next_state
                if done:
                    break
            rewards.append(total_reward)
            successes.append(1 if total_reward > 0 else 0) # Assuming positive reward is success
            steps_list.append(steps)

        self.epsilon = curr_epsilon
        self.online_model.train()
        avg_reward = np.mean(rewards)
        success_rate = np.mean(successes)
        avg_steps = np.mean(steps_list)
        agent_logger.info(f"Evaluation over {num_episodes} episodes: "
                         f"Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}, Avg Steps: {avg_steps:.2f}")
        return avg_reward

    def get_policy(self):
        """Return the current greedy policy."""
        policy = {}
        self.online_model.eval()
        with torch.no_grad():
            # Iterate through possible states (assuming GridWorldEnv exposes dimensions)
            try:
                height, width = self.env.grid_height, self.env.grid_width
                for i in range(height):
                    for j in range(width):
                        state_key = (i, j)
                        # Create a state representation compatible with your env's reset/step
                        # This might need adjustment based on how your GridWorldEnv works
                        # Example: env.reset() might accept an initial position
                        # For now, assume we can query the Q-values for a given (i,j) state
                        # This requires knowing how the state vector maps to (i,j)
                        # A more robust way is to iterate through the actual state space
                        # if your env supports it, or query specific known states.

                        # Simplified approach: Assume state vector is [i, j, ...] or similar
                        # This is highly dependent on your GridWorldEnv implementation.
                        # You might need to create a state representation like:
                        # state_vector = np.array([i, j, ...]) # Fill based on env's state structure
                        # Or query the env in a specific way.

                        # Placeholder logic - needs to be adapted:
                        # Let's assume the first two elements of the state vector are row/col
                        # and the rest are fixed or don't matter for action selection in static grids.
                        # You need to create a valid state vector for (i,j).
                        # Example (if env state is just [row, col]):
                        test_state = np.array([float(i), float(j)], dtype=np.float32)
                        # --- Critical: Ensure test_state format matches what your env/model expects ---
                        # If your env state includes more info (e.g., agent state, goal state),
                        # you need to construct that correctly here.
                        # This is a common source of bugs.

                        state_tensor = torch.FloatTensor(test_state).unsqueeze(0).to(self.device)
                        q_values = self.online_model(state_tensor)
                        action = q_values.argmax().item()
                        policy[state_key] = int(action)
            except (AttributeError, NotImplementedError):
                 agent_logger.warning("Could not automatically generate policy. Env does not expose grid dimensions or state construction is complex.")
                 # Fallback or alternative method needed if grid dimensions aren't easily accessible
        self.online_model.train()
        return policy

# For backward compatibility
DQN = DuelingDQN
__all__ = ["DQNAgent", "DQN"]

# --- Example Usage (assuming imports are fixed) ---
# if __name__ == "__main__":
#     # Load your config
#     config = DQN_AGENT_CONFIG # Make sure this is imported/defined
#     # Create your GridWorldEnv instance
#     env = GridWorldEnv(...) # Initialize with appropriate parameters
#     # Create agent
#     agent = DQNAgent(env, config)
#     # Train
#     agent.train(num_episodes=500)
#     # Evaluate
#     final_score = agent._evaluate(num_episodes=20)
#     # Get final policy
#     final_policy = agent.get_policy()
#     print("Final Policy Sample:", dict(list(final_policy.items())[:5])) # Print first 5 entries
