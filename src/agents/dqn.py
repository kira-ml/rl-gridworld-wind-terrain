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
from envs.gridworld import GridWorldEnv
from config import DQN_AGENT_CONFIG

# Set up logger
agent_logger = logging.getLogger('gridworld_rl.agent')

# Set device for computation with CPU optimization settings
device = torch.device("cpu")  # Force CPU for optimized implementation
torch.set_num_threads(multiprocessing.cpu_count())  # Use all available CPU cores
torch.set_num_interop_threads(multiprocessing.cpu_count())  # Optimize parallel execution

# Dueling DQN architecture
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):  # Smaller networks for CPU
        super().__init__()
        
        # CPU-optimized feature extraction network
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),  # Lower momentum for stability
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU()
            # Removed third layer and dropout for efficiency
        )
        
        # Simplified value stream for CPU
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
            # Removed extra layer
        )
        
        # Simplified advantage stream for CPU
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
            # Removed extra layer
        )
        
    def forward(self, x):
        # Use contiguous memory layout for better CPU performance
        x = self.feature(x.contiguous())
        value = self.value(x)
        advantage = self.advantage(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
        
    def _initialize_weights(self):
        # CPU-efficient weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Fast initialization for CPU training
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


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
        self.sync_frequency = config.get("sync_frequency", 1)  # More frequent target updates (was 5)
        self.gamma = config.get("gamma", 0.99)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_decay = config.get("epsilon_decay", 0.998)  # Slower decay (was 0.995)
        self.epsilon_min = config.get("epsilon_min", 0.1)  # Higher minimum exploration (was 0.05)
        self.reward_step_penalty = config.get("reward_step_penalty", -1.0)
        self.max_steps = config.get("max_steps", 200)
        
        # Initialize models with CPU optimization
        self.online_model, self.target_model = self._build_models()
        self.online_model.train()
        
        # Apply weight initialization for better CPU performance
        self.online_model._initialize_weights()
        self.target_model._initialize_weights()
        
        # Use lightweight optimizer for CPU efficiency (RMSprop uses less memory than Adam)
        self.optimizer = optim.RMSprop(
            self.online_model.parameters(), 
            lr=self.learning_rate,
            alpha=0.95,  # Higher smoothing for stability
            eps=1e-5     # Numerical stability
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10,  # Reduced patience
            threshold=0.01, threshold_mode='rel', cooldown=5, min_lr=1e-6
        )
        
        # Performance tracking for adaptive batch sizing
        self.batch_update_times = []
        self.last_performance_check = time.time()
        
        # CPU-optimized fixed-size numpy arrays for replay buffer
        self.buffer_idx = 0
        self.buffer_full = False
        
        # For backward compatibility with main.py and other modules
        # We keep both the optimized arrays and traditional deque to maintain compatibility
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.buffer = self.replay_buffer  # Alias for the new implementation
        
        # Pre-allocate memory for buffer arrays (more efficient than deques on CPU)
        # Using float32 instead of float64 to reduce memory usage by half
        state_shape = (self.buffer_size, self.state_size)
        self.state_buffer = np.zeros(state_shape, dtype=np.float32)
        self.action_buffer = np.zeros(self.buffer_size, dtype=np.int32)
        self.reward_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.next_state_buffer = np.zeros(state_shape, dtype=np.float32)
        self.done_buffer = np.zeros(self.buffer_size, dtype=np.bool_)  # Use boolean for done flags
        
        # Prioritization support with float32
        self.priorities = np.ones(self.buffer_size, dtype=np.float32) * 1e-6
        self.alpha = config.get("priority_alpha", 0.6)  # Priority exponent
        self.beta = config.get("priority_beta", 0.4)   # Importance sampling weight
        self.beta_increment = config.get("beta_increment", 0.001)  # Annealing parameter
        
        # N-step learning with preallocated arrays
        self.n_steps = min(config.get("n_steps", 3), 5)  # Cap for memory efficiency
        self.n_step_states = np.zeros((self.n_steps, self.state_size), dtype=np.float32)
        self.n_step_actions = np.zeros(self.n_steps, dtype=np.int32)
        self.n_step_rewards = np.zeros(self.n_steps, dtype=np.float32)
        self.n_step_dones = np.zeros(self.n_steps, dtype=np.bool_)
        self.n_step_count = 0
        
        # Adaptive batch sizing based on CPU performance
        self.dynamic_batch_size = self.batch_size
        self.target_update_time = 0.05  # 50ms target for batch processing





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
        Select an action using epsilon-greedy policy with CPU optimizations.
        
        Args:
            state: Current state
            
        Returns:
            int: Selected action
        """
        state = unwrap_state(state)
        
        # Fast path for exploration (avoid unnecessary computation)
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        
        # CPU optimization: reuse tensor if possible
        if not hasattr(self, '_cached_state_tensor') or self._cached_state_tensor.shape[0] != 1:
            self._cached_state_tensor = torch.zeros((1, self.state_size), 
                                                 dtype=torch.float32, 
                                                 device=self.device)
        
        # Copy state data directly
        with torch.no_grad():
            self._cached_state_tensor[0] = torch.tensor(state, dtype=torch.float32)
            q_values = self.online_model(self._cached_state_tensor)
            return q_values.argmax().item()
    
    def batch_act(self, states, deterministic=False):
        """
        CPU-optimized batch action selection for multiple states.
        
        Args:
            states: Batch of states
            deterministic: Whether to use deterministic policy (no exploration)
            
        Returns:
            numpy.ndarray: Selected actions
        """
        batch_size = len(states)
        
        # Fast exploration for non-deterministic actions
        if not deterministic:
            # Vectorized random sampling
            random_actions = np.random.randint(0, self.action_size, size=batch_size)
            # Create exploration mask
            explore = np.random.random(batch_size) < self.epsilon
            
            if explore.all():
                return random_actions
                
            # If all actions are from policy, skip the random part
            if not explore.any():
                deterministic = True
        
        # Process states in a single batch for CPU efficiency
        unwrapped_states = np.array([unwrap_state(s) for s in states], dtype=np.float32)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(unwrapped_states).to(self.device)
            q_values = self.online_model(state_tensor)
            policy_actions = q_values.argmax(dim=1).cpu().numpy()
        
        # If deterministic, return policy actions directly
        if deterministic:
            return policy_actions
            
        # Otherwise, combine random and policy actions
        actions = np.where(explore, random_actions, policy_actions)
        return actions
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer with memory-efficient arrays."""
        # Store in the buffer at current index
        state = unwrap_state(state)
        next_state = unwrap_state(next_state)
        
        # Update the deque buffer for backward compatibility with main.py
        self.replay_buffer.append((state, action, reward, next_state, float(done)))
        
        # If optimized arrays are available, update them too
        if hasattr(self, 'state_buffer') and hasattr(self, 'buffer_idx'):
            self.state_buffer[self.buffer_idx] = state
            self.action_buffer[self.buffer_idx] = action
            self.reward_buffer[self.buffer_idx] = reward
            self.next_state_buffer[self.buffer_idx] = next_state
            self.done_buffer[self.buffer_idx] = done
            
            # Update priority for new transition
            self.priorities[self.buffer_idx] = max(self.priorities.max(), 1e-6)
            
            # Update buffer index and full flag
            self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
            if not self.buffer_full and self.buffer_idx == 0:
                self.buffer_full = True
        
        return len(self.replay_buffer)
    
    def _sample_minibatch(self):
        """Sample a minibatch with prioritization optimized for CPU performance."""
        # Check if using the optimized array-based buffer or traditional deque
        if hasattr(self, 'buffer_full') and hasattr(self, 'buffer_idx'):
            # Optimized array-based implementation
            size = self.buffer_size if self.buffer_full else self.buffer_idx
        else:
            # Backward compatible approach with deque
            size = len(self.replay_buffer)
        
        if size == 0:
            return None
            
        # Calculate sampling probabilities - use vectorized operations
        if self.alpha == 0:
            # Uniform sampling (avoid unnecessary computation)
            indices = np.random.choice(size, min(self.batch_size, size), replace=False)
            weights = np.ones(len(indices), dtype=np.float32)
        else:
            # Priority sampling
            priorities = self.priorities[:size]
            probs = priorities ** self.alpha
            probs /= probs.sum()
            
            # Sample transitions with vectorized numpy operations
            # Use batch_size instead of dynamic_batch_size for compatibility
            indices = np.random.choice(size, min(self.batch_size, size), p=probs, replace=False)
            
            # Calculate importance sampling weights
            weights = (size * probs[indices]) ** (-self.beta)
            weights /= weights.max()  # Normalize weights
        
        # For compatibility with both implementations:
        # 1. Check if we're using optimized arrays or traditional deque
        if hasattr(self, 'buffer_full') and hasattr(self, 'state_buffer') and size > 0:
            # Get batch data using efficient slicing operations
            states = self.state_buffer[indices]
            actions = self.action_buffer[indices]
            rewards = self.reward_buffer[indices]
            next_states = self.next_state_buffer[indices]
            dones = self.done_buffer[indices]
        else:
            # Fallback to traditional deque approach for compatibility
            minibatch = [self.replay_buffer[idx] for idx in indices]
            states = np.array([transition[0] for transition in minibatch])
            actions = np.array([transition[1] for transition in minibatch])
            rewards = np.array([transition[2] for transition in minibatch])
            next_states = np.array([transition[3] for transition in minibatch])
            dones = np.array([transition[4] for transition in minibatch])
        
        # Convert to tensors with efficient memory layout and dtype
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones.astype(np.float32)).to(device)
        weights = torch.FloatTensor(weights).to(device)

        return states, actions, rewards, next_states, dones, indices, weights
    
    def train(self, num_episodes=None):
        """
        Train the agent for the specified number of episodes with CPU optimizations.
        
        Args:
            num_episodes: Number of episodes to train for. Defaults to config value.
        """
        if num_episodes is None:
            num_episodes = self.config.get("num_episodes", 500)
            
        agent_logger.info(f"Starting CPU-optimized DQN training for {num_episodes} episodes")
        agent_logger.info(f"Using {multiprocessing.cpu_count()} CPU cores with batch size {self.dynamic_batch_size}")
        
        # Pre-allocate memory for tracking variables
        rewards_history = np.zeros(num_episodes, dtype=np.float32)
        steps_history = np.zeros(num_episodes, dtype=np.int32)
        
        # Measure training time
        training_start = time.time()
        last_log_time = training_start
        
        for episode in range(num_episodes):
            episode_start = time.time()
            state = self.env.reset()
            state = unwrap_state(state)
            total_reward = 0
            
            for t in range(self.max_steps):
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = unwrap_state(next_state)
                
                # CPU-efficient reward shaping with minimal memory usage
                # Vectorized operations for distance calculation
                goal_pos = np.array(self.env.config["goal_pos"], dtype=np.float32)
                curr_pos = np.array(state, dtype=np.float32)
                next_pos = np.array(next_state, dtype=np.float32)
                
                # Fast squared distance calculation (avoid sqrt for speed)
                curr_dist_sq = np.sum((curr_pos - goal_pos)**2)
                next_dist_sq = np.sum((next_pos - goal_pos)**2)
                
                # Progressive distance reward (use sqrt only once)
                next_dist = np.sqrt(next_dist_sq)
                dist_factor = 1.0 / (1.0 + next_dist)
                
                # Simpler reward shaping for CPU efficiency
                dist_improvement = curr_dist_sq - next_dist_sq
                distance_reward = 0.1 * np.sign(dist_improvement) * dist_factor
                shaped_reward = reward + self.reward_step_penalty + distance_reward
                
                # Store transition directly in preallocated buffer
                idx = self.store_transition(state, action, shaped_reward, next_state, done)
                
                # Efficient n-step learning with array operations
                # Store in circular n-step buffer
                n_idx = self.n_step_count % self.n_steps
                self.n_step_states[n_idx] = np.array(state, dtype=np.float32)
                self.n_step_actions[n_idx] = action
                self.n_step_rewards[n_idx] = shaped_reward
                self.n_step_dones[n_idx] = done
                self.n_step_count += 1
                
                # If n-step buffer is filled, process n-step return
                if self.n_step_count >= self.n_steps:
                    # Get first state and action (n steps ago)
                    n_start_idx = (n_idx + 1) % self.n_steps
                    first_state = self.n_step_states[n_start_idx]
                    first_action = self.n_step_actions[n_start_idx]
                    
                    # Vectorized n-step reward calculation
                    indices = np.arange(self.n_steps)
                    discount_factors = self.gamma ** indices
                    rewards_array = np.roll(self.n_step_rewards, -n_start_idx)[:self.n_steps]
                    n_step_reward = np.sum(discount_factors * rewards_array)
                    
                    # Store the n-step transition
                    self.store_transition(
                        first_state, 
                        first_action, 
                        n_step_reward, 
                        next_state, 
                        done
                    )
                
                # If episode terminates before n steps, flush buffer with adjusted rewards
                if done and len(self.n_step_buffer) < self.n_steps:
                    first_state = self.n_step_buffer[0][0]
                    first_action = self.n_step_buffer[0][1]
                    
                    # Calculate n-step reward for partial sequence
                    n_step_reward = 0
                    for i in range(len(self.n_step_buffer)):
                        n_step_reward += (self.gamma ** i) * self.n_step_buffer[i][2]
                    
                    # Store transition with final state and done flag
                    self.buffer.append((first_state, first_action, n_step_reward, next_state, float(done)))
                    
                    # Clear n-step buffer
                    self.n_step_buffer.clear()
                
                # Update state and total reward
                state = next_state
                total_reward += shaped_reward
                
                # Train on minibatch if buffer is large enough
                # Check replay_buffer length for backward compatibility with main.py
                if len(self.replay_buffer) >= self.batch_size:
                    self._update_network()
                    
                    # Anneal beta parameter for importance sampling
                    self.beta = min(1.0, self.beta + self.beta_increment)
                
                if done:
                    break
            
            # Track episode stats efficiently
            rewards_history[episode] = total_reward
            steps_history[episode] = t + 1
            
            # Sync target network with delayed copy for CPU efficiency
            if episode % self.sync_frequency == 0:
                # Use state_dict() with CPU optimization
                with torch.no_grad():  # Prevent tracking unnecessary gradients
                    for target_param, online_param in zip(
                            self.target_model.parameters(), 
                            self.online_model.parameters()):
                        target_param.data.copy_(online_param.data)
            
            # Decay epsilon with vectorized operations
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Update learning rate scheduler every 20 episodes
            if episode % 20 == 0 and episode > 0:
                # Evaluate only if enough time has passed (CPU efficiency)
                current_time = time.time()
                if current_time - self.last_performance_check > 60:  # 1 minute
                    eval_reward = self._evaluate(3)  # Reduced evaluations for CPU
                    self.scheduler.step(eval_reward)
                    self.last_performance_check = current_time
                    
                    # Log training stats with useful CPU metrics
                    episode_time = (current_time - training_start) / (episode + 1)
                    eta_minutes = (num_episodes - episode) * episode_time / 60
                    
                    # Recent performance metrics (last 100 episodes)
                    recent_idx = max(0, episode-100)
                    recent_rewards = rewards_history[recent_idx:episode]
                    recent_steps = steps_history[recent_idx:episode]
                    
                    agent_logger.info(
                        f"DQN Episode {episode}/{num_episodes} | "
                        f"Reward: {total_reward:.1f} | "
                        f"Recent Avg: {np.mean(recent_rewards):.1f} | "
                        f"Steps: {np.mean(recent_steps):.1f} | "
                        f"Batch: {self.dynamic_batch_size} | "
                        f"ETA: {eta_minutes:.1f}min"
                    )
            
            # More efficient progress tracking
            if episode % 100 == 99:
                # Save CPU resources by doing less frequent full logs
                curr_lr = self.optimizer.param_groups[0]['lr']
                buffer_size = self.buffer_size if self.buffer_full else self.buffer_idx
                agent_logger.info(
                    f"[CPU Stats] Episode {episode+1}/{num_episodes} | "
                    f"Epsilon: {self.epsilon:.3f} | "
                    f"LR: {curr_lr:.6f} | "
                    f"Buffer: {buffer_size}/{self.buffer_size} | "
                    f"Time/ep: {(time.time() - training_start)/(episode+1):.3f}s"
                )
                
        # Final statistics
        training_time = (time.time() - training_start) / 60
        agent_logger.info(
            f"DQN Training completed successfully in {training_time:.2f} minutes | "
            f"Final Avg(100) Reward: {np.mean(rewards_history[-100:]):.2f}"
        )
    
    def _update_network(self):
        """Update the network parameters using a sampled minibatch with CPU optimizations."""
        start_time = time.time()
        
        # Get a minibatch
        result = self._sample_minibatch()
        if result is None:
            return  # Not enough samples yet
        
        states, actions, rewards, next_states, dones, indices, weights = result
        
        # CPU optimization: Use smaller precision where possible
        states = states.to(torch.float32)
        next_states = next_states.to(torch.float32)
        
        # Double DQN target with n-step returns adjustment and batch processing
        with torch.no_grad():
            # Split large batches into smaller chunks to avoid memory spikes on CPU
            chunk_size = 64  # Process in chunks of 64 for better CPU cache utilization
            num_samples = states.shape[0]
            next_q_list = []
            
            for i in range(0, num_samples, chunk_size):
                end = min(i + chunk_size, num_samples)
                chunk_states = next_states[i:end]
                
                # CPU optimization: Compute next actions with online model
                next_actions_chunk = self.online_model(chunk_states).argmax(1)
                
                # Use target network for value estimation
                next_q_chunk = self.target_model(chunk_states).gather(
                    1, next_actions_chunk.unsqueeze(1)).view(-1)
                next_q_list.append(next_q_chunk)
            
            # Combine results
            next_q = torch.cat(next_q_list)
            
            # For n-step returns, use gamma^n for the bootstrap
            gamma_n = self.gamma ** self.n_steps
            target = rewards + (1 - dones) * gamma_n * next_q
        
        # Compute Q-values with efficient chunking for CPU
        q_value_list = []
        for i in range(0, num_samples, chunk_size):
            end = min(i + chunk_size, num_samples)
            chunk_states = states[i:end]
            chunk_actions = actions[i:end]
            
            # Get Q-values for this chunk
            q_values_chunk = self.online_model(chunk_states)
            q_value_chunk = q_values_chunk.gather(1, chunk_actions.unsqueeze(1)).view(-1)
            q_value_list.append(q_value_chunk)
        
        # Combine results
        q_value = torch.cat(q_value_list)
        
        # Calculate TD errors for prioritization (detach for memory efficiency)
        with torch.no_grad():
            td_errors = torch.abs(q_value.detach() - target).cpu().numpy()
        
        # Update priorities in the buffer with vectorized operations
        self.priorities[indices] = (td_errors + 1e-6) ** self.alpha
        
        # Apply importance sampling weights to the loss with Huber loss
        # CPU optimization: use built-in smooth L1 loss with reduced memory usage
        elementwise_loss = nn.functional.smooth_l1_loss(q_value, target, reduction='none')
        loss = (elementwise_loss * weights).mean()
        
        # Optimize the network with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        # Use a lower clip value for CPU stability
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), 
                                      self.config.get("max_grad_norm", 0.5))
        self.optimizer.step()
        
        # Track update time for adaptive batch sizing
        update_time = time.time() - start_time
        self.batch_update_times.append(update_time)
        
        # Periodically adjust batch size based on CPU performance
        if len(self.batch_update_times) >= 10:
            avg_time = np.mean(self.batch_update_times)
            
            # If updates are too slow, decrease batch size
            if avg_time > self.target_update_time and self.dynamic_batch_size > self.min_batch_size:
                self.dynamic_batch_size = max(self.min_batch_size, self.dynamic_batch_size - 8)
            # If updates are fast, increase batch size
            elif avg_time < 0.8 * self.target_update_time and self.dynamic_batch_size < self.max_batch_size:
                self.dynamic_batch_size = min(self.max_batch_size, self.dynamic_batch_size + 8)
                
            # Reset timing buffer
            self.batch_update_times = []
        
        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), 
                                       self.config.get("max_grad_norm", 1.0))
        self.optimizer.step()
    
    def _evaluate(self, num_episodes=3):  # Reduced default for CPU efficiency
        """
        Evaluate the agent's performance over a number of episodes with CPU optimizations.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            float: Average reward across episodes
        """
        # Pre-allocate arrays for CPU efficiency
        rewards = np.zeros(num_episodes, dtype=np.float32)
        success = np.zeros(num_episodes, dtype=np.bool_)
        steps = np.zeros(num_episodes, dtype=np.int32)
        
        # Save current epsilon and set to evaluation mode
        curr_epsilon = self.epsilon
        self.epsilon = 0.05  # Small epsilon for minimal exploration during evaluation
        
        # Tell PyTorch this is evaluation mode to skip unnecessary computations
        self.online_model.eval()
        
        with torch.no_grad():  # Disable gradient computation during evaluation
            for ep in range(num_episodes):
                state = self.env.reset()
                state = unwrap_state(state)
                
                for t in range(self.max_steps):
                    # Get action with minimal computation
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action = self.online_model(state_tensor).argmax().item()
                    
                    # Take step
                    next_state, reward, done, info = self.env.step(action)
                    next_state = unwrap_state(next_state)
                    
                    # Update tracking variables
                    rewards[ep] += reward
                    state = next_state
                    
                    if done:
                        steps[ep] = t + 1
                        if reward > 0:  # Assuming positive reward means success
                            success[ep] = True
                        break
                    
                if not done:
                    steps[ep] = self.max_steps
        
        # Switch back to training mode
        self.online_model.train()
        self.epsilon = curr_epsilon
        
        # Calculate statistics efficiently with numpy
        avg_reward = np.mean(rewards)
        success_rate = np.mean(success)
        avg_steps = np.mean(steps)
        
        # Log less frequently to save CPU resources
        if random.random() < 0.2:  # Log only ~20% of evaluations
            agent_logger.info(
                f"Eval: Reward={avg_reward:.1f}, Success={success_rate:.2f}, "
                f"Steps={avg_steps:.1f}, LR={self.optimizer.param_groups[0]['lr']:.6f}"
            )
        
        return float(avg_reward)  # Convert numpy type to Python float
    
    def get_policy(self):
        """Return the current greedy policy with CPU-optimized batch processing."""
        policy = {}
        
        # Get grid dimensions
        height = self.env.grid_height
        width = self.env.grid_width
        
        # Create all states in one batch for vectorized processing
        states = []
        state_to_idx = {}
        idx = 0
        
        for i in range(height):
            for j in range(width):
                state = (i, j)
                state_np = unwrap_state(state)
                states.append(state_np)
                state_to_idx[state] = idx
                idx += 1
        
        # Process all states in efficient batches
        batch_size = 128  # Process in reasonable batches for CPU memory efficiency
        num_states = len(states)
        states_array = np.array(states, dtype=np.float32)
        
        # Switch to evaluation mode
        self.online_model.eval()
        
        with torch.no_grad():
            for start_idx in range(0, num_states, batch_size):
                end_idx = min(start_idx + batch_size, num_states)
                batch_states = states_array[start_idx:end_idx]
                
                # Forward pass with batch
                state_tensor = torch.FloatTensor(batch_states).to(self.device)
                q_values = self.online_model(state_tensor)
                actions = q_values.argmax(dim=1).cpu().numpy()
                
                # Store in policy
                for i, action in enumerate(actions):
                    global_idx = start_idx + i
                    for state, idx in state_to_idx.items():
                        if idx == global_idx:
                            policy[state] = int(action)  # Convert to Python int
                            break
        
        # Switch back to training mode
        self.online_model.train()
        return policy

# For backward compatibility with main.py
DQN = DuelingDQN

__all__ = ["DQNAgent", "DQN"]