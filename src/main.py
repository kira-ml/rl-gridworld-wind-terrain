import random
import logging
import logging.config
from logging.handlers import RotatingFileHandler
import sys
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import time
import threading
import queue
import argparse
import traceback
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Any

# Delay matplotlib import until needed for specific visualization backend
# This prevents unnecessary initialization when using PyGame
import queue
import argparse
import traceback
from typing import Dict, List, Tuple, Optional, Any

# Import agents from src/agents module
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.dqn import DuelingDQN as DQN, DQNAgent
from agents.value_iteration import ValueIterationAgent
from agents.policy_iteration import PolicyIterationAgent

# Import configurations
from config import (
    DEFAULT_ENV_CONFIG,
    QL_AGENT_CONFIG,
    DQN_AGENT_CONFIG,
    SARSA_AGENT_CONFIG
)

# Import metrics logging utilities
from utils.metrics_logger import save_metrics_to_csv, generate_enhanced_plot

# ===================== LOGGING CONFIGURATION =====================
def setup_logging():
    """
    Configure centralized logging for the entire application.
    Sets up both console and file handlers with appropriate formatting.
    """
    # Create logs directory if it doesn't exist
    import os
    from datetime import datetime
    from pathlib import Path
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"gridworld_training_{timestamp}.log"
    
    # Configure logging
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s | %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': str(log_file),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            '': {  # root logger
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        }
    }
    
    logging.config.dictConfig(logging_config)
    
    # Create module-specific loggers
    main_logger = logging.getLogger('gridworld_rl.main')
    env_logger = logging.getLogger('gridworld_rl.environment')
    agent_logger = logging.getLogger('gridworld_rl.agent')
    vis_logger = logging.getLogger('gridworld_rl.visualization')
    
    # Log system startup information
    main_logger.info("=" * 60)
    main_logger.info("GridWorld RL Training Session Started")
    main_logger.info(f"Python version: {sys.version}")
    main_logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    main_logger.info(f"Log file: {log_file}")
    main_logger.info("=" * 60)
    
    return main_logger, env_logger, agent_logger, vis_logger

# Initialize logging system
main_logger, env_logger, agent_logger, vis_logger = setup_logging()

# ===================== GLOBAL SETTINGS =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PYGAME_MAX_FPS = 30.0
METRICS_UPDATE_INTERVAL = 1.0  # Slower updates for better performance
METRICS_MIN_UPDATE_INTERVAL = 0.3  # Minimum time between matplotlib updates
METRICS_FRAME_SKIP = 8  # Skip more frames for high-frequency training

# ===================== GRIDWORLD ENVIRONMENT =====================
class GridWorldEnv:
    """
    Enhanced GridWorld environment with wind zones and terrain effects.
    """
    
    def __init__(self, config=None):
        self.config = config or DEFAULT_ENV_CONFIG
        self.grid_height, self.grid_width = self.config.get("grid_size", (7, 7))
        
        # Define action space properties directly
        self.action_space_size = 4
        self.sample_action = lambda: random.randint(0, 3)
        
        # Action to direction mapping: 0=Up, 1=Right, 2=Down, 3=Left
        self.action_to_dir = {
            0: (-1, 0),   # Up
            1: (0, 1),    # Right
            2: (1, 0),    # Down
            3: (0, -1)    # Left
        }
        
        # Provide action space interface for compatibility
        self.action_space = SimpleNamespace(
            n=self.action_space_size,
            sample=self.sample_action
        )
        
        self.agent_pos = None
        self.steps_taken = 0
        
        # Log environment initialization
        env_logger.info(f"GridWorld environment initialized with grid size {self.grid_height}x{self.grid_width}")
        env_logger.debug(f"Start position: {self.config.get('start_pos')}, Goal position: {self.config.get('goal_pos')}")
        
        self.reset()

    def reset(self):
        """Reset the environment to initial state."""
        self.agent_pos = np.array(self.config.get("start_pos", (0, 0)))
        self.steps_taken = 0
        return self.agent_pos.copy()

    def _is_valid_position(self, pos):
        """Check if position is within grid bounds."""
        return (0 <= pos[0] < self.grid_height and 0 <= pos[1] < self.grid_width)

    def _check_wind_effect(self, position):
        """Apply wind effects if agent is in a wind zone."""
        pos = position.copy()
        for wind_zone in self.config.get("wind_zones", []):
            if tuple(pos) in wind_zone["area"]:
                if random.random() < wind_zone["strength"]:
                    dy, dx = wind_zone["direction"]
                    push = wind_zone["push_distance"]
                    new_pos = pos + np.array([dy * push, dx * push])
                    if self._is_valid_position(new_pos):
                        pos = new_pos
        return pos

    def _check_terrain_effect(self, position):
        """Apply terrain effects based on current position."""
        pos = position.copy()
        additional_reward = 0
        pos_tuple = tuple(pos)
        terrain = self.config.get("terrain", {})
        
        # Ice effect (random slipping)
        if "ice" in terrain and pos_tuple in terrain["ice"]["positions"]:
            if random.random() < terrain["ice"]["slip_prob"]:
                slip_dir = random.choice(list(self.action_to_dir.values()))
                new_pos = pos + np.array(slip_dir)
                if self._is_valid_position(new_pos):
                    pos = new_pos
        
        # Mud effect (movement penalty)
        if "mud" in terrain and pos_tuple in terrain["mud"]["positions"]:
            if random.random() < terrain["mud"]["slow_prob"]:
                additional_reward += terrain["mud"]["step_cost"]
        
        # Quicksand effect (possible trapping)
        if "quicksand" in terrain and pos_tuple in terrain["quicksand"]["positions"]:
            if random.random() < terrain["quicksand"]["trap_prob"]:
                additional_reward += terrain["quicksand"]["escape_cost"]
        
        return pos, additional_reward

    def step(self, action):
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
        new_pos, terrain_reward = self._check_terrain_effect(new_pos)
        reward += terrain_reward
        
        # Update position
        self.agent_pos = new_pos
        
        # Check if goal reached or max steps exceeded
        goal_pos = self.config.get("goal_pos", (self.grid_height-1, self.grid_width-1))
        done = (
            np.array_equal(self.agent_pos, goal_pos) or
            self.steps_taken >= self.config.get("max_steps", 200)
        )
        
        # Add goal reward if applicable
        if np.array_equal(self.agent_pos, goal_pos):
            reward += self.config.get("rewards", {}).get("goal", 100.0)
        
        return self.agent_pos.copy(), reward, done, {}

# ===================== THREAD-SAFE METRICS VISUALIZER =====================
class LiveMetricsVisualizer(threading.Thread):
    """
    Thread-safe live metrics visualizer that saves plots to files
    to avoid GUI threading issues with Pygame.
    """
    
    def __init__(self, agent_type: str, update_interval: float = 0.5):
        super().__init__(daemon=True)
        self.agent_type = agent_type
        self.update_interval = update_interval
        self.metrics = {"rewards": [], "steps": [], "successes": []}
        self.lock = threading.Lock()
        self.running = True
        self._update_pending = False
        self._last_update_time = 0
        self._min_update_interval = 0.5  # Slower updates for file I/O
        self._frame_skip_counter = 0
        self._frame_skip_threshold = 10  # Skip more frames for file-based updates
        
        # File-based approach to avoid GUI threading issues
        self.save_path = "metrics_plots.png"
        
        # Import and configure matplotlib on demand
        import matplotlib
        import matplotlib.pyplot as plt
        self.plt = plt  # Store plt as instance variable
        
        # Initialize Matplotlib for headless rendering
        matplotlib.use('Agg')  # Non-interactive backend
        plt.ioff()  # Turn off interactive mode
        matplotlib.rcParams.update({
            'figure.max_open_warning': 0,
            'animation.embed_limit': 20,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 1.5,
            'font.size': 9,
            'figure.autolayout': True,
            'savefig.dpi': 80,
            'figure.dpi': 80
        })
        
        vis_logger.debug(f"Metrics will be saved to {self.save_path} every {update_interval}s")
        
        # Start the update thread
        self.start()

    def run(self):
        """Main thread loop for updating plots with file-based rendering."""
        while self.running:
            try:
                with self.lock:
                    # Only update if there's pending data and enough time has passed
                    current_time = time.time()
                    if (not self._update_pending or 
                        current_time - self._last_update_time < self._min_update_interval):
                        time.sleep(0.1)
                        continue
                    
                    # Create and save plot
                    self._create_and_save_plot()
                    
                    # Reset flags and update timestamp
                    self._update_pending = False
                    self._last_update_time = current_time
                    
            except Exception as e:
                vis_logger.error(f"Metrics update error: {e}")
            
            time.sleep(self.update_interval)

    def _create_and_save_plot(self):
        """Create and save plot to file efficiently."""
        try:
            # Create new figure for each update to avoid threading issues
            fig, (r_ax, s_ax, succ_ax) = self.plt.subplots(1, 3, figsize=(15, 4), dpi=80)
            fig.suptitle(f"{self.agent_type} Training Metrics (Live)", fontsize=12)
            
            # Plot rewards
            if self.metrics["rewards"]:
                episodes = list(range(len(self.metrics["rewards"])))
                r_ax.plot(episodes, self.metrics["rewards"], 'b-', linewidth=1.5)
                r_ax.set_title("Rewards per Episode", fontsize=10)
                r_ax.set_xlabel("Episode", fontsize=9)
                r_ax.set_ylabel("Total Reward", fontsize=9)
                r_ax.grid(True, alpha=0.3)
            
            # Plot steps
            if self.metrics["steps"]:
                episodes = list(range(len(self.metrics["steps"])))
                s_ax.plot(episodes, self.metrics["steps"], 'g-', linewidth=1.5)
                s_ax.set_title("Steps per Episode", fontsize=10)
                s_ax.set_xlabel("Episode", fontsize=9)
                s_ax.set_ylabel("Steps", fontsize=9)
                s_ax.grid(True, alpha=0.3)
            
            # Plot success rate (moving average)
            if self.metrics["successes"]:
                window = min(10, len(self.metrics["successes"]))
                if window > 0:
                    rate = np.convolve(self.metrics["successes"], 
                                     np.ones(window)/window, mode='valid')
                    episodes = list(range(len(rate)))
                    succ_ax.plot(episodes, rate, 'r-', linewidth=1.5)
                    succ_ax.set_title("Success Rate (10-ep avg)", fontsize=10)
                    succ_ax.set_xlabel("Episode", fontsize=9)
                    succ_ax.set_ylabel("Success Rate", fontsize=9)
                    succ_ax.set_ylim(-0.1, 1.1)
                    succ_ax.grid(True, alpha=0.3)
            
            # Save to file
            fig.tight_layout()
            fig.savefig(self.save_path, dpi=80, bbox_inches='tight')
            self.plt.close(fig)  # Important: close figure to free memory
            
        except Exception as e:
            vis_logger.error(f"Error creating plot: {e}")

    def setup_axes(self):
        """Placeholder for compatibility - not needed for file-based approach."""
        pass

    def update(self, metrics: Dict[str, List]):
        """Thread-safe update of metrics data with intelligent throttling."""
        current_time = time.time()
        
        # Implement frame skipping for high-frequency updates
        self._frame_skip_counter += 1
        if self._frame_skip_counter < self._frame_skip_threshold:
            return  # Skip this update
        
        # Reset counter
        self._frame_skip_counter = 0
        
        # Time-based throttling
        if current_time - self._last_update_time < self._min_update_interval:
            return  # Too soon since last update
        
        with self.lock:
            # Only update if not currently pending
            if not self._update_pending:
                # Deep copy only the data we need, limit history for memory efficiency
                max_history = 500  # Keep last 500 episodes max for file-based approach
                self.metrics = {
                    k: v[-max_history:].copy() if len(v) > max_history else v.copy() 
                    for k, v in metrics.items()
                }
                self._update_pending = True

    def _reset_update_flag(self):
        """Reset the update pending flag."""
        with self.lock:
            self._update_pending = False

    def get_performance_stats(self):
        """Get performance statistics for debugging."""
        with self.lock:
            return {
                'update_pending': self._update_pending,
                'last_update_time': self._last_update_time,
                'frame_skip_counter': self._frame_skip_counter,
                'metrics_length': len(self.metrics.get('rewards', [])),
                'thread_alive': self.is_alive(),
                'save_path': self.save_path
            }

    def set_throttling(self, min_interval: float = 0.5, frame_skip: int = 10):
        """Dynamically adjust throttling parameters."""
        with self.lock:
            self._min_update_interval = min_interval
            self._frame_skip_threshold = frame_skip

    def stop(self):
        """Stop the update thread and clean up resources."""
        self.running = False
        try:
            self.join(timeout=2.0)
        except:
            pass
        
        # Clean up any remaining matplotlib figures
        try:
            self.plt.close('all')
        except:
            pass

# ===================== VISUALIZATION UTILS =====================
def import_pygame_visualizer():
    """
    Dynamically import PyGame visualizer only when needed.
    This prevents unnecessary initialization when using Matplotlib.
    """
    try:
        from utils.game_visual import GridWorldVisualizer
        return GridWorldVisualizer
    except ImportError:
        vis_logger.warning("Could not import GridWorldVisualizer from utils.game_visual")
        return None

class NullVisualizer:
    """
    A no-op visualizer that implements the necessary interface but doesn't render anything.
    Used when render_mode='none' to maximize training speed.
    """
    def __init__(self):
        pass
    
    def reset(self, env=None):
        pass
    
    def update(self, env=None, agent=None, state=None, action=None, reward=None, next_state=None, done=None):
        pass
    
    def render(self, *args, **kwargs):
        pass
    
    def close(self):
        pass
        
    def save(self, *args, **kwargs):
        pass

# ===================== VISUALIZATION INITIALIZATION =====================
def initialize_visualization(backend, agent_type, metrics=None):
    """
    Initialize visualization backend based on user selection.
    Returns appropriate visualizer objects and settings.
    
    Args:
        backend: "matplotlib", "pygame", or "none"
        agent_type: Type of agent being trained
        metrics: Optional metrics dictionary to initialize visualizer
        
    Returns:
        vis: Primary visualizer object
        metrics_vis: Metrics visualizer object
        use_pygame: Boolean indicating if PyGame should be used
        render_enabled: Boolean indicating if rendering is enabled
    """
    vis_logger.info(f"Initializing {backend} visualization for {agent_type} agent")
    
    if backend == "matplotlib":
        # Import matplotlib-specific modules only when needed
        from utils.plotting import GridWorldVisualizer, MetricsVisualizer
        
        # Configure matplotlib for non-interactive use
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        matplotlib.rcParams.update({
            'figure.max_open_warning': 0,
            'animation.embed_limit': 20,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 1.5,
            'font.size': 9,
            'figure.autolayout': True,
            'savefig.dpi': 80,
            'figure.dpi': 80
        })
        
        vis_logger.debug("Matplotlib configured for non-interactive use")
        
        # Create visualizers
        vis = GridWorldVisualizer()
        metrics_vis = MetricsVisualizer(agent_type.upper())
        if metrics:
            metrics_vis.update(metrics)
            
        return vis, metrics_vis, False, True  # vis, metrics_vis, use_pygame, render_enabled
        
    elif backend == "pygame":
        # Import pygame-specific modules only when needed
        PyGameVisualizer = import_pygame_visualizer()
        
        if not PyGameVisualizer:
            vis_logger.error("PyGame visualization is not available")
            raise ImportError("PyGame visualization is not available")
        
        # Create visualizers
        pygame_vis = PyGameVisualizer()
        metrics_vis = LiveMetricsVisualizer(agent_type.upper())
        
        return pygame_vis, metrics_vis, True, True  # vis, metrics_vis, use_pygame, render_enabled
    
    elif backend == "none":
        # Use null visualizer for maximum training speed
        vis_logger.info("Visualization disabled for maximum training speed")
        vis = NullVisualizer()
        metrics_vis = LiveMetricsVisualizer(agent_type.upper())
        
        return vis, metrics_vis, False, False  # vis, metrics_vis, use_pygame, render_enabled
    
    vis_logger.error(f"Unknown visualization backend: {backend}")
    raise ValueError(f"Unknown visualization backend: {backend}")

# ===================== TRAINING UTILITIES =====================
def initialize_metrics():
    """Initialize empty metrics dictionary."""
    return {"rewards": [], "steps": [], "successes": []}

def record_metrics(metrics, reward, steps, goal_reached):
    """Record episode metrics."""
    metrics["rewards"].append(reward)
    metrics["steps"].append(steps)
    metrics["successes"].append(1 if goal_reached else 0)

def unwrap_state(state):
    """Unwrap state from tuple if needed."""
    while isinstance(state, tuple):
        state = state[0]
    return state

# ===================== TRAINING FUNCTIONS =====================
def train_policy_iteration(env, metrics, use_pygame=False, metrics_vis=None, pygame_vis=None, render_enabled=True):
    """Train Policy Iteration agent."""
    agent_logger.info("Initializing Policy Iteration agent")
    
    # Import here to avoid circular imports
    from config import PI_AGENT_CONFIG
    
    agent = PolicyIterationAgent(env, PI_AGENT_CONFIG)
    goal = tuple(env.config["goal_pos"])
    num_episodes = PI_AGENT_CONFIG.get("num_episodes", 50)
    
    agent_logger.debug(f"Policy Iteration parameters: gamma={PI_AGENT_CONFIG['gamma']}, "
                      f"theta={PI_AGENT_CONFIG['theta']}, max_iterations={PI_AGENT_CONFIG['max_iterations']}")
    
    # Run policy iteration algorithm once
    agent_logger.info("Running policy iteration...")
    agent.run_policy_iteration()
    agent_logger.info("Policy iteration completed")
    
    # Now evaluate the policy for specified episodes
    for ep in range(num_episodes):
        obs = env.reset()
        total_reward, done, steps = 0, False, 0
        
        while not done:
            try:
                action = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                obs = next_obs
                total_reward += reward
                steps += 1
                
                # Update visualizations
                if render_enabled and pygame_vis and steps % 2 == 0:  # Reduce frequency to prevent blocking
                    pygame_vis.render(env, agent, episode=ep, step=steps, reward=total_reward)
            except Exception as e:
                agent_logger.error(f"Error during policy iteration execution: {str(e)}")
                agent_logger.error(f"State type: {type(obs)}, value: {obs}")
                break
        
        # Update metrics
        record_metrics(metrics, total_reward, steps, tuple(env.agent_pos) == goal)
        if metrics_vis:
            metrics_vis.update(metrics)
        
        if ep % 10 == 0:
            agent_logger.info(f"Policy Iteration Evaluation - Episode {ep}/{num_episodes} | "
                             f"Reward: {total_reward:.2f}, Steps: {steps}")
    
    agent_logger.info(f"Policy Iteration evaluation completed - {num_episodes} episodes")
    
    # Initialize experiment
    exp_logger, checkpoint_mgr = initialize_experiment("policy_iteration")
    
    # Save final state
    save_info = save_training_state(
        agent=agent,
        metrics=metrics,
        episode=num_episodes,
        agent_name="policy_iteration",
        logger=exp_logger,
        checkpoint_manager=checkpoint_mgr,
        extra_data={
            "config": PI_AGENT_CONFIG,
            "episodes_trained": num_episodes,
            "environment": env.config
        }
    )
    
    # Display successful completion message
    exp_logger.info(f"Agent: POLICY_ITERATION successfully trained")
    exp_logger.info(f"Episodes trained: {num_episodes}")
    exp_logger.info(f"Checkpoint saved: {save_info['checkpoint_path']}")
    exp_logger.info(f"Training plot: {save_info['plot_path']}")
    # Add model_path to save_info to prevent KeyError
    save_info['model_path'] = save_info['checkpoint_path']
    agent_logger.info(f"Training completed successfully.")
    
    return agent, None, save_info

def train_q_learning(env, metrics, use_pygame=False, metrics_vis=None, pygame_vis=None, render_enabled=True):
    """Train Q-Learning agent."""
    agent_logger.info("Initializing Q-Learning agent")
    agent = QLearningAgent(env, QL_AGENT_CONFIG)
    goal = tuple(env.config["goal_pos"])
    num_episodes = QL_AGENT_CONFIG["num_episodes"]
    
    agent_logger.debug(f"Q-Learning parameters: alpha={QL_AGENT_CONFIG['alpha']}, "
                      f"gamma={QL_AGENT_CONFIG['gamma']}, epsilon={QL_AGENT_CONFIG['epsilon']}")
    
    for ep in range(num_episodes):
        obs = env.reset()
        total_reward, done, steps = 0, False, 0
        
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.update(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward
            steps += 1
            
            # Update visualizations
            if render_enabled and pygame_vis and steps % 2 == 0:  # Reduce frequency to prevent blocking
                pygame_vis.render(env, agent, episode=ep, step=steps, reward=total_reward)
        
        # Update metrics
        record_metrics(metrics, total_reward, steps, tuple(env.agent_pos) == goal)
        if metrics_vis:
            metrics_vis.update(metrics)
        
        agent.decay_epsilon()
        
        if ep % 10 == 0:
            agent_logger.info(f"Q-Learning Episode {ep}/{num_episodes} | "
                             f"Reward: {total_reward:.2f}, Steps: {steps}, "
                             f"Epsilon: {agent.epsilon:.3f}")
    
    agent_logger.info(f"Q-Learning training completed - {num_episodes} episodes")
    
    # Initialize experiment
    exp_logger, checkpoint_mgr = initialize_experiment("q_learning")
    
    # Save final state
    save_info = save_training_state(
        agent=agent,
        metrics=metrics,
        episode=num_episodes,
        agent_name="q_learning",
        logger=exp_logger,
        checkpoint_manager=checkpoint_mgr,
        extra_data={
            "config": QL_AGENT_CONFIG,
            "episodes_trained": num_episodes,
            "environment": env.config
        }
    )
    
    # Display successful completion message
    exp_logger.info(f"Agent: Q_LEARNING successfully trained")
    exp_logger.info(f"Episodes trained: {num_episodes}")
    exp_logger.info(f"Checkpoint saved: {save_info['checkpoint_path']}")
    exp_logger.info(f"Training plot: {save_info['plot_path']}")
    
    # Add model_path to save_info to prevent KeyError
    save_info['model_path'] = save_info['checkpoint_path']
    agent_logger.info(f"Training completed successfully.")
    
    return agent, None, save_info

def train_sarsa(env, metrics, use_pygame=False, metrics_vis=None, pygame_vis=None, render_enabled=True):
    """Train SARSA agent."""
    agent_logger.info("Initializing SARSA agent")
    agent = SarsaAgent(env, SARSA_AGENT_CONFIG)
    goal = tuple(env.config["goal_pos"])
    num_episodes = SARSA_AGENT_CONFIG["num_episodes"]
    
    agent_logger.debug(f"SARSA parameters: alpha={SARSA_AGENT_CONFIG['alpha']}, "
                      f"gamma={SARSA_AGENT_CONFIG['gamma']}, epsilon={SARSA_AGENT_CONFIG['epsilon']}")
    
    for ep in range(num_episodes):
        obs = env.reset()
        action = agent.act(obs)
        total_reward, done, steps = 0, False, 0
        
        while not done:
            next_obs, reward, done, _ = env.step(action)
            next_action = agent.act(next_obs) if not done else None
            agent.update(obs, action, reward, next_obs, next_action)
            
            obs, action = next_obs, next_action
            total_reward += reward
            steps += 1
            
            # Update visualizations
            if render_enabled and pygame_vis and steps % 2 == 0:  # Reduce frequency to prevent blocking
                pygame_vis.render(env, agent, episode=ep, step=steps, reward=total_reward)
        
        # Update metrics
        record_metrics(metrics, total_reward, steps, tuple(env.agent_pos) == goal)
        if metrics_vis:
            metrics_vis.update(metrics)
        
        agent.decay_epsilon()
        
        if ep % 10 == 0:
            agent_logger.info(f"SARSA Episode {ep}/{num_episodes} | "
                             f"Reward: {total_reward:.2f}, Steps: {steps}, "
                             f"Epsilon: {agent.epsilon:.3f}")
    
    agent_logger.info(f"SARSA training completed - {num_episodes} episodes")
    
    # Initialize experiment
    exp_logger, checkpoint_mgr = initialize_experiment("sarsa")
    
    # Save final state
    save_info = save_training_state(
        agent=agent,
        metrics=metrics,
        episode=num_episodes,
        agent_name="sarsa",
        logger=exp_logger,
        checkpoint_manager=checkpoint_mgr,
        extra_data={
            "config": SARSA_AGENT_CONFIG,
            "episodes_trained": num_episodes,
            "environment": env.config
        }
    )
    
    # Display successful completion message
    exp_logger.info(f"Agent: SARSA successfully trained")
    exp_logger.info(f"Episodes trained: {num_episodes}")
    exp_logger.info(f"Checkpoint saved: {save_info['checkpoint_path']}")
    exp_logger.info(f"Training plot: {save_info['plot_path']}")
    # Add model_path to save_info to prevent KeyError
    save_info['model_path'] = save_info['checkpoint_path']
    agent_logger.info(f"Training completed successfully.")
    
    return agent, None, save_info

def update_dqn(batch, online_model, target_model, optimizer, gamma):
    """Update DQN models using batch of experiences."""
    # Pre-allocate numpy arrays for better performance
    batch_size = len(batch)
    state_dim = len(batch[0][0])
    
    # Create numpy arrays first, then convert to tensors
    states_np = np.array([s[0] for s in batch], dtype=np.float32)
    actions_np = np.array([s[1] for s in batch], dtype=np.int64)
    rewards_np = np.array([s[2] for s in batch], dtype=np.float32)
    next_states_np = np.array([s[3] for s in batch], dtype=np.float32)
    dones_np = np.array([s[4] for s in batch], dtype=np.float32)
    
    # Convert to tensors (more efficient than from lists)
    states = torch.from_numpy(states_np).to(DEVICE)
    actions = torch.from_numpy(actions_np).to(DEVICE)
    rewards = torch.from_numpy(rewards_np).to(DEVICE)
    next_states = torch.from_numpy(next_states_np).to(DEVICE)
    dones = torch.from_numpy(dones_np).to(DEVICE)

    with torch.no_grad():
        next_actions = online_model(next_states).argmax(1)
        next_q = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + (1 - dones) * gamma * next_q

    q_values = online_model(states)
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = torch.nn.MSELoss()(q_selected, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_dqn(env, metrics, use_pygame=False, metrics_vis=None, pygame_vis=None, render_enabled=True):
    """Train DQN agent with configurable visualization."""
    agent_logger.info("Initializing DQN agent")
    
    # Import here to avoid circular imports
    from agents.dqn import DQNAgent
    
    # Create DQN agent
    agent = DQNAgent(env, DQN_AGENT_CONFIG)
    goal = tuple(env.config["goal_pos"])
    num_episodes = DQN_AGENT_CONFIG.get("num_episodes", 500)
    max_steps = DQN_AGENT_CONFIG.get("max_steps", 200)
    
    agent_logger.debug(f"DQN parameters: hidden_dim={DQN_AGENT_CONFIG['hidden_dim']}, "
                      f"lr={DQN_AGENT_CONFIG['learning_rate']}, gamma={DQN_AGENT_CONFIG['gamma']}")
    
    # Initialize tracking variables
    total_steps = 0
    
    # Hyperparameters
    epsilon = DQN_AGENT_CONFIG["epsilon_start"]
    epsilon_min = DQN_AGENT_CONFIG["epsilon_min"]
    decay = DQN_AGENT_CONFIG["epsilon_decay"]
    # Train the agent for specified episodes
    for ep in range(num_episodes):
        state = env.reset()
        total_reward, done, steps = 0, False, 0
        
        while not done and steps < max_steps:
            # Select action using agent's policy
            action = agent.act(state)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store transition in agent's replay buffer (handled internally)
            # Update is handled internally in agent._update_network
            
            # Update visualizations if enabled
            if render_enabled and pygame_vis and steps % 3 == 0:  # Reduce frequency to prevent blocking
                pygame_vis.render(env, agent, episode=ep, step=steps, reward=total_reward)
            
            # Update state and metrics
            state = next_state
            total_reward += reward
            steps += 1
            
            # Let the agent update its network
            if len(agent.replay_buffer) >= agent.batch_size:
                agent._update_network()
                
            if done:
                break
        
        # Update metrics
        record_metrics(metrics, total_reward, steps, tuple(env.agent_pos) == goal)
        if metrics_vis:
            metrics_vis.update(metrics)
        
        # Sync target network periodically
        if ep % agent.sync_frequency == 0:
            agent.target_model.load_state_dict(agent.online_model.state_dict())
        
        # Decay epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        if ep % 10 == 0:
            agent_logger.info(f"DQN Episode {ep}/{num_episodes} | "
                              f"Reward: {total_reward:.2f}, Steps: {steps}, "
                              f"Epsilon: {agent.epsilon:.3f}")
    
    agent_logger.info(f"DQN training completed - {num_episodes} episodes")
    
    # Initialize experiment
    exp_logger, checkpoint_mgr = initialize_experiment("dqn")
    
    # Save final state
    save_info = save_training_state(
        agent=agent,
        metrics=metrics,
        episode=num_episodes,
        agent_name="dqn",
        logger=exp_logger,
        checkpoint_manager=checkpoint_mgr,
        extra_data={
            "config": DQN_AGENT_CONFIG,
            "episodes_trained": num_episodes,
            "environment": env.config
        }
    )
    
    # Display successful completion message
    exp_logger.info(f"Agent: DQN successfully trained")
    exp_logger.info(f"Episodes trained: {num_episodes}")
    exp_logger.info(f"Checkpoint saved: {save_info['checkpoint_path']}")
    exp_logger.info(f"Training plot: {save_info['plot_path']}")
    # Add model_path to save_info to prevent KeyError
    save_info['model_path'] = save_info['checkpoint_path']
    agent_logger.info(f"Training completed successfully.")
    
    # For compatibility with other agents
    return agent, agent.online_model, save_info

# ===================== EXPERIMENT MANAGEMENT =====================
from utils.checkpoint import CheckpointManager
from utils.logger import ExperimentLogger

def initialize_experiment(agent_name: str, keep_best_k: int = 5, keep_last_k: int = 2):
    """
    Initialize experiment with logging and checkpointing.
    
    Args:
        agent_name: Name of the agent (e.g., 'q_learning')
        keep_best_k: Number of best checkpoints to keep
        keep_last_k: Number of most recent checkpoints to keep
    
    Returns:
        tuple: (ExperimentLogger, CheckpointManager)
    """
    # Initialize experiment logger
    logger = ExperimentLogger(
        experiment_name=agent_name,
        base_dir="logs",
        use_tensorboard=False,  # Can be made configurable
        console_level="INFO"
    )
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        base_dir="logs",
        experiment_name=agent_name,
        keep_best_k=keep_best_k,
        keep_last_k=keep_last_k,
        metric_name="reward",
        metric_goal="max"
    )
    
    logger.info(f"Initialized experiment for {agent_name}")
    logger.info(f"Checkpoints will be saved in: {checkpoint_manager.checkpoint_dir}")
    
    return logger, checkpoint_manager

def save_training_state(
    agent,
    metrics: Dict[str, List[float]],
    episode: int,
    agent_name: str,
    logger: ExperimentLogger,
    checkpoint_manager: CheckpointManager,
    extra_data: Optional[Dict] = None
) -> Dict[str, str]:
    """
    Save complete training state including model, metrics, and plots.
    
    Args:
        agent: The trained agent
        metrics: Dictionary of training metrics
        episode: Current episode number
        agent_name: Name of the agent
        logger: ExperimentLogger instance
        checkpoint_manager: CheckpointManager instance
        extra_data: Optional additional data to save
    
    Returns:
        dict: Paths where data was saved
    """
    try:
        # Handle Q-table or model state based on agent type
        if hasattr(agent, 'state_dict'):
            # For DQN-like agents that have PyTorch models
            model_state = agent.state_dict()
        elif hasattr(agent, 'q_table'):
            # For tabular agents like Q-Learning
            model_state = {
                'q_table': agent.q_table,
                'epsilon': agent.epsilon,
                'alpha': agent.alpha if hasattr(agent, 'alpha') else None,
                'gamma': agent.gamma if hasattr(agent, 'gamma') else None
            }
        else:
            # For other agents, store essential parameters
            model_state = {
                attr: getattr(agent, attr)
                for attr in ['epsilon', 'alpha', 'gamma']
                if hasattr(agent, attr)
            }
        
        # Prepare state dictionary with only serializable components
        state_dict = {
            "model_state": model_state,
            "training_info": {
                "episode": episode,
                "agent_type": agent_name,
                "parameters": {
                    k: v for k, v in (extra_data or {}).items()
                    if isinstance(v, (dict, list, str, int, float, bool, type(None)))
                }
            }
        }
        
        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(
            state_dict=state_dict,
            metrics=metrics,
            episode=episode,
            is_best=False  # Will be determined by CheckpointManager
        )
        
        # Log checkpoint creation
        logger.info(f"Saved checkpoint at episode {episode}: {checkpoint_path}")
        
        # Generate and save training plot if metrics exist
        plot_path = None
        if metrics:
            try:
                plot_path = generate_enhanced_plot(
                    metrics,
                    agent_name,
                    logs_dir=str(checkpoint_manager.checkpoint_dir)
                )
                logger.info(f"Generated training plot: {plot_path}")
            except Exception as e:
                logger.error(f"Failed to generate plot: {str(e)}")
        
        return {
            "checkpoint_path": checkpoint_path,
            "plot_path": plot_path,
            "metrics": checkpoint_manager.get_best_metrics()
        }
        
    except Exception as e:
        logger.error(f"Failed to save training state: {str(e)}")
        raise

# ===================== COMMAND LINE INTERFACE =====================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GridWorld RL with Live Visualization")
    parser.add_argument(
        "--agent", "-a",
        choices=["q_learning", "dqn", "sarsa", "policy_iteration"],
        default=None,
        help="Type of agent to train (Q-Learning, DQN, SARSA, or Policy Iteration)"
    )
    parser.add_argument(
        "--no-pygame",
        action="store_true",
        help="Disable PyGame visualization (deprecated, use --render-mode instead)"
    )
    parser.add_argument(
        "--visualization", "-v",
        choices=["pygame", "matplotlib"],
        default=None,
        help="Visualization mode to use (deprecated, use --render-mode instead)"
    )
    parser.add_argument(
        "--render-mode", "-r",
        choices=["pygame", "matplotlib", "none"],
        default=None,
        help="Rendering mode: pygame (fast), matplotlib (detailed), or none (fastest training)"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=None,
        help="Number of episodes to train. Recommended: Q-Learning=400, DQN=1200, SARSA=500, Policy Iteration=150"
    )
    parser.add_argument(
        "--safe-metrics", "-s",
        action="store_true",
        help="Enable thread-safe metrics visualization"
    )
    return parser.parse_args()

# ===================== VISUALIZATION SELECTION =====================
def select_visualization_backend():
    main_logger.info("Requesting visualization backend selection")
    print("\nSelect visualization backend:")
    print("1. Matplotlib (static/animated plots, no live dashboard)")
    print("2. Pygame (interactive visualization)")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            main_logger.info("Matplotlib backend selected")
            return "matplotlib"
        elif choice == "2":
            main_logger.info("Pygame backend selected")
            return "pygame"
        else:
            print("Invalid input. Please enter 1 or 2.")

# ===================== HEADLESS CHECK =====================
def check_headless_environment():
    """
    Check if we're running in a headless environment (no display).
    
    Returns:
        bool: True if running in headless environment, False otherwise
    """
    # First, check if DISPLAY variable is set (Unix)
    import os
    if os.name == "posix" and "DISPLAY" not in os.environ:
        return True
    
    # For Windows, try to import display-related modules
    try:
        if os.name == "nt":
            # On Windows, try to get display info
            from ctypes import windll
            try:
                windll.user32.GetSystemMetrics(0)  # Width of primary monitor
                return False
            except:
                return True
    except:
        pass
    
    # As a fallback, try to initialize pygame display
    try:
        import pygame
        pygame.init()
        pygame.display.init()
        pygame.display.quit()
        pygame.quit()
        return False
    except:
        return True

# ===================== MAIN FUNCTION =====================
def main():
    """
    Main execution function.
    
    Key improvements:
    1. Agent selection happens FIRST, before any training initialization
    2. Thread-safe Matplotlib visualization prevents conflicts with PyGame
    3. All code is self-contained in this single file
    4. Clear error handling and resource cleanup
    5. Production-grade logging throughout
    """
    try:
        main_logger.info("Starting GridWorld RL application")
        # Parse command line arguments
        args = parse_args()
        
        # Handle visualization mode selection
        render_mode = None
        
        # First check for the new render-mode flag
        if args.render_mode:
            render_mode = args.render_mode.lower()
            main_logger.info(f"Using render mode from command line: {render_mode}")
        # Fall back to the old visualization flag if render-mode is not set
        elif args.visualization:
            render_mode = args.visualization.lower()
            main_logger.info(f"Using visualization mode (deprecated): {render_mode}")
        # Finally, default to matplotlib or pygame based on no_pygame flag
        else:
            render_mode = "matplotlib" if args.no_pygame else "pygame"
            main_logger.info(f"Defaulting to render mode: {render_mode}")
            
        # Set the visualization flags based on render_mode
        use_pygame = render_mode == "pygame"
        use_matplotlib = render_mode == "matplotlib"
        render_enabled = render_mode != "none"
        
        # Check if running in headless environment
        is_headless = check_headless_environment()
        if is_headless and use_pygame:
            main_logger.warning("Running in headless environment. Falling back to Matplotlib visualization.")
            use_pygame = False
            use_matplotlib = True
            
        # Select appropriate backend based on visualization choice
        if use_matplotlib:
            backend = select_visualization_backend()
        
        # ===== FIRST: GET AGENT SELECTION (NO TRAINING YET) =====
        print("\n" + "="*60)
        print("           GridWorld Reinforcement Learning")
        print("="*60)
        print("\nAvailable agents:")
        print("1. Q-Learning - Classic tabular Q-learning algorithm")
        print("2. DQN - Deep Q-Network with neural network")
        print("3. SARSA - State-Action-Reward-State-Action learning")
        print("4. Policy Iteration - Model-based dynamic programming approach")
        print("-" * 60)
        
        agent_type = args.agent
        if not agent_type:
            while not agent_type:
                choice = input("\nSelect agent (1-4 or q_learning/dqn/sarsa/policy_iteration): ").strip().lower()
                if choice in ["1", "q_learning", "q", "ql"]:
                    agent_type = "q_learning"
                elif choice in ["2", "dqn", "d"]:
                    agent_type = "dqn"
                elif choice in ["3", "sarsa", "s"]:
                    agent_type = "sarsa"
                elif choice in ["4", "policy_iteration", "pi"]:
                    agent_type = "policy_iteration"
                else:
                    print("[ERROR] Invalid selection. Please choose a valid agent.")
        
        main_logger.info(f"Agent selected: {agent_type}")
        
        # Get number of episodes
        num_episodes = args.episodes
        if not num_episodes:
            try:
                # Show default episodes based on agent type
                default_episodes = {
                    "q_learning": 400,    # Optimized for Q-learning
                    "dqn": 1200,         # Optimized for DQN
                    "sarsa": 500,        # Optimized for SARSA
                    "policy_iteration": 150  # Optimized for Policy Iteration
                }.get(agent_type, 50)
                
                ep_input = input(f"\nNumber of training episodes (recommended for {agent_type}: {default_episodes}): ").strip()
                if ep_input:
                    num_episodes = int(ep_input)
                else:
                    num_episodes = default_episodes
            except ValueError:
                main_logger.warning("Invalid episode count provided, using default value")
                num_episodes = 50
        
        # Update config with selected episodes
        if agent_type == "dqn":
            DQN_AGENT_CONFIG["num_episodes"] = num_episodes
        elif agent_type == "q_learning":
            QL_AGENT_CONFIG["num_episodes"] = num_episodes
        elif agent_type == "sarsa":
            SARSA_AGENT_CONFIG["num_episodes"] = num_episodes
        elif agent_type == "policy_iteration":
            # Import here to avoid circular imports
            from config import PI_AGENT_CONFIG
            PI_AGENT_CONFIG["num_episodes"] = num_episodes
        
        main_logger.info(f"Training configuration: episodes={num_episodes}, "
                        f"visualization={'Pygame' if use_pygame else 'Matplotlib'}")
        
        print(f"\nSelected: {agent_type.upper()} agent")
        print(f"Episodes: {num_episodes}")
        print(f"Visualization: {'PyGame' if use_pygame else 'Matplotlib'}")
        print(f"Live metrics: {'Thread-safe mode' if args.safe_metrics else 'Standard mode'}")
        
        print("\n" + "-" * 60)
        print("Training will start after you press Enter...")
        print("Press Ctrl+C at any time to stop training.")
        print("-" * 60)
        input("\nPress Enter to begin training...")
        
        # ===== NOW INITIALIZE EVERYTHING FOR TRAINING =====
        main_logger.info(f"Starting {agent_type.upper()} training...")
        
        # Initialize environment and metrics
        env = GridWorldEnv(config=DEFAULT_ENV_CONFIG)
        metrics = initialize_metrics()
        
        # Initialize visualizers based on selected modes
        try:
            visualizer = None
            metrics_vis = None
            render_enabled = render_mode != "none"
            
            if render_mode == "pygame":
                # Initialize PyGame visualization
                main_logger.info("Initializing PyGame visualization")
                from utils.game_visual import GridWorldVisualizer
                visualizer = GridWorldVisualizer()
                
                # Create a secondary thread for metrics if thread-safe mode is enabled
                if args.safe_metrics:
                    import matplotlib
                    matplotlib.use('Agg')  # Non-interactive backend
                    from utils.plotting import AsyncMetricsPlotter
                    metrics_vis = AsyncMetricsPlotter(metrics, thread_safe=True)
            elif render_mode == "matplotlib":
                # Initialize Matplotlib visualization
                main_logger.info("Initializing Matplotlib visualization")
                import matplotlib
                matplotlib.use('TkAgg')  # Interactive backend
                from utils.plotting import AsyncMetricsPlotter
                metrics_vis = AsyncMetricsPlotter(metrics, thread_safe=args.safe_metrics)
            elif render_mode == "none":
                # No visualization for maximum training speed
                main_logger.info("Visualization disabled for maximum training speed")
                visualizer = NullVisualizer()
                
                # Still set up metrics for logging, but without visualization
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                from utils.plotting import AsyncMetricsPlotter
                metrics_vis = AsyncMetricsPlotter(metrics, thread_safe=True, update_interval=5.0)  # Slower updates
                
        except ImportError as e:
            main_logger.error(f"Visualization initialization failed: {e}")
            print(f"Error: {e}")
            print("Falling back to no visualization.")
            visualizer = NullVisualizer()
            metrics_vis = None
            render_enabled = False
        
        agent, model, save_info = None, None, None
        
        # Train with the selected agent type
        if agent_type == "dqn":
            agent, model, save_info = train_dqn(env, metrics, use_pygame=use_pygame, 
                                              metrics_vis=metrics_vis, pygame_vis=visualizer,
                                              render_enabled=render_enabled)
        elif agent_type == "q_learning":
            agent, model, save_info = train_q_learning(env, metrics, use_pygame=use_pygame, 
                                                     metrics_vis=metrics_vis, pygame_vis=visualizer,
                                                     render_enabled=render_enabled)
        elif agent_type == "sarsa":
            agent, model, save_info = train_sarsa(env, metrics, use_pygame=use_pygame, 
                                                metrics_vis=metrics_vis, pygame_vis=visualizer,
                                                render_enabled=render_enabled)
        elif agent_type == "policy_iteration":
            agent, model, save_info = train_policy_iteration(env, metrics, use_pygame=use_pygame,
                                                           metrics_vis=metrics_vis, pygame_vis=visualizer,
                                                           render_enabled=render_enabled)
        
        # Display training summary
        if save_info:
            display_training_summary(agent_type, num_episodes, metrics, save_info)
            
        # Post-training visualization based on backend
        if use_matplotlib and visualizer is not None:
            # For matplotlib: render gridworld visualization after training
            vis_logger.info("Generating post-training visualizations")
            visualizer.plot_gridworld(env, agent=agent, episode=num_episodes)
            if metrics_vis:
                metrics_vis.update(metrics)
                metrics_vis.save("metrics_matplotlib.png")
                main_logger.info("Matplotlib plots saved as 'metrics_matplotlib.png'")
            plot_path = generate_enhanced_plot(metrics, agent_type, logs_dir='models/logs')
            main_logger.info(f"Enhanced metrics saved to models/logs/ directory")
            
            input("Press Enter to exit...")
        elif use_pygame:
            # PyGame visualization happens during training, nothing extra needed here
            pass
        
        # ===== TRAINING COMPLETE =====
        main_logger.info(f"{agent_type.upper()} training completed successfully!")
        
        # Print summary statistics
        print("\n" + "="*50)
        print("           TRAINING SUMMARY")
        print("="*50)
        try:
            avg_reward = np.mean(metrics['rewards'])
            avg_steps = np.mean(metrics['steps'])
            success_rate = np.mean(metrics['successes'])
            
            # Log detailed statistics
            main_logger.info(f"Training summary - Avg Reward: {avg_reward:.2f}, "
                           f"Avg Steps: {avg_steps:.2f}, Success Rate: {success_rate:.2%}")
            
            print(f"Average Reward:  {avg_reward:.2f}")
            print(f"Average Steps:   {avg_steps:.2f}")
            print(f"Success Rate:    {success_rate:.2%}")
            
            # Show final performance
            if len(metrics['rewards']) >= 10:
                final_10_reward = np.mean(metrics['rewards'][-10:])
                final_10_success = np.mean(metrics['successes'][-10:])
                main_logger.info(f"Final 10 episodes - Reward: {final_10_reward:.2f}, "
                               f"Success Rate: {final_10_success:.2%}")
                print(f"\nFinal 10 episodes:")
                print(f"  Average Reward: {final_10_reward:.2f}")
                print(f"  Success Rate:   {final_10_success:.2%}")
                
            # For PyGame visualization, we need to save metrics here
            if use_pygame:
                # Save detailed training metrics to CSV and enhanced plots
                csv_path = save_metrics_to_csv(metrics, agent_type, logs_dir='models/logs')
                plot_path = generate_enhanced_plot(metrics, agent_type, logs_dir='models/logs')
                main_logger.info("Enhanced metrics saved to models/logs/ directory")
                
        except Exception as summary_err:
            main_logger.error(f"Failed to compute summary statistics: {summary_err}")
        
        print("="*50)
        
        # Keep metrics visualization running
        if metrics_vis:
            print(f"\n📊 Live metrics are being saved to '{metrics_vis.save_path}'")
            print("You can view the updated plots by opening this file periodically.")
            print("Training metrics will continue updating during training.")
            print("Press Ctrl+C to exit.")
            
            # Wait for user to exit
            try:
                while metrics_vis.running:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass
        
    except KeyboardInterrupt:
        main_logger.warning("Training interrupted by user")
    except Exception as e:
        main_logger.critical(f"Unhandled exception: {e}", exc_info=True)
    finally:
        # ===== CLEANUP =====
        main_logger.info("Cleaning up resources...")
        
        # Stop metrics visualizer
        try:
            if 'metrics_vis' in locals() and metrics_vis is not None:
                metrics_vis.stop()
        except Exception as cleanup_err:
            main_logger.error(f"Error stopping metrics visualizer: {cleanup_err}")
        
        # Close Matplotlib properly if it was imported
        try:
            if 'use_matplotlib' in locals() and use_matplotlib:
                import matplotlib.pyplot as plt
                plt.close('all')
                plt.ioff()
        except Exception as cleanup_err:
            main_logger.error(f"Error cleaning up matplotlib: {cleanup_err}")
        
        main_logger.info("GridWorld RL Training Session Ended")
        main_logger.info("=" * 60)

def display_training_summary(agent_type, num_episodes, metrics, save_info):
    """
    Display a summary of training results.
    
    Args:
        agent_type: String identifier of the agent type
        num_episodes: Number of episodes trained
        metrics: Dictionary containing training metrics
        save_info: Dictionary containing paths where data was saved
    """
    import numpy as np
    
    # Calculate summary statistics
    rewards = metrics.get("cumulative_rewards", [])
    steps = metrics.get("episode_lengths", [])
    success_rate = metrics.get("success_rates", [])[-1] if metrics.get("success_rates") else 0.0
    
    # Display summary header
    print("\n" + "="*60)
    print(f"     TRAINING SUMMARY: {agent_type.upper()}")
    print("="*60)
    
    # Display metrics summary
    if rewards:
        print(f"• Average reward (last 10%): {np.mean(rewards[-int(len(rewards)*0.1):]):.2f}")
        print(f"• Best episode reward: {max(rewards):.2f}")
        
    if steps:
        print(f"• Average steps per episode (last 10%): {np.mean(steps[-int(len(steps)*0.1):]):.1f}")
        print(f"• Minimum steps: {min(steps)}")
    
    print(f"• Final success rate: {success_rate:.2%}")
    print(f"• Episodes trained: {num_episodes}")
    
    # Display save information
    print("\nSAVED OUTPUTS:")
    # Use get() to safely access dictionary keys
    print(f"• Model: {save_info.get('model_path', save_info.get('checkpoint_path', 'N/A'))}")
    print(f"• Metrics: {save_info.get('metrics_path', 'N/A')}")
    print(f"• Plot: {save_info['plot_path']}")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
