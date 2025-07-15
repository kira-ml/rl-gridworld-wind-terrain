import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import torch
from typing import Dict, Any, List, Tuple, Optional, Union
from agents.dqn import DQNAgent

# Use CPU/GPU appropriately
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GridWorldVisualizer:
    """Visualizer class for GridWorld environment and agents."""
    
    TERRAIN_COLORS = {
        'ice': {'color': '#b3e0ff', 'alpha': 0.5},
        'mud': {'color': '#8b4513', 'alpha': 0.3},
        'quicksand': {'color': '#c2b280', 'alpha': 0.4}
    }
    
    WIND_ARROWS = {
        (0, 1): '→',   # Right
        (0, -1): '←',  # Left
        (-1, 0): '↑',  # Up
        (1, 0): '↓',   # Down
        (-1, 1): '↗',  # Up-Right
        (-1, -1): '↖', # Up-Left
        (1, 1): '↘',   # Down-Right
        (1, -1): '↙'   # Down-Left
    }

    def __init__(self):
        self.terrain_cache = {}
    
    def plot_gridworld(self, env, agent=None, fig=None, ax=None, show_value_heatmap=True,
                      trajectory=None, episode=None, step=None, cbar_ax=None):
        """Plot GridWorld environment with agent, terrain, and policy visualization.
        
        Args:
            env: GridWorld environment instance
            agent: RL agent instance (Q-Learning, SARSA, DQN, etc.)
            fig: Optional matplotlib figure
            ax: Optional matplotlib axis
            show_value_heatmap: Whether to show value/Q-value heatmap
            trajectory: List of positions showing agent's path
            episode: Current episode number
            step: Current step number
            cbar_ax: Optional colorbar axis
        """
        if not hasattr(env, 'config'):
            raise ValueError("Environment must have config attribute")
        
        grid_height, grid_width = env.config["grid_size"]
        
        # Create or clear axes
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            close_fig = True
        else:
            ax.clear()
            close_fig = False
            
        # Set up grid
        ax.set_xlim(-0.5, grid_width - 0.5)
        ax.set_ylim(-0.5, grid_height - 0.5)
        ax.set_xticks(range(grid_width))
        ax.set_yticks(range(grid_height))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        
        # Draw terrain
        self._draw_terrain(ax, env)
        
        # Draw wind zones
        self._draw_wind_zones(ax, env)
        
        # Draw value/Q-value heatmap
        if agent is not None and show_value_heatmap:
            self._draw_value_heatmap(ax, env, agent, cbar_ax)
        
        # Draw policy arrows
        if agent is not None:
            self._draw_policy_arrows(ax, env, agent)
        
        # Draw trajectory
        if trajectory:
            self._draw_trajectory(ax, trajectory)
        
        # Draw goal and agent
        self._draw_goal_and_agent(ax, env)
        
        # Add title and info
        self._add_plot_info(ax, episode, step, env)
        
        plt.tight_layout()
        if close_fig:
            plt.pause(0.1)
            plt.close(fig)
        else:
            fig.canvas.draw_idle()
            plt.pause(0.01)

    def _draw_terrain(self, ax, env):
        """Draw different terrain types."""
        for terrain_type, data in env.config.get("terrain", {}).items():
            if terrain_type in self.TERRAIN_COLORS:
                color_info = self.TERRAIN_COLORS[terrain_type]
                for pos in data["positions"]:
                    ax.add_patch(patches.Rectangle(
                        (pos[1]-0.5, pos[0]-0.5), 1, 1,
                        color=color_info['color'],
                        alpha=color_info['alpha'],
                        label=terrain_type.capitalize(),
                        zorder=2
                    ))

    def _draw_wind_zones(self, ax, env):
        """Draw wind zones with directional indicators."""
        for wind_zone in env.config.get("wind_zones", []):
            direction = wind_zone["direction"]
            strength = wind_zone["strength"]
            arrow = self.WIND_ARROWS.get(direction, '•')
            
            for pos in wind_zone["area"]:
                ax.add_patch(patches.Rectangle(
                    (pos[1]-0.5, pos[0]-0.5), 1, 1,
                    color='lightblue', alpha=0.2, zorder=1
                ))
                ax.text(pos[1], pos[0], f'{arrow}{strength:.1f}',
                       ha='center', va='center', color='navy',
                       fontsize=10, zorder=3)

    def _draw_value_heatmap(self, ax, env, agent, cbar_ax):
        """Draw value function or Q-value heatmap."""
        values = None
        
        if hasattr(agent, 'Q'):  # Q-Learning or SARSA
            values = np.max(agent.Q, axis=2)
        elif hasattr(agent, 'V'):  # Value/Policy Iteration
            values = agent.V
        elif hasattr(agent, 'online_net'):  # DQN
            # For DQN, we'll need to get Q-values for all states
            values = np.zeros(env.config["grid_size"])
            for i in range(env.grid_height):
                for j in range(env.grid_width):
                    state = np.array([i, j])
                    q_values = agent.online_net(torch.FloatTensor(state).to(DEVICE))
                    values[i, j] = q_values.max().item()
        
        if values is not None:
            im = ax.imshow(values, cmap='YlOrRd', alpha=0.3,
                         origin='upper', extent=[-0.5, env.grid_width-0.5,
                                               env.grid_height-0.5, -0.5],
                         zorder=2)
            if cbar_ax is not None:
                cbar_ax.clear()
                plt.colorbar(im, cax=cbar_ax)
                cbar_ax.set_ylabel('Value')

    def _draw_policy_arrows(self, ax, env, agent):
        """Draw policy arrows showing the best action in each state."""
        action_to_arrow = {
            0: (0, 0.3),    # Up
            1: (0.3, 0),    # Right
            2: (0, -0.3),   # Down
            3: (-0.3, 0)    # Left
        }
        
        for i in range(env.grid_height):
            for j in range(env.grid_width):
                if (i, j) == tuple(env.config["goal_pos"]):
                    continue
                    
                action = None
                if hasattr(agent, 'Q'):  # Q-Learning or SARSA
                    action = np.argmax(agent.Q[i, j])
                elif hasattr(agent, 'policy'):  # Value/Policy Iteration
                    action = agent.policy[i, j]
                elif hasattr(agent, 'online_net'):  # DQN
                    state = torch.FloatTensor([i, j]).to(DEVICE)
                    action = agent.online_net(state).argmax().item()
                
                if action is not None:
                    dx, dy = action_to_arrow[action]
                    ax.arrow(j, i, dx, dy, head_width=0.15, head_length=0.15,
                            fc='blue', ec='blue', alpha=0.6, zorder=4,
                            length_includes_head=True)

    def _draw_trajectory(self, ax, trajectory):
        """Draw agent's trajectory through the environment."""
        if len(trajectory) > 1:
            traj = np.array(trajectory)
            ax.plot(traj[:, 1], traj[:, 0], 'magenta',
                   linewidth=2, alpha=0.7, label='Trajectory',
                   zorder=5)

    def _draw_goal_and_agent(self, ax, env):
        """Draw goal and current agent position."""
        # Draw goal
        goal_row, goal_col = env.config["goal_pos"]
        ax.add_patch(patches.Rectangle(
            (goal_col-0.3, goal_row-0.3), 0.6, 0.6,
            color='green', label='Goal', zorder=6
        ))
        
        # Draw agent
        agent_row, agent_col = env.agent_pos
        ax.add_patch(patches.Circle(
            (agent_col, agent_row), 0.3,
            color='orange', ec='black', lw=1.5,
            label='Agent', zorder=7
        ))

    def _add_plot_info(self, ax, episode, step, env):
        """Add title, legend, and other information to the plot."""
        title = "GridWorld Environment"
        if episode is not None:
            title += f" | Episode {episode+1}"
        if step is not None:
            title += f" | Step {step+1}"
        
        ax.set_title(title)
        
        # Add legend with no duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.
        )

class MetricsVisualizer:
    """Class for visualizing training metrics."""
    
    def __init__(self, agent_type: str):
        """
        Args:
            agent_type: Type of agent ('q_learning', 'sarsa', 'dqn', etc.)
        """
        self.agent_type = agent_type.upper()
        plt.ion()
        self.fig, (self.r_ax, self.s_ax, self.succ_ax) = plt.subplots(1, 3, figsize=(15, 4))
        self.setup_axes()
        
    def setup_axes(self):
        """Set up the axes with proper labels and titles."""
        self.fig.suptitle(f"{self.agent_type} Training Metrics")
        
        self.r_ax.set_title("Rewards per Episode")
        self.r_ax.set_xlabel("Episode")
        self.r_ax.set_ylabel("Total Reward")
        
        self.s_ax.set_title("Steps per Episode")
        self.s_ax.set_xlabel("Episode")
        self.s_ax.set_ylabel("Steps")
        
        self.succ_ax.set_title("Success Rate")
        self.succ_ax.set_xlabel("Episode")
        self.succ_ax.set_ylabel("Success Rate (10-ep avg)")
        
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update(self, metrics: Dict[str, List]):
        """Update plots with new metrics."""
        self.r_ax.clear()
        self.s_ax.clear()
        self.succ_ax.clear()
        
        # Plot rewards
        self.r_ax.plot(metrics["rewards"], 'b-')
        self.r_ax.grid(True)
        
        # Plot steps
        self.s_ax.plot(metrics["steps"], 'g-')
        self.s_ax.grid(True)
        
        # Plot success rate
        if metrics["successes"]:
            window = min(10, len(metrics["successes"]))
            rate = np.convolve(metrics["successes"], 
                             np.ones(window)/window, mode='valid')
            self.succ_ax.plot(rate, 'r-')
            self.succ_ax.set_ylim(-0.1, 1.1)
        self.succ_ax.grid(True)
        
        self.setup_axes()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save(self, filepath: str):
        """Save the current figure to a file."""
        self.fig.savefig(filepath, bbox_inches='tight', dpi=300)

def plot_comparison(metrics_list: List[Dict], agent_names: List[str],
                   save_path: Optional[str] = None):
    """Plot comparison of different agents' performance.
    
    Args:
        metrics_list: List of metrics dictionaries from different agents
        agent_names: List of agent names for the legend
        save_path: Optional path to save the comparison plot
    """
    fig, (r_ax, s_ax, sr_ax) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Agent Performance Comparison")
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(agent_names)))
    
    # Plot rewards
    r_ax.set_title("Rewards per Episode")
    r_ax.set_xlabel("Episode")
    r_ax.set_ylabel("Total Reward")
    
    # Plot steps
    s_ax.set_title("Steps per Episode")
    s_ax.set_xlabel("Episode")
    s_ax.set_ylabel("Steps")
    
    # Plot success rates
    sr_ax.set_title("Success Rate")
    sr_ax.set_xlabel("Episode")
    sr_ax.set_ylabel("Success Rate (10-ep avg)")
    
    for metrics, name, color in zip(metrics_list, agent_names, colors):
        # Smooth the metrics for clearer visualization
        window = min(10, len(metrics["rewards"]))
        reward_smooth = np.convolve(metrics["rewards"],
                                  np.ones(window)/window, mode='valid')
        steps_smooth = np.convolve(metrics["steps"],
                                 np.ones(window)/window, mode='valid')
        
        r_ax.plot(reward_smooth, color=color, label=name)
        s_ax.plot(steps_smooth, color=color, label=name)
        
        if metrics["successes"]:
            success_rate = np.convolve(metrics["successes"],
                                     np.ones(window)/window, mode='valid')
            sr_ax.plot(success_rate, color=color, label=name)
    
    for ax in [r_ax, s_ax, sr_ax]:
        ax.grid(True)
        ax.legend()
    
    sr_ax.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def animate_gridworld_episode(env, agent, online_model, metrics, num_episodes, agent_type="unknown"):
    """Animate the gridworld environment and update metrics in real-time."""
    plt.ion()  # Enable interactive mode
    
    # Create visualizers
    grid_vis = GridWorldVisualizer()
    metrics_vis = MetricsVisualizer(agent_type)
    
    # Create main environment figure with colorbar
    env_fig = plt.figure(figsize=(9, 8))
    gs = GridSpec(1, 2, width_ratios=[20, 1], figure=env_fig)
    env_ax = env_fig.add_subplot(gs[0])
    cbar_ax = env_fig.add_subplot(gs[1])
    
    try:
        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            step_count = 0
            trajectory = [env.agent_pos]

            while not done:
                # Get action based on agent type
                if agent:
                    action = agent.act(obs)
                else:  # DQN
                    flat_obs = obs if not isinstance(obs, tuple) else obs[0]
                    # Create a temporary DQNAgent instance to use its act method
                    temp_agent = DQNAgent(online_model)
                    action = temp_agent.act(flat_obs, epsilon=0.01)
                
                # Take step in environment
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                step_count += 1
                trajectory.append(env.agent_pos)

                # Update visualizations
                env_ax.clear()
                grid_vis.plot_gridworld(
                    env, agent, env_fig, env_ax,
                    trajectory=trajectory,
                    episode=ep,
                    step=step_count,
                    cbar_ax=cbar_ax
                )
                metrics_vis.update(metrics)
                plt.pause(0.05)  # Control animation speed
                
            plt.pause(0.5)  # Pause between episodes
    finally:
        plt.ioff()  # Disable interactive mode

import matplotlib.pyplot as plt
import numpy as np

class LivePlotter:
    def __init__(self):
        plt.ion()
        self.fig, (self.r_ax, self.s_ax, self.succ_ax) = plt.subplots(1, 3, figsize=(15, 4))
        self.fig.suptitle("Training Metrics (Live)")
        self.r_ax.set_title("Rewards")
        self.s_ax.set_title("Steps")
        self.succ_ax.set_title("Success Rate")
        self.r_ax.set_xlabel("Episode")
        self.s_ax.set_xlabel("Episode")
        self.succ_ax.set_xlabel("Episode")
        self.r_ax.set_ylabel("Reward")
        self.s_ax.set_ylabel("Steps")
        self.succ_ax.set_ylabel("Success Rate")
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update(self, metrics):
        self.r_ax.clear()
        self.s_ax.clear()
        self.succ_ax.clear()
        self.r_ax.plot(metrics["rewards"], 'b-')
        self.s_ax.plot(metrics["steps"], 'g-')
        if metrics["successes"]:
            window = min(10, len(metrics["successes"]))
            rate = np.convolve(metrics["successes"], np.ones(window)/window, mode='valid')
            self.succ_ax.plot(rate, 'r-')
            self.succ_ax.set_ylim(-0.1, 1.1)
        self.r_ax.set_title("Rewards")
        self.s_ax.set_title("Steps")
        self.succ_ax.set_title("Success Rate")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
