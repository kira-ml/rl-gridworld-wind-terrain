"""
Evaluation script for trained reinforcement learning agents.

This module provides utilities to load trained agents and evaluate their
performance in the GridWorld environment.
"""

import os
import sys
import argparse
import numpy as np
import torch
import time
import logging
from pathlib import Path

# Add project root to sys.path if running this script directly
if __name__ == "__main__":
    root_path = str(Path(__file__).resolve().parent.parent)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

# Import project modules
from envs.gridworld import GridWorldEnv
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from agents.sarsa import SarsaAgent
from agents.policy_iteration import PolicyIterationAgent
from config import DEFAULT_ENV_CONFIG
from utils.game_visual import GridWorldVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("evaluation")

def load_agent(agent_type, checkpoint_path, env):
    """
    Load a trained agent from checkpoint file.
    
    Args:
        agent_type: Type of agent ('q_learning', 'dqn', 'sarsa', 'policy_iteration')
        checkpoint_path: Path to the saved checkpoint
        env: GridWorld environment instance
        
    Returns:
        Loaded agent instance
    """
    logger.info(f"Loading {agent_type.upper()} agent from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path)
        model_state = checkpoint["model_state"]
        
        if agent_type == "q_learning":
            agent = QLearningAgent(env, {})
            agent.q_table = model_state["q_table"]
            agent.epsilon = 0.01  # Set to low value for evaluation
            agent.alpha = model_state.get("alpha", 0.1)
            agent.gamma = model_state.get("gamma", 0.99)
            
        elif agent_type == "dqn":
            # Import DQN agent config
            from config import DQN_AGENT_CONFIG
            agent = DQNAgent(env, DQN_AGENT_CONFIG)
            agent.online_model.load_state_dict(model_state)
            agent.target_model.load_state_dict(model_state)  # Also update target
            agent.epsilon = 0.01  # Set to low value for evaluation
            
        elif agent_type == "sarsa":
            agent = SarsaAgent(env, {})
            agent.q_table = model_state["q_table"]
            agent.epsilon = 0.01  # Set to low value for evaluation
            agent.alpha = model_state.get("alpha", 0.1)
            agent.gamma = model_state.get("gamma", 0.99)
            
        elif agent_type == "policy_iteration":
            agent = PolicyIterationAgent(env, {})
            agent.policy = model_state["policy"]
            agent.value_function = model_state["value_function"]
            
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        logger.info(f"Agent loaded successfully")
        return agent
        
    except Exception as e:
        logger.error(f"Error loading agent: {str(e)}")
        raise

def evaluate_agent(agent, env, num_episodes=10, render=True, delay=0.2):
    """
    Evaluate an agent's performance over multiple episodes.
    
    Args:
        agent: The agent to evaluate
        env: GridWorld environment instance
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment
        delay: Delay between steps when rendering (seconds)
        
    Returns:
        dict: Evaluation metrics
    """
    total_rewards = []
    total_steps = []
    success_rate = 0
    
    visualizer = None
    if render:
        visualizer = GridWorldVisualizer()
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done:
            # Get action from agent
            action = agent.act(obs)
            
            # Take action in environment
            next_obs, reward, done, _ = env.step(action)
            
            # Update stats
            ep_reward += reward
            steps += 1
            
            # Render if required
            if render and visualizer:
                visualizer.render(env, agent, episode=ep, step=steps, reward=ep_reward)
                time.sleep(delay)  # Add delay for visualization
            
            # Update observation
            obs = next_obs
        
        # Check if goal reached
        goal_pos = tuple(env.config["goal_pos"])
        success = tuple(env.agent_pos) == goal_pos
        
        # Update statistics
        total_rewards.append(ep_reward)
        total_steps.append(steps)
        success_rate += 1 if success else 0
        
        logger.info(f"Episode {ep+1}/{num_episodes}: Reward={ep_reward:.1f}, Steps={steps}, Success={'Yes' if success else 'No'}")
    
    # Calculate average metrics
    success_rate = success_rate / num_episodes * 100
    logger.info(f"\nEvaluation Results ({num_episodes} episodes):")
    logger.info(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    logger.info(f"Average Steps: {np.mean(total_steps):.2f} ± {np.std(total_steps):.2f}")
    logger.info(f"Success Rate: {success_rate:.1f}%")
    
    # Clean up
    if visualizer:
        visualizer.close()
    
    return {
        'avg_reward': np.mean(total_rewards),
        'avg_steps': np.mean(total_steps),
        'success_rate': success_rate
    }

def compare_agents(agents_dict, env, num_episodes=50):
    """
    Compare multiple agents on the same environment.
    
    Args:
        agents_dict: Dictionary of {agent_name: agent_instance}
        env: GridWorld environment instance
        num_episodes: Number of evaluation episodes per agent
        
    Returns:
        dict: Comparison results
    """
    results = {}
    
    for name, agent in agents_dict.items():
        logger.info(f"\nEvaluating {name}...")
        results[name] = evaluate_agent(agent, env, num_episodes, render=False)
    
    # Print comparison table
    logger.info("\n" + "="*50)
    logger.info("Agent Comparison Results")
    logger.info("="*50)
    logger.info(f"{'Agent':<15} {'Avg Reward':<15} {'Avg Steps':<15} {'Success Rate':<15}")
    logger.info("-"*50)
    
    for name, metrics in results.items():
        logger.info(f"{name:<15} {metrics['avg_reward']:<15.2f} {metrics['avg_steps']:<15.2f} {metrics['success_rate']:<15.1f}%")
    
    return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained reinforcement learning agents")
    
    parser.add_argument(
        "--agent", "-a",
        choices=["q_learning", "dqn", "sarsa", "policy_iteration"],
        required=True,
        help="Type of agent to evaluate"
    )
    
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Path to the checkpoint file"
    )
    
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    
    parser.add_argument(
        "--render", "-r",
        action="store_true",
        help="Enable visualization during evaluation"
    )
    
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=0.2,
        help="Delay between steps when rendering (seconds)"
    )
    
    return parser.parse_args()

def main():
    """Main function for evaluating agents from the command line."""
    args = parse_args()
    
    try:
        # Create environment
        env = GridWorldEnv(config=DEFAULT_ENV_CONFIG)
        
        # Load agent
        agent = load_agent(args.agent, args.checkpoint, env)
        
        # Evaluate agent
        evaluate_agent(
            agent, 
            env, 
            num_episodes=args.episodes, 
            render=args.render,
            delay=args.delay
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
