"""
Run a visual test of the enhanced GridWorld visualization.

This script loads the textures, initializes the environment and 
runs the visualization with a pre-trained or random agent.
"""

import os
import sys
import time
from datetime import datetime
import pygame
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

from utils.metrics_saver import save_metrics_to_json
import random
import torch
import numpy as np
import argparse

# Add parent directory to path to ensure imports work properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from envs.gridworld import GridWorldEnv
from agents.q_learning import QLearningAgent
from agents.dqn import DQN
from agents.sarsa import SarsaAgent
from config import DEFAULT_ENV_CONFIG, QL_AGENT_CONFIG, DQN_AGENT_CONFIG, SARSA_AGENT_CONFIG
from utils.game_visual import GridWorldVisualizer
from utils.texture_generator import TextureGenerator
from utils.font_downloader import main as download_fonts


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GridWorld Visual Test")
    parser.add_argument("--agent", "-a", choices=["q_learning", "dqn", "sarsa", "random"], 
                        default="q_learning", help="Agent type to use")
    parser.add_argument("--episodes", "-e", type=int, default=5, 
                        help="Number of episodes to run")
    parser.add_argument("--train", "-t", action="store_true", 
                        help="Train the agent before visualization")
    parser.add_argument("--train-steps", "-ts", type=int, default=1000, 
                        help="Number of steps to train (if --train is specified)")
    parser.add_argument("--generate-assets", "-g", action="store_true", 
                        help="Force regeneration of textures and download of fonts")
    parser.add_argument("--window-size", "-w", type=str, default="1024x768", 
                        help="Window size in format WIDTHxHEIGHT")
    return parser.parse_args()


def ensure_assets(force_generate=False):
    """Ensure all assets (textures, fonts) are available."""
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check and create asset directories
    textures_dir = os.path.join(project_root, "assets", "textures")
    fonts_dir = os.path.join(project_root, "assets", "fonts")
    
    os.makedirs(textures_dir, exist_ok=True)
    os.makedirs(fonts_dir, exist_ok=True)
    
    # Generate textures if needed
    if force_generate or len(os.listdir(textures_dir)) == 0:
        print("Generating textures...")
        generator = TextureGenerator(textures_dir)
        generator.generate_all()
    
    # Download fonts if needed
    if force_generate or len(os.listdir(fonts_dir)) == 0:
        print("Downloading fonts...")
        download_fonts()


def train_agent(agent_type, env, num_episodes):
    """Train an agent for the specified number of episodes."""
    print(f"Training {agent_type} agent for {num_episodes} episodes...")
    
    # Create run directory for this training session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("logs", agent_type, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    if agent_type == "q_learning":
        agent = QLearningAgent(env, QL_AGENT_CONFIG)
    elif agent_type == "sarsa":
        agent = SarsaAgent(env, SARSA_AGENT_CONFIG)
    elif agent_type == "dqn":
        state_size = 2  # (row, col) for GridWorld
        action_size = env.action_space.n
        agent = DQN(state_size, action_size, DQN_AGENT_CONFIG)
    else:
        # Random agent doesn't need training
        return RandomAgent(env.action_space)
    
    # Train the agent
    agent.train(num_episodes)
    
    # Save metrics
    if hasattr(agent, 'get_metrics'):
        metrics = agent.get_metrics()
        metrics_file = save_metrics_to_json(
            metrics,
            agent_type,
            agent.config,
            run_dir
        )
        print(f"Metrics saved to: {metrics_file}")
        
    return agent
    if hasattr(agent, 'load'):
        try:
            agent.load(os.path.join("models", f"{agent_type}_latest.pt"))
            print(f"Loaded pre-trained {agent_type} agent")
        except:
            print(f"No pre-trained model found for {agent_type}")
            
        # Take step in environment
        next_state, reward, done, _ = env.step(action)
        
        # Update agent
        if agent_type == "q_learning":
            agent.update(state, action, reward, next_state)
        elif agent_type == "sarsa":
            next_action = agent.act(next_state) if not done else None
            agent.update(state, action, reward, next_state, next_action)
        elif agent_type == "dqn":
            # Simple DQN update (would normally use a replay buffer)
            agent.update(state, action, reward, next_state, done)
            
        state = next_state
        
        if done:
            state = env.reset()
            
        # Print progress
        if step % 100 == 0:
            print(f"Training step {step}/{train_steps}")
            
    return agent


class RandomAgent:
    """A simple random agent for testing."""
    
    def __init__(self, action_space):
        self.action_space = action_space
        # Initialize dummy Q values for visualization
        self.Q = np.random.uniform(-1, 1, (7, 7, 4))  # Assuming 7x7 grid with 4 actions
        
    def act(self, state):
        return self.action_space.sample()


def main():
    """Main function."""
    args = parse_args()
    
    # Parse window size
    try:
        width, height = map(int, args.window_size.split('x'))
        window_size = (width, height)
    except:
        print("Invalid window size format. Using default 1024x768.")
        window_size = (1024, 768)
    
    # Ensure assets are available
    ensure_assets(args.generate_assets)
    
    # Create environment
    env = GridWorldEnv(DEFAULT_ENV_CONFIG)
    
    # Create or train agent
    if args.agent == "random":
        agent = RandomAgent(env.action_space)
    elif args.train:
        agent = train_agent(args.agent, env, args.train_steps)
    else:
        # Create pre-initialized agent
        if args.agent == "q_learning":
            agent = QLearningAgent(env.observation_space, env.action_space,
                                 env.config["grid_size"], QL_AGENT_CONFIG)
        elif args.agent == "sarsa":
            agent = SarsaAgent(env.observation_space, env.action_space,
                             env.config["grid_size"], SARSA_AGENT_CONFIG)
        elif args.agent == "dqn":
            state_size = 2  # (row, col) for GridWorld
            action_size = env.action_space.n
            agent = DQN(state_size, action_size, DQN_AGENT_CONFIG["hidden_dim"])
        else:
            agent = RandomAgent(env.action_space)
    
    # Create visualizer and run simulation
    print(f"Starting visualization with {args.agent} agent...")
    vis = GridWorldVisualizer(window_size=window_size)
    vis.simulate(env, agent, num_episodes=args.episodes)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting due to user interrupt")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Make sure PyGame quits properly
        import pygame
        pygame.quit()
