# Trained Models Usage Guide

This document explains how to load and use trained reinforcement learning models in the GridWorld environment.

## Loading Trained Models

### Using the Evaluation Script

The simplest way to load and evaluate a trained model is to use the provided `evaluate.py` script:

```bash
# Evaluate a trained Q-Learning agent
python src/evaluate.py --agent q_learning --checkpoint logs/checkpoints/q_learning/checkpoint_ep400_20250718_234343.pt --episodes 20 --render

# Evaluate with custom settings
python src/evaluate.py --agent q_learning --checkpoint logs/checkpoints/q_learning/checkpoint_ep400_20250718_234343.pt --episodes 50 --render --delay 0.1
```

### Loading Programmatically

You can also load and use trained models in your own Python code:

```python
import torch
from envs.gridworld import GridWorldEnv
from config import DEFAULT_ENV_CONFIG

# Q-Learning Agent
from agents.q_learning import QLearningAgent

# Create environment
env = GridWorldEnv(config=DEFAULT_ENV_CONFIG)

# Create agent
agent = QLearningAgent(env, {})

# Load checkpoint
checkpoint_path = "logs/checkpoints/q_learning/checkpoint_ep400_20250718_234343.pt"
checkpoint = torch.load(checkpoint_path)
model_state = checkpoint["model_state"]

# Option 1: Load components individually
agent.q_table = model_state["q_table"]
agent.epsilon = 0.01  # Low exploration for inference

# Option 2: Use built-in method
# agent.load_state_dict(model_state)

# Use the agent
obs = env.reset()
done = False
total_reward = 0

while not done:
    action = agent.act(obs)
    obs, reward, done, _ = env.step(action)
    total_reward += reward

print(f"Episode finished with reward: {total_reward}")
```

### For DQN Agent

```python
from agents.dqn import DQNAgent
from config import DQN_AGENT_CONFIG

# Create agent with appropriate architecture
agent = DQNAgent(env, DQN_AGENT_CONFIG)

# Load checkpoint
checkpoint = torch.load("logs/checkpoints/dqn/checkpoint_epXXX.pt")
agent.online_model.load_state_dict(checkpoint["model_state"])
agent.target_model.load_state_dict(checkpoint["model_state"])
agent.epsilon = 0.01  # Set exploration rate low for evaluation
```

## Comparing Multiple Agents

You can compare the performance of different trained agents using the evaluation utility:

```python
from src.evaluate import evaluate_agent, compare_agents

# Load your agents
q_learning_agent = load_q_learning("path/to/checkpoint.pt")
dqn_agent = load_dqn("path/to/dqn_checkpoint.pt")
sarsa_agent = load_sarsa("path/to/sarsa_checkpoint.pt")

# Create comparison dictionary
agents = {
    "Q-Learning": q_learning_agent,
    "DQN": dqn_agent,
    "SARSA": sarsa_agent
}

# Compare over 50 episodes
results = compare_agents(agents, env, num_episodes=50)

# The results variable contains metrics for each agent
```

## Visualization

For visualizing agent behavior:

```python
from utils.game_visual import GridWorldVisualizer
import time

visualizer = GridWorldVisualizer()

obs = env.reset()
done = False

while not done:
    action = agent.act(obs)
    obs, reward, done, _ = env.step(action)
    
    # Render the current state
    visualizer.render(env, agent, episode=0, step=steps, reward=total_reward)
    time.sleep(0.2)  # Add delay for visualization
```

## Checkpoint Format

Each checkpoint file (.pt) contains:

1. **model_state**: The agent's learned parameters
   - For Q-Learning/SARSA: Q-table and parameters
   - For DQN: Neural network weights
   - For Policy Iteration: Policy and value function

2. **training_info**: Training metadata
   - `episode`: Number of training episodes
   - `agent_type`: Type of agent
   - Other configuration parameters

3. **metrics**: Performance metrics
   - Rewards per episode
   - Steps per episode
   - Success rate

## Available Checkpoints

The following trained models are available:

- Q-Learning: `logs/checkpoints/q_learning/checkpoint_ep400_20250718_234343.pt`
- DQN: *To be trained*
- Policy Iteration: *To be trained*
- SARSA: *To be trained*
