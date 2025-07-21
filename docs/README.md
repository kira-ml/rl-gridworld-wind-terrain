# GridWorld RL Documentation

This directory contains documentation for the GridWorld Reinforcement Learning project.

## Contents

- [Model Usage Guide](./model_usage_guide.md): How to load and use trained models
- [Configuration Reference](../config/hyperparameters.yaml): Complete hyperparameter reference

## Agent Implementations

The project includes the following agent implementations:

### Q-Learning Agent
A tabular reinforcement learning algorithm that learns state-action values through experience.

- Implementation: `src/agents/q_learning.py`
- Configuration: `config/hyperparameters.yaml` (q_learning section)
- Checkpoint format: PyTorch .pt file containing Q-table and parameters

### Deep Q-Network (DQN) Agent
A deep reinforcement learning algorithm using neural networks to approximate Q-values.

- Implementation: `src/agents/dqn.py`
- Configuration: `config/hyperparameters.yaml` (dqn section)
- Checkpoint format: PyTorch .pt file containing neural network weights

### SARSA Agent
A tabular on-policy reinforcement learning algorithm.

- Implementation: `src/agents/sarsa.py`
- Configuration: `config/hyperparameters.yaml` (sarsa section)
- Checkpoint format: PyTorch .pt file containing Q-table and parameters

### Policy Iteration Agent
A model-based dynamic programming approach to find optimal policies.

- Implementation: `src/agents/policy_iteration.py`
- Configuration: `config/hyperparameters.yaml` (policy_iteration section)
- Checkpoint format: PyTorch .pt file containing policy and value function

## Environment

The GridWorld environment features:
- Variable grid size
- Customizable start and goal positions
- Wind zones that push the agent
- Different terrain types (ice, mud, quicksand)
- Negative rewards for steps, collisions
- Positive reward for reaching the goal

Configuration is available in `config/hyperparameters.yaml` under the environment section.

## Training and Evaluation

- Training: `python src/main.py --agent q_learning --episodes 400`
- Evaluation: `python src/evaluate.py --agent q_learning --checkpoint path/to/checkpoint.pt`
