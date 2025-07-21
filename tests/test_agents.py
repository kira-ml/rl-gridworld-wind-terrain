"""
Unit tests for agent implementations.
"""
import pytest
import numpy as np
from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent

def test_q_learning_initialization(test_env, test_q_learning_agent):
    """Test Q-learning agent initialization."""
    assert test_q_learning_agent.env == test_env
    assert test_q_learning_agent.alpha == 0.1
    assert test_q_learning_agent.gamma == 0.99
    assert test_q_learning_agent.epsilon == 1.0

def test_q_learning_act(test_q_learning_agent):
    """Test Q-learning agent action selection."""
    state = (0, 0)
    action = test_q_learning_agent.act(state)
    assert 0 <= action < 4  # Valid action

def test_q_learning_update(test_q_learning_agent):
    """Test Q-learning update rule."""
    state = (0, 0)
    action = 1  # Right
    next_state = (0, 1)
    reward = -1
    
    # Get initial Q-value
    initial_q = test_q_learning_agent.q_table[state][action]
    
    # Perform update
    test_q_learning_agent.update(state, action, reward, next_state)
    
    # Check Q-value changed
    assert test_q_learning_agent.q_table[state][action] != initial_q

def test_dqn_initialization(test_env, test_dqn_agent):
    """Test DQN agent initialization."""
    assert test_dqn_agent.env == test_env
    assert len(test_dqn_agent.replay_buffer) == 0
    assert test_dqn_agent.epsilon == 1.0

def test_dqn_act(test_dqn_agent):
    """Test DQN agent action selection."""
    state = np.zeros(test_dqn_agent.state_dim)
    action = test_dqn_agent.act(state)
    assert 0 <= action < 4  # Valid action

@pytest.mark.slow
def test_dqn_learning(test_env, test_dqn_agent):
    """Test DQN learning over a few episodes."""
    n_episodes = 5
    max_steps = 10
    
    for _ in range(n_episodes):
        state = test_env.reset()
        for _ in range(max_steps):
            action = test_dqn_agent.act(state)
            next_state, reward, done, _ = test_env.step(action)
            test_dqn_agent.update(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
    
    # Check that replay buffer contains experiences
    assert len(test_dqn_agent.replay_buffer) > 0
