"""
Evaluation pipeline for RL agents.
Provides standardized evaluation procedures and metrics collection.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import asdict

from .metrics import (
    EpisodeMetrics,
    EvaluationMetrics,
    compute_episode_metrics,
    compute_evaluation_metrics
)

class AgentEvaluator:
    """Handles evaluation of trained RL agents."""
    
    def __init__(
        self,
        env,
        agent,
        eval_episodes: int = 100,
        seed: Optional[int] = None,
        save_dir: Optional[str] = None,
        optimal_path_length: Optional[int] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            env: OpenAI Gym-style environment
            agent: RL agent with act() method
            eval_episodes: Number of episodes to evaluate
            seed: Random seed for reproducibility
            save_dir: Directory to save evaluation results
            optimal_path_length: Known optimal path length (if available)
        """
        self.env = env
        self.agent = agent
        self.eval_episodes = eval_episodes
        self.optimal_path_length = optimal_path_length
        self.save_dir = save_dir or "evaluation_results"
        
        if seed is not None:
            np.random.seed(seed)
            env.seed(seed)
        
        os.makedirs(self.save_dir, exist_ok=True)
    
    def evaluate_episode(self) -> Tuple[EpisodeMetrics, List[Dict]]:
        """Run a single evaluation episode."""
        state = self.env.reset()
        done = False
        rewards = []
        trajectory = []
        q_values = []
        
        while not done:
            # Get action from agent
            action = self.agent.act(state)
            
            # Store Q-values if available
            if hasattr(self.agent, "Q"):
                q_values.append(np.max(self.agent.Q[state[0], state[1]]))
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store step information
            rewards.append(reward)
            trajectory.append({
                "state": state.tolist() if isinstance(state, np.ndarray) else state,
                "action": action,
                "reward": reward,
                "next_state": next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
                "done": done
            })
            
            state = next_state
        
        # Determine if episode was successful
        success = False
        if hasattr(self.env, "is_successful"):
            success = self.env.is_successful()
        else:
            # Assume success if agent reached goal state in GridWorld
            success = np.array_equal(state, self.env.config.get("goal_pos", (self.env.grid_height-1, self.env.grid_width-1)))
        
        # Get final epsilon if available
        epsilon = getattr(self.agent, "epsilon", None)
        
        metrics = compute_episode_metrics(
            rewards=rewards,
            episode_length=len(trajectory),
            success=success,
            q_values=q_values if q_values else None,
            epsilon=epsilon
        )
        
        return metrics, trajectory
    
    def evaluate(self) -> EvaluationMetrics:
        """Run full evaluation across multiple episodes."""
        episode_metrics = []
        all_trajectories = []
        
        for episode in range(self.eval_episodes):
            metrics, trajectory = self.evaluate_episode()
            episode_metrics.append(metrics)
            all_trajectories.append(trajectory)
        
        # Compute aggregated metrics
        eval_metrics = compute_evaluation_metrics(
            episode_metrics,
            optimal_path_length=self.optimal_path_length
        )
        
        # Save results
        self._save_results(eval_metrics, episode_metrics, all_trajectories)
        
        return eval_metrics
    
    def _save_results(
        self,
        eval_metrics: EvaluationMetrics,
        episode_metrics: List[EpisodeMetrics],
        trajectories: List[List[Dict]]
    ):
        """Save evaluation results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results dictionary
        results = {
            "timestamp": timestamp,
            "num_episodes": self.eval_episodes,
            "aggregated_metrics": asdict(eval_metrics),
            "episode_metrics": [asdict(m) for m in episode_metrics],
            "config": {
                "env": self.env.config if hasattr(self.env, "config") else {},
                "agent": self.agent.config if hasattr(self.agent, "config") else {}
            }
        }
        
        # Save main results
        results_file = os.path.join(self.save_dir, f"eval_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save trajectories separately (can be large)
        traj_file = os.path.join(self.save_dir, f"trajectories_{timestamp}.json")
        with open(traj_file, 'w') as f:
            json.dump(trajectories, f)
