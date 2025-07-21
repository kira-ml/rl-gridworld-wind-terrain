"""
Metrics computation utilities for RL agent evaluation.
Provides standardized metric calculations for comparing agent performance.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class EpisodeMetrics:
    """Container for metrics from a single episode."""
    total_reward: float
    episode_length: int
    success: bool
    average_q_value: Optional[float] = None
    min_q_value: Optional[float] = None
    max_q_value: Optional[float] = None
    final_epsilon: Optional[float] = None

@dataclass
class EvaluationMetrics:
    """Container for aggregated metrics across multiple episodes."""
    # Reward statistics
    mean_reward: float
    median_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    
    # Episode length statistics
    mean_episode_length: float
    median_episode_length: float
    std_episode_length: float
    
    # Success metrics
    success_rate: float
    
    # Q-value statistics (if available)
    mean_q_value: Optional[float] = None
    std_q_value: Optional[float] = None
    
    # Additional metrics
    mean_steps_to_goal: Optional[float] = None
    optimal_path_ratio: Optional[float] = None

def compute_episode_metrics(
    rewards: List[float],
    episode_length: int,
    success: bool,
    q_values: Optional[List[float]] = None,
    epsilon: Optional[float] = None
) -> EpisodeMetrics:
    """Compute metrics for a single episode."""
    total_reward = sum(rewards)
    
    # Q-value statistics if available
    avg_q = min_q = max_q = None
    if q_values:
        avg_q = np.mean(q_values)
        min_q = np.min(q_values)
        max_q = np.max(q_values)
    
    return EpisodeMetrics(
        total_reward=total_reward,
        episode_length=episode_length,
        success=success,
        average_q_value=avg_q,
        min_q_value=min_q,
        max_q_value=max_q,
        final_epsilon=epsilon
    )

def compute_evaluation_metrics(
    episode_metrics: List[EpisodeMetrics],
    optimal_path_length: Optional[int] = None
) -> EvaluationMetrics:
    """Compute aggregated metrics across multiple episodes."""
    rewards = [m.total_reward for m in episode_metrics]
    lengths = [m.episode_length for m in episode_metrics]
    successes = [m.success for m in episode_metrics]
    
    # Basic statistics
    eval_metrics = EvaluationMetrics(
        mean_reward=np.mean(rewards),
        median_reward=np.median(rewards),
        std_reward=np.std(rewards),
        min_reward=np.min(rewards),
        max_reward=np.max(rewards),
        mean_episode_length=np.mean(lengths),
        median_episode_length=np.median(lengths),
        std_episode_length=np.std(lengths),
        success_rate=np.mean(successes)
    )
    
    # Compute steps to goal for successful episodes
    successful_lengths = [l for l, s in zip(lengths, successes) if s]
    if successful_lengths:
        eval_metrics.mean_steps_to_goal = np.mean(successful_lengths)
    
    # Compute optimal path ratio if optimal length is provided
    if optimal_path_length is not None and successful_lengths:
        optimal_ratios = [optimal_path_length / l for l in successful_lengths]
        eval_metrics.optimal_path_ratio = np.mean(optimal_ratios)
    
    # Compute Q-value statistics if available
    q_values = [m.average_q_value for m in episode_metrics if m.average_q_value is not None]
    if q_values:
        eval_metrics.mean_q_value = np.mean(q_values)
        eval_metrics.std_q_value = np.std(q_values)
    
    return eval_metrics

def format_metrics(metrics: EvaluationMetrics) -> str:
    """Format evaluation metrics as a human-readable string."""
    lines = [
        "=== Evaluation Metrics ===",
        f"Rewards: mean={metrics.mean_reward:.2f} ± {metrics.std_reward:.2f}",
        f"        median={metrics.median_reward:.2f} [{metrics.min_reward:.2f}, {metrics.max_reward:.2f}]",
        f"Episode Length: mean={metrics.mean_episode_length:.2f} ± {metrics.std_episode_length:.2f}",
        f"Success Rate: {metrics.success_rate*100:.1f}%"
    ]
    
    if metrics.mean_steps_to_goal is not None:
        lines.append(f"Mean Steps to Goal: {metrics.mean_steps_to_goal:.2f}")
    
    if metrics.optimal_path_ratio is not None:
        lines.append(f"Optimal Path Ratio: {metrics.optimal_path_ratio:.2f}")
    
    if metrics.mean_q_value is not None:
        lines.append(f"Q-Values: mean={metrics.mean_q_value:.2f} ± {metrics.std_q_value:.2f}")
    
    return "\n".join(lines)
