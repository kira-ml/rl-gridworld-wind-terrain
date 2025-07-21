import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def save_metrics_to_json(metrics: Dict[str, Any], agent_type: str, config: Dict[str, Any],
                        run_dir: str) -> str:
    """
    Save training metrics to a JSON file with proper formatting and numpy handling.
    
    Args:
        metrics: Dictionary containing all metrics from training
        agent_type: Type of agent (e.g., 'q_learning', 'dqn', 'sarsa')
        config: Agent configuration dictionary
        run_dir: Directory to save the metrics file
        
    Returns:
        str: Path to the saved metrics file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create metrics dictionary with metadata
    metrics_data = {
        "timestamp": timestamp,
        "agent_type": agent_type,
        "config": config,
        "episodes_trained": metrics.get("total_episodes", 0),
        "total_steps": metrics.get("total_steps", 0),
        "final_metrics": {
            "avg_reward_last_10": float(np.mean(metrics["current_metrics"]["cumulative_rewards"][-10:])),
            "avg_steps_last_10": float(np.mean(metrics["current_metrics"]["episode_lengths"][-10:])),
            "success_rate_last_10": float(np.mean(metrics["current_metrics"]["success_rates"][-10:])) * 100,
            "final_epsilon": float(metrics["current_metrics"]["epsilon_values"][-1]),
            "final_learning_rate": float(metrics["current_metrics"]["learning_rates"][-1])
        },
        "training_history": {
            "cumulative_rewards": metrics["current_metrics"]["cumulative_rewards"],
            "episode_lengths": metrics["current_metrics"]["episode_lengths"],
            "success_rates": metrics["current_metrics"]["success_rates"],
            "epsilon_values": metrics["current_metrics"]["epsilon_values"],
            "learning_rates": metrics["current_metrics"]["learning_rates"],
            "average_q_values": metrics["current_metrics"]["average_q_values"],
            "episode_max_q_values": metrics["current_metrics"]["episode_max_q_values"]
        }
    }
    
    # Save to file
    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, cls=NumpyEncoder, indent=2)
        
    return metrics_file
