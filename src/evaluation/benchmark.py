"""
Benchmarking utilities for comparing multiple RL agents.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

from .evaluate import AgentEvaluator
from .metrics import EvaluationMetrics, format_metrics

class AgentBenchmark:
    """Benchmark multiple agents on the same environment."""
    
    def __init__(
        self,
        env_creator,
        agent_dict: Dict[str, Any],
        eval_episodes: int = 100,
        seeds: Optional[List[int]] = None,
        save_dir: Optional[str] = None
    ):
        """
        Initialize benchmarking suite.
        
        Args:
            env_creator: Function that creates a new environment instance
            agent_dict: Dictionary mapping agent names to agent instances
            eval_episodes: Number of episodes for evaluation
            seeds: List of random seeds for multiple trials
            save_dir: Directory to save benchmark results
        """
        self.env_creator = env_creator
        self.agent_dict = agent_dict
        self.eval_episodes = eval_episodes
        self.seeds = seeds or [None]
        self.save_dir = save_dir or "benchmark_results"
        
        os.makedirs(self.save_dir, exist_ok=True)
    
    def run_benchmark(self) -> Dict[str, List[EvaluationMetrics]]:
        """Run benchmark for all agents across all seeds."""
        results = defaultdict(list)
        
        for agent_name, agent in self.agent_dict.items():
            print(f"\nEvaluating {agent_name}...")
            
            for seed in self.seeds:
                # Create fresh environment for each trial
                env = self.env_creator()
                
                # Create evaluator
                evaluator = AgentEvaluator(
                    env=env,
                    agent=agent,
                    eval_episodes=self.eval_episodes,
                    seed=seed,
                    save_dir=os.path.join(self.save_dir, agent_name)
                )
                
                # Run evaluation
                metrics = evaluator.evaluate()
                results[agent_name].append(metrics)
                
                print(f"Seed {seed if seed else 'None'} completed")
                print(format_metrics(metrics))
        
        self._save_benchmark_results(results)
        self._plot_comparisons(results)
        
        return results
    
    def _save_benchmark_results(self, results: Dict[str, List[EvaluationMetrics]]):
        """Save benchmark results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to serializable format
        serializable_results = {
            agent_name: [vars(m) for m in metrics_list]
            for agent_name, metrics_list in results.items()
        }
        
        # Add metadata
        benchmark_data = {
            "timestamp": timestamp,
            "num_episodes": self.eval_episodes,
            "seeds": self.seeds,
            "results": serializable_results
        }
        
        # Save to file
        results_file = os.path.join(self.save_dir, f"benchmark_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
    
    def _plot_comparisons(self, results: Dict[str, List[EvaluationMetrics]]):
        """Generate comparison plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prepare data
        agents = list(results.keys())
        metrics_data = {
            'reward': ([m.mean_reward for m in results[a]] for a in agents),
            'success': ([m.success_rate for m in results[a]] for a in agents),
            'steps': ([m.mean_episode_length for m in results[a]] for a in agents),
            'efficiency': ([m.optimal_path_ratio for m in results[a]] for a in agents)
        }
        
        # Plot reward comparison
        ax1.boxplot([list(d) for d in metrics_data['reward']], labels=agents)
        ax1.set_title('Mean Reward Distribution')
        ax1.set_ylabel('Mean Reward')
        
        # Plot success rate comparison
        ax2.boxplot([list(d) for d in metrics_data['success']], labels=agents)
        ax2.set_title('Success Rate Distribution')
        ax2.set_ylabel('Success Rate')
        
        # Plot episode length comparison
        ax3.boxplot([list(d) for d in metrics_data['steps']], labels=agents)
        ax3.set_title('Episode Length Distribution')
        ax3.set_ylabel('Mean Episode Length')
        
        # Plot efficiency comparison (if available)
        if any(any(r is not None for r in d) for d in metrics_data['efficiency']):
            ax4.boxplot([list(d) for d in metrics_data['efficiency']], labels=agents)
            ax4.set_title('Path Efficiency Distribution')
            ax4.set_ylabel('Optimal Path Ratio')
        else:
            ax4.set_visible(False)
        
        # Adjust layout and save
        plt.tight_layout()
        plot_file = os.path.join(self.save_dir, f"benchmark_comparison_{timestamp}.png")
        plt.savefig(plot_file)
        plt.close()

def load_benchmark_results(results_file: str) -> Dict[str, List[EvaluationMetrics]]:
    """Load benchmark results from a file."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = {}
    for agent_name, metrics_list in data["results"].items():
        results[agent_name] = [
            EvaluationMetrics(**metrics_dict)
            for metrics_dict in metrics_list
        ]
    
    return results
