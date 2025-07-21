"""
Filesystem-based experiment tracking system.
Stores experiment data in structured JSON files and provides comparison tools.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class ExperimentMetadata:
    """Metadata for an experiment run."""
    experiment_id: str
    agent_type: str
    start_time: str
    end_time: Optional[str] = None
    status: str = "running"
    git_commit: Optional[str] = None
    config: Dict = None

@dataclass
class ExperimentMetrics:
    """Metrics collected during an experiment."""
    rewards: List[float]
    steps: List[int]
    success_rates: List[float]
    final_performance: Dict[str, float]

class ExperimentTracker:
    """Local filesystem-based experiment tracker."""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_experiment: Optional[str] = None
    
    def start_experiment(
        self,
        agent_type: str,
        config: Dict[str, Any]
    ) -> str:
        """Start a new experiment and return its ID."""
        # Create unique experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{agent_type}_{timestamp}"
        
        # Create experiment directory
        exp_dir = self.base_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Store metadata
        metadata = ExperimentMetadata(
            experiment_id=exp_id,
            agent_type=agent_type,
            start_time=timestamp,
            config=config
        )
        
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)
        
        self.current_experiment = exp_id
        return exp_id
    
    def log_metrics(
        self,
        metrics: Dict[str, List[float]],
        step: int
    ):
        """Log metrics for current experiment."""
        if not self.current_experiment:
            raise ValueError("No active experiment")
        
        exp_dir = self.base_dir / self.current_experiment
        metrics_file = exp_dir / "metrics.csv"
        
        # Convert metrics to row format
        row = {"step": step}
        row.update({k: v[-1] if isinstance(v, list) else v 
                   for k, v in metrics.items()})
        
        # Write to CSV
        write_header = not metrics_file.exists()
        with open(metrics_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    
    def end_experiment(
        self,
        final_metrics: Dict[str, float]
    ):
        """Mark experiment as completed and store final metrics."""
        if not self.current_experiment:
            raise ValueError("No active experiment")
        
        exp_dir = self.base_dir / self.current_experiment
        
        # Update metadata
        with open(exp_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        metadata["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata["status"] = "completed"
        metadata["final_metrics"] = final_metrics
        
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.current_experiment = None
    
    def compare_experiments(
        self,
        experiment_ids: Optional[List[str]] = None,
        agent_type: Optional[str] = None,
        metric_names: Optional[List[str]] = None,
        last_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments and return analysis.
        Can filter by agent type and select specific metrics.
        """
        results = {}
        
        # Get list of experiments to compare
        if experiment_ids:
            exp_dirs = [self.base_dir / exp_id for exp_id in experiment_ids]
        else:
            exp_dirs = list(self.base_dir.glob("*"))
            if agent_type:
                exp_dirs = [d for d in exp_dirs if agent_type in d.name]
            if last_n:
                exp_dirs = sorted(exp_dirs, key=lambda x: x.stat().st_mtime)[-last_n:]
        
        for exp_dir in exp_dirs:
            if not exp_dir.is_dir():
                continue
            
            try:
                # Load metadata
                with open(exp_dir / "metadata.json", "r") as f:
                    metadata = json.load(f)
                
                # Load metrics
                metrics_data = []
                with open(exp_dir / "metrics.csv", "r") as f:
                    reader = csv.DictReader(f)
                    metrics_data = list(reader)
                
                # Calculate summary statistics
                metrics_summary = {}
                if metric_names:
                    for metric in metric_names:
                        values = [float(row[metric]) for row in metrics_data 
                                if metric in row]
                        if values:
                            metrics_summary[metric] = {
                                "mean": np.mean(values),
                                "std": np.std(values),
                                "min": np.min(values),
                                "max": np.max(values),
                                "final": values[-1]
                            }
                
                results[exp_dir.name] = {
                    "metadata": metadata,
                    "metrics_summary": metrics_summary
                }
                
            except Exception as e:
                print(f"Error processing experiment {exp_dir.name}: {e}")
                continue
        
        return results

    def get_best_experiments(
        self,
        metric_name: str,
        agent_type: Optional[str] = None,
        n_best: int = 5,
        maximize: bool = True
    ) -> List[Dict[str, Any]]:
        """Get the N best experiments based on a specific metric."""
        all_experiments = self.compare_experiments(
            agent_type=agent_type,
            metric_names=[metric_name]
        )
        
        # Sort experiments by metric
        sorted_experiments = sorted(
            all_experiments.items(),
            key=lambda x: x[1]["metrics_summary"][metric_name]["final"],
            reverse=maximize
        )
        
        return [{"id": exp_id, **data} 
                for exp_id, data in sorted_experiments[:n_best]]
