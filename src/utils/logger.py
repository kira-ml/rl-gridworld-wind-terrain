"""
Structured logging system for reinforcement learning experiments.
Supports file, console, and optional experiment tracking backends.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import defaultdict

class ExperimentLogger:
    """Unified logging interface for RL experiments."""
    
    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "logs",
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        console_level: str = "INFO"
    ):
        # Set UTF-8 encoding for logging to handle special characters
        import sys
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        """
        Initialize logger with multiple backends.
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for logs
            use_tensorboard: Whether to log to TensorBoard
            use_wandb: Whether to log to Weights & Biases
            console_level: Logging level for console output
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = defaultdict(list)
        
        # Create experiment directory
        self.exp_dir = Path(base_dir) / f"{experiment_name}_{self.timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file and console logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler (DEBUG level)
        fh = logging.FileHandler(self.exp_dir / "experiment.log")
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
        fh.setFormatter(fh_formatter)
        self.logger.addHandler(fh)
        
        # Console handler (configurable level)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, console_level.upper()))
        ch_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        ch.setFormatter(ch_formatter)
        self.logger.addHandler(ch)
        
        # Initialize experiment trackers
        self._setup_tensorboard() if use_tensorboard else None
        self._setup_wandb() if use_wandb else None
        
        # Create metrics file
        self.metrics_file = self.exp_dir / "metrics.jsonl"
        
        self.logger.info(f"Experiment {experiment_name} initialized")
        self.logger.info(f"Logging to {self.exp_dir}")
    
    # Delegate standard logging methods to the internal logger
    def debug(self, msg, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(msg, *args, **kwargs)
    
    def _setup_tensorboard(self):
        """Initialize TensorBoard writer if available."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(self.exp_dir / "tensorboard")
            self.logger.info("TensorBoard logging enabled")
        except ImportError:
            self.logger.warning("TensorBoard not available")
            self.tb_writer = None
    
    def _setup_wandb(self):
        """Initialize Weights & Biases if available."""
        try:
            import wandb
            wandb.init(
                project=self.experiment_name,
                name=f"run_{self.timestamp}",
                dir=self.exp_dir
            )
            self.wandb = wandb
            self.logger.info("Weights & Biases logging enabled")
        except ImportError:
            self.logger.warning("Weights & Biases not available")
            self.wandb = None
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to all active backends."""
        # Log to file
        self.logger.info(f"Hyperparameters: {json.dumps(params, indent=2)}")
        
        # Log to experiment trackers
        if hasattr(self, 'tb_writer') and self.tb_writer:
            self.tb_writer.add_hparams(params, {})
        
        if hasattr(self, 'wandb') and self.wandb:
            self.wandb.config.update(params)
        
        # Save to separate file
        with open(self.exp_dir / "hyperparameters.json", 'w') as f:
            json.dump(params, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics to all active backends."""
        # Add prefix to metric names if provided
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()} if prefix else metrics
        
        # Store metrics
        for name, value in metrics.items():
            self.metrics[name].append(value)
        
        # Log to experiment trackers
        if hasattr(self, 'tb_writer') and self.tb_writer:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, value, step)
        
        if hasattr(self, 'wandb') and self.wandb:
            self.wandb.log(metrics, step=step)
        
        # Append to metrics file
        with open(self.metrics_file, 'a') as f:
            record = {"step": step, "timestamp": datetime.now().isoformat(), **metrics}
            f.write(json.dumps(record) + "\n")
    
    def log_episode(self, episode_idx: int, metrics: Dict[str, float]):
        """Log episode-specific metrics."""
        self.log_metrics(metrics, step=episode_idx, prefix="episode/")
        
        # Log episode summary
        summary = " | ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
        self.logger.info(f"Episode {episode_idx}: {summary}")
    
    def get_metric_statistics(self, metric_name: str, window: int = 100) -> Dict[str, float]:
        """Compute statistics for a metric over recent episodes."""
        values = self.metrics[metric_name][-window:]
        if not values:
            return {}
            
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values))
        }
    
    def save_metrics_summary(self):
        """Save summary statistics for all metrics."""
        summary = {}
        for metric_name in self.metrics:
            summary[metric_name] = self.get_metric_statistics(metric_name)
        
        with open(self.exp_dir / "metrics_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def close(self):
        """Clean up and close all logging backends."""
        self.save_metrics_summary()
        
        if hasattr(self, 'tb_writer') and self.tb_writer:
            self.tb_writer.close()
        
        if hasattr(self, 'wandb') and self.wandb:
            self.wandb.finish()
        
        self.logger.info("Experiment completed")
