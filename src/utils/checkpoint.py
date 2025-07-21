"""
Robust checkpointing system for model and experiment state management.
"""

import os
import json
import shutil
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
from dataclasses import asdict, is_dataclass

class CheckpointManager:
    """Manages model checkpoints with versioning and retention policies."""
    
    def __init__(
        self,
        base_dir: str,
        experiment_name: str,
        keep_best_k: int = 5,
        keep_last_k: int = 2,
        metric_name: str = "reward",
        metric_goal: str = "max"
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            base_dir: Base directory for checkpoints
            experiment_name: Name of the experiment
            keep_best_k: Number of best checkpoints to keep
            keep_last_k: Number of most recent checkpoints to keep
            metric_name: Metric to use for keeping best checkpoints
            metric_goal: Whether to maximize or minimize the metric
        """
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        self.keep_best_k = keep_best_k
        self.keep_last_k = keep_last_k
        self.metric_name = metric_name
        self.metric_goal = metric_goal
        
        # Create checkpoint directory
        self.checkpoint_dir = self.base_dir / "checkpoints" / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint registry
        self.registry_file = self.checkpoint_dir / "checkpoint_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load or create checkpoint registry."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {
            "checkpoints": [],
            "best_metric": None if self.metric_goal == "max" else float('inf')
        }
    
    def _save_registry(self):
        """Save checkpoint registry."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _is_better_metric(self, new_value: float, old_value: Optional[float]) -> bool:
        """Check if new metric value is better than old value."""
        if old_value is None:
            return True
        return new_value > old_value if self.metric_goal == "max" else new_value < old_value
    
    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        metrics: Dict[str, float],
        episode: int,
        is_best: bool = False
    ) -> str:
        """
        Save a checkpoint with all relevant information.
        
        Args:
            state_dict: Model and training state
            metrics: Current evaluation metrics
            episode: Current episode number
            is_best: Whether this is the best model so far
            
        Returns:
            str: Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_ep{episode}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Convert dataclass instances to dictionaries
        processed_state = {
            k: asdict(v) if is_dataclass(v) else v
            for k, v in state_dict.items()
        }
        
        # Save checkpoint
        checkpoint = {
            "state_dict": processed_state,
            "metrics": metrics,
            "episode": episode,
            "timestamp": timestamp
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Update registry
        metric_value = metrics.get(self.metric_name)
        registry_entry = {
            "path": str(checkpoint_path),
            "episode": episode,
            "metrics": metrics,
            "timestamp": timestamp,
            "is_best": is_best
        }
        self.registry["checkpoints"].append(registry_entry)
        
        # Update best metric if applicable
        if metric_value is not None and self._is_better_metric(
            metric_value, self.registry["best_metric"]
        ):
            self.registry["best_metric"] = metric_value
            registry_entry["is_best"] = True
        
        self._save_registry()
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        identifier: Optional[Union[str, int]] = None,
        load_best: bool = False
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            identifier: Checkpoint path, episode number, or None for latest
            load_best: Whether to load the best checkpoint
            
        Returns:
            dict: Loaded checkpoint
        """
        if not self.registry["checkpoints"]:
            raise ValueError("No checkpoints found")
        
        # Determine which checkpoint to load
        if load_best:
            checkpoint_info = max(
                self.registry["checkpoints"],
                key=lambda x: x["metrics"].get(self.metric_name, float('-inf'))
                if self.metric_goal == "max"
                else -x["metrics"].get(self.metric_name, float('inf'))
            )
        elif isinstance(identifier, int):
            # Load by episode number
            matches = [c for c in self.registry["checkpoints"] if c["episode"] == identifier]
            if not matches:
                raise ValueError(f"No checkpoint found for episode {identifier}")
            checkpoint_info = matches[-1]
        elif isinstance(identifier, str):
            # Load by path
            matches = [c for c in self.registry["checkpoints"] if c["path"] == identifier]
            if not matches:
                raise ValueError(f"Checkpoint not found: {identifier}")
            checkpoint_info = matches[0]
        else:
            # Load latest
            checkpoint_info = self.registry["checkpoints"][-1]
        
        # Load checkpoint file
        checkpoint = torch.load(checkpoint_info["path"])
        return checkpoint
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints based on retention policy."""
        if not self.registry["checkpoints"]:
            return
        
        # Sort checkpoints by metric value and timestamp
        checkpoints = self.registry["checkpoints"]
        checkpoints.sort(
            key=lambda x: (
                x["metrics"].get(self.metric_name, float('-inf')),
                x["timestamp"]
            ),
            reverse=self.metric_goal == "max"
        )
        
        # Determine which checkpoints to keep
        keep_indices = set()
        
        # Keep best K
        for idx in range(min(self.keep_best_k, len(checkpoints))):
            keep_indices.add(idx)
        
        # Keep last K
        for idx in range(max(0, len(checkpoints) - self.keep_last_k), len(checkpoints)):
            keep_indices.add(idx)
        
        # Remove checkpoints not in keep set
        for idx in range(len(checkpoints) - 1, -1, -1):
            if idx not in keep_indices:
                checkpoint = checkpoints[idx]
                try:
                    os.remove(checkpoint["path"])
                    checkpoints.pop(idx)
                except OSError as e:
                    print(f"Error removing checkpoint: {e}")
        
        self._save_registry()
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get metrics from the best checkpoint."""
        if not self.registry["checkpoints"]:
            return {}
            
        best_checkpoint = max(
            self.registry["checkpoints"],
            key=lambda x: x["metrics"].get(self.metric_name, float('-inf'))
            if self.metric_goal == "max"
            else -x["metrics"].get(self.metric_name, float('inf'))
        )
        return best_checkpoint["metrics"]
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Get list of all checkpoint information."""
        return self.registry["checkpoints"]
