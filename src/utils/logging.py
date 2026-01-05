"""
Performance logging utilities for RootCNN v2.
Provides structured logging for training and inference operations.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class PerformanceLogger:
    """
    Logger for tracking performance metrics during training and inference.
    Logs are saved as JSON files with structured data for dashboard visualization.
    """
    
    def __init__(self, log_file: str, log_type: str = "generic"):
        """
        Initialize the performance logger.
        
        Args:
            log_file: Path to the output log file (JSON)
            log_type: Type of logging - 'training', 'detection', 'tracking', or 'generic'
        """
        self.log_file = Path(log_file)
        self.log_type = log_type
        self.entries: List[Dict[str, Any]] = []
        self.start_time = time.time()
        
        # Create parent directory if it doesn't exist
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
    def log_training_epoch(self, epoch: int, train_loss: float, val_loss: float,
                          train_accuracy: Optional[float] = None,
                          val_accuracy: Optional[float] = None,
                          epoch_time: Optional[float] = None):
        """
        Log metrics for a training epoch.
        
        Args:
            epoch: Epoch number (1-indexed)
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch
            train_accuracy: Optional training accuracy
            val_accuracy: Optional validation accuracy
            epoch_time: Optional time taken for this epoch in seconds
        """
        entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        }
        
        if train_accuracy is not None:
            entry["train_accuracy"] = float(train_accuracy)
        if val_accuracy is not None:
            entry["val_accuracy"] = float(val_accuracy)
        if epoch_time is not None:
            entry["epoch_time_seconds"] = float(epoch_time)
            
        self.entries.append(entry)
        
    def log_image_processing(self, image_name: str, num_tips: int,
                            processing_time: float,
                            image_timestamp: Optional[str] = None,
                            num_associations: Optional[int] = None,
                            additional_metrics: Optional[Dict[str, Any]] = None):
        """
        Log metrics for processing a single image.
        
        Args:
            image_name: Name/path of the processed image
            num_tips: Number of tips detected/tracked
            processing_time: Time taken to process this image in seconds
            image_timestamp: Optional timestamp extracted from image filename
            num_associations: Optional number of successful tip associations
            additional_metrics: Optional dictionary of additional metrics to log
        """
        entry = {
            "image_name": image_name,
            "timestamp": datetime.now().isoformat(),
            "num_tips": int(num_tips),
            "processing_time_seconds": float(processing_time),
        }
        
        if image_timestamp is not None:
            entry["image_timestamp"] = image_timestamp
        if num_associations is not None:
            entry["num_associations"] = int(num_associations)
        if additional_metrics:
            entry.update(additional_metrics)
            
        self.entries.append(entry)
        
    def log_custom(self, **kwargs):
        """
        Log custom metrics as key-value pairs.
        
        Args:
            **kwargs: Arbitrary key-value pairs to log
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.entries.append(entry)
        
    def save(self):
        """
        Save all logged entries to the log file as JSON.
        """
        total_time = time.time() - self.start_time
        
        log_data = {
            "log_type": self.log_type,
            "created_at": datetime.now().isoformat(),
            "total_duration_seconds": float(total_time),
            "num_entries": len(self.entries),
            "entries": self.entries
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
            
        print(f"Performance log saved to {self.log_file} ({len(self.entries)} entries)")
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically save on exit."""
        self.save()
        return False


def calculate_pixel_accuracy(pred: 'torch.Tensor', target: 'torch.Tensor', 
                             threshold: float = 0.5) -> float:
    """
    Calculate pixel-wise binary accuracy for heatmap predictions.
    
    Args:
        pred: Predicted heatmap (after sigmoid), shape (B, 1, H, W)
        target: Target heatmap, shape (B, 1, H, W)
        threshold: Threshold for binarizing predictions
        
    Returns:
        Pixel-wise accuracy as a float between 0 and 1
    """
    import torch
    
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    correct = (pred_binary == target_binary).float()
    accuracy = correct.mean().item()
    
    return accuracy
