"""
Training callbacks for modular control flow.
"""
from abc import ABC, abstractmethod
from pathlib import Path
import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TrainingCallback(ABC):
    """Base class for training callbacks"""
    
    def on_train_start(self, trainer: 'Trainer') -> None:
        """Called at the start of training"""
        pass
    
    def on_train_end(self, trainer: 'Trainer') -> None:
        """Called at the end of training"""
        pass
    
    def on_epoch_start(self, epoch: int, trainer: 'Trainer') -> None:
        """Called at the start of each epoch"""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: 'Trainer') -> None:
        """Called at the end of each epoch"""
        pass
    
    def on_batch_end(self, batch_idx: int, loss: float, trainer: 'Trainer') -> None:
        """Called at the end of each batch"""
        pass


class EarlyStoppingCallback(TrainingCallback):
    """
    Early stopping callback.
    
    Stops training if monitored metric doesn't improve for `patience` epochs.
    """
    
    def __init__(
        self,
        patience: int = 10,
        metric: str = 'val_loss',
        mode: str = 'min'
    ):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: 'Trainer') -> None:
        """Check if should stop training"""
        if self.metric not in metrics:
            logger.warning(f"Metric '{self.metric}' not found in metrics")
            return
        
        current_value = metrics[self.metric]
        
        # Check if improved
        if self.mode == 'min':
            improved = current_value < self.best_value
        else:
            improved = current_value > self.best_value
        
        if improved:
            self.best_value = current_value
            self.counter = 0
            logger.info(f"Metric improved: {self.metric}={current_value:.6f}")
        else:
            self.counter += 1
            logger.info(f"No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                trainer.stop_training = True
                self.stopped_epoch = epoch


class CheckpointCallback(TrainingCallback):
    """
    Checkpointing callback.
    
    Saves model checkpoints at regular intervals and/or when metric improves.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_interval: int = 5,
        save_best_only: bool = False,
        metric: str = 'val_loss',
        mode: str = 'min'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.save_best_only = save_best_only
        self.metric = metric
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: 'Trainer') -> None:
        """Save checkpoint if needed"""
        should_save = False
        is_best = False
        
        # Check if should save by interval
        if not self.save_best_only and epoch % self.save_interval == 0:
            should_save = True
        
        # Check if best model
        if self.metric in metrics:
            current_value = metrics[self.metric]
            
            if self.mode == 'min':
                is_best = current_value < self.best_value
            else:
                is_best = current_value > self.best_value
            
            if is_best:
                self.best_value = current_value
                should_save = True
        
        if should_save:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'metrics': metrics,
                'config': trainer.config
            }
            
            if trainer.scheduler is not None:
                checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
            
            # Save checkpoint
            if is_best:
                path = self.checkpoint_dir / 'best_model.pth'
                torch.save(checkpoint, path)
                logger.info(f"Saved best model (epoch {epoch})")
            
            if not self.save_best_only:
                path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
                torch.save(checkpoint, path)
                logger.info(f"Saved checkpoint (epoch {epoch})")


class MetricsLoggingCallback(TrainingCallback):
    """
    Logging callback for metrics.
    
    Can integrate with various logging backends (TensorBoard, W&B, etc.)
    """
    
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.batch_losses = []
    
    def on_batch_end(self, batch_idx: int, loss: float, trainer: 'Trainer') -> None:
        """Log batch loss"""
        self.batch_losses.append(loss)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: 'Trainer') -> None:
        """Log epoch metrics"""
        # Compute average batch loss
        if self.batch_losses:
            avg_batch_loss = sum(self.batch_losses) / len(self.batch_losses)
            self.batch_losses = []
        
        # Format metrics for display
        metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        logger.info(f"Epoch {epoch:03d} | {metrics_str}")