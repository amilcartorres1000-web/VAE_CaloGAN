"""
Checkpoint management utilities.
"""
import torch
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Optional[object] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
):
    """
    Save a training checkpoint.
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        metrics: Dictionary of metrics
        config: Optional configuration object
        scheduler: Optional learning rate scheduler
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Create directory if needed
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load a training checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to
        
    Returns:
        Checkpoint dictionary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(path, map_location=device)
    
    # Load model
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from {path}")
    
    # Load optimizer
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Loaded optimizer state")
    
    # Load scheduler
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("Loaded scheduler state")
    
    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Find all checkpoint files
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    
    if not checkpoints:
        return None
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
    
    latest = str(checkpoints[-1])
    logger.info(f"Found latest checkpoint: {latest}")
    
    return latest