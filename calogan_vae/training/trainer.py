"""
Main training loop with clean separation of concerns.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Optional
import logging
import gc

from .callbacks import TrainingCallback
from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles VAE training with modular callbacks.
    
    Responsibilities:
    - Execute training epochs
    - Manage optimizer and scheduler
    - Call callbacks at appropriate times
    - Handle errors gracefully
    - Generate validation samples
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        config,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        metrics_calculator: Optional[MetricsCalculator] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        
        self.device = torch.device(config.training.device)
        self.model.to(self.device)
        
        self.stop_training = False
        self.current_epoch = 0
        self.metrics_history = defaultdict(list)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of average metrics for the epoch
        """
        self.model.train()
        epoch_metrics = defaultdict(float)
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch:03d}')
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            
            try:
                # Forward pass
                recon, mu, logvar = self.model(batch)
                loss, loss_dict = self.loss_fn(recon, batch, mu, logvar, self.current_epoch)
                
                # Check for NaN/Inf
                if not self._is_valid_loss(loss):
                    logger.warning(f"Invalid loss at batch {batch_idx}: {loss.item()}")
                    self._handle_invalid_loss(batch_idx, loss)
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip
                )
                
                # Check for invalid gradients
                if not torch.isfinite(grad_norm):
                    logger.warning(f"Invalid gradients at batch {batch_idx}")
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue
                
                self.optimizer.step()
                
                # Accumulate metrics
                for key, value in loss_dict.items():
                    epoch_metrics[key] += value
                epoch_metrics['grad_norm'] += grad_norm.item()
                n_batches += 1
                
                # Callbacks
                for callback in self.callbacks:
                    callback.on_batch_end(batch_idx, loss.item(), self)
                
                # Update progress bar
                if batch_idx % self.config.training.log_interval == 0:
                    pbar.set_postfix(self._format_metrics(loss_dict, grad_norm.item()))
                
                # Periodic cleanup
                if batch_idx % 50 == 0:
                    del recon, mu, logvar, loss
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"OOM at batch {batch_idx}, skipping")
                    self._handle_oom()
                    continue
                else:
                    raise e
        
        # Average metrics
        if n_batches > 0:
            epoch_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
        
        return dict(epoch_metrics)
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation with BOTH reconstruction AND generation.
        """
        self.model.eval()
        val_loss = 0.0
        n_batches = 0
        
        # Collect real and RECONSTRUCTED data
        real_data_list = []
        recon_data_list = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                batch = batch.to(self.device)
                
                # Forward pass (RECONSTRUCTION)
                recon, mu, logvar = self.model(batch)
                loss, loss_dict = self.loss_fn(recon, batch, mu, logvar, self.current_epoch)
                
                val_loss += loss.item()
                n_batches += 1
                
                # Collect samples for metrics
                if len(real_data_list) < self.config.validation.num_samples // batch.size(0):
                    real_data_list.append(batch.cpu())
                    recon_data_list.append(recon.cpu())
        
        # Average loss
        val_metrics = {'val_loss': val_loss / n_batches if n_batches > 0 else 0.0}
        
        # Compute reconstruction metrics
        if real_data_list:
            real_data = torch.cat(real_data_list, dim=0)[:self.config.validation.num_samples]
            recon_data = torch.cat(recon_data_list, dim=0)[:self.config.validation.num_samples]
            
            recon_metrics = self.metrics_calculator.calculate(real_data, recon_data)
            val_metrics.update({f'val_recon_{k}': v for k, v in recon_metrics.items()})
        
        # **NEW: GENERATION VALIDATION**
        # Generate from random noise
        with torch.no_grad():
            z = torch.randn(
                self.config.validation.num_samples, 
                self.model.z_dim, 
                device=self.device
            )
            gen_data = self.model.decode(z).cpu()
        
        # Compute generation metrics
        gen_metrics = self.metrics_calculator.calculate(real_data, gen_data)
        val_metrics.update({f'val_gen_{k}': v for k, v in gen_metrics.items()})
        
        # Log both
        logger.info(f"Reconstruction KS: {val_metrics.get('val_recon_ks_stat', 0):.4f}")
        logger.info(f"Generation KS: {val_metrics.get('val_gen_ks_stat', 0):.4f}")
        
        return val_metrics
    
    def fit(self):
        """
        Main training loop.
        """
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.config.training.epochs}")
        logger.info(f"Batch size: {self.config.data.batch_size}")
        
        # Callbacks: train start
        for callback in self.callbacks:
            callback.on_train_start(self)
        
        try:
            for epoch in range(1, self.config.training.epochs + 1):
                self.current_epoch = epoch
                    # Callbacks: epoch start
                for callback in self.callbacks:
                    callback.on_epoch_start(epoch, self)
                
                # Train
                train_metrics = self.train_epoch()
                
                # Validate
                if epoch % self.config.validation.val_interval == 0:
                    val_metrics = self.validate()
                else:
                    val_metrics = {}
                
                # Combine metrics
                metrics = {**train_metrics, **val_metrics}
                
                # Store history
                for key, value in metrics.items():
                    self.metrics_history[key].append(value)
                
                # Learning rate scheduling
                if self.scheduler is not None and 'val_loss' in metrics:
                    self.scheduler.step(metrics['val_loss'])
                    metrics['lr'] = self.optimizer.param_groups[0]['lr']
                
                # Callbacks: epoch end
                for callback in self.callbacks:
                    callback.on_epoch_end(epoch, metrics, self)
                
                # Check early stopping
                if self.stop_training:
                    logger.info(f"Training stopped at epoch {epoch}")
                    break
                
        except KeyboardInterrupt:
            logger.info("WARNING: Training interrupted by user")
        except Exception as e:
            logger.error(f"ERROR: Training failed: {e}", exc_info=True)
            raise
        finally:
            # Callbacks: train end
            for callback in self.callbacks:
                callback.on_train_end(self)
            
            logger.info("Training complete!")

    def _is_valid_loss(self, loss: torch.Tensor) -> bool:
        """Check if loss is valid (not NaN/Inf)"""
        return torch.isfinite(loss).all()

    def _handle_invalid_loss(self, batch_idx: int, loss: torch.Tensor):
        """Handle invalid loss"""
        logger.error(f"Invalid loss at batch {batch_idx}: {loss.item()}")
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        gc.collect()

    def _handle_oom(self):
        """Handle out-of-memory error"""
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        gc.collect()

    def _format_metrics(self, loss_dict: Dict, grad_norm: float) -> Dict[str, str]:
        """Format metrics for progress bar"""
        formatted = {k: f'{v:.4f}' for k, v in loss_dict.items() if k != 'kl_weight'}
        formatted['grad'] = f'{grad_norm:.2f}'
        
        # Add memory usage if CUDA
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            formatted['mem'] = f'{mem_alloc:.1f}G'
        
        return formatted