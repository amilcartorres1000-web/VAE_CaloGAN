"""
Loss functions for VAE training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class SparseVAELoss(nn.Module):
    """
    Custom loss for sparse calorimeter data.
    
    Components:
    1. Reconstruction loss (weighted BCE)
    2. Sparsity regularization
    3. KL divergence (with annealing)
    4. Energy conservation loss
    """
    
    def __init__(
        self,
        recon_weight: float = 1.0,
        sparsity_weight: float = 5.0,
        energy_weight: float = 0.5,
        kl_annealing: bool = True,
        kl_warmup_epochs: int = 20,
        kl_max_weight: float = 0.1,
        nonzero_weight: float = 50.0
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.sparsity_weight = sparsity_weight
        self.energy_weight = energy_weight
        self.kl_annealing = kl_annealing
        self.kl_warmup_epochs = kl_warmup_epochs
        self.kl_max_weight = kl_max_weight
        self.nonzero_weight = nonzero_weight
    
    def _get_kl_weight(self, epoch: int) -> float:
        """Compute KL weight with annealing schedule"""
        if not self.kl_annealing:
            return self.kl_max_weight
        
        if epoch < self.kl_warmup_epochs:
            # Linear warmup
            return self.kl_max_weight * (epoch / self.kl_warmup_epochs)
        else:
            # Gradual increase after warmup
            extra = min((epoch - self.kl_warmup_epochs) * 0.002, self.kl_max_weight)
            return min(self.kl_max_weight + extra, 1.0)
    
    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss.
        
        Args:
            recon: Reconstructed output (B, 3, H, W)
            target: Target input (B, 3, H, W)
            mu: Latent mean (B, z_dim)
            logvar: Latent log variance (B, z_dim)
            epoch: Current epoch (for KL annealing)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        batch_size = target.size(0)
        
        # 1. Reconstruction loss with adaptive weighting
        # Use MSE instead of BCE since decoder uses softplus (not sigmoid)
        weights = torch.ones_like(target)
        non_zero_mask = target > 1e-6
        weights[non_zero_mask] = self.nonzero_weight
        
        # Weighted MSE loss
        recon_loss = F.mse_loss(recon, target, reduction='none')
        recon_loss = (recon_loss * weights).mean()
        recon_loss = recon_loss * self.recon_weight
        
        # 2. Sparsity regularization (encourage zeros)
        sparsity_loss = torch.mean(torch.abs(recon)) * self.sparsity_weight
        
        # 3. KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_weight = self._get_kl_weight(epoch)
        kl_loss_weighted = kl_loss * kl_weight
        
        # 4. Energy conservation loss
        E_target = target.sum(dim=(1, 2, 3))
        E_recon = recon.sum(dim=(1, 2, 3))
        energy_loss = F.mse_loss(E_recon, E_target) * self.energy_weight
        
        # Total loss
        total_loss = recon_loss + sparsity_loss + kl_loss_weighted + energy_loss
        
        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'sparsity': sparsity_loss.item(),
            'kl': kl_loss.item(),
            'kl_weighted': kl_loss_weighted.item(),
            'energy': energy_loss.item(),
            'kl_weight': kl_weight
        }
        
        return total_loss, loss_dict


class BetaVAELoss(nn.Module):
    """
    Beta-VAE loss (simpler alternative).
    
    L = Recon + beta * KL
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        epoch: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute Beta-VAE loss"""
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, target, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item()
        }
        
        return total_loss, loss_dict


def get_loss_fn(loss_type: str, config) -> nn.Module:
    """
    Factory function for loss functions.
    
    Args:
        loss_type: Type of loss ('sparse_vae', 'beta_vae')
        config: LossConfig instance
        
    Returns:
        Loss function module
    """
    if loss_type == 'sparse_vae':
        return SparseVAELoss(
            recon_weight=config.recon_weight,
            sparsity_weight=config.sparsity_weight,
            energy_weight=config.energy_weight,
            kl_annealing=config.kl_annealing,
            kl_warmup_epochs=config.kl_warmup_epochs,
            kl_max_weight=config.kl_max_weight,
            nonzero_weight=config.nonzero_weight
        )
    elif loss_type == 'beta_vae':
        return BetaVAELoss(beta=config.kl_max_weight)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")