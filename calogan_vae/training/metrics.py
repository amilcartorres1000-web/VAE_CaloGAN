"""
Validation metrics for calorimeter data.
"""
import torch
import numpy as np
from scipy.stats import ks_2samp
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_energy_distribution(data: torch.Tensor) -> np.ndarray:
    """
    Compute total energy for each event.
    
    Args:
        data: Tensor of shape (N, 3, H, W)
        
    Returns:
        Array of total energies (N,)
    """
    return data.sum(dim=(1, 2, 3)).cpu().numpy()


def compute_ks_statistic(real_data: torch.Tensor, gen_data: torch.Tensor) -> Tuple[float, float]:
    """
    Compute Kolmogorov-Smirnov test statistic for energy distributions.
    
    Args:
        real_data: Real calorimeter data (N, 3, H, W)
        gen_data: Generated calorimeter data (N, 3, H, W)
        
    Returns:
        statistic: KS test statistic
        pvalue: P-value of the test
    """
    real_energy = compute_energy_distribution(real_data)
    gen_energy = compute_energy_distribution(gen_data)
    
    if len(real_energy) == 0 or len(gen_energy) == 0:
        logger.warning("Empty energy distribution, returning KS=1.0")
        return 1.0, 0.0
    
    statistic, pvalue = ks_2samp(real_energy, gen_energy)
    return float(statistic), float(pvalue)


def compute_sparsity(data: torch.Tensor, threshold: float = 1e-6) -> float:
    """
    Compute fraction of near-zero values.
    
    Args:
        data: Tensor of shape (N, 3, H, W)
        threshold: Values below this are considered zero
        
    Returns:
        Sparsity fraction (0 to 1)
    """
    return float((data < threshold).float().mean())


def compute_energy_stats(data: torch.Tensor) -> Dict[str, float]:
    """
    Compute energy distribution statistics.
    
    Args:
        data: Tensor of shape (N, 3, H, W)
        
    Returns:
        Dictionary of statistics
    """
    energies = compute_energy_distribution(data)
    
    return {
        'energy_mean': float(energies.mean()),
        'energy_std': float(energies.std()),
        'energy_min': float(energies.min()),
        'energy_max': float(energies.max()),
        'energy_median': float(np.median(energies))
    }


def compute_layer_energies(data: torch.Tensor) -> Dict[str, float]:
    """
    Compute per-layer energy statistics.
    
    Args:
        data: Tensor of shape (N, 3, H, W)
        
    Returns:
        Dictionary of per-layer statistics
    """
    layer_stats = {}
    
    for i in range(data.size(1)):
        layer_data = data[:, i, :, :]
        layer_energy = layer_data.sum(dim=(1, 2)).cpu().numpy()
        
        layer_stats[f'layer_{i}_mean'] = float(layer_energy.mean())
        layer_stats[f'layer_{i}_std'] = float(layer_energy.std())
    
    return layer_stats


class MetricsCalculator:
    """
    Unified metrics calculator for validation.
    """
    
    def __init__(
        self,
        compute_ks: bool = True,
        compute_energy_stats: bool = True,
        compute_sparsity: bool = True,
        compute_layer_stats: bool = False
    ):
        self.compute_ks = compute_ks
        self.compute_energy_stats_flag = compute_energy_stats
        self.compute_sparsity_flag = compute_sparsity
        self.compute_layer_stats = compute_layer_stats
    
    def calculate(
        self,
        real_data: torch.Tensor,
        gen_data: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate all requested metrics.
        
        Args:
            real_data: Real data (N, 3, H, W)
            gen_data: Generated data (N, 3, H, W)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # KS test
        if self.compute_ks:
            ks_stat, ks_pval = compute_ks_statistic(real_data, gen_data)
            metrics['ks_stat'] = ks_stat
            metrics['ks_pval'] = ks_pval
        
        # Energy statistics
        if self.compute_energy_stats_flag:
            real_stats = compute_energy_stats(real_data)
            gen_stats = compute_energy_stats(gen_data)
            
            for key, value in real_stats.items():
                metrics[f'real_{key}'] = value
            for key, value in gen_stats.items():
                metrics[f'gen_{key}'] = value
        
        # Sparsity
        if self.compute_sparsity_flag:
            metrics['real_sparsity'] = compute_sparsity(real_data)
            metrics['gen_sparsity'] = compute_sparsity(gen_data)
        
        # Layer statistics
        if self.compute_layer_stats:
            real_layer_stats = compute_layer_energies(real_data)
            gen_layer_stats = compute_layer_energies(gen_data)
            
            for key, value in real_layer_stats.items():
                metrics[f'real_{key}'] = value
            for key, value in gen_layer_stats.items():
                metrics[f'gen_{key}'] = value
        
        return metrics