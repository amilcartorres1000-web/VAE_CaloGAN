"""
Visualization functions for calorimeter data and training results.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 10)
):
    """
    Plot training history curves.
    
    Args:
        history: Dictionary mapping metric names to lists of values
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Plot loss components
    loss_keys = [k for k in history.keys() if 'loss' in k.lower() and 'val' not in k]
    if loss_keys and len(axes) > 0:
        ax = axes[0]
        for key in loss_keys:
            ax.plot(history[key], label=key, alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot validation metrics
    val_keys = [k for k in history.keys() if 'val' in k and k != 'val_loss']
    if val_keys and len(axes) > 1:
        ax = axes[1]
        for key in val_keys[:3]:  # Limit to 3 metrics
            ax.plot(history[key], label=key.replace('val_', ''), alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.set_title('Validation Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot learning rate
    if 'lr' in history and len(axes) > 2:
        ax = axes[2]
        ax.plot(history['lr'], color='orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # Plot KS statistic
    if 'val_ks_stat' in history and len(axes) > 3:
        ax = axes[3]
        ax.plot(history['val_ks_stat'], color='green', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KS Statistic')
        ax.set_title('KS Test Statistic')
        ax.grid(True, alpha=0.3)
    
    # Plot gradient norm
    if 'grad_norm' in history and len(axes) > 4:
        ax = axes[4]
        ax.plot(history['grad_norm'], color='red', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm')
        ax.grid(True, alpha=0.3)
    
    # Plot energy comparison
    if 'val_real_energy_mean' in history and 'val_gen_energy_mean' in history and len(axes) > 5:
        ax = axes[5]
        ax.plot(history['val_real_energy_mean'], label='Real', linewidth=2)
        ax.plot(history['val_gen_energy_mean'], label='Generated', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Energy')
        ax.set_title('Energy Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
    
    plt.close()


def plot_energy_distributions(
    real_data: np.ndarray,
    gen_data: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5)
):
    """
    Plot energy distribution comparison.
    
    Args:
        real_data: Real calorimeter data (N, 3, H, W)
        gen_data: Generated calorimeter data (N, 3, H, W)
        save_path: Optional path to save figure
        figsize: Figure size
    """
    # Compute energies
    real_energy = real_data.sum(axis=(1, 2, 3))
    gen_energy = gen_data.sum(axis=(1, 2, 3))
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax = axes[0]
    bins = 50
    ax.hist(real_energy, bins=bins, alpha=0.7, label='Real', density=True, histtype='step', linewidth=2)
    ax.hist(gen_energy, bins=bins, alpha=0.7, label='Generated', density=True, histtype='step', linewidth=2)
    ax.set_xlabel('Total Energy')
    ax.set_ylabel('Probability Density')
    ax.set_title('Energy Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # CDF
    ax = axes[1]
    sorted_real = np.sort(real_energy)
    sorted_gen = np.sort(gen_energy)
    ax.plot(sorted_real, np.linspace(0, 1, len(sorted_real)), label='Real', linewidth=2)
    ax.plot(sorted_gen, np.linspace(0, 1, len(sorted_gen)), label='Generated', linewidth=2)
    ax.set_xlabel('Total Energy')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved energy distribution plot to {save_path}")
    
    plt.close()


def plot_sample_comparison(
    real_data: np.ndarray,
    gen_data: np.ndarray,
    n_samples: int = 4,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 12)
):
    """
    Plot side-by-side comparison of real and generated samples.
    
    Args:
        real_data: Real calorimeter data (N, 3, H, W)
        gen_data: Generated calorimeter data (N, 3, H, W)
        n_samples: Number of samples to plot
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(3, n_samples, figsize=figsize)
    
    for i in range(n_samples):
        # Real sample (layer 0)
        axes[0, i].imshow(np.log1p(real_data[i, 0]), cmap='hot', origin='lower')
        axes[0, i].set_title(f'Real Sample {i+1}', fontsize=10)
        axes[0, i].axis('off')
        
        # Generated sample (layer 0)
        axes[1, i].imshow(np.log1p(gen_data[i, 0]), cmap='hot', origin='lower')
        axes[1, i].set_title(f'Generated Sample {i+1}', fontsize=10)
        axes[1, i].axis('off')
        
        # Projection comparison
        axes[2, i].plot(real_data[i, 0].sum(axis=0), label='Real', alpha=0.7)
        axes[2, i].plot(gen_data[i, 0].sum(axis=0), label='Generated', alpha=0.7)
        axes[2, i].set_title(f'X Projection {i+1}', fontsize=10)
        axes[2, i].legend(fontsize=8)
        axes[2, i].grid(True, alpha=0.3)
    
    plt.suptitle('Sample Comparisons (Layer 0)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved sample comparison plot to {save_path}")
    
    plt.close()


def plot_layer_energies(
    real_data: np.ndarray,
    gen_data: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4)
):
    """
    Plot per-layer energy distributions.
    
    Args:
        real_data: Real calorimeter data (N, 3, H, W)
        gen_data: Generated calorimeter data (N, 3, H, W)
        save_path: Optional path to save figure
        figsize: Figure size
    """
    n_layers = real_data.shape[1]
    fig, axes = plt.subplots(1, n_layers, figsize=figsize)
    
    if n_layers == 1:
        axes = [axes]
    
    for i in range(n_layers):
        real_layer_energy = real_data[:, i, :, :].sum(axis=(1, 2))
        gen_layer_energy = gen_data[:, i, :, :].sum(axis=(1, 2))
        
        axes[i].hist(real_layer_energy, bins=40, alpha=0.7, label='Real', density=True)
        axes[i].hist(gen_layer_energy, bins=40, alpha=0.7, label='Generated', density=True)
        axes[i].set_xlabel('Layer Energy')
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'Layer {i}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved layer energy plot to {save_path}")
    
    plt.close()


def create_training_report(
    history: Dict[str, List[float]],
    real_data: np.ndarray,
    gen_data: np.ndarray,
    output_dir: str
):
    """
    Create a comprehensive training report with all plots.
    
    Args:
        history: Training history dictionary
        real_data: Real calorimeter data
        gen_data: Generated calorimeter data
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating training report...")
    
    # Training history
    plot_training_history(
        history,
        save_path=str(output_dir / 'training_history.png')
    )
    
    # Energy distributions
    plot_energy_distributions(
        real_data,
        gen_data,
        save_path=str(output_dir / 'energy_distributions.png')
    )
    
    # Sample comparisons
    plot_sample_comparison(
        real_data,
        gen_data,
        n_samples=4,
        save_path=str(output_dir / 'sample_comparisons.png')
    )
    
    # Layer energies
    plot_layer_energies(
        real_data,
        gen_data,
        save_path=str(output_dir / 'layer_energies.png')
    )
    
    logger.info(f"Training report saved to {output_dir}")