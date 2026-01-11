"""
Improved evaluation with smart sparsification.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from scipy.ndimage import label

from calogan_vae.config import ExperimentConfig
from calogan_vae.data import CaloDataset, get_preprocessor
from calogan_vae.models import build_vae_from_config
from calogan_vae.training.metrics import MetricsCalculator
from calogan_vae.visualization import (
    plot_energy_distributions,
    plot_sample_comparison,
    plot_layer_energies
)
from calogan_vae.utils import setup_logging, get_logger

logger = get_logger(__name__)


def sparsify_energy_preserving(
    samples: np.ndarray,
    target_energy_fraction: float = 0.98,
    min_sparsity: float = 0.85
) -> np.ndarray:
    """
    Sparsify while preserving total energy.
    
    Keep values that contribute to target_energy_fraction of total energy.
    
    Args:
        samples: Generated samples (N, C, H, W)
        target_energy_fraction: Fraction of energy to preserve (0.98 = 98%)
        min_sparsity: Minimum sparsity to achieve
        
    Returns:
        Sparsified samples
    """
    batch_size = samples.shape[0]
    samples_sparse = np.zeros_like(samples)
    
    for i in range(batch_size):
        sample = samples[i]  # (C, H, W)
        sample_flat = sample.flatten()
        
        # Sort by value (descending)
        sorted_indices = np.argsort(sample_flat)[::-1]
        sorted_values = sample_flat[sorted_indices]
        
        # Cumulative energy
        cumsum = np.cumsum(sorted_values)
        total_energy = cumsum[-1]
        
        # Find cutoff index for target energy
        cutoff_idx = np.searchsorted(cumsum, target_energy_fraction * total_energy)
        
        # Ensure minimum sparsity
        max_keep = int(sample.size * (1 - min_sparsity))
        cutoff_idx = min(cutoff_idx, max_keep)
        
        # Keep top values
        keep_indices = sorted_indices[:cutoff_idx]
        
        sparse_flat = np.zeros_like(sample_flat)
        sparse_flat[keep_indices] = sample_flat[keep_indices]
        
        samples_sparse[i] = sparse_flat.reshape(sample.shape)
    
    return samples_sparse


def sparsify_per_layer(
    samples: np.ndarray,
    real_samples: np.ndarray
) -> np.ndarray:
    """
    Apply layer-specific sparsification based on real data statistics.
    """
    # Compute sparsity per layer from real data
    layer_sparsities = []
    for i in range(real_samples.shape[1]):
        sparsity = (real_samples[:, i] < 1e-6).mean()
        layer_sparsities.append(sparsity)
    
    logger.info(f"Target layer sparsities: {[f'{s:.3f}' for s in layer_sparsities]}")
    
    samples_sparse = np.zeros_like(samples)
    
    for layer_idx in range(samples.shape[1]):
        target_sparsity = layer_sparsities[layer_idx]
        
        for sample_idx in range(samples.shape[0]):
            sample = samples[sample_idx, layer_idx]  # (H, W)
            sample_flat = sample.flatten()
            
            n_total = len(sample_flat)
            n_keep = max(1, int(n_total * (1 - target_sparsity)))
            
            # Keep top n_keep values
            if n_keep < n_total:
                top_indices = np.argpartition(sample_flat, -n_keep)[-n_keep:]
                sparse_flat = np.zeros_like(sample_flat)
                sparse_flat[top_indices] = sample_flat[top_indices]
                samples_sparse[sample_idx, layer_idx] = sparse_flat.reshape(sample.shape)
            else:
                samples_sparse[sample_idx, layer_idx] = sample
    
    return samples_sparse


def main(args):
    """Improved evaluation"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"eval_improved_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'evaluation.log'
    setup_logging(log_file=str(log_file), level='INFO')
    
    logger.info("="*60)
    logger.info("IMPROVED VAE EVALUATION")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Sparsification: {args.sparsify_method}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', ExperimentConfig())
    
    # Build model
    model = build_vae_from_config(config.encoder, config.decoder)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    # CRITICAL FIX: Force BatchNorm to eval mode
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = True
            module.eval()
    
    logger.info("Model in eval mode with BatchNorm frozen")
    logger.info(f"Loaded from epoch {checkpoint.get('epoch', '?')}")
    
    # Load data
    preprocessor = get_preprocessor(config.data.preprocessing, percentile=config.data.percentile)
    
    dataset = CaloDataset(
        h5_path=args.h5_path or config.data.h5_path,
        layer_keys=config.data.layer_keys,
        max_events=args.max_events,
        loading_mode='memory',
        preprocessor=None
    )
    
    # Fit preprocessor
    sample_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    sample_data = []
    for i, batch in enumerate(sample_loader):
        sample_data.append(batch.numpy())
        if i >= 10:
            break
    sample_data = np.concatenate(sample_data, axis=0)
    preprocessor.fit(sample_data)
    dataset.preprocessor = preprocessor
    
    # Get real data
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    real_data = []
    for batch in dataloader:
        real_data.append(batch.numpy())
        if len(real_data) * args.batch_size >= args.num_samples:
            break
    real_data = np.concatenate(real_data, axis=0)[:args.num_samples]
    
    # Generate with FIXED random seed for reproducibility
    logger.info(f"Generating {args.num_samples} samples (seed={args.seed})...")
    torch.manual_seed(args.seed)
    with torch.no_grad():
        z = torch.randn(args.num_samples, model.z_dim, device=device)
        gen_data = model.decode(z).cpu().numpy()
    
    logger.info(f"Generated data stats:")
    logger.info(f"  Mean: {gen_data.mean():.4f}")
    logger.info(f"  Std: {gen_data.std():.4f}")
    logger.info(f"  Min: {gen_data.min():.4f}")
    logger.info(f"  Max: {gen_data.max():.4f}")
    
    # Sparsify
    logger.info(f"Applying {args.sparsify_method} sparsification...")
    
    if args.sparsify_method == 'energy':
        gen_data_sparse = sparsify_energy_preserving(
            gen_data,
            target_energy_fraction=0.98,
            min_sparsity=0.85
        )
    elif args.sparsify_method == 'layer':
        gen_data_sparse = sparsify_per_layer(gen_data, real_data)
    else:  # 'global'
        # Original method
        batch_size = gen_data.shape[0]
        gen_data_sparse = np.zeros_like(gen_data)
        target_sparsity = 0.92
        
        for i in range(batch_size):
            sample_flat = gen_data[i].flatten()
            n_keep = int(len(sample_flat) * (1 - target_sparsity))
            top_indices = np.argpartition(sample_flat, -n_keep)[-n_keep:]
            sparse_flat = np.zeros_like(sample_flat)
            sparse_flat[top_indices] = sample_flat[top_indices]
            gen_data_sparse[i] = sparse_flat.reshape(gen_data[i].shape)
    
    # Stats
    sparsity_before = (gen_data < 1e-6).mean()
    sparsity_after = (gen_data_sparse < 1e-6).mean()
    logger.info(f"Sparsity: {sparsity_before:.2%} → {sparsity_after:.2%}")
    
    energy_before = gen_data.sum(axis=(1,2,3)).mean()
    energy_after = gen_data_sparse.sum(axis=(1,2,3)).mean()
    logger.info(f"Mean energy: {energy_before:.2f} → {energy_after:.2f}")
    
    # Compute metrics
    metrics_calc = MetricsCalculator(
        compute_ks=True,
        compute_energy_stats=True,
        compute_sparsity=True,
        compute_layer_stats=True
    )
    
    metrics_dense = metrics_calc.calculate(
        torch.from_numpy(real_data),
        torch.from_numpy(gen_data)
    )
    
    metrics_sparse = metrics_calc.calculate(
        torch.from_numpy(real_data),
        torch.from_numpy(gen_data_sparse)
    )
    
    # Print results
    logger.info("="*60)
    logger.info("RESULTS")
    logger.info("="*60)
    
    logger.info("\n--- WITHOUT Sparsification ---")
    logger.info(f"KS: {metrics_dense['ks_stat']:.4f} (p={metrics_dense['ks_pval']:.4e})")
    logger.info(f"Energy: Real={metrics_dense['real_energy_mean']:.2f}±{metrics_dense['real_energy_std']:.2f}, "
                f"Gen={metrics_dense['gen_energy_mean']:.2f}±{metrics_dense['gen_energy_std']:.2f}")
    logger.info(f"Sparsity: Real={metrics_dense['real_sparsity']:.3f}, Gen={metrics_dense['gen_sparsity']:.3f}")
    
    logger.info("\n--- WITH Sparsification ---")
    logger.info(f"KS: {metrics_sparse['ks_stat']:.4f} (p={metrics_sparse['ks_pval']:.4e})")
    logger.info(f"Energy: Real={metrics_sparse['real_energy_mean']:.2f}±{metrics_sparse['real_energy_std']:.2f}, "
                f"Gen={metrics_sparse['gen_energy_mean']:.2f}±{metrics_sparse['gen_energy_std']:.2f}")
    logger.info(f"Sparsity: Real={metrics_sparse['real_sparsity']:.3f}, Gen={metrics_sparse['gen_sparsity']:.3f}")
    
    # Save plots
    plot_energy_distributions(real_data, gen_data_sparse,
                              save_path=str(output_dir / 'energy_dist.png'))
    plot_sample_comparison(real_data, gen_data_sparse, n_samples=4,
                          save_path=str(output_dir / 'samples.png'))
    plot_layer_energies(real_data, gen_data_sparse,
                       save_path=str(output_dir / 'layers.png'))
    
    # Save metrics
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({'dense': metrics_dense, 'sparse': metrics_sparse}, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--h5_path', type=str)
    parser.add_argument('--max_events', type=int, default=10000)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--sparsify_method', type=str, default='energy',
                       choices=['energy', 'layer', 'global'])
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)