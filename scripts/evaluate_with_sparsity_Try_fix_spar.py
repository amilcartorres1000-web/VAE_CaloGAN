"""
Evaluation script with post-processing sparsification.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime

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


def sparsify_samples(samples: np.ndarray, target_sparsity: float = 0.92) -> np.ndarray:
    """
    Post-process generated samples to match target sparsity.
    
    Strategy: Keep only the top (1 - target_sparsity) percentage of values
    
    Args:
        samples: Generated samples (N, C, H, W)
        target_sparsity: Target fraction of zeros (default 0.92 = 92%)
        
    Returns:
        Sparsified samples
    """
    batch_size = samples.shape[0]
    samples_sparse = np.zeros_like(samples)
    
    for i in range(batch_size):
        sample = samples[i]  # (C, H, W)
        
        # Flatten
        sample_flat = sample.flatten()
        
        # Calculate how many values to keep
        n_total = len(sample_flat)
        n_keep = int(n_total * (1 - target_sparsity))
        
        # Get indices of top values
        top_indices = np.argpartition(sample_flat, -n_keep)[-n_keep:]
        
        # Create sparse output
        sparse_flat = np.zeros_like(sample_flat)
        sparse_flat[top_indices] = sample_flat[top_indices]
        
        # Reshape back
        samples_sparse[i] = sparse_flat.reshape(sample.shape)
    
    return samples_sparse


def main(args):
    """Evaluate trained VAE with sparsification"""
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"eval_sparse_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'evaluation.log'
    setup_logging(log_file=str(log_file), level='INFO')
    
    logger.info("="*60)
    logger.info("EVALUATING VAE WITH SPARSIFICATION")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Target sparsity: {args.target_sparsity:.2%}")
    logger.info(f"Output directory: {output_dir}")
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        logger.warning("Config not found in checkpoint, using default")
        config = ExperimentConfig()
    
    # Build model
    logger.info("Building model...")
    model = build_vae_from_config(config.encoder, config.decoder)
    
    # Load weights (handle missing keys gracefully)
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Error loading model: {e}")
        logger.warning("Attempting to load with strict=False")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load real data
    logger.info("Loading real data for comparison...")
    preprocessor = get_preprocessor(
        config.data.preprocessing,
        percentile=config.data.percentile
    )
    
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
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Loaded {len(dataset)} real samples")
    
    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")
    with torch.no_grad():
        z = torch.randn(args.num_samples, model.z_dim, device=device)
        gen_data = model.decode(z).cpu().numpy()
    
    logger.info(f"Generated {args.num_samples} samples")
    
    # Collect real data
    logger.info("Collecting real data...")
    real_data = []
    for batch in dataloader:
        real_data.append(batch.numpy())
        if len(real_data) * args.batch_size >= args.num_samples:
            break
    real_data = np.concatenate(real_data, axis=0)[:args.num_samples]
    
    # Keep data in preprocessed space for comparison (same as training)
    real_data_preprocessed = real_data  # Already preprocessed from dataset
    gen_data_preprocessed = gen_data    # Model outputs in preprocessed scale
    
    logger.info("Comparing in PREPROCESSED space (same as training)")
    logger.info(f"Real data range: [{real_data_preprocessed.min():.4f}, {real_data_preprocessed.max():.4f}]")
    logger.info(f"Gen data range: [{gen_data_preprocessed.min():.4f}, {gen_data_preprocessed.max():.4f}]")
    
    # Apply sparsification IN PREPROCESSED SPACE
    logger.info(f"Applying sparsification (target: {args.target_sparsity:.2%})...")
    gen_data_sparse = sparsify_samples(gen_data_preprocessed, target_sparsity=args.target_sparsity)
    
    # Print before/after sparsity
    sparsity_before = (gen_data_preprocessed < 1e-6).mean()
    sparsity_after = (gen_data_sparse < 1e-6).mean()
    logger.info(f"Sparsity before: {sparsity_before:.2%}")
    logger.info(f"Sparsity after: {sparsity_after:.2%}")
    
    # Compute metrics (both versions)
    logger.info("Calculating metrics...")
    metrics_calc = MetricsCalculator(
        compute_ks=True,
        compute_energy_stats=True,
        compute_sparsity=True,
        compute_layer_stats=True
    )
    
    # Metrics WITHOUT sparsification
    metrics_dense = metrics_calc.calculate(
        torch.from_numpy(real_data_preprocessed),
        torch.from_numpy(gen_data_preprocessed)
    )
    
    # Metrics WITH sparsification
    metrics_sparse = metrics_calc.calculate(
        torch.from_numpy(real_data_preprocessed),
        torch.from_numpy(gen_data_sparse)
    )

    # Print comparison
    logger.info("="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    
    logger.info("\n--- WITHOUT Sparsification ---")
    logger.info(f"KS Statistic: {metrics_dense['ks_stat']:.4f}")
    logger.info(f"KS P-value: {metrics_dense['ks_pval']:.4e}")
    logger.info(f"Real Energy: {metrics_dense['real_energy_mean']:.2f} ± {metrics_dense['real_energy_std']:.2f}")
    logger.info(f"Gen Energy:  {metrics_dense['gen_energy_mean']:.2f} ± {metrics_dense['gen_energy_std']:.2f}")
    logger.info(f"Real Sparsity: {metrics_dense['real_sparsity']:.3f}")
    logger.info(f"Gen Sparsity:  {metrics_dense['gen_sparsity']:.3f}")
    
    logger.info("\n--- WITH Sparsification ---")
    logger.info(f"KS Statistic: {metrics_sparse['ks_stat']:.4f}")
    logger.info(f"KS P-value: {metrics_sparse['ks_pval']:.4e}")
    logger.info(f"Real Energy: {metrics_sparse['real_energy_mean']:.2f} ± {metrics_sparse['real_energy_std']:.2f}")
    logger.info(f"Gen Energy:  {metrics_sparse['gen_energy_mean']:.2f} ± {metrics_sparse['gen_energy_std']:.2f}")
    logger.info(f"Real Sparsity: {metrics_sparse['real_sparsity']:.3f}")
    logger.info(f"Gen Sparsity:  {metrics_sparse['gen_sparsity']:.3f}")
    
    # Create plots
    logger.info("\nCreating visualizations...")
    
    # Dense version
    plot_energy_distributions(
        real_data_preprocessed,
        gen_data_preprocessed,
        save_path=str(output_dir / 'energy_distributions_dense.png')
    )
    
    plot_sample_comparison(
        real_data_preprocessed,
        gen_data_preprocessed,
        n_samples=4,
        save_path=str(output_dir / 'sample_comparison_dense.png')
    )
    
    # Sparse version
    plot_energy_distributions(
        real_data_preprocessed,
        gen_data_sparse,
        save_path=str(output_dir / 'energy_distributions_sparse.png')
    )
    
    plot_sample_comparison(
        real_data_preprocessed,
        gen_data_sparse,
        n_samples=4,
        save_path=str(output_dir / 'sample_comparison_sparse.png')
    )
    
    plot_layer_energies(
        real_data_preprocessed,
        gen_data_sparse,
        save_path=str(output_dir / 'layer_energies.png')
    )
    
    # Save metrics
    import json
    with open(output_dir / 'metrics_comparison.json', 'w') as f:
        json.dump({
            'dense': metrics_dense,
            'sparse': metrics_sparse
        }, f, indent=2)
    
    logger.info(f"Visualizations saved to {output_dir}")
    
    logger.info("="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate VAE with sparsification')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--h5_path', type=str,
                       help='Path to HDF5 file (overrides config)')
    parser.add_argument('--max_events', type=int, default=10000,
                       help='Maximum events to load')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for plots')
    parser.add_argument('--target_sparsity', type=float, default=0.92,
                       help='Target sparsity (fraction of zeros)')
    
    args = parser.parse_args()
    main(args)