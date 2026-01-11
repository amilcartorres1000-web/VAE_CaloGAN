"""
Evaluation in preprocessed space (same as training).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime

from calogan_vae.data import CaloDataset, get_preprocessor
from calogan_vae.models import build_vae_from_config
from calogan_vae.training.metrics import MetricsCalculator
from calogan_vae.utils import setup_logging, get_logger

logger = get_logger(__name__)


def sparsify_samples(samples: np.ndarray, target_sparsity: float = 0.92) -> np.ndarray:
    """Sparsify samples by keeping top values."""
    batch_size = samples.shape[0]
    samples_sparse = np.zeros_like(samples)
    
    for i in range(batch_size):
        sample = samples[i]
        sample_flat = sample.flatten()
        n_total = len(sample_flat)
        n_keep = int(n_total * (1 - target_sparsity))
        
        if n_keep > 0:
            top_indices = np.argpartition(sample_flat, -n_keep)[-n_keep:]
            sparse_flat = np.zeros_like(sample_flat)
            sparse_flat[top_indices] = sample_flat[top_indices]
            samples_sparse[i] = sparse_flat.reshape(sample.shape)
    
    return samples_sparse


def main(args):
    """Evaluate in preprocessed space"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"eval_preprocessed_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_file=str(output_dir / 'evaluation.log'), level='INFO')
    
    logger.info("="*60)
    logger.info("EVALUATION IN PREPROCESSED SPACE")
    logger.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config')
    
    # Build model
    model = build_vae_from_config(config.encoder, config.decoder)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    
    # Load and preprocess data
    preprocessor = get_preprocessor(config.data.preprocessing, percentile=config.data.percentile)
    
    dataset = CaloDataset(
        h5_path=config.data.h5_path,
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
    
    # Get real data (already preprocessed)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    real_data = []
    for batch in dataloader:
        real_data.append(batch.numpy())
        if len(real_data) * 64 >= args.num_samples:
            break
    real_data = np.concatenate(real_data, axis=0)[:args.num_samples]
    
    # Generate data (in preprocessed space)
    logger.info(f"Generating {args.num_samples} samples...")
    with torch.no_grad():
        z = torch.randn(args.num_samples, model.z_dim, device=device)
        gen_data = model.decode(z).cpu().numpy()
    
    logger.info("NOTE: Evaluation in PREPROCESSED space (same scale as training)")
    
    # Apply sparsification
    logger.info(f"Applying sparsification...")
    gen_data_sparse = sparsify_samples(gen_data, target_sparsity=args.target_sparsity)
    
    sparsity_before = (gen_data < 1e-6).mean()
    sparsity_after = (gen_data_sparse < 1e-6).mean()
    logger.info(f"Sparsity: {sparsity_before:.2%} → {sparsity_after:.2%}")
    
    # Compute metrics
    metrics_calc = MetricsCalculator(
        compute_ks=True,
        compute_energy_stats=True,
        compute_sparsity=True
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
    logger.info("RESULTS (Preprocessed Space)")
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
    
    # Save metrics
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({'dense': metrics_dense, 'sparse': metrics_sparse}, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--max_events', type=int, default=10000)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--target_sparsity', type=float, default=0.92)
    
    args = parser.parse_args()
    main(args)