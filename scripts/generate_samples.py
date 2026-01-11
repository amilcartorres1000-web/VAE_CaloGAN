"""
Generate calorimeter shower samples using trained VAE.

This is the main script for generating synthetic data.

Usage:
    # Generate 10000 samples
    python scripts/generate_samples.py \
        --checkpoint checkpoints/fix_generation/best_model.pth \
        --output generated_showers.h5 \
        --num_samples 10000
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import h5py
import numpy as np
from datetime import datetime
import json

from calogan_vae.models import build_vae_from_config
from calogan_vae.data import CaloDataset, get_preprocessor
from calogan_vae.utils import setup_logging, get_logger
from torch.utils.data import DataLoader

logger = get_logger(__name__)


def generate_samples(
    model: torch.nn.Module,
    preprocessor,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    seed: int = 42
) -> np.ndarray:
    """
    Generate samples from trained VAE.
    
    Returns samples in ORIGINAL scale (not preprocessed).
    """
    model.eval()
    torch.manual_seed(seed)
    
    all_samples = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            n = min(batch_size, num_samples - i)
            
            # Sample from prior
            z = torch.randn(n, model.z_dim, device=device)
            
            # Decode (outputs are in preprocessed scale)
            samples_preprocessed = model.decode(z).cpu().numpy()
            
            # Convert back to original scale
            samples_original = preprocessor.inverse(samples_preprocessed)
            
            all_samples.append(samples_original)
            
            if (i + n) % 1000 == 0:
                logger.info(f"Generated {i + n}/{num_samples} samples")
    
    return np.concatenate(all_samples, axis=0)


def compute_statistics(samples: np.ndarray, real_samples: np.ndarray = None) -> dict:
    """Compute statistics of generated samples."""
    stats = {
        'num_samples': samples.shape[0],
        'shape': list(samples.shape),
        'total_energy': {
            'mean': float(samples.sum(axis=(1,2,3)).mean()),
            'std': float(samples.sum(axis=(1,2,3)).std()),
            'min': float(samples.sum(axis=(1,2,3)).min()),
            'max': float(samples.sum(axis=(1,2,3)).max()),
        },
        'sparsity': float((samples < 1e-6).mean()),
        'layer_energies': {}
    }
    
    # Per-layer stats
    for i in range(samples.shape[1]):
        layer_energy = samples[:, i].sum(axis=(1,2))
        stats['layer_energies'][f'layer_{i}'] = {
            'mean': float(layer_energy.mean()),
            'std': float(layer_energy.std()),
        }
    
    # Compare to real data if provided
    if real_samples is not None:
        from scipy.stats import ks_2samp
        
        real_energy = real_samples.sum(axis=(1,2,3))
        gen_energy = samples.sum(axis=(1,2,3))
        
        ks_stat, ks_pval = ks_2samp(real_energy, gen_energy)
        
        stats['comparison_to_real'] = {
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pval),
            'real_energy_mean': float(real_energy.mean()),
            'real_energy_std': float(real_energy.std()),
        }
    
    return stats


def main(args):
    """Main generation function"""
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    log_file = output_path.parent / f'generation_{timestamp}.log'
    setup_logging(log_file=str(log_file), level='INFO')
    
    logger.info("="*60)
    logger.info("GENERATING CALORIMETER SHOWER SAMPLES")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Number of samples: {args.num_samples}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load checkpoint
    logger.info("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    model = build_vae_from_config(config.encoder, config.decoder)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from epoch {checkpoint.get('epoch', '?')}")
    
    # Setup preprocessor
    logger.info("Setting up preprocessor...")
    preprocessor = get_preprocessor(
        config.data.preprocessing,
        percentile=config.data.percentile
    )
    
    # Fit preprocessor on real data
    dataset = CaloDataset(
        h5_path=config.data.h5_path,
        layer_keys=config.data.layer_keys,
        max_events=1000,  # Use 1000 samples for fitting
        loading_mode='memory',
        preprocessor=None
    )
    
    sample_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    sample_data = []
    for batch in sample_loader:
        sample_data.append(batch.numpy())
    sample_data = np.concatenate(sample_data, axis=0)
    preprocessor.fit(sample_data)
    
    logger.info("Preprocessor fitted")
    
    # Load some real data for comparison
    real_samples = None
    if args.compute_comparison:
        logger.info("Loading real data for comparison...")
        dataset.preprocessor = preprocessor
        real_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        real_samples = []
        for batch in real_loader:
            real_samples.append(batch.numpy())
        real_samples = np.concatenate(real_samples, axis=0)
        real_samples = preprocessor.inverse(real_samples)
        logger.info(f"Loaded {len(real_samples)} real samples")
    
    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")
    generated_samples = generate_samples(
        model=model,
        preprocessor=preprocessor,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed
    )
    
    logger.info(f"Generated {generated_samples.shape[0]} samples")
    
    # Compute statistics
    logger.info("Computing statistics...")
    stats = compute_statistics(generated_samples, real_samples)
    
    logger.info("\n" + "="*60)
    logger.info("GENERATED SAMPLES STATISTICS")
    logger.info("="*60)
    logger.info(f"Shape: {stats['shape']}")
    logger.info(f"Total Energy: {stats['total_energy']['mean']:.2f} ± {stats['total_energy']['std']:.2f}")
    logger.info(f"Energy Range: [{stats['total_energy']['min']:.2f}, {stats['total_energy']['max']:.2f}]")
    logger.info(f"Sparsity: {stats['sparsity']:.3f}")
    
    for layer, layer_stats in stats['layer_energies'].items():
        logger.info(f"{layer}: {layer_stats['mean']:.2f} ± {layer_stats['std']:.2f}")
    
    if 'comparison_to_real' in stats:
        logger.info("\nComparison to Real Data:")
        logger.info(f"KS Statistic: {stats['comparison_to_real']['ks_statistic']:.4f}")
        logger.info(f"KS P-value: {stats['comparison_to_real']['ks_pvalue']:.4e}")
        logger.info(f"Real Energy: {stats['comparison_to_real']['real_energy_mean']:.2f} ± {stats['comparison_to_real']['real_energy_std']:.2f}")
    
    # Save to HDF5
    logger.info(f"\nSaving to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        # Save generated data
        f.create_dataset('generated_showers', data=generated_samples, compression='gzip')
        
        # Save metadata
        f.attrs['num_samples'] = generated_samples.shape[0]
        f.attrs['shape'] = str(generated_samples.shape)
        f.attrs['checkpoint'] = str(args.checkpoint)
        f.attrs['generation_date'] = timestamp
        f.attrs['seed'] = args.seed
        
        # Save statistics as JSON in attributes
        f.attrs['statistics'] = json.dumps(stats)
    
    # Also save statistics as JSON
    stats_file = output_path.parent / f'statistics_{timestamp}.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Samples saved to: {output_path}")
    logger.info(f"Statistics saved to: {stats_file}")
    
    logger.info("="*60)
    logger.info("GENERATION COMPLETE")
    logger.info("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate calorimeter shower samples',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='generated_showers.h5',
                       help='Output HDF5 file path')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Generation batch size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--compute_comparison', action='store_true',
                       help='Compare generated samples to real data')
    
    args = parser.parse_args()
    main(args)