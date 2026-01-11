"""
Generate samples from trained VAE.

Usage:
    python scripts/generate.py --checkpoint checkpoints/best_model.pth --num_samples 1000
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import h5py
import numpy as np

from calogan_vae.config import ExperimentConfig
from calogan_vae.data import get_preprocessor
from calogan_vae.models import build_vae_from_config
from calogan_vae.utils import setup_logging, get_logger

logger = get_logger(__name__)


def main(args):
    """Generate samples from VAE"""
    
    # Setup logging
    setup_logging(level='INFO')
    
    logger.info("="*60)
    logger.info("GENERATING SAMPLES")
    logger.info("="*60)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        logger.warning("Config not found in checkpoint, using default")
        config = ExperimentConfig()
    
    # Build model
    logger.info("Building model...")
    model = build_vae_from_config(config.encoder, config.decoder)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Create preprocessor (for inverse transform)
    preprocessor = get_preprocessor(
        config.data.preprocessing,
        percentile=config.data.percentile
    )
    
    # Need to fit preprocessor - load some real data
    if args.fit_preprocessor:
        logger.info("Fitting preprocessor on real data...")
        from calogan_vae.data import CaloDataset
        from torch.utils.data import DataLoader
        
        dataset = CaloDataset(
            h5_path=config.data.h5_path,
            layer_keys=config.data.layer_keys,
            max_events=1000,
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
    
    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")
    
    all_samples = []
    batch_size = args.batch_size
    
    with torch.no_grad():
        for i in range(0, args.num_samples, batch_size):
            n = min(batch_size, args.num_samples - i)
            
            # Sample latent vectors
            z = torch.randn(n, model.z_dim, device=device)
            
            # Generate
            samples = model.decode(z).cpu().numpy()
            
            # Inverse preprocessing (if fitted)
            if args.fit_preprocessor:
                samples = preprocessor.inverse(samples)
            
            all_samples.append(samples)
            
            if (i + n) % 500 == 0:
                logger.info(f"Generated {i + n}/{args.num_samples} samples")
    
    # Concatenate
    all_samples = np.concatenate(all_samples, axis=0)
    
    logger.info(f"Generated {all_samples.shape[0]} samples")
    logger.info(f"Sample shape: {all_samples.shape}")
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == 'npy':
        np.save(output_path, all_samples)
        logger.info(f"Saved samples to {output_path}")
    
    elif args.format == 'h5':
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('samples', data=all_samples, compression='gzip')
            
            # Save metadata
            f.attrs['num_samples'] = all_samples.shape[0]
            f.attrs['shape'] = all_samples.shape
            f.attrs['checkpoint'] = str(args.checkpoint)
            
        logger.info(f"Saved samples to {output_path}")
    
    else:
        raise ValueError(f"Unknown format: {args.format}")
    
    # Print statistics
    logger.info("\nSample Statistics:")
    logger.info(f"  Mean:   {all_samples.mean():.6f}")
    logger.info(f"  Std:    {all_samples.std():.6f}")
    logger.info(f"  Min:    {all_samples.min():.6f}")
    logger.info(f"  Max:    {all_samples.max():.6f}")
    logger.info(f"  Sparsity: {(all_samples < 1e-6).mean():.3f}")
    
    logger.info("="*60)
    logger.info("GENERATION COMPLETE")
    logger.info("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate samples from VAE')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Generation batch size')
    parser.add_argument('--output', type=str, default='generated_samples.npy',
                       help='Output file path')
    parser.add_argument('--format', type=str, choices=['npy', 'h5'], default='npy',
                       help='Output format')
    parser.add_argument('--fit_preprocessor', action='store_true',
                       help='Fit preprocessor on real data for inverse transform')
    
    args = parser.parse_args()
    main(args)