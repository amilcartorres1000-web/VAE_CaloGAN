"""
Evaluation script for trained VAE models.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --num_samples 5000
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import json
from datetime import datetime

from calogan_vae.config import ExperimentConfig
from calogan_vae.data import CaloDataset, get_preprocessor
from calogan_vae.models import build_vae_from_config
from calogan_vae.training import MetricsCalculator
from calogan_vae.visualization import (
    plot_energy_distributions,
    plot_sample_comparison,
    plot_layer_energies
)
from calogan_vae.utils import setup_logging, get_logger

logger = get_logger(__name__)


def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        model: Loaded VAE model
        config: Experiment configuration
        preprocessor: Data preprocessor
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Use default config if not saved
        logger.warning("No config found in checkpoint, using defaults")
        config = ExperimentConfig()
    
    # Build model
    model = build_vae_from_config(config.encoder, config.decoder)
    
    # Load state dict with strict=False to allow missing keys (e.g., new parameters)
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model_state_dict'], 
        strict=False
    )
    
    if missing_keys:
        logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        logger.warning("These will be initialized with default values")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Load preprocessor if available
    preprocessor = None
    if 'preprocessor_state' in checkpoint:
        preprocessor = get_preprocessor(
            config.data.preprocessing,
            percentile=config.data.percentile
        )
        preprocessor.load_state(checkpoint['preprocessor_state'])
        logger.info("Preprocessor loaded from checkpoint")
    
    return model, config, preprocessor


def run_diagnostics(model, device: str):
    """
    Run diagnostic tests on the decoder.
    
    Args:
        model: VAE model
        device: Device to use
    """
    logger.info("\n" + "="*60)
    logger.info("DECODER DIAGNOSTICS")
    logger.info("="*60)
    
    model.eval()
    with torch.no_grad():
        # Sample some latent vectors
        z = torch.randn(10, model.z_dim, device=device)
        
        # Generate
        gen = model.decode(z)
        
        logger.info("\nGenerated output statistics:")
        logger.info(f"  Mean: {gen.mean().item():.6f}")
        logger.info(f"  Std: {gen.std().item():.6f}")
        logger.info(f"  Min: {gen.min().item():.6f}")
        logger.info(f"  Max: {gen.max().item():.6f}")
        logger.info(f"  % zeros: {(gen < 1e-6).float().mean().item():.3f}")
        
        # Check decoder output before activation
        x = model.decoder.fc(z)
        x = x.view(x.size(0), -1, 3, 3)
        x = model.decoder.deconv_layers(x)
        
        logger.info("\nBefore final activation:")
        logger.info(f"  Mean: {x.mean().item():.6f}")
        logger.info(f"  Std: {x.std().item():.6f}")
        logger.info(f"  Min: {x.min().item():.6f}")
        logger.info(f"  Max: {x.max().item():.6f}")
        
        # Check learned thresholds if available
        if hasattr(model.decoder, 'threshold'):
            logger.info("\nLearned thresholds per channel:")
            for i, thresh in enumerate(model.decoder.threshold):
                logger.info(f"  Channel {i}: {thresh.item():.6f}")
    
    logger.info("="*60 + "\n")


def load_real_data(config: ExperimentConfig, preprocessor=None, max_samples: int = None):
    """
    Load real data for comparison.
    
    Args:
        config: Experiment configuration
        preprocessor: Optional preprocessor
        max_samples: Maximum number of samples to load
        
    Returns:
        real_data: Real calorimeter data
    """
    logger.info("Loading real data for comparison...")
    
    # Create dataset
    dataset = CaloDataset(
        h5_path=config.data.h5_path,
        layer_keys=config.data.layer_keys,
        max_events=max_samples,
        loading_mode='auto',
        preprocessor=preprocessor
    )
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Load all data
    real_data = []
    for batch in loader:
        real_data.append(batch)
        if max_samples and len(real_data) * config.data.batch_size >= max_samples:
            break
    
    real_data = torch.cat(real_data, dim=0)
    
    if max_samples:
        real_data = real_data[:max_samples]
    
    logger.info(f"Loaded {len(real_data)} real samples")
    return real_data


def generate_samples(model, num_samples: int, device: str, batch_size: int = 64):
    """
    Generate samples from the model.
    
    Args:
        model: VAE model
        num_samples: Number of samples to generate
        device: Device to use
        batch_size: Batch size for generation
        
    Returns:
        generated_data: Generated samples
    """
    logger.info(f"Generating {num_samples} samples...")
    
    model.eval()
    generated_data = []
    
    with torch.no_grad():
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # Sample from latent space
            z = torch.randn(current_batch_size, model.z_dim, device=device)
            
            # Generate
            gen_batch = model.decode(z)
            generated_data.append(gen_batch.cpu())
    
    generated_data = torch.cat(generated_data, dim=0)[:num_samples]
    
    logger.info(f"Generated {len(generated_data)} samples")
    return generated_data


def evaluate_model(
    checkpoint_path: str,
    num_samples: int = 1000,
    output_dir: str = 'evaluation_results',
    device: str = 'cuda'
):
    """
    Evaluate a trained VAE model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of samples to use for evaluation
        output_dir: Directory to save results
        device: Device to use
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = output_path / f"eval_{timestamp}"
    eval_dir.mkdir(exist_ok=True)
    
    logger.info("="*60)
    logger.info("EVALUATING VAE MODEL")
    logger.info("="*60)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output directory: {eval_dir}")
    
    # Load model
    model, config, preprocessor = load_checkpoint(checkpoint_path, device)
    
    # Run diagnostics
    run_diagnostics(model, device)
    
    # Load real data
    real_data = load_real_data(config, preprocessor, max_samples=num_samples)
    real_data = real_data.to(device)
    
    # Generate samples
    gen_data = generate_samples(model, num_samples, device, config.data.batch_size)
    gen_data = gen_data.to(device)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics_calculator = MetricsCalculator(
        compute_ks=True,
        compute_energy_stats=True,
        compute_sparsity=True,
        compute_layer_stats=True
    )
    
    metrics = metrics_calculator.calculate(real_data, gen_data)
    
    # Log metrics
    logger.info("\n" + "="*60)
    logger.info("EVALUATION METRICS")
    logger.info("="*60)
    
    # KS test
    if 'ks_stat' in metrics:
        logger.info(f"\nKolmogorov-Smirnov Test:")
        logger.info(f"  Statistic: {metrics['ks_stat']:.4f}")
        logger.info(f"  P-value: {metrics['ks_pval']:.4f}")
    
    # Energy statistics
    logger.info(f"\nEnergy Statistics:")
    logger.info(f"  Real - Mean: {metrics.get('real_energy_mean', 0):.4f}, Std: {metrics.get('real_energy_std', 0):.4f}")
    logger.info(f"  Gen  - Mean: {metrics.get('gen_energy_mean', 0):.4f}, Std: {metrics.get('gen_energy_std', 0):.4f}")
    
    # Sparsity
    logger.info(f"\nSparsity:")
    logger.info(f"  Real: {metrics.get('real_sparsity', 0):.4f}")
    logger.info(f"  Gen:  {metrics.get('gen_sparsity', 0):.4f}")
    
    # Layer energies
    logger.info(f"\nPer-Layer Energy:")
    for i in range(3):
        real_mean = metrics.get(f'real_layer_{i}_mean', 0)
        gen_mean = metrics.get(f'gen_layer_{i}_mean', 0)
        logger.info(f"  Layer {i} - Real: {real_mean:.4f}, Gen: {gen_mean:.4f}")
    
    # Save metrics to JSON
    metrics_file = eval_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nMetrics saved to {metrics_file}")
    
    # Create visualizations
    logger.info("\nCreating visualizations...")
    
    # Move data to CPU for plotting
    real_data_np = real_data.cpu().numpy()
    gen_data_np = gen_data.cpu().numpy()
    
    # Inverse preprocessing if available
    if preprocessor:
        real_data_np = preprocessor.inverse(real_data_np)
        gen_data_np = preprocessor.inverse(gen_data_np)
    
    # Energy distributions
    plot_energy_distributions(
        real_data_np,
        gen_data_np,
        save_path=str(eval_dir / 'energy_distributions.png')
    )
    
    # Sample comparison
    plot_sample_comparison(
        real_data_np,
        gen_data_np,
        n_samples=4,
        save_path=str(eval_dir / 'sample_comparison.png')
    )
    
    # Layer energies
    plot_layer_energies(
        real_data_np,
        gen_data_np,
        save_path=str(eval_dir / 'layer_energies.png')
    )
    
    logger.info(f"Visualizations saved to {eval_dir}")
    
    # Create summary report
    summary_file = eval_dir / 'evaluation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("VAE Model Evaluation Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Evaluation Date: {timestamp}\n")
        f.write(f"Number of Samples: {num_samples}\n\n")
        
        f.write("Metrics:\n")
        f.write("-"*60 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
    
    logger.info(f"\nSummary saved to {summary_file}")
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved to: {eval_dir}")
    
    return metrics, eval_dir


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained VAE model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to use for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level='INFO')
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Run evaluation
    evaluate_model(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
