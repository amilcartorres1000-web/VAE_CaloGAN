"""
Main training script for VAE.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/experiments/quick_test.yaml
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
from dataclasses import asdict
import argparse

from calogan_vae.config import ExperimentConfig, DataConfig
from calogan_vae.data import CaloDataset, get_preprocessor
from calogan_vae.models import build_vae_from_config, get_loss_fn
from calogan_vae.training import (
    Trainer,
    EarlyStoppingCallback,
    CheckpointCallback,
    MetricsLoggingCallback,
    MetricsCalculator
)
from calogan_vae.utils import setup_logging, get_logger
from calogan_vae.visualization import create_training_report

logger = get_logger(__name__)


def load_config(config_path: str) -> ExperimentConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to config object
    # This is simplified - you might want to use a proper config library like Hydra
    config = ExperimentConfig()
    
    # Update from YAML
    if 'name' in config_dict:
        config.name = config_dict['name']
    if 'output_dir' in config_dict:
        config.output_dir = config_dict['output_dir']
    
    # Update sub-configs
    for key in ['data', 'encoder', 'decoder', 'loss', 'optimizer', 'training', 'validation']:
        if key in config_dict:
            config_attr = getattr(config, key)
            for k, v in config_dict[key].items():
                if hasattr(config_attr, k):
                    setattr(config_attr, k, v)
    
    return config


def create_dataloaders(config: ExperimentConfig):
    """Create train and validation dataloaders"""
    logger.info("Creating dataloaders...")
    
    # Create preprocessor
    preprocessor = get_preprocessor(
        config.data.preprocessing,
        percentile=config.data.percentile
    )
    
    # Create dataset
    dataset = CaloDataset(
        h5_path=config.data.h5_path,
        layer_keys=config.data.layer_keys,
        max_events=config.data.max_events,
        loading_mode=config.data.loading_mode,
        preprocessor=None  # Will apply after fitting
    )
    
    # Fit preprocessor on a sample of training data
    logger.info("Fitting preprocessor...")
    sample_loader = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=False, num_workers=0)
    sample_data = []
    for i, batch in enumerate(sample_loader):
        sample_data.append(batch.numpy())
        if i >= 10:  # Use 10 batches for fitting
            break
    sample_data = torch.cat([torch.from_numpy(d) for d in sample_data], dim=0).numpy()
    preprocessor.fit(sample_data)
    
    # Now set preprocessor
    dataset.preprocessor = preprocessor
    
    # Split into train/val
    train_size = int(config.data.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers // 2,
        pin_memory=config.data.pin_memory
    )
    
    return train_loader, val_loader, preprocessor


def main(args):
    """Main training function"""
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = ExperimentConfig()
    
    # Override with command line args
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    
    # Setup logging
    log_file = Path(config.output_dir) / 'training.log'
    setup_logging(log_file=str(log_file), level='INFO')
    
    logger.info("="*60)
    logger.info("TRAINING VAE FOR CALOGAN")
    logger.info("="*60)
    logger.info(f"Experiment: {config.name}")
    logger.info(f"Output dir: {config.output_dir}")
    
    # Set seed
    torch.manual_seed(config.training.seed)
    
    # Create dataloaders
    train_loader, val_loader, preprocessor = create_dataloaders(config)
    
    # Build model
    logger.info("Building model...")
    model = build_vae_from_config(config.encoder, config.decoder)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        betas=config.optimizer.betas
    )
    
    # Create scheduler
    scheduler = None
    if config.optimizer.use_scheduler:
        if config.optimizer.scheduler_type == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.optimizer.scheduler_factor,
                patience=config.optimizer.scheduler_patience
            )
    
    # Create loss function
    loss_fn = get_loss_fn(config.loss.loss_type, config.loss)
    
    # Create callbacks
    callbacks = [
        MetricsLoggingCallback(log_interval=config.training.log_interval),
        CheckpointCallback(
            checkpoint_dir=config.training.checkpoint_dir,
            save_interval=config.training.checkpoint_interval,
            save_best_only=config.training.save_best_only,
            metric=config.training.monitor_metric,
            mode='min'
        )
    ]
    
    if config.training.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                patience=config.training.patience,
                metric=config.training.monitor_metric,
                mode='min'
            )
        )
    
    # Create metrics calculator
    metrics_calculator = MetricsCalculator(
        compute_ks=config.validation.compute_ks,
        compute_energy_stats=config.validation.compute_energy_stats,
        compute_sparsity=config.validation.compute_sparsity
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        scheduler=scheduler,
        callbacks=callbacks,
        metrics_calculator=metrics_calculator
    )
    
    # Train
    trainer.fit()
    
    # Generate final report
    logger.info("Generating final report...")
    
    # Load best model
    best_model_path = Path(config.training.checkpoint_dir) / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded best model for evaluation")
    
    # Generate samples
    model.eval()
    device = torch.device(config.training.device)
    model.to(device)
    
    with torch.no_grad():
        # Generate samples
        z = torch.randn(config.validation.num_samples, model.z_dim, device=device)
        gen_data = model.decode(z).cpu().numpy()
        
        # Collect real data
        real_data = []
        for batch in val_loader:
            real_data.append(batch.numpy())
            if len(real_data) * config.data.batch_size >= config.validation.num_samples:
                break
        real_data = torch.cat([torch.from_numpy(d) for d in real_data], dim=0).numpy()
        real_data = real_data[:config.validation.num_samples]
    
    # Inverse preprocessing
    real_data_orig = preprocessor.inverse(real_data)
    gen_data_orig = preprocessor.inverse(gen_data)
    
    # Create report
    figures_dir = Path(config.output_dir) / 'figures'
    create_training_report(
        history=trainer.metrics_history,
        real_data=real_data_orig,
        gen_data=gen_data_orig,
        output_dir=str(figures_dir)
    )
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Checkpoints: {config.training.checkpoint_dir}")
    logger.info(f"Figures: {figures_dir}")
    logger.info(f"Logs: {log_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE for CaloGAN')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    
    args = parser.parse_args()
    main(args)