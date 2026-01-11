"""
Configuration dataclasses for the VAE project.
Uses dataclasses for type safety and validation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal


@dataclass
class DataConfig:
    """Dataset configuration"""
    h5_path: str = "eplus.hdf5"
    layer_keys: List[str] = field(default_factory=lambda: ["layer_0", "layer_1", "layer_2"])
    max_events: Optional[int] = None
    
    # Data loading strategy
    loading_mode: Literal['auto', 'memory', 'cache', 'disk'] = 'auto'
    cache_size: int = 1000
    
    # Train/val split
    train_split: float = 0.9
    
    # DataLoader settings
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    
    # Preprocessing
    preprocessing: str = 'log1p_norm'  # 'log1p_norm', 'standardize', etc.
    percentile: float = 99.0


@dataclass
class EncoderConfig:
    """Encoder architecture configuration"""
    input_channels: int = 3
    channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    z_dim: int = 64
    use_batchnorm: bool = True
    activation: str = 'leaky_relu'
    dropout: float = 0.0


@dataclass
class DecoderConfig:
    """Decoder architecture configuration"""
    z_dim: int = 64
    channels: List[int] = field(default_factory=lambda: [128, 64, 32, 16])
    output_channels: int = 3
    output_size: tuple = (12, 96)
    use_batchnorm: bool = True
    activation: str = 'leaky_relu'
    dropout: float = 0.0
    
    # Sparsity control
    sparsity_threshold: float = 0.1
    output_scale: float = 0.05


@dataclass
class LossConfig:
    """Loss function configuration"""
    loss_type: str = 'sparse_vae'  # 'sparse_vae', 'beta_vae', etc.
    
    # Loss weights
    recon_weight: float = 1.0
    sparsity_weight: float = 5.0
    energy_weight: float = 0.5
    
    # KL annealing
    kl_annealing: bool = True
    kl_warmup_epochs: int = 20
    kl_max_weight: float = 0.1
    
    # Non-zero pixel weighting
    nonzero_weight: float = 50.0


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    optimizer: str = 'adamw'
    lr: float = 3e-4
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = 'reduce_on_plateau'
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5


@dataclass
class TrainingConfig:
    """Training loop configuration"""
    epochs: int = 100
    device: str = 'cuda'
    seed: int = 42
    
    # Gradient management
    grad_clip: float = 1.0
    
    # Logging
    log_interval: int = 10
    
    # Checkpointing
    checkpoint_interval: int = 5
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = False
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    monitor_metric: str = 'val_ks'
    
    # Mixed precision
    use_amp: bool = False


@dataclass
class ValidationConfig:
    """Validation configuration"""
    val_interval: int = 1  # Validate every N epochs
    num_samples: int = 1000
    
    # Metrics to compute
    compute_ks: bool = True
    compute_energy_stats: bool = True
    compute_sparsity: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    name: str = "vae_experiment"
    output_dir: str = "outputs"
    
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.encoder.z_dim == self.decoder.z_dim, \
            "Encoder and decoder z_dim must match"
        
        assert 0 < self.data.train_split < 1, \
            "train_split must be between 0 and 1"
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)