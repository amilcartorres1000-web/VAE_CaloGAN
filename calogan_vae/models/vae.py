"""
VAE wrapper that combines encoder and decoder.
"""
import torch
import torch.nn as nn
from typing import Tuple


class VAE(nn.Module):
    """
    Variational Autoencoder for calorimeter showers.
    
    Combines encoder and decoder with reparameterization trick.
    """
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.z_dim = encoder.z_dim
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        
        Args:
            mu: Mean of latent distribution (B, z_dim)
            logvar: Log variance of latent distribution (B, z_dim)
            
        Returns:
            z: Sampled latent vector (B, z_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor (B, 3, 12, 96)
            
        Returns:
            recon: Reconstructed input (B, 3, 12, 96)
            mu: Latent mean (B, z_dim)
            logvar: Latent log variance (B, z_dim)
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon = self.decoder(z)
        
        return recon, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space (deterministic).
        
        Args:
            x: Input tensor (B, 3, 12, 96)
            
        Returns:
            mu: Latent mean (B, z_dim)
        """
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to output.
        
        Args:
            z: Latent vector (B, z_dim)
            
        Returns:
            Reconstructed output (B, 3, 12, 96)
        """
        return self.decoder(z)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate samples from prior.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated samples (num_samples, 3, 12, 96)
        """
        z = torch.randn(num_samples, self.z_dim, device=device)
        return self.decode(z)


def build_vae_from_config(encoder_config, decoder_config) -> VAE:
    """
    Factory function to build VAE from configs.
    
    Args:
        encoder_config: EncoderConfig instance
        decoder_config: DecoderConfig instance
        
    Returns:
        VAE instance
    """
    from .encoder import Encoder
    from .decoder import Decoder
    
    encoder = Encoder(
        input_channels=encoder_config.input_channels,
        channels=encoder_config.channels,
        z_dim=encoder_config.z_dim,
        use_batchnorm=encoder_config.use_batchnorm,
        activation=encoder_config.activation,
        dropout=encoder_config.dropout
    )
    
    decoder = Decoder(
        z_dim=decoder_config.z_dim,
        channels=decoder_config.channels,
        output_channels=decoder_config.output_channels,
        output_size=tuple(decoder_config.output_size),
        use_batchnorm=decoder_config.use_batchnorm,
        activation=decoder_config.activation,
        dropout=decoder_config.dropout,
        sparsity_threshold=decoder_config.sparsity_threshold,
        output_scale=decoder_config.output_scale
    )
    
    return VAE(encoder, decoder)