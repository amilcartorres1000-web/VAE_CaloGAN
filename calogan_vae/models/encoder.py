"""
Encoder architectures for VAE.
"""
import torch
import torch.nn as nn
from typing import Tuple

class Encoder(nn.Module):
    """
    Convolutional encoder for calorimeter data.
    
    Architecture:
    Input (3, 12, 96) -> Conv layers -> (z_dim,) mu and logvar
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        channels: list = None,
        z_dim: int = 64,
        use_batchnorm: bool = True,
        activation: str = 'leaky_relu',
        dropout: float = 0.0
    ):
        super().__init__()
        
        if channels is None:
            channels = [32, 64, 128]
        
        self.z_dim = z_dim
        
        # Build convolutional layers
        layers = []
        in_ch = input_channels
        
        for out_ch in channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = out_ch
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))
        
        # Fully connected layers
        fc_input_dim = channels[-1] * 3 * 3
        self.fc_mu = nn.Linear(fc_input_dim, z_dim)
        self.fc_logvar = nn.Linear(fc_input_dim, z_dim)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif name == 'elu':
            return nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (B, 3, 12, 96)
            
        Returns:
            mu: Mean of latent distribution (B, z_dim)
            logvar: Log variance of latent distribution (B, z_dim)
        """
        # Convolutional encoding
        x = self.conv_layers(x)  # (B, C, H', W')
        
        # Adaptive pooling
        x = self.adaptive_pool(x)  # (B, C, 3, 3)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (B, C*3*3)
        
        # Get distribution parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar