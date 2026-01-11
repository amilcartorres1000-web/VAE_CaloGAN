"""
Decoder architectures for VAE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    Convolutional decoder for calorimeter data.
    
    Architecture:
    (z_dim,) -> FC -> Conv transpose layers -> (3, 12, 96)
    """
    
    def __init__(
        self,
        z_dim: int = 64,
        channels: list = None,
        output_channels: int = 3,
        output_size: tuple = (12, 96),
        use_batchnorm: bool = True,
        activation: str = 'leaky_relu',
        dropout: float = 0.0,
        sparsity_threshold: float = 0.1,
        output_scale: float = 0.05
    ):
        super().__init__()
        
        if channels is None:
            channels = [128, 64, 32, 16]
        
        self.z_dim = z_dim
        self.output_channels = output_channels
        self.output_size = output_size
        self.sparsity_threshold = sparsity_threshold
        self.output_scale = output_scale
        
        # Learnable threshold (one per output channel)
        # Initialize with higher values to match softplus output distribution
        # Softplus typically outputs values in range [0.01, 0.2], so threshold should be ~0.05
        self.threshold = nn.Parameter(torch.ones(output_channels) * 0.12)
        
        # Initial FC layer
        self.fc = nn.Linear(z_dim, channels[0] * 3 * 3)
        
        # Build deconvolutional layers
        layers = []
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
        
        # Final layer (no activation, no batchnorm)
        layers.append(nn.ConvTranspose2d(channels[-1], output_channels, kernel_size=3, stride=1, padding=1))
        
        self.deconv_layers = nn.Sequential(*layers)
    
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to calorimeter shower."""
        # FC layer
        x = self.fc(z)
        x = x.view(x.size(0), -1, 3, 3)
        
        # Deconvolutional layers
        x = self.deconv_layers(x)
        
        # Resize to target size
        if x.shape[2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        
        # Apply softplus activation (smooth, always positive)
        x = F.softplus(x)
        
        # Apply learned threshold per channel
        # This allows the model to learn optimal sparsity for each layer
        for i in range(self.output_channels):
            # Threshold values below learned threshold to zero
            x[:, i, :, :] = torch.where(
                x[:, i, :, :] > self.threshold[i],
                x[:, i, :, :],
                torch.zeros_like(x[:, i, :, :])
            )
        
        return x