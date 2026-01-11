"""
Tests for model architectures.
"""
import pytest
import torch

from calogan_vae.models import Encoder, Decoder, VAE


def test_encoder_forward():
    """Test encoder forward pass"""
    encoder = Encoder(
        input_channels=3,
        channels=[32, 64, 128],
        z_dim=64
    )
    
    x = torch.randn(8, 3, 12, 96)
    mu, logvar = encoder(x)
    
    assert mu.shape == (8, 64)
    assert logvar.shape == (8, 64)


def test_decoder_forward():
    """Test decoder forward pass"""
    decoder = Decoder(
        z_dim=64,
        channels=[128, 64, 32, 16],
        output_channels=3,
        output_size=(12, 96)
    )
    
    z = torch.randn(8, 64)
    recon = decoder(z)
    
    assert recon.shape == (8, 3, 12, 96)


def test_vae_forward():
    """Test VAE forward pass"""
    encoder = Encoder(z_dim=64)
    decoder = Decoder(z_dim=64)
    vae = VAE(encoder, decoder)
    
    x = torch.randn(8, 3, 12, 96)
    recon, mu, logvar = vae(x)
    
    assert recon.shape == x.shape
    assert mu.shape == (8, 64)
    assert logvar.shape == (8, 64)


def test_vae_sample():
    """Test VAE sampling"""
    encoder = Encoder(z_dim=64)
    decoder = Decoder(z_dim=64)
    vae = VAE(encoder, decoder)
    
    samples = vae.sample(num_samples=10, device=torch.device('cpu'))
    
    assert samples.shape == (10, 3, 12, 96)


if __name__ == '__main__':
    pytest.main([__file__])