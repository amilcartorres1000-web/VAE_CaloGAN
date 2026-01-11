"""
Quick diagnostic to check if thresholds are being applied during generation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from calogan_vae.models import build_vae_from_config
from calogan_vae.config import ExperimentConfig

# Load checkpoint
print("Loading checkpoint...")
checkpoint = torch.load('checkpoints/fix_generation/best_model.pth', map_location='cpu')

# Get config
config = checkpoint.get('config', ExperimentConfig())

# Build model
print("Building model...")
model = build_vae_from_config(config.encoder, config.decoder)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

# Check thresholds
print("\n" + "="*60)
print("DECODER THRESHOLD VALUES")
print("="*60)
if hasattr(model.decoder, 'threshold'):
    for i, thresh in enumerate(model.decoder.threshold):
        print(f"Channel {i}: {thresh.item():.6f}")
else:
    print("❌ No threshold attribute found!")
print("="*60)

# Generate samples
print("\nGenerating 100 samples...")
with torch.no_grad():
    z = torch.randn(100, model.z_dim)
    gen = model.decode(z)

# Analyze output
print("\n" + "="*60)
print("GENERATED OUTPUT ANALYSIS")
print("="*60)
print(f"Shape: {gen.shape}")
print(f"Mean: {gen.mean().item():.6f}")
print(f"Std: {gen.std().item():.6f}")
print(f"Min: {gen.min().item():.6f}")
print(f"Max: {gen.max().item():.6f}")
print(f"Sparsity (< 1e-6): {(gen < 1e-6).float().mean().item():.4f} ({(gen < 1e-6).float().mean().item()*100:.2f}%)")
print(f"Sparsity (== 0): {(gen == 0).float().mean().item():.4f} ({(gen == 0).float().mean().item()*100:.2f}%)")

# Check per-channel
print("\nPer-channel analysis:")
for i in range(gen.shape[1]):
    channel_data = gen[:, i, :, :]
    sparsity = (channel_data < 1e-6).float().mean().item()
    print(f"  Channel {i}: sparsity={sparsity:.4f} ({sparsity*100:.2f}%), mean={channel_data.mean().item():.6f}")

# Check if values are exactly at threshold
print("\nValues near threshold (0.01):")
near_threshold = ((gen > 0.009) & (gen < 0.011)).sum().item()
print(f"  Values in range [0.009, 0.011]: {near_threshold}")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

gen_sparsity = (gen < 1e-6).float().mean().item()
if gen_sparsity > 0.5:
    print("✅ Thresholds ARE being applied!")
    print(f"   Generated sparsity: {gen_sparsity*100:.2f}%")
else:
    print("❌ Thresholds are NOT being applied!")
    print(f"   Generated sparsity: {gen_sparsity*100:.2f}%")
    print("   Expected: >50% if thresholds work")
    
    # Additional checks
    print("\nPossible reasons:")
    print("1. Threshold values too low (all values pass threshold)")
    print("2. Model outputs are all above threshold")
    print("3. Softplus activation produces values > 0.01 everywhere")
    
    # Check raw decoder output before softplus
    print("\nChecking decoder output BEFORE softplus...")
    with torch.no_grad():
        x = model.decoder.fc(z)
        x = x.view(x.size(0), -1, 3, 3)
        x = model.decoder.deconv_layers(x)
        print(f"  Before softplus - Mean: {x.mean().item():.6f}, Min: {x.min().item():.6f}, Max: {x.max().item():.6f}")
        
        x_after_softplus = torch.nn.functional.softplus(x)
        print(f"  After softplus - Mean: {x_after_softplus.mean().item():.6f}, Min: {x_after_softplus.mean().item():.6f}")
        print(f"  % values > 0.01: {(x_after_softplus > 0.01).float().mean().item()*100:.2f}%")

print("="*60)
