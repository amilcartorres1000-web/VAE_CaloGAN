"""
Check learned threshold values from new training.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch

# Load new checkpoint
print("Loading NEW checkpoint (with threshold=0.05 initialization)...")
checkpoint = torch.load('checkpoints/fix_generation/best_model.pth', map_location='cpu')

state_dict = checkpoint['model_state_dict']

print("\n" + "="*60)
print("LEARNED THRESHOLD VALUES")
print("="*60)
if 'decoder.threshold' in state_dict:
    thresholds = state_dict['decoder.threshold']
    print(f"Thresholds: {thresholds}")
    for i, thresh in enumerate(thresholds):
        print(f"  Channel {i}: {thresh.item():.6f}")
    print(f"\nMean threshold: {thresholds.mean().item():.6f}")
    print(f"Min threshold: {thresholds.min().item():.6f}")
    print(f"Max threshold: {thresholds.max().item():.6f}")
else:
    print("❌ No thresholds found!")

print("="*60)

# Compare with initialization
print("\nComparison:")
print(f"  Initialization: 0.050000")
print(f"  After training: {thresholds.mean().item():.6f}")
print(f"  Change: {(thresholds.mean().item() - 0.05):.6f}")

if thresholds.mean().item() > 0.05:
    print("\n✅ Thresholds INCREASED during training (good!)")
    print("   Model learned it needs higher thresholds for sparsity")
elif thresholds.mean().item() < 0.05:
    print("\n⚠️ Thresholds DECREASED during training")
    print("   Model may need different initialization or loss weights")
else:
    print("\n⚠️ Thresholds didn't change much")
    print("   May need stronger sparsity loss or more training")
