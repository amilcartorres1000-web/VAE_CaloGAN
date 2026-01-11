import torch

checkpoint = torch.load('checkpoints/fix_generation/best_model.pth')
state_dict = checkpoint['model_state_dict']

# Check for threshold parameter
if 'decoder.threshold' in state_dict:
    print("✅ Threshold found in checkpoint")
    print(f"Values: {state_dict['decoder.threshold']}")
else:
    print("❌ Threshold NOT in checkpoint")
    print("This checkpoint was saved BEFORE adding learnable thresholds")