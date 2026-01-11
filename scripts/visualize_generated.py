"""
Visualize generated calorimeter showers.

Usage:
    python scripts/visualize_generated.py --input generated_showers.h5
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

def load_generated_data(h5_path: str):
    """Load generated samples from HDF5"""
    with h5py.File(h5_path, 'r') as f:
        data = np.array(f['generated_showers'])
        print(f"Loaded {data.shape[0]} generated samples")
        print(f"Shape: {data.shape}")
    return data

def plot_energy_distribution(data: np.ndarray, save_path: str = None):
    """Plot total energy distribution"""
    energies = data.sum(axis=(1, 2, 3))
    
    plt.figure(figsize=(10, 6))
    plt.hist(energies, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Total Energy', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Generated Shower Energy Distribution\n(Mean: {energies.mean():.1f} ± {energies.std():.1f})', 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def plot_layer_energies(data: np.ndarray, save_path: str = None):
    """Plot per-layer energy distributions"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i in range(3):
        layer_energy = data[:, i, :, :].sum(axis=(1, 2))
        
        axes[i].hist(layer_energy, bins=40, alpha=0.7, edgecolor='black', color=f'C{i}')
        axes[i].set_xlabel('Energy', fontsize=11)
        axes[i].set_ylabel('Count', fontsize=11)
        axes[i].set_title(f'Layer {i}\n(Mean: {layer_energy.mean():.1f} ± {layer_energy.std():.1f})', 
                         fontsize=12)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def plot_sample_showers(data: np.ndarray, n_samples: int = 6, save_path: str = None):
    """Plot example shower images"""
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 2.5*n_samples))
    
    for i in range(n_samples):
        for layer in range(3):
            ax = axes[i, layer] if n_samples > 1 else axes[layer]
            
            # Plot log-scale for better visibility
            img = np.log1p(data[i, layer, :, :])
            
            im = ax.imshow(img, cmap='hot', aspect='auto', origin='lower')
            ax.set_title(f'Sample {i+1}, Layer {layer}', fontsize=10)
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Generated Calorimeter Showers (log scale)', fontsize=14, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def plot_projections(data: np.ndarray, n_samples: int = 4, save_path: str = None):
    """Plot X and Y projections of showers"""
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3*n_samples))
    
    for i in range(n_samples):
        # Sum over all layers
        shower = data[i].sum(axis=0)  # (12, 96)
        
        # X projection (sum over Y axis)
        x_proj = shower.sum(axis=0)
        
        # Y projection (sum over X axis)
        y_proj = shower.sum(axis=1)
        
        # Plot X projection
        axes[i, 0].plot(x_proj, linewidth=2, color='C0')
        axes[i, 0].fill_between(range(len(x_proj)), x_proj, alpha=0.3)
        axes[i, 0].set_xlabel('X Position', fontsize=10)
        axes[i, 0].set_ylabel('Energy', fontsize=10)
        axes[i, 0].set_title(f'Sample {i+1} - X Projection', fontsize=11)
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot Y projection
        axes[i, 1].plot(y_proj, linewidth=2, color='C1')
        axes[i, 1].fill_between(range(len(y_proj)), y_proj, alpha=0.3)
        axes[i, 1].set_xlabel('Y Position', fontsize=10)
        axes[i, 1].set_ylabel('Energy', fontsize=10)
        axes[i, 1].set_title(f'Sample {i+1} - Y Projection', fontsize=11)
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def plot_sparsity_analysis(data: np.ndarray, save_path: str = None):
    """Analyze sparsity patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Overall sparsity
    threshold = 1e-6
    sparsity = (data < threshold).mean()
    
    axes[0, 0].text(0.5, 0.5, f'Overall Sparsity\n{sparsity:.1%}', 
                   ha='center', va='center', fontsize=24, 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].axis('off')
    
    # Per-layer sparsity
    layer_sparsity = [(data[:, i] < threshold).mean() for i in range(3)]
    axes[0, 1].bar(range(3), layer_sparsity, color=['C0', 'C1', 'C2'], alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Layer', fontsize=11)
    axes[0, 1].set_ylabel('Sparsity', fontsize=11)
    axes[0, 1].set_title('Sparsity per Layer', fontsize=12)
    axes[0, 1].set_xticks(range(3))
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Non-zero values distribution
    non_zero_values = data[data > threshold].flatten()
    axes[1, 0].hist(non_zero_values, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Energy Value (non-zero)', fontsize=11)
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title('Distribution of Non-Zero Values', fontsize=12)
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sparsity per sample
    sample_sparsity = (data < threshold).mean(axis=(1, 2, 3))
    axes[1, 1].hist(sample_sparsity, bins=40, alpha=0.7, edgecolor='black', color='C2')
    axes[1, 1].set_xlabel('Sparsity', fontsize=11)
    axes[1, 1].set_ylabel('Count', fontsize=11)
    axes[1, 1].set_title('Sparsity Distribution Across Samples', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def main(args):
    """Main visualization function"""
    print("="*60)
    print("VISUALIZING GENERATED CALORIMETER SHOWERS")
    print("="*60)
    print(f"Input file: {args.input}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data
    print("\nLoading data...")
    data = load_generated_data(args.input)
    
    print(f"\nStatistics:")
    print(f"  Total energy: {data.sum(axis=(1,2,3)).mean():.1f} ± {data.sum(axis=(1,2,3)).std():.1f}")
    print(f"  Sparsity: {(data < 1e-6).mean():.3f}")
    print(f"  Min/Max: [{data.min():.2e}, {data.max():.2e}]")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    print("1. Energy distribution...")
    plot_energy_distribution(data, 
        save_path=output_dir / f'energy_distribution_{timestamp}.png')
    
    print("2. Layer energies...")
    plot_layer_energies(data, 
        save_path=output_dir / f'layer_energies_{timestamp}.png')
    
    print("3. Sample showers...")
    plot_sample_showers(data, n_samples=args.n_samples,
        save_path=output_dir / f'sample_showers_{timestamp}.png')
    
    print("4. Projections...")
    plot_projections(data, n_samples=4,
        save_path=output_dir / f'projections_{timestamp}.png')
    
    print("5. Sparsity analysis...")
    plot_sparsity_analysis(data, 
        save_path=output_dir / f'sparsity_analysis_{timestamp}.png')
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"All plots saved to: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize generated calorimeter showers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', type=str, default='generated_showers.h5',
                       help='Path to generated HDF5 file')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for plots')
    parser.add_argument('--n_samples', type=int, default=6,
                       help='Number of sample showers to plot')
    
    args = parser.parse_args()
    main(args)