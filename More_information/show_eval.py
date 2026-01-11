import json
import sys

# Read the metrics file
with open('evaluation_results/eval_sparse_20251210_183600/metrics_comparison.json', 'r') as f:
    data = json.load(f)

print("="*80)
print("EVALUATION RESULTS SUMMARY")
print("="*80)

for mode in ['dense', 'sparse']:
    if mode in data:
        print(f"\n{mode.upper()} MODE:")
        print("-"*80)
        metrics = data[mode]
        
        # Energy metrics
        print(f"\nEnergy Statistics:")
        print(f"  Real  - Mean: {metrics.get('real_energy_mean', 0):.2f}, Std: {metrics.get('real_energy_std', 0):.2f}")
        print(f"  Gen   - Mean: {metrics.get('gen_energy_mean', 0):.2f}, Std: {metrics.get('gen_energy_std', 0):.2f}")
        print(f"  Error: {abs(metrics.get('real_energy_mean', 0) - metrics.get('gen_energy_mean', 0)):.2f}")
        
        # KS test
        print(f"\nKS Test:")
        print(f"  Statistic: {metrics.get('ks_stat', 0):.4f}")
        print(f"  P-value: {metrics.get('ks_pval', 0):.6f}")
        
        # Sparsity
        print(f"\nSparsity:")
        print(f"  Real: {metrics.get('real_sparsity', 0):.4f} ({metrics.get('real_sparsity', 0)*100:.2f}%)")
        print(f"  Gen:  {metrics.get('gen_sparsity', 0):.4f} ({metrics.get('gen_sparsity', 0)*100:.2f}%)")

print("\n" + "="*80)
