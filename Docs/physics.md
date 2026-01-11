# Physics Documentation for CaloGAN VAE

## 1. Calorimeter Data Structure

The dataset consists of simulated calorimeter showers from the CaloGAN dataset. A calorimeter measures the energy deposits of particles as they traverse through layers of material (e.g., lead/liquid argon).

### Layers
The detector is segmented into 3 layers with varying granularity:
- **Layer 0**: High granularity, focuses on early shower development. Shape: $3 \times 96$ cells.
- **Layer 1**: Finest granularity, captures the shower core. Shape: $12 \times 12$ cells.
- **Layer 2**: Coarser granularity, captures the tail of the shower. Shape: $12 \times 6$ cells.

### Energy Deposition
Each cell contains a value representing the energy deposited (in MeV or GeV) by the incident particle and the secondary particles produced in the shower cascade.

## 2. VAE Physics Interpretation

The Variational Autoencoder (VAE) learns a compressed representation (latent space) of the calorimeter showers.

### Encoder (Recognition Model)
Maps high-dimensional shower images $x$ to a probabilistic latent space $z$.
- **Physics intuition**: The latent variables $z$ should capture independent physical characteristics of the shower, such as:
  - Total incident energy
  - Angle of incidence
  - Shower depth / longitudinal profile
  - Lateral spread

### Decoder (Generative Model)
Reconstructs the shower image $\hat{x}$ from a latent vector $z$.
- **Physics intuition**: Learned to produce realistic shower shapes that obey physical constraints (e.g., contiguous energy deposits, exponential decay in depth).

## 3. Evaluation Metrics

To validate the physics fidelity of the generated showers, we compare them to real Geant4 simulated showers using several metrics.

### Energy Distributions
- **Total Energy**: Sum of energy across all cells. The distribution should match the incoming particle energy spectrum.
- **Layer Energies**: Energy deposited in each layer. Important for particle identification (e.g., electrons deposit more energy earlier than pions).

### Sparsity
Calorimeter images are sparse (many zero-energy cells).
- **Metric**: Fraction of cells with zero energy.
- **Goal**: Generated images should reproduce the sparsity pattern of real data, avoiding "gray goo" (small non-zero values everywhere).

### Structural Fidelity (SSIM / MSE)
- While MSE measures pixel-level accuracy, it doesn't capture structural similarity well.
- We rely more on distribution comparisons (1D histograms) than pixel-wise error for generation quality.

### Consistency Check
- **Conservation of Energy**: The reconstructed energy should correlate strongly with the input energy.
