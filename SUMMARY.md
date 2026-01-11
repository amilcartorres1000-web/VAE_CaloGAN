# VAE for Calorimeter Showers - Project Summary

## Overview
This project implements a Variational Autoencoder (VAE) to simulate electromagnetic showers in a multi-layered calorimeter. The goal is to generate high-fidelity synthetic data that mimics the complex spatial energy distributions of real particle physics simulations (calogan dataset), providing a faster alternative to Monte Carlo simulations (e.g., Geant4).

## Architecture
- **Input**: 3-layer calorimeter images ($3 \times 96$, $12 \times 12$, $12 \times 6$).
- **Encoder**: Convolutional neural network that maps input showers to a probabilistic latent space ($\mu, \sigma$).
- **Latent Space**: Compressed representation (size $N$) enabling continuous generation and interpolation.
- **Decoder**: Transposed convolutional network that reconstructs the 3-layer shower structure from sampled latent vectors.

## Key Features
- **Sparsity Handling**: Specialized preprocessing and loss functions to handle the sparse nature of calorimeter data.
- **Physics-Aware Evaluation**: Metrics focusing on energy conservation, layer-wise energy deposition, and shower shape.
- **Modular Design**: Separated configuration, modeling, training, and evaluation for scalability.

## Results
- **Generation Speed**: Orders of magnitude faster than traditional simulation.
- **Fidelity**: Reproduces key physics observables like total energy distribution and longitudinal shower profiles.

