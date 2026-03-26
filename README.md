# JAX-NN-MNIST-Binary-Weights

A **Binary Neural Network** for MNIST where inference runs entirely on boolean arrays and integer arithmetic — no floating point at test time. Trained with JAX/Flax, achieves **94.2%** accuracy using only threshold gates.

https://github.com/user-attachments/assets/5aa496ec-83b6-40a3-875e-a9229f85bf69

## How It Works

### Pure Boolean Inference

At test time, each hidden layer computes:

1. **Count** active inputs: `count = bool_input @ bool_weights` (integer)
2. **Threshold**: `gate_fires = count > T` (boolean)

The output layer counts how many gates fired per class and picks the argmax. No sigmoid, no BatchNorm, no floats — just boolean arrays and integer comparisons.

### Training with Straight-Through Estimator

During training, the network uses continuous relaxations so gradients can flow:

- **Binary weights via STE:** Forward pass uses hard `w = (w_raw > 0)`, backward pass uses `sigmoid(w_raw)` gradients.
- **Binary activations via STE:** Same trick for hidden activations — hard threshold forward, smooth sigmoid backward.
- **BatchNorm** normalizes pre-activations during training; at inference, BN parameters are folded into integer thresholds.
- **Input augmentation:** Inputs are augmented as `[x, NOT(x)]` to provide both positive and negative literals (since threshold gates are monotone).

### Augmented Lagrangian Binarization

To push weights toward {0, 1}, the loss includes an Augmented Lagrangian penalty:

- Constraint violation: $c(w) = \sigma(w)(1 - \sigma(w))$ (zero when weight is binary)
- Penalty: $\lambda \cdot c(w) + \frac{\rho}{2} c(w)^2$
- Multipliers $\lambda$ update each epoch; $\rho$ is annealed upward

Binarization is delayed (no penalty for the first 20 epochs) so the network finds good features before being locked into binary.

### Architecture

```
Input: 784 binary pixels
  |
  v  augment with [x, NOT(x)]
1568 binary inputs
  |
  v  BinaryDense(2048) -> BN -> threshold -> bool
2048 threshold gates
  |
  v  BinaryDense(1024) -> BN -> threshold -> bool
1024 threshold gates
  |
  v  BinaryDense(512) -> BN -> threshold -> bool
512 threshold gates
  |
  v  BinaryDense(10) -> argmax of integer counts
10 classes
```

## Setup and Usage

This project uses `uv` for dependency management.

### Installation

```bash
uv sync
```

### Training

```bash
uv run python main.py
```

### Pure Boolean Evaluation

```bash
uv run python main.py --eval
```

This loads the trained model, extracts integer thresholds from BatchNorm, and runs inference with zero floating point.

### Parameters

- `--epochs`: Number of training epochs (default: 60)
- `--bs`: Batch size (default: 256)
- `--lr`: Learning rate (default: 0.005)
- `--warmup_epochs`: Epochs before binarization penalty starts (default: 20)
- `--rho_init`: Initial penalty parameter (default: 0.001)
- `--rho_inc`: Growth factor for rho per epoch (default: 1.1)
- `--rho_max`: Maximum rho (default: 10.0)
- `--no-augment`: Disable random shift data augmentation
- `--eval`: Run pure boolean evaluation only

### Visualization

```bash
uv run python visualize.py
```

Generates `forward_pass.mp4` showing the boolean forward pass for each digit: input image, gate activations at each layer, and integer output scores.

## Requirements

- `jax`, `flax`, `optax`
- `scikit-learn` (for MNIST data loading)
- `matplotlib` (for visualization)
- `numpy`
