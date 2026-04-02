# JAX-NN-MNIST-Binary-Weights

A neural network for MNIST where inference runs entirely on **boolean arrays** and **integer arithmetic** — no floating point at test time. Trained end-to-end with JAX/Flax using an Augmented Lagrangian method that pushes continuous weights to {0, 1} and continuous thresholds to integers. Achieves ~92% accuracy.

https://github.com/user-attachments/assets/5aa496ec-83b6-40a3-875e-a9229f85bf69

## How It Works

### Inference (no floats)

At test time, every value is either a boolean or a small integer. Each hidden layer computes:

1. **Augment** with complements: `x_aug = [x, NOT(x)]` — doubles the input width so gates can match on both presence and absence of features
2. **Popcount**: `count = x_aug @ w` where `x_aug` is a boolean vector and `w` is a boolean matrix — this is just counting how many of the selected inputs are active (an integer)
3. **Threshold**: `output = (count > t)` where `t` is a learned integer threshold — produces a boolean

The output layer does the same popcount (`bool_hidden @ bool_weights`) to get an integer score per class, then picks the argmax. The entire forward pass is: boolean ANDs, integer additions (from the popcount), and integer comparisons.

### Training (continuous relaxation)

During training, everything is continuous so gradients can flow:

- **Weights** are continuous values in [0, 1], clipped after each step. The popcount `x @ w` becomes a soft weighted sum.
- **Threshold gates** use `sigmoid(alpha * (count - t))` as a smooth approximation of the hard `count > t` step function.
- **Thresholds** `t` are continuous learned parameters, initialized to the mean popcount observed on a calibration sample. This centers each gate so roughly half the training examples activate it.
- **Alpha annealing**: `alpha` (the sigmoid steepness) is linearly annealed from 3 to 100 over training, starting soft (smooth gradients) and ending hard (close to the discrete step function). Alpha is scaled by `1/sqrt(n_inputs)` to keep gradients stable regardless of layer width.
- **Per-layer alpha multipliers**: Deeper layers use progressively sharper sigmoids (1.0x, 1.5x, 2.25x) to compensate for receiving softer inputs from earlier layers.
- **Complement augmentation**: `[x, 1-x]` before each hidden layer gives gates access to both positive and negative literals.
- **Data augmentation**: Random +/-2 pixel shifts on the binary 28x28 images.

### Augmented Lagrangian binarization

After a warmup period (first 24 epochs), penalty terms gradually push the continuous parameters toward discrete values:

- **Weight penalty**: $c(w) = w^2(1 - w)^2$ — zero at 0 and 1, maximum at 0.5
- **Threshold penalty**: $c(t) = \sin^2(\pi t)$ — zero at every integer, maximum at half-integers
- **AL form**: $\lambda \cdot c + \frac{\rho}{2} c^2$ — the Lagrange multiplier $\lambda$ accumulates over epochs, while $\rho$ grows exponentially (1.1x per epoch, capped at 10)

This is more effective than simply adding a penalty because the multiplier $\lambda$ remembers past constraint violations, creating persistent pressure toward discreteness even for parameters that are slow to converge.

At the end of training, weights are rounded to {0, 1} (`w > 0.5`) and thresholds are rounded to the nearest integer. The network then runs with pure boolean/integer arithmetic.

### Architecture

```
Input: 784 binary pixels (28x28, thresholded at 0.5)
  |
  v  [x, NOT(x)]
1568 → ThresholdGate(1024): popcount > t → bool
  |
  v  [x, NOT(x)]
2048 → ThresholdGate(512):  popcount > t → bool
  |
  v  [x, NOT(x)]
1024 → ThresholdGate(256):  popcount > t → bool
  |
  v  popcount per class → argmax
10 classes
```

Each `ThresholdGate(n)` layer has:
- A boolean weight matrix (which inputs to count)
- An integer threshold vector (how many must be active for the gate to fire)

Total parameters: ~2.9M continuous during training, stored as ~2.9M bits + a few hundred integers after binarization.

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

Trains a 3-layer model (1024 → 512 → 256) end-to-end. Boolean accuracy is evaluated every 10 epochs after warmup, and the best model is saved to `best_model.pkl`.

### Pure Boolean Evaluation

```bash
uv run python main.py --eval
```

Loads the trained model, rounds weights to {0, 1} and thresholds to integers, and runs inference with zero floating point.

### Parameters

- `--epochs`: Number of training epochs (default: 80)
- `--warmup_epochs`: Epochs before binarization penalty starts (default: 24)
- `--lr`: Learning rate (default: 0.002)
- `--alpha_start`: Initial sigmoid steepness (default: 3.0)
- `--alpha_end`: Final sigmoid steepness (default: 100.0)
- `--rho_init`: Initial AL penalty parameter (default: 0.01)
- `--rho_inc`: Growth factor for rho per epoch (default: 1.1)
- `--rho_max`: Maximum rho (default: 10.0)
- `--layers`: Hidden layer sizes (default: 1024 512 256)
- `--no-augment`: Disable random shift data augmentation
- `--eval`: Run pure boolean evaluation only
- `--seed`: Random seed (default: 42)

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
