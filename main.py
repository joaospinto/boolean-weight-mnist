import jax
jax.config.update('jax_platforms', 'cpu')
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
import argparse
import pickle
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# --- Data Loading ---

def get_datasets():
    cache_file = 'mnist_cache.npz'
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return (data['x_train'], data['y_train']), (data['x_test'], data['y_test'])

    print("Fetching MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    x, y = mnist['data'], mnist['target'].astype(int)
    x = (x / 255.0 > 0.5).astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=10000, random_state=42)
    np.savez(cache_file, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return (x_train, y_train), (x_test, y_test)


def augment_batch(images, rng):
    """Random ±2 pixel shifts on binary 28x28 images."""
    batch = images.reshape(-1, 28, 28)
    n = len(batch)
    shifts = rng.integers(-2, 3, size=(n, 2))  # shift_y, shift_x
    out = np.zeros_like(batch)
    for i in range(n):
        dy, dx = shifts[i]
        sy = max(0, dy)
        ey = min(28, 28 + dy)
        sx = max(0, dx)
        ex = min(28, 28 + dx)
        ty = max(0, -dy)
        tx = max(0, -dx)
        out[i, ty:ty+(ey-sy), tx:tx+(ex-sx)] = batch[i, sy:ey, sx:ex]
    return out.reshape(-1, 784)


# --- Initialization ---

def sparse_init(num_active=50, active_val=3.0):
    """Per output: num_active inputs start ON (w_raw=active_val), rest OFF (w_raw=-3)."""
    def init(key, shape, dtype=jnp.float32):
        scores = jax.random.uniform(key, shape)
        k = min(num_active, shape[0])
        threshold = jnp.sort(scores, axis=0)[-k]
        return jnp.where(scores >= threshold[None, :], active_val, -3.0).astype(dtype)
    return init


# --- Model ---

class BinaryDense(nn.Module):
    """Dense layer with binary (0/1) weights via Straight-Through Estimator."""
    features: int
    num_active: int = 50

    @nn.compact
    def __call__(self, x):
        shape = (x.shape[-1], self.features)
        w_raw = self.param('w', sparse_init(self.num_active, active_val=3.0), shape)

        # STE: binary forward, sigmoid backward
        w_soft = jax.nn.sigmoid(w_raw)
        w_hard = (w_raw > 0.0).astype(jnp.float32)
        w = w_soft + jax.lax.stop_gradient(w_hard - w_soft)

        return x @ w  # no bias — BatchNorm handles centering


def binary_activation(x, alpha=4.0):
    """STE for activations: binary (0/1) in forward, sigmoid gradient in backward."""
    soft = jax.nn.sigmoid(alpha * x)
    hard = (x > 0.0).astype(jnp.float32)
    return soft + jax.lax.stop_gradient(hard - soft)


class BinaryNN(nn.Module):
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, train=True):
        # Augment with negations: [x, NOT(x)] gives both pos and neg literals
        x = jnp.concatenate([x, 1.0 - x], axis=-1)
        # Layer 1: 1568 → 2048
        x = BinaryDense(2048, num_active=25)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = binary_activation(x, alpha=4.0)
        # Layer 2: 2048 → 1024
        x = BinaryDense(1024, num_active=30)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = binary_activation(x, alpha=4.0)
        # Layer 3: 1024 → 512
        x = BinaryDense(512, num_active=35)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = binary_activation(x, alpha=3.0)
        # Output: 512 → 10
        return BinaryDense(self.num_classes, num_active=80)(x)


# --- Training State (with batch_stats for BatchNorm) ---

class TrainState(train_state.TrainState):
    batch_stats: dict


# --- Loss ---

def loss_fn(params, batch_stats, images, labels, rho, lmbda):
    logits, updates = BinaryNN().apply(
        {'params': params, 'batch_stats': batch_stats},
        images, train=True,
        mutable=['batch_stats'])
    new_batch_stats = updates['batch_stats']

    ce = optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, 10)).mean()

    # Augmented Lagrangian: push weights toward {0,1}
    def penalty(p, l):
        if p.ndim < 2:  # skip biases and BN params (1D)
            return jnp.float32(0.0)
        w = jax.nn.sigmoid(p)
        c = w * (1.0 - w)
        return jnp.sum(l * c + (rho / 2.0) * c ** 2)

    pen = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(penalty, params, lmbda)))
    return ce + pen, (ce, new_batch_stats)


@jax.jit
def train_step(state, images, labels, rho, lmbda):
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (ce, new_batch_stats)), grads = grad_fn(
        state.params, state.batch_stats, images, labels, rho, lmbda)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)
    return state, ce


def evaluate(state, images, labels):
    correct = 0
    for i in range(0, len(images), 1000):
        logits = BinaryNN().apply(
            {'params': state.params, 'batch_stats': state.batch_stats},
            images[i:i+1000], train=False)
        correct += int(jnp.sum(jnp.argmax(logits, -1) == labels[i:i+1000]))
    return correct / len(images)


def binariness(params):
    """Fraction of weights near binary (1.0=all binary, 0.0=all at midpoint)."""
    vals = []
    for l in jax.tree_util.tree_leaves(params):
        if l.ndim >= 2:
            vals.append(float(jnp.mean(jax.nn.sigmoid(l) * (1 - jax.nn.sigmoid(l)))))
    return 1.0 - 4.0 * np.mean(vals) if vals else 1.0


# --- Pure Boolean/Integer Inference ---

def extract_boolean_model(params, batch_stats):
    """Extract binary weights and integer thresholds from trained model.

    Automatically detects the number of BinaryDense and BatchNorm layers.
    Hidden layers have 'w', 'threshold', 'scale_positive'.
    Output layer (last BinaryDense, no BN) has only 'w'.
    """
    # Find all BinaryDense and BatchNorm layers
    dense_names = sorted([k for k in params if k.startswith('BinaryDense_')],
                         key=lambda s: int(s.split('_')[1]))
    bn_names = sorted([k for k in params if k.startswith('BatchNorm_')],
                      key=lambda s: int(s.split('_')[1]))

    layers = []
    # Hidden layers: each has a matching BN
    for layer_name, bn_name in zip(dense_names[:-1], bn_names):
        w_raw = np.array(params[layer_name]['w'])
        w_bin = w_raw > 0.0

        scale = np.array(params[bn_name]['scale'])
        bias = np.array(params[bn_name]['bias'])
        mean = np.array(batch_stats[bn_name]['mean'])
        var = np.array(batch_stats[bn_name]['var'])
        eps = 1e-5
        std = np.sqrt(var + eps)

        raw_threshold = mean - bias * std / scale
        threshold = np.floor(raw_threshold).astype(np.int32)
        scale_positive = scale > 0

        layers.append({
            'w': w_bin,
            'threshold': threshold,
            'scale_positive': scale_positive,
        })

    # Output layer: no BN
    w_raw = np.array(params[dense_names[-1]]['w'])
    layers.append({'w': w_raw > 0.0})

    return layers


def boolean_forward(x_bool, layers):
    """Pure boolean/integer forward pass. No floating point."""
    x = np.concatenate([x_bool, ~x_bool], axis=-1)

    for layer in layers[:-1]:
        w = layer['w']
        threshold = layer['threshold']
        scale_pos = layer['scale_positive']

        counts = x.astype(np.int32) @ w.astype(np.int32)
        fired = counts > threshold[None, :]
        x = np.where(scale_pos[None, :], fired, ~fired)

    w_out = layers[-1]['w']
    scores = x.astype(np.int32) @ w_out.astype(np.int32)
    return np.argmax(scores, axis=-1)


def boolean_evaluate(model_path='best_model.pkl'):
    """Load trained model, extract boolean circuit, evaluate on test set."""
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)

    def to_numpy_dict(d):
        out = {}
        for k, v in d.items():
            if hasattr(v, 'items'):
                out[k] = {kk: np.array(vv) for kk, vv in v.items()}
            else:
                out[k] = np.array(v)
        return out

    params = to_numpy_dict(saved['params'])
    batch_stats = to_numpy_dict(saved['batch_stats'])

    layers = extract_boolean_model(params, batch_stats)

    print("Boolean network structure:")
    for i, layer in enumerate(layers[:-1]):
        w = layer['w']
        active = w.sum(axis=0)
        print(f"  Layer {i+1}: {w.shape[0]} bool inputs -> {w.shape[1]} threshold gates")
        print(f"    Connections per gate: min={int(active.min())}, "
              f"median={int(np.median(active))}, max={int(active.max())}")
        print(f"    Thresholds: min={int(layer['threshold'].min())}, "
              f"max={int(layer['threshold'].max())}")
        neg = (~layer['scale_positive']).sum()
        if neg > 0:
            print(f"    Inverted gates (negative BN scale): {neg}")
    w_out = layers[-1]['w']
    active_per_class = w_out.sum(axis=0)
    print(f"  Output: {w_out.shape[0]} bool inputs -> {w_out.shape[1]} classes (argmax of int counts)")
    print(f"    Connections per class: {[int(x) for x in active_per_class]}")
    total_ones = sum(int(l['w'].sum()) for l in layers)
    total_weights = sum(l['w'].size for l in layers)
    print(f"\nTotal binary weights: {total_ones} ones out of {total_weights}")

    (_, _), (x_test, y_test) = get_datasets()
    x_bool = x_test > 0.5
    preds = boolean_forward(x_bool, layers)
    correct = int((preds == y_test).sum())
    print(f"\nPure boolean accuracy: {correct}/{len(y_test)} = {correct/len(y_test):.4f}")
    return correct / len(y_test)


# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help='Epochs before binarization penalty kicks in')
    parser.add_argument('--rho_init', type=float, default=0.001)
    parser.add_argument('--rho_inc', type=float, default=1.1)
    parser.add_argument('--rho_max', type=float, default=10.0)
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.add_argument('--eval', action='store_true',
                        help='Run pure boolean evaluation (no floating point)')
    args = parser.parse_args()

    if args.eval:
        boolean_evaluate()
        return

    (x_train, y_train), (x_test, y_test) = get_datasets()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    model = BinaryNN()
    key = jax.random.PRNGKey(42)
    variables = model.init(key, jnp.ones((1, 784)), train=True)

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(variables['params']))
    print(f"Parameters: {n_params}")

    num_batches = len(x_train) // args.bs
    total_steps = args.epochs * num_batches
    warmup_steps = 500
    schedule = optax.join_schedules([
        optax.linear_schedule(0.0, args.lr, warmup_steps),
        optax.cosine_decay_schedule(args.lr, total_steps - warmup_steps, alpha=0.01),
    ], [warmup_steps])
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        batch_stats=variables['batch_stats'])

    rho = 0.0  # Start with no binarization pressure
    lmbda = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    best_acc = 0.0
    aug_rng = np.random.default_rng(42)

    for epoch in range(args.epochs):
        perm = np.random.permutation(len(x_train))
        losses = []

        for i in range(0, len(x_train) - args.bs + 1, args.bs):
            idx = perm[i:i + args.bs]
            batch_x = x_train[idx]
            if args.augment:
                batch_x = augment_batch(batch_x, aug_rng)
            state, ce = train_step(state, batch_x, y_train[idx], rho, lmbda)
            losses.append(float(ce))

        # After warmup, start binarization pressure
        if epoch >= args.warmup_epochs:
            lmbda = jax.tree_util.tree_map(
                lambda l, p: l + rho * jax.nn.sigmoid(p) * (1.0 - jax.nn.sigmoid(p))
                if p.ndim >= 2 else l,
                lmbda, state.params)
            if rho == 0.0:
                rho = args.rho_init
            rho = min(rho * args.rho_inc, args.rho_max)

        acc = evaluate(state, x_test, y_test)
        bm = binariness(state.params)

        print(f"Ep {epoch:2d}: CE={np.mean(losses):.3f} Acc={acc:.3f} "
              f"Bin={bm:.3f} rho={rho:.4f}")

        if acc > best_acc:
            best_acc = acc
            with open('best_model.pkl', 'wb') as f:
                pickle.dump({
                    'params': state.params,
                    'batch_stats': state.batch_stats,
                }, f)

    print(f"\nBest accuracy: {best_acc:.3f}")
    print("\nRunning pure boolean evaluation...")
    boolean_evaluate()


if __name__ == '__main__':
    main()
