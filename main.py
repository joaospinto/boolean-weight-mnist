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
    """Random +/-2 pixel shifts on binary 28x28 images."""
    batch = images.reshape(-1, 28, 28)
    n = len(batch)
    shifts = rng.integers(-2, 3, size=(n, 2))
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


# --- Model ---

class ThresholdGateLayer(nn.Module):
    """Layer of threshold gates with continuous weights in [0,1].

    Weights pushed to {0,1} by AL penalty w^2*(1-w)^2.
    Thresholds pushed to integers by AL penalty sin^2(pi*t).
    Gate fires when: sum(w * x) > t
    """
    features: int

    @nn.compact
    def __call__(self, x, alpha=5.0):
        n_in = x.shape[-1]

        def w_init(key, shape, dtype=jnp.float32):
            return jax.random.uniform(key, shape, dtype=dtype)

        w = self.param('w', w_init, (n_in, self.features))
        w = jnp.clip(w, 0.0, 1.0)
        counts = x @ w

        def t_init(key, shape):
            noise = jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)
            return jnp.full(shape, jnp.float32(n_in) * 0.25) + noise

        t = self.param('t', t_init, (self.features,))

        scaled_alpha = alpha / jnp.sqrt(jnp.float32(n_in))
        return jax.nn.sigmoid(scaled_alpha * (counts - t))


class OutputLayer(nn.Module):
    """Output layer: raw weighted sum -> logits."""
    features: int

    @nn.compact
    def __call__(self, x):
        n_in = x.shape[-1]

        def w_init(key, shape, dtype=jnp.float32):
            return jax.random.uniform(key, shape, dtype=dtype)

        w = self.param('w', w_init, (n_in, self.features))
        w = jnp.clip(w, 0.0, 1.0)
        return (x @ w) * (10.0 / n_in)


class BinaryNN(nn.Module):
    """End-to-end binary neural network with complement augmentation between layers.

    Each hidden layer: [x, 1-x] -> ThresholdGateLayer
    Output: hidden -> OutputLayer (no complements)
    """
    hidden_layers: tuple = (1024, 512, 256)
    num_classes: int = 10
    # Per-layer alpha multipliers: deeper layers use sharper sigmoid
    # to compensate for softer inputs from previous layers
    alpha_mults: tuple = (1.0, 1.5, 2.25)

    @nn.compact
    def __call__(self, x, alpha=5.0):
        for i, hidden in enumerate(self.hidden_layers):
            x = jnp.concatenate([x, 1.0 - x], axis=-1)
            mult = self.alpha_mults[i] if i < len(self.alpha_mults) else 1.0
            x = ThresholdGateLayer(hidden, name=f'ThresholdGateLayer_{i}')(
                x, alpha=alpha * mult)
        return OutputLayer(self.num_classes, name='OutputLayer_0')(x)


def calibrate_thresholds(params, x_sample, alpha=5.0, alpha_mults=(1.0, 1.5, 2.0)):
    """Set each threshold to the mean count observed on a sample of data."""
    x = x_sample
    new_params = dict(params)
    layer_names = sorted([k for k in params if k.startswith('ThresholdGateLayer_')],
                         key=lambda s: int(s.split('_')[1]))
    for idx, name in enumerate(layer_names):
        x = jnp.concatenate([x, 1.0 - x], axis=-1)
        w = params[name]['w']
        n_in = w.shape[0]
        counts = x @ w
        mean_counts = jnp.mean(counts, axis=0)
        new_layer = dict(params[name])
        new_layer['t'] = mean_counts
        new_params[name] = new_layer
        mult = alpha_mults[idx] if idx < len(alpha_mults) else 1.0
        scaled_alpha = (alpha * mult) / jnp.sqrt(jnp.float32(n_in))
        x = jax.nn.sigmoid(scaled_alpha * (counts - mean_counts))
    return new_params


# --- Loss ---

def loss_fn(params, model, images, labels, rho, lmbda, alpha):
    logits = model.apply({'params': params}, images, alpha=alpha)
    ce = optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, 10)).mean()

    def penalty(p, l):
        if p.ndim >= 2:
            c = p ** 2 * (1.0 - p) ** 2
            return jnp.sum(l * c + (rho / 2.0) * c ** 2)
        elif p.ndim == 1 and p.shape[0] > 10:
            c = jnp.sin(jnp.pi * p) ** 2
            return jnp.sum(l * c + (rho / 2.0) * c ** 2)
        return jnp.float32(0.0)

    pen = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(penalty, params, lmbda)))
    return ce + pen, ce


def make_train_step(model):
    @jax.jit
    def train_step(state, images, labels, rho, lmbda, alpha):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, ce), grads = grad_fn(
            state.params, model, images, labels, rho, lmbda, alpha)
        state = state.apply_gradients(grads=grads)
        return state, ce
    return train_step


def evaluate(model, state, images, labels, alpha):
    correct = 0
    for i in range(0, len(images), 1000):
        logits = model.apply(
            {'params': state.params}, images[i:i+1000], alpha=alpha)
        correct += int(jnp.sum(jnp.argmax(logits, -1) == labels[i:i+1000]))
    return correct / len(images)


def discreteness(params):
    """How close weights are to {0,1} and thresholds to integers."""
    w_scores, t_scores = [], []
    for l in jax.tree_util.tree_leaves(params):
        if l.ndim >= 2:
            w_scores.append(float(1.0 - jnp.mean(l ** 2 * (1.0 - l) ** 2) / 0.0625))
        elif l.ndim == 1 and l.shape[0] > 10:
            t_scores.append(float(1.0 - jnp.mean(jnp.sin(jnp.pi * l) ** 2)))
    w_disc = np.mean(w_scores) if w_scores else 1.0
    t_disc = np.mean(t_scores) if t_scores else 1.0
    return w_disc, t_disc


# --- Pure Boolean/Integer Inference ---

def extract_boolean_model(params):
    """Extract binary weights and integer thresholds from trained model."""
    gate_names = sorted([k for k in params if k.startswith('ThresholdGateLayer_')],
                        key=lambda s: int(s.split('_')[1]))
    layers = []
    for name in gate_names:
        w = np.array(params[name]['w'])
        t = np.array(params[name]['t'])
        layers.append({'w': w > 0.5, 't': np.round(t).astype(np.int32)})

    out_w = np.array(params['OutputLayer_0']['w'])
    layers.append({'w': out_w > 0.5})
    return layers


def boolean_apply_layer(x_bool, layer):
    """Apply one boolean threshold gate layer: [x, ~x] @ w > t."""
    x = np.concatenate([x_bool, ~x_bool], axis=-1)
    w = layer['w']
    t = layer['t']
    counts = x.astype(np.int32) @ w.astype(np.int32)
    return counts > t[None, :]


def boolean_forward(x_bool, layers, return_activations=False):
    """Pure boolean/integer forward pass through stacked layers.

    Each hidden layer: [x, ~x] @ bool_w > int_t -> bool
    Output layer: x @ bool_w -> int_scores -> argmax
    """
    x = x_bool
    activations = [] if return_activations else None

    for layer in layers[:-1]:
        x = boolean_apply_layer(x, layer)
        if return_activations:
            activations.append(x)

    w_out = layers[-1]['w']
    scores = x.astype(np.int32) @ w_out.astype(np.int32)

    if return_activations:
        return np.argmax(scores, axis=-1), activations, scores
    return np.argmax(scores, axis=-1)


def print_boolean_structure(layers):
    """Print the structure of a boolean network."""
    for i, layer in enumerate(layers[:-1]):
        w = layer['w']
        t = layer['t']
        active = w.sum(axis=0)
        print(f"  Layer {i+1}: {w.shape[0]} bool inputs -> {w.shape[1]} threshold gates")
        print(f"    Connections per gate: min={int(active.min())}, "
              f"median={int(np.median(active))}, max={int(active.max())}")
        print(f"    Thresholds: min={int(t.min())}, median={int(np.median(t))}, max={int(t.max())}")
    w_out = layers[-1]['w']
    active_per_class = w_out.sum(axis=0)
    print(f"  Output: {w_out.shape[0]} bool inputs -> {w_out.shape[1]} classes")
    print(f"    Connections per class: {[int(x) for x in active_per_class]}")
    total_ones = sum(int(l['w'].sum()) for l in layers)
    total_weights = sum(l['w'].size for l in layers)
    print(f"\nTotal binary weights: {total_ones} ones out of {total_weights}")


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

    # Handle both old format (saved['layers']) and new (saved['params'])
    if 'layers' in saved:
        layers = saved['layers']
    else:
        params = to_numpy_dict(saved['params'])
        layers = extract_boolean_model(params)

    print("Boolean network structure:")
    print_boolean_structure(layers)

    (_, _), (x_test, y_test) = get_datasets()
    x_bool = x_test > 0.5
    preds = boolean_forward(x_bool, layers)
    correct = int((preds == y_test).sum())
    print(f"\nPure boolean accuracy: {correct}/{len(y_test)} = {correct/len(y_test):.4f}")
    return correct / len(y_test)


# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--warmup_epochs', type=int, default=24,
                        help='Epochs before discretization penalty kicks in')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--alpha_start', type=float, default=3.0)
    parser.add_argument('--alpha_end', type=float, default=100.0)
    parser.add_argument('--rho_init', type=float, default=0.01)
    parser.add_argument('--rho_inc', type=float, default=1.1)
    parser.add_argument('--rho_max', type=float, default=10.0)
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.add_argument('--layers', type=int, nargs='+', default=[1024, 512, 256],
                        help='Hidden layer sizes')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval', action='store_true',
                        help='Run pure boolean evaluation (no floating point)')
    args = parser.parse_args()

    if args.eval:
        boolean_evaluate()
        return

    (x_train, y_train), (x_test, y_test) = get_datasets()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    hidden = tuple(args.layers)
    # Per-layer alpha multipliers: deeper layers sharper to compensate for soft inputs
    alpha_mults = tuple(1.5 ** i for i in range(len(hidden)))
    print(f"Architecture: {hidden}, alpha multipliers: {tuple(f'{m:.2f}' for m in alpha_mults)}")

    model = BinaryNN(
        hidden_layers=hidden,
        alpha_mults=alpha_mults,
    )

    key = jax.random.PRNGKey(args.seed)
    alpha_start = args.alpha_start
    variables = model.init(key, jnp.ones((1, 784)), alpha=alpha_start)

    # Calibrate thresholds
    cal_sample = jnp.array(x_train[:1000])
    variables = {'params': calibrate_thresholds(
        variables['params'], cal_sample, alpha=alpha_start, alpha_mults=alpha_mults)}

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(variables['params']))
    print(f"Parameters: {n_params}")

    num_batches = len(x_train) // args.bs
    total_steps = args.epochs * num_batches
    warmup_steps = 500
    schedule = optax.join_schedules([
        optax.linear_schedule(0.0, args.lr, warmup_steps),
        optax.cosine_decay_schedule(args.lr, total_steps - warmup_steps, alpha=0.1),
    ], [warmup_steps])
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables['params'], tx=tx)

    train_step = make_train_step(model)
    rho = 0.0
    lmbda = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    best_bool_acc = 0.0
    aug_rng = np.random.default_rng(42)

    for epoch in range(args.epochs):
        t = min(epoch / max(args.epochs - 1, 1), 1.0)
        alpha = args.alpha_start + (args.alpha_end - args.alpha_start) * t
        alpha_jnp = jnp.float32(alpha)

        perm = np.random.permutation(len(x_train))
        losses = []

        for i in range(0, len(x_train) - args.bs + 1, args.bs):
            idx = perm[i:i + args.bs]
            batch_x = x_train[idx]
            if args.augment:
                batch_x = augment_batch(batch_x, aug_rng)
            state, ce = train_step(
                state, batch_x, y_train[idx], rho, lmbda, alpha_jnp)
            losses.append(float(ce))

        if epoch >= args.warmup_epochs:
            def update_lmbda(l, p):
                if p.ndim >= 2:
                    return l + rho * p ** 2 * (1.0 - p) ** 2
                elif p.ndim == 1 and p.shape[0] > 10:
                    return l + rho * jnp.sin(jnp.pi * p) ** 2
                return l
            lmbda = jax.tree_util.tree_map(update_lmbda, lmbda, state.params)
            if rho == 0.0:
                rho = args.rho_init
            rho = min(rho * args.rho_inc, args.rho_max)

        soft_acc = evaluate(model, state, x_test, y_test, alpha_jnp)
        w_disc, t_disc = discreteness(state.params)

        # Periodic boolean evaluation (every 10 epochs after warmup)
        bool_acc_str = ""
        if epoch >= args.warmup_epochs and epoch % 10 == 0:
            params_np = {k: ({kk: np.array(vv) for kk, vv in v.items()}
                            if hasattr(v, 'items') else np.array(v))
                        for k, v in state.params.items()}
            bool_layers = extract_boolean_model(params_np)
            preds = boolean_forward(x_test > 0.5, bool_layers)
            bool_acc = (preds == y_test).mean()
            bool_acc_str = f" Bool={bool_acc:.3f}"
            if bool_acc > best_bool_acc:
                best_bool_acc = bool_acc
                with open('best_model.pkl', 'wb') as f:
                    pickle.dump({'params': state.params}, f)

        # Always save if best soft accuracy and no bool eval yet
        if epoch < args.warmup_epochs and soft_acc > best_bool_acc:
            with open('best_model.pkl', 'wb') as f:
                pickle.dump({'params': state.params}, f)

        print(f"Ep {epoch:3d}: CE={np.mean(losses):.3f} Acc={soft_acc:.3f} "
              f"W={w_disc:.3f} T={t_disc:.3f} rho={rho:.4f} "
              f"alpha={alpha:.1f}{bool_acc_str}")

    print(f"\nBest boolean accuracy: {best_bool_acc:.3f}")
    print("\nRunning final pure boolean evaluation...")
    boolean_evaluate()


if __name__ == '__main__':
    main()
