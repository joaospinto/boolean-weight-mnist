import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from main import extract_boolean_model, get_datasets


def boolean_forward_with_activations(x_bool, layers):
    """Pure boolean forward pass returning intermediate activations."""
    x = np.concatenate([x_bool, ~x_bool], axis=-1)
    activations = []

    for layer in layers[:-1]:
        w = layer['w']
        scale = layer['ln_scale']
        bias = layer['ln_bias']
        N = w.shape[1]

        counts = x.astype(np.float32) @ w.astype(np.float32)
        counts = np.round(counts).astype(np.int64)

        S = counts.sum(axis=1, keepdims=True)
        S2 = (counts ** 2).sum(axis=1, keepdims=True)
        Q = N * S2 - S ** 2

        PRECISION = 512
        scale_int = np.round(scale * PRECISION).astype(np.int64)
        bias_int = np.round(bias * PRECISION).astype(np.int64)

        A = scale_int[None, :] * (N * counts - S)
        A2 = A ** 2
        B2Q = bias_int[None, :] ** 2 * Q

        a_pos = A >= 0
        b_pos = bias_int[None, :] >= 0

        fired = ((a_pos & b_pos) |
                 (a_pos & ~b_pos & (A2 > B2Q)) |
                 (~a_pos & b_pos & (B2Q > A2)))

        x = fired
        activations.append(x)

    w_out = layers[-1]['w']
    scores = x.astype(np.float32) @ w_out.astype(np.float32)
    scores = np.round(scores).astype(np.int32)
    return activations, scores


def main():
    with open('best_model.pkl', 'rb') as f:
        saved = pickle.load(f)

    params = {}
    for k, v in saved['params'].items():
        if hasattr(v, 'items'):
            params[k] = {kk: np.array(vv) for kk, vv in v.items()}
        else:
            params[k] = np.array(v)

    layers = extract_boolean_model(params)
    (_, _), (x_test, y_test) = get_datasets()

    # Pick 10 diverse examples (one per digit)
    indices = []
    for digit in range(10):
        idx = np.where(y_test == digit)[0][0]
        indices.append(idx)

    x_bool = x_test[indices] > 0.5
    labels = y_test[indices]
    activations, scores = boolean_forward_with_activations(x_bool, layers)

    n_hidden = len(activations)
    n_panels = 1 + n_hidden + 1  # input + hidden layers + output
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5))
    fig.suptitle("Boolean Neural Network — Pure Integer/Boolean Forward Pass (91.9% accuracy)",
                 fontsize=14)

    # Grid shapes for displaying each hidden layer as a 2D boolean image
    grid_shapes = [(32, 64), (32, 32), (16, 32)]
    layer_colors = ['Greens', 'Blues', 'Purples']
    layer_sizes = [2048, 1024, 512]

    def update(frame):
        for ax in axes.flatten():
            ax.clear()

        # Panel 0: Input image (28x28 boolean)
        img = x_bool[frame].reshape(28, 28)
        axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f"Input\n(label={labels[frame]})", fontsize=11)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # Panels 1..n_hidden: hidden layer activations
        for i in range(n_hidden):
            h, w = grid_shapes[i]
            grid = activations[i][frame].astype(np.uint8).reshape(h, w)
            axes[i + 1].imshow(grid, cmap=layer_colors[i], vmin=0, vmax=1,
                               interpolation='nearest')
            active = int(activations[i][frame].sum())
            axes[i + 1].set_title(
                f"Layer {i+1}\n{active}/{layer_sizes[i]} gates", fontsize=11)
            axes[i + 1].set_xticks([])
            axes[i + 1].set_yticks([])

        # Last panel: output scores
        ax_out = axes[n_hidden + 1]
        s = scores[frame]
        pred = int(np.argmax(s))
        colors = ['#2ecc71' if i == pred else '#95a5a6' for i in range(10)]
        ax_out.bar(range(10), s, color=colors)
        ax_out.set_xticks(range(10))
        ax_out.set_xlabel("Digit class")
        ax_out.set_ylabel("Integer count")
        correct = "correct" if pred == labels[frame] else "WRONG"
        ax_out.set_title(f"Output\n(pred={pred}, {correct})", fontsize=11)

        plt.tight_layout(rect=[0, 0, 1, 0.92])

    ani = FuncAnimation(fig, update, frames=len(indices), repeat=False)
    writer = FFMpegWriter(fps=1)
    print("Saving video to forward_pass.mp4...")
    ani.save("forward_pass.mp4", writer=writer)
    plt.close()
    print("Video saved.")


if __name__ == '__main__':
    main()
