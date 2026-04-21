"""
Step 0 – Simple ANN (pure NumPy, no framework needed)
======================================================
Architecture : Input(4) → Dense(5, ReLU) → Dense(5, ReLU) → Output(1, linear)

This file intentionally avoids TensorFlow / PyTorch so the model internals
(weights, pre- and post-activation values) are completely transparent.
It serves as the agreed baseline before adding the Alternatives Generator.

Run:
    python step0_simple_ann_numpy.py
"""

import numpy as np

# ── reproducibility ────────────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)


# ══════════════════════════════════════════════════════════════════════════════
# Activation functions
# ══════════════════════════════════════════════════════════════════════════════

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)

def relu_deriv(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)

def linear(z: np.ndarray) -> np.ndarray:
    return z

def linear_deriv(z: np.ndarray) -> np.ndarray:
    return np.ones_like(z)


# ══════════════════════════════════════════════════════════════════════════════
# Dense layer
# ══════════════════════════════════════════════════════════════════════════════

class DenseLayer:
    """
    A fully-connected layer.

    Attributes
    ----------
    W  : weight matrix  shape (n_in, n_out)   — rows = incoming, cols = neurons
    b  : bias vector    shape (n_out,)
    """

    def __init__(self, n_in: int, n_out: int, activation: str = "relu", name: str = ""):
        # Glorot-uniform initialisation (same default as Keras)
        limit = np.sqrt(6.0 / (n_in + n_out))
        self.W    = RNG.uniform(-limit, limit, size=(n_in, n_out))
        self.b    = np.zeros(n_out)
        self.name = name

        if activation == "relu":
            self._act       = relu
            self._act_deriv = relu_deriv
            self.activation = "relu"
        else:  # linear
            self._act       = linear
            self._act_deriv = linear_deriv
            self.activation = "linear"

        # cache for backprop
        self._z_cache : np.ndarray | None = None
        self._a_cache : np.ndarray | None = None
        self._x_cache : np.ndarray | None = None

    # ── forward pass ──────────────────────────────────────────────────────────
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : shape (batch, n_in)

        Returns
        -------
        a : shape (batch, n_out)   post-activation values
        """
        z = x @ self.W + self.b     # pre-activation
        a = self._act(z)            # post-activation
        # cache for backprop
        self._z_cache = z
        self._a_cache = a
        self._x_cache = x
        return a

    # ── backward pass ─────────────────────────────────────────────────────────
    def backward(self, d_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        d_x  : gradient w.r.t. input  (batch, n_in)
        d_W  : gradient w.r.t. W      (n_in, n_out)
        d_b  : gradient w.r.t. b      (n_out,)
        """
        d_z = d_out * self._act_deriv(self._z_cache)   # (batch, n_out)
        d_W = self._x_cache.T @ d_z                    # (n_in, n_out)
        d_b = d_z.sum(axis=0)                          # (n_out,)
        d_x = d_z @ self.W.T                           # (batch, n_in)
        return d_x, d_W, d_b


# ══════════════════════════════════════════════════════════════════════════════
# Simple ANN model
# ══════════════════════════════════════════════════════════════════════════════

class SimpleANN:
    """
    Two hidden layers (5 neurons, ReLU) + linear output.
    Trained with mini-batch gradient descent (Adam).
    """

    def __init__(self, input_dim: int = 4):
        self.layers = [
            DenseLayer(input_dim, 5, activation="relu",   name="hidden_1"),
            DenseLayer(5,         5, activation="relu",   name="hidden_2"),
            DenseLayer(5,         1, activation="linear", name="output"),
        ]
        # Adam state (m, v per parameter)
        self._init_adam()

    def _init_adam(self):
        self._adam_m = [{"W": np.zeros_like(l.W), "b": np.zeros_like(l.b)} for l in self.layers]
        self._adam_v = [{"W": np.zeros_like(l.W), "b": np.zeros_like(l.b)} for l in self.layers]
        self._adam_t = 0

    # ── forward pass ──────────────────────────────────────────────────────────
    def predict(self, x: np.ndarray) -> np.ndarray:
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    # ── MSE loss ──────────────────────────────────────────────────────────────
    @staticmethod
    def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    # ── single training step ──────────────────────────────────────────────────
    def _train_step(self, x_batch: np.ndarray, y_batch: np.ndarray,
                    lr: float = 1e-3, beta1: float = 0.9,
                    beta2: float = 0.999, eps: float = 1e-8):
        batch = x_batch.shape[0]
        y_pred = self.predict(x_batch)

        # dL/dy_pred  for MSE = 2(y_pred - y_true) / N
        d_out = 2 * (y_pred - y_batch) / batch   # (batch, 1)

        self._adam_t += 1
        for i in reversed(range(len(self.layers))):
            d_out, d_W, d_b = self.layers[i].backward(d_out)
            # Adam update
            m, v = self._adam_m[i], self._adam_v[i]
            m["W"] = beta1 * m["W"] + (1 - beta1) * d_W
            m["b"] = beta1 * m["b"] + (1 - beta1) * d_b
            v["W"] = beta2 * v["W"] + (1 - beta2) * d_W ** 2
            v["b"] = beta2 * v["b"] + (1 - beta2) * d_b ** 2
            m_hat_W = m["W"] / (1 - beta1 ** self._adam_t)
            m_hat_b = m["b"] / (1 - beta1 ** self._adam_t)
            v_hat_W = v["W"] / (1 - beta2 ** self._adam_t)
            v_hat_b = v["b"] / (1 - beta2 ** self._adam_t)
            self.layers[i].W -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
            self.layers[i].b -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 20, batch_size: int = 32, lr: float = 1e-3):
        n = X.shape[0]
        for epoch in range(1, epochs + 1):
            idx = RNG.permutation(n)
            X_s, y_s = X[idx], y[idx]
            for start in range(0, n, batch_size):
                self._train_step(X_s[start:start + batch_size],
                                 y_s[start:start + batch_size], lr=lr)
            loss = self.mse_loss(self.predict(X), y)
            print(f"  Epoch {epoch:>3}/{epochs}   MSE loss = {loss:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# Inspection helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_layer_weights(layer: DenseLayer, label: str = "") -> None:
    """Pretty-print W and b for one Dense layer."""
    title = label or layer.name
    n_in, n_out = layer.W.shape
    print(f"\n{'─'*65}")
    print(f"  [{title}]   W: ({n_in}×{n_out})   b: ({n_out},)   "
          f"activation: {layer.activation}")
    print(f"{'─'*65}")
    print(f"  Weight matrix W  (rows = input neurons, cols = this layer's neurons):")
    header = "  ".join(f"  n{j:<4}" for j in range(n_out))
    print(f"          {header}")
    for i, row in enumerate(layer.W):
        vals = "  ".join(f"{v:+7.4f}" for v in row)
        print(f"  in[{i}] [ {vals} ]")
    bias_str = "  ".join(f"{v:+7.4f}" for v in layer.b)
    print(f"  bias   [ {bias_str} ]")


def inspect_activations(model: SimpleANN, x_sample: np.ndarray) -> None:
    """
    Forward-pass a single sample and show:
      z  – pre-activation  (raw linear combination Wx + b)
      a  – post-activation (after applying the activation function)
    """
    print("\n" + "═"*65)
    print("  ACTIVATION INSPECTION  (single sample forward pass)")
    print("═"*65)
    print(f"  Input:  {x_sample.flatten()}")

    a = x_sample                    # shape (1, input_dim)
    for layer in model.layers:
        z = a @ layer.W + layer.b   # pre-activation
        act_fn = relu if layer.activation == "relu" else linear
        a = act_fn(z)               # post-activation

        print(f"\n  ┌─ Layer: {layer.name}  [{layer.activation}]")
        print(f"  │  {'Neuron':<8}  {'z  (pre-activation)':>22}  "
              f"{'a  (post-activation)':>22}")
        print(f"  │  {'──────':<8}  {'────────────────────':>22}  "
              f"{'────────────────────':>22}")
        for n in range(z.shape[1]):
            marker = "←active" if a[0, n] > 0 else "       "
            print(f"  │  {n:<8}  {z[0, n]:>22.6f}  {a[0, n]:>22.6f}  {marker}")
        print(f"  └─ layer output shape: {a.shape}")

    print(f"\n  Final model output: {a[0, 0]:.6f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "█"*65)
    print("  STEP 0 – Simple ANN baseline (pure NumPy)")
    print("  Architecture: Input(4) → Dense(5,ReLU) → Dense(5,ReLU) → Dense(1)")
    print("█"*65)

    model = SimpleANN(input_dim=4)

    # ── weights BEFORE training ────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  WEIGHTS  (initial – before any training)")
    print("="*65)
    for layer in model.layers:
        print_layer_weights(layer)

    # ── synthetic dataset  y = sum(x) + small noise ───────────────────────────
    X_train = RNG.random((200, 4)).astype(np.float32)
    y_train = X_train.sum(axis=1, keepdims=True) + 0.05 * RNG.standard_normal((200, 1))

    print("\n\n" + "="*65)
    print("  TRAINING  (20 epochs, Adam, MSE loss)")
    print("="*65)
    model.fit(X_train, y_train, epochs=20, batch_size=32, lr=1e-3)

    # ── weights AFTER training ─────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  WEIGHTS  (after training)")
    print("="*65)
    for layer in model.layers:
        print_layer_weights(layer, label=f"{layer.name} [trained]")

    # ── activation inspection for one sample ──────────────────────────────────
    x_sample = np.array([[0.2, 0.5, 0.8, 0.1]], dtype=np.float32)
    inspect_activations(model, x_sample)

    print(f"  Expected (sum of inputs): {x_sample.sum():.6f}")
    print()


if __name__ == "__main__":
    main()
