"""
Step 0 – Simple ANN (PyTorch)
======================================================
Architecture : Input(4) → Dense(5, ReLU) → Dense(5, ReLU) → Output(1, linear)

Displays weight values before and after training, and shows
pre-activation (z) vs post-activation (a) values for a sample input.

Run:
    pip install torch
    python step0_simple_ann.py
"""

import torch
import torch.nn as nn
import numpy as np

# ── reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# Model definition
# ══════════════════════════════════════════════════════════════════════════════

class SimpleANN(nn.Module):
    """
    Two hidden layers (5 neurons, ReLU) + linear output.
    Architecture: Input(4) → Dense(5,ReLU) → Dense(5,ReLU) → Dense(1,linear)
    """

    def __init__(self, input_dim: int = 4):
        super().__init__()
        self.hidden_1 = nn.Linear(input_dim, 5)   # W: (4x5), b: (5,)
        self.hidden_2 = nn.Linear(5, 5)            # W: (5x5), b: (5,)
        self.output   = nn.Linear(5, 1)            # W: (5x1), b: (1,)
        self.relu     = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.hidden_1(x))
        x = self.relu(self.hidden_2(x))
        x = self.output(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Weight inspection
# ══════════════════════════════════════════════════════════════════════════════

def print_layer_weights(layer: nn.Linear, label: str = "") -> None:
    """Pretty-print W and b for one Linear layer."""
    # PyTorch stores W as (out, in) — transpose to (in, out) for display
    W = layer.weight.detach().numpy().T     # (n_in, n_out)
    b = layer.bias.detach().numpy()         # (n_out,)
    n_in, n_out = W.shape

    title = label or layer.__class__.__name__
    print(f"\n{'─'*65}")
    print(f"  [{title}]   W: ({n_in}x{n_out})   b: ({n_out},)")
    print(f"{'─'*65}")
    print(f"  Weight matrix W  (rows = input neurons, cols = this layer's neurons):")
    header = "  ".join(f"  n{j:<4}" for j in range(n_out))
    print(f"          {header}")
    for i, row in enumerate(W):
        vals = "  ".join(f"{v:+7.4f}" for v in row)
        print(f"  in[{i}] [ {vals} ]")
    bias_str = "  ".join(f"{v:+7.4f}" for v in b)
    print(f"  bias   [ {bias_str} ]")


# ══════════════════════════════════════════════════════════════════════════════
# Activation inspection
# ══════════════════════════════════════════════════════════════════════════════

def inspect_activations(model: SimpleANN, x_sample: torch.Tensor) -> None:
    """
    Forward-pass a single sample and show:
      z  -- pre-activation  (raw linear combination Wx + b)
      a  -- post-activation (after ReLU or linear)
    """
    print("\n" + "="*65)
    print("  ACTIVATION INSPECTION  (single sample forward pass)")
    print("="*65)
    print(f"  Input:  {x_sample.numpy().flatten()}")

    layers = [
        ("hidden_1", model.hidden_1, "relu"),
        ("hidden_2", model.hidden_2, "relu"),
        ("output",   model.output,   "linear"),
    ]

    a = x_sample
    for name, layer, activation in layers:
        z = layer(a)                                         # pre-activation
        a = torch.relu(z) if activation == "relu" else z    # post-activation

        z_np = z.detach().numpy().flatten()
        a_np = a.detach().numpy().flatten()

        print(f"\n  Layer: {name}  [{activation}]")
        print(f"  {'Neuron':<8}  {'z  (pre-activation)':>22}  {'a  (post-activation)':>22}")
        print(f"  {'------':<8}  {'--------------------':>22}  {'--------------------':>22}")
        for n in range(len(z_np)):
            marker = "active" if a_np[n] > 0 else "zeroed"
            print(f"  {n:<8}  {z_np[n]:>22.6f}  {a_np[n]:>22.6f}  [{marker}]")
        print(f"  layer output shape: {tuple(a.shape)}")

    print(f"\n  Final model output: {a.detach().item():.6f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(model, X, y, epochs=20, batch_size=32, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()
    n         = X.shape[0]

    for epoch in range(1, epochs + 1):
        idx  = torch.randperm(n)
        X_s, y_s = X[idx], y[idx]

        for start in range(0, n, batch_size):
            x_batch = X_s[start : start + batch_size]
            y_batch = y_s[start : start + batch_size]

            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            epoch_loss = loss_fn(model(X), y).item()
        print(f"  Epoch {epoch:>3}/{epochs}   MSE loss = {epoch_loss:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "#"*65)
    print("  STEP 0 -- Simple ANN baseline (PyTorch)")
    print("  Architecture: Input(4) -> Dense(5,ReLU) -> Dense(5,ReLU) -> Dense(1)")
    print("#"*65)

    model = SimpleANN(input_dim=4)
    print(f"\n{model}\n")

    # weights BEFORE training
    print("="*65)
    print("  WEIGHTS  (initial -- before any training)")
    print("="*65)
    print_layer_weights(model.hidden_1, label="hidden_1")
    print_layer_weights(model.hidden_2, label="hidden_2")
    print_layer_weights(model.output,   label="output")

    # synthetic dataset: y = sum(x) + small noise
    X_train = torch.rand(200, 4)
    y_train = X_train.sum(dim=1, keepdim=True) + 0.05 * torch.randn(200, 1)

    print("\n\n" + "="*65)
    print("  TRAINING  (20 epochs, Adam, MSE loss)")
    print("="*65)
    train(model, X_train, y_train, epochs=20, batch_size=32, lr=1e-3)

    # weights AFTER training
    print("\n\n" + "="*65)
    print("  WEIGHTS  (after training)")
    print("="*65)
    print_layer_weights(model.hidden_1, label="hidden_1 [trained]")
    print_layer_weights(model.hidden_2, label="hidden_2 [trained]")
    print_layer_weights(model.output,   label="output   [trained]")

    # activation inspection
    x_sample = torch.tensor([[0.2, 0.5, 0.8, 0.1]])
    inspect_activations(model, x_sample)

    print(f"  Expected (sum of inputs): {x_sample.sum().item():.6f}")
    print()


if __name__ == "__main__":
    main()
