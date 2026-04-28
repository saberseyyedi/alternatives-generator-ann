"""
run_masking.py
==============
Demonstration script for LogitMaskingLayer.

This script is FIXED for demonstration purposes.
The model architecture and input data are predefined.
The user only controls two parameters:
    - num_masks        (how many alternatives per logit)
    - connection_ratio (how many connections each mask keeps)

Fixed internally:
    base_layer = nn.Linear(8, 3)   8 inputs, 3 output logits
    x          = torch.randn(4, 8)  4 sample inputs

Run:
    python run_masking.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
from alternatives_generator import LogitMaskingLayer


# ══════════════════════════════════════════════════════════════════════════════
# Fixed demo configuration
# ══════════════════════════════════════════════════════════════════════════════

# These values are fixed — not asked from the user.
# They define the demo model and input.
IN_FEATURES  = 8    # number of input neurons
OUT_FEATURES = 3    # number of output logits
BATCH_SIZE   = 4    # number of input samples
MODEL_SEED   = 42   # seed for model weights (fixed, reproducible)
INPUT_SEED   = 7    # seed for input data   (fixed, reproducible)


# ══════════════════════════════════════════════════════════════════════════════
# Print helpers
# ══════════════════════════════════════════════════════════════════════════════

W = 65   # line width

def line(char="═"):
    print(char * W)

def section(title):
    print()
    line()
    print(f"  {title}")
    line()


# ══════════════════════════════════════════════════════════════════════════════
# User input — only 2 questions
# ══════════════════════════════════════════════════════════════════════════════

def ask_num_masks() -> int:
    while True:
        raw = input("  Number of masks per logit [default: 3]: ").strip()
        if raw == "":
            return 3
        try:
            val = int(raw)
            if 1 <= val <= 10:
                return val
            print("  Please enter a value between 1 and 10.")
        except ValueError:
            print("  Please enter a whole number.")


def ask_connection_ratio() -> float:
    while True:
        raw = input("  Connection ratio (0.1 – 0.9) [default: 0.5]: ").strip()
        if raw == "":
            return 0.5
        try:
            val = float(raw)
            if 0.1 <= val <= 0.9:
                return val
            print("  Please enter a value between 0.1 and 0.9.")
        except ValueError:
            print("  Please enter a decimal number like 0.5.")


# ══════════════════════════════════════════════════════════════════════════════
# Output printing
# ══════════════════════════════════════════════════════════════════════════════

def print_input_parameters(num_masks, connection_ratio, n_connections):
    section("INPUT PARAMETERS")
    print(f"\n  Model (fixed for demo):")
    print(f"    base_layer   : nn.Linear({IN_FEATURES}, {OUT_FEATURES})")
    print(f"    input x      : torch.randn({BATCH_SIZE}, {IN_FEATURES})")
    print()
    print(f"  User parameters:")
    print(f"    num_masks        : {num_masks}")
    print(f"    connection_ratio : {connection_ratio}")
    print(f"    n_connections    : {n_connections}  "
          f"(= round({IN_FEATURES} × {connection_ratio}) per mask per logit)")


def print_original(output):
    section("ORIGINAL LOGITS  —  fully connected, no masking")
    print(f"\n  Shape: {list(output.original.shape)}"
          f"  →  {BATCH_SIZE} samples × {OUT_FEATURES} logits\n")

    # Header
    header = "  ".join(f"  logit_{i}" for i in range(OUT_FEATURES))
    print(f"  {'Sample':<10}  {header}")
    print(f"  {'──────':<10}  " +
          "  ".join("────────" for _ in range(OUT_FEATURES)))

    for b in range(BATCH_SIZE):
        vals = "  ".join(
            f"{output.original[b, i].item():>+8.4f}"
            for i in range(OUT_FEATURES)
        )
        print(f"  sample_{b:<3}   {vals}")


def print_masked(output, num_masks):
    section("MASKED LOGITS  —  partial connections per mask")
    print(f"\n  Shape: {list(output.masked.shape)}"
          f"  →  {BATCH_SIZE} samples × {OUT_FEATURES} logits × {num_masks} masks\n")

    for b in range(BATCH_SIZE):
        print(f"  ── Sample {b} " + "─" * 50)
        print()

        # Column headers
        header = "  ".join(f"  logit_{i}" for i in range(OUT_FEATURES))
        print(f"  {'':12}  {header}")
        print(f"  {'':12}  " +
              "  ".join("────────" for _ in range(OUT_FEATURES)))

        # Original row
        orig = "  ".join(
            f"{output.original[b, i].item():>+8.4f}"
            for i in range(OUT_FEATURES)
        )
        print(f"  {'original':<12}  {orig}")

        # One row per mask
        for m in range(num_masks):
            row = "  ".join(
                f"{output.masked[b, i, m].item():>+8.4f}"
                for i in range(OUT_FEATURES)
            )
            print(f"  {f'mask_{m}':<12}  {row}")

        print()


def print_mean(output):
    section("MEAN  —  average across original + all masks")
    print(f"\n  Shape: {list(output.mean.shape)}\n")

    header = "  ".join(f"  logit_{i}" for i in range(OUT_FEATURES))
    print(f"  {'Sample':<10}  {header}")
    print(f"  {'──────':<10}  " +
          "  ".join("────────" for _ in range(OUT_FEATURES)))

    for b in range(BATCH_SIZE):
        vals = "  ".join(
            f"{output.mean[b, i].item():>+8.4f}"
            for i in range(OUT_FEATURES)
        )
        print(f"  sample_{b:<3}   {vals}")


def print_spread(output):
    section("SPREAD  —  max − min across alternatives per logit")
    print()
    print("  Spread measures how much the alternatives disagree.")
    print("  HIGH spread  →  masks disagree  →  UNCERTAIN output")
    print("  LOW  spread  →  masks agree     →  CONSISTENT output")
    print()

    header = "  ".join(f"  logit_{i}" for i in range(OUT_FEATURES))
    print(f"  {'Sample':<10}  {header}    {'Uncertainty':>12}  Interpretation")
    print(f"  {'──────':<10}  " +
          "  ".join("────────" for _ in range(OUT_FEATURES)) +
          "    " + "─" * 12 + "  " + "─" * 14)

    for b in range(BATCH_SIZE):
        spread_vals = "  ".join(
            f"{output.spread[b, i].item():>8.4f}"
            for i in range(OUT_FEATURES)
        )
        u    = output.uncertainty[b].item()
        flag = "UNCERTAIN    " if u > 0.5 else "consistent   "
        print(f"  sample_{b:<3}   {spread_vals}    {u:>12.4f}  {flag}")

    # Visual bar chart for uncertainty
    print()
    line("─")
    print("  Uncertainty per sample (visual):")
    print()
    for b in range(BATCH_SIZE):
        u   = output.uncertainty[b].item()
        bar = "█" * min(40, int(u * 15))
        print(f"  sample_{b}: {u:.4f}  {bar}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── header ────────────────────────────────────────────────────────────────
    line("█")
    print("  Alternatives Generator for ANN")
    print("  LogitMaskingLayer — Demonstration")
    print("  RCSE Master Research Project, TU Ilmenau")
    line("█")

    # ── ask only 2 questions ──────────────────────────────────────────────────
    section("USER INPUT")
    print()
    num_masks        = ask_num_masks()
    connection_ratio = ask_connection_ratio()

    # ── build fixed demo model ────────────────────────────────────────────────
    torch.manual_seed(MODEL_SEED)
    base_layer = nn.Linear(IN_FEATURES, OUT_FEATURES)

    layer = LogitMaskingLayer(
        base_layer       = base_layer,
        num_masks        = num_masks,
        connection_ratio = connection_ratio,
        seed             = MODEL_SEED,
    )

    # ── fixed demo input ──────────────────────────────────────────────────────
    torch.manual_seed(INPUT_SEED)
    x = torch.randn(BATCH_SIZE, IN_FEATURES)

    # ── forward pass ──────────────────────────────────────────────────────────
    with torch.no_grad():
        output = layer(x)

    # ── print all results ─────────────────────────────────────────────────────
    print_input_parameters(num_masks, connection_ratio, layer.n_connections)
    print_original(output)
    print_masked(output, num_masks)
    print_mean(output)
    print_spread(output)

    # ── closing message ───────────────────────────────────────────────────────
    line("█")
    print("  The SPREAD values above are the foundation of uncertainty")
    print("  estimation. They feed into the Weighted Probability (Step 3)")
    print("  and Prospect Certainty (Step 4) computations.")
    line("█")
    print()


if __name__ == "__main__":
    main()
