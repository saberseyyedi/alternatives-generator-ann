"""
demo/step2_logit_masking_module.py
===================================
Programmatic demo of LogitMaskingLayer.

Shows how to:
  1. Wrap an existing nn.Linear with LogitMaskingLayer
  2. Access original logits, masked logits, spread, uncertainty
  3. Integrate the layer into a full model
  4. Save and reload the module

Run:
    python demo/step2_logit_masking_module.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from alternatives_generator import LogitMaskingLayer, MaskingOutput


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1 — Minimal usage (the one you show first)
# ══════════════════════════════════════════════════════════════════════════════

def example_minimal():
    print("\n" + "█"*65)
    print("  EXAMPLE 1 — Minimal usage")
    print("█"*65)

    # A standard PyTorch linear layer — could be from any trained model
    linear = nn.Linear(128, 10)

    # Wrap it — this is ALL the user needs to do
    layer = LogitMaskingLayer(
        base_layer       = linear,
        num_masks        = 3,
        connection_ratio = 0.5,
        seed             = 42,
    )

    print(f"\n  Layer: {layer}\n")

    # Create a batch of 5 samples
    x = torch.randn(5, 128)

    # Forward pass
    with torch.no_grad():
        output = layer(x)

    print(f"  Input shape:    {list(x.shape)}")
    print(f"  Original shape: {list(output.original.shape)}")
    print(f"  Masked shape:   {list(output.masked.shape)}")
    print(f"  Spread shape:   {list(output.spread.shape)}")
    print(f"  Uncertainty:    {output.uncertainty.tolist()}")
    print()
    print("  Interpretation:")
    for b in range(5):
        u = output.uncertainty[b].item()
        level = "HIGH uncertainty" if u > 1.0 else "LOW uncertainty"
        print(f"    Sample {b}: uncertainty = {u:.4f}  → {level}")


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2 — Small network (matches supervisor's requirement: 4 inputs, 2 logits)
# ══════════════════════════════════════════════════════════════════════════════

def example_small_network():
    print("\n" + "█"*65)
    print("  EXAMPLE 2 — Small network (4 inputs, 2 logits, 3 masks)")
    print("  This matches the Step 1 demo exactly but uses the module")
    print("█"*65)

    torch.manual_seed(42)
    linear = nn.Linear(4, 2)

    layer = LogitMaskingLayer(
        base_layer       = linear,
        num_masks        = 3,
        connection_ratio = 0.5,   # 0.5 × 4 = 2 connections per mask
        seed             = 42,
    )

    # Same input as Step 1 demo
    x = torch.tensor([[0.8, 0.3, 0.6, 0.1]])   # 1 sample, 4 features

    with torch.no_grad():
        output = layer(x)

    print(f"\n  Input:    {x.tolist()[0]}")
    print(f"\n  Original logits:")
    for i in range(2):
        print(f"    logit_{i} = {output.original[0, i].item():+.6f}")

    print(f"\n  Masked alternatives:")
    for i in range(2):
        print(f"\n    logit_{i} group:")
        print(f"      original = {output.original[0, i].item():+.6f}")
        for m in range(3):
            print(f"      mask_{i}_{m} = {output.masked[0, i, m].item():+.6f}")
        print(f"      mean     = {output.mean[0, i].item():+.6f}")
        print(f"      spread   = {output.spread[0, i].item():.6f}  "
              f"{'← uncertain' if output.spread[0, i].item() > 0.5 else '← consistent'}")

    print(f"\n  Overall uncertainty score: {output.uncertainty[0].item():.6f}")

    print(f"\n  Binary masks (which connections each mask uses):")
    print(f"  {'':10} " + " ".join(f"in{i}" for i in range(4)))
    for m in range(3):
        for i in range(2):
            bits = "  ".join(
                "1" if layer.binary_masks[m, i, j].item() == 1 else "·"
                for j in range(4)
            )
            print(f"  mask_{m}_logit{i}:  {bits}")


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 3 — Full model integration
# ══════════════════════════════════════════════════════════════════════════════

def example_full_model():
    print("\n" + "█"*65)
    print("  EXAMPLE 3 — Full model with LogitMaskingLayer as output")
    print("█"*65)

    # Build a simple two-layer model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(8, 16)
            self.relu   = nn.ReLU()
            self.output = nn.Linear(16, 3)   # 3 output logits
            # Wrap the output layer with masking
            self.masking = LogitMaskingLayer(
                base_layer       = self.output,
                num_masks        = 3,
                connection_ratio = 0.5,
                seed             = 42,
            )

        def forward(self, x):
            h = self.relu(self.hidden(x))
            return self.masking(h)   # returns MaskingOutput

    torch.manual_seed(42)
    model = SimpleModel()
    print(f"\n  Model architecture:")
    print(f"    Input(8) → Linear(8,16) → ReLU → Linear(16,3) + LogitMaskingLayer")
    print(f"\n  {model}")

    x = torch.randn(4, 8)   # 4 samples
    with torch.no_grad():
        output = model(x)

    print(f"\n  Input:         {list(x.shape)}")
    print(f"  Original out:  {list(output.original.shape)}")
    print(f"  Masked out:    {list(output.masked.shape)}")
    print(f"  Spread:        {list(output.spread.shape)}")
    print(f"\n  Uncertainty scores per sample:")
    for b in range(4):
        u = output.uncertainty[b].item()
        print(f"    Sample {b}: {u:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 4 — Effect of connection_ratio on spread
# ══════════════════════════════════════════════════════════════════════════════

def example_ratio_effect():
    print("\n" + "█"*65)
    print("  EXAMPLE 4 — Effect of connection_ratio on spread")
    print("  Lower ratio → more diversity → higher spread")
    print("█"*65)

    torch.manual_seed(42)
    linear = nn.Linear(8, 3)
    x      = torch.randn(10, 8)

    ratios = [0.2, 0.4, 0.6, 0.8]
    print(f"\n  {'Ratio':<8}  {'Avg Spread':<14}  {'Avg Uncertainty':<16}  Interpretation")
    print(f"  {'─────':<8}  {'──────────':<14}  {'───────────────':<16}  ──────────────")

    for ratio in ratios:
        layer = LogitMaskingLayer(linear, num_masks=3,
                                  connection_ratio=ratio, seed=42)
        with torch.no_grad():
            output = layer(x)
        avg_spread = output.spread.mean().item()
        avg_unc    = output.uncertainty.mean().item()
        note = "high diversity" if ratio < 0.5 else "low diversity"
        print(f"  {ratio:<8.1f}  {avg_spread:<14.4f}  {avg_unc:<16.4f}  {note}")

    print(f"\n  Insight: lower ratio → each mask sees less → more disagreement")
    print(f"           → higher spread → richer uncertainty signal")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█"*65)
    print("  STEP 2 — LogitMaskingLayer Module Demo")
    print("  Alternatives Generator for ANN")
    print("  RCSE Master Research Project, TU Ilmenau")
    print("█"*65)

    example_minimal()
    example_small_network()
    example_full_model()
    example_ratio_effect()

    print("\n" + "█"*65)
    print("  All examples completed.")
    print("  Module is ready for integration with Steps 3 and 4.")
    print("█"*65 + "\n")


if __name__ == "__main__":
    main()
