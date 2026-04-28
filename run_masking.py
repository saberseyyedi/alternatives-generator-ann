"""
run_masking.py
==============
Interactive command-line script to demonstrate the LogitMaskingLayer.

This script:
  1. Asks the user for configuration (num_masks, connection_ratio)
  2. Creates a simple nn.Linear model
  3. Wraps it with LogitMaskingLayer
  4. Runs a forward pass with a sample input
  5. Prints results clearly

Run:
    python run_masking.py
"""

import sys
import os

# Make sure the src package is importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
from alternatives_generator import LogitMaskingLayer


# ══════════════════════════════════════════════════════════════════════════════
# Pretty print helpers
# ══════════════════════════════════════════════════════════════════════════════

def separator(char="═", width=65):
    print(char * width)

def section(title):
    print()
    separator()
    print(f"  {title}")
    separator()

def subsection(title):
    print(f"\n  {'─'*61}")
    print(f"  {title}")
    print(f"  {'─'*61}")


# ══════════════════════════════════════════════════════════════════════════════
# User input helpers
# ══════════════════════════════════════════════════════════════════════════════

def ask_int(prompt, default, min_val=1, max_val=20):
    while True:
        raw = input(f"  {prompt} [default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"  Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("  Please enter a whole number.")

def ask_float(prompt, default, min_val=0.1, max_val=0.9):
    while True:
        raw = input(f"  {prompt} [default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
            if min_val <= val <= max_val:
                return val
            print(f"  Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("  Please enter a decimal number (e.g. 0.5).")


# ══════════════════════════════════════════════════════════════════════════════
# Print results
# ══════════════════════════════════════════════════════════════════════════════

def print_results(output, in_features, out_features, num_masks, batch_size):
    """Print all results from a MaskingOutput in a readable format."""

    # ── original logits ───────────────────────────────────────────────────────
    section("ORIGINAL LOGITS  (unmasked, fully connected)")
    print(f"\n  Shape: {list(output.original.shape)}  "
          f"→ {batch_size} samples × {out_features} logits\n")
    print(f"  {'':5} " + "  ".join(f"logit_{i:>2}" for i in range(out_features)))
    print(f"  {'':5} " + "  ".join("─────────" for _ in range(out_features)))
    for b in range(batch_size):
        row = "  ".join(f"{output.original[b, i].item():>+9.4f}"
                        for i in range(out_features))
        print(f"  s{b:<4} {row}")

    # ── masked logits ─────────────────────────────────────────────────────────
    section("MASKED LOGITS  (per mask, per logit)")
    print(f"\n  Shape: {list(output.masked.shape)}  "
          f"→ {batch_size} samples × {out_features} logits × {num_masks} masks\n")

    for b in range(batch_size):
        subsection(f"Sample {b}")
        print(f"\n  {'':10} " +
              "  ".join(f"logit_{i:>2}" for i in range(out_features)))
        print(f"  {'':10} " +
              "  ".join("─────────" for _ in range(out_features)))

        # original row
        orig_row = "  ".join(
            f"{output.original[b, i].item():>+9.4f}" for i in range(out_features)
        )
        print(f"  {'original':<10} {orig_row}")

        # one row per mask
        for m in range(num_masks):
            mask_row = "  ".join(
                f"{output.masked[b, i, m].item():>+9.4f}"
                for i in range(out_features)
            )
            print(f"  {f'mask_{m}':<10} {mask_row}")

        # mean row
        mean_row = "  ".join(
            f"{output.mean[b, i].item():>+9.4f}" for i in range(out_features)
        )
        print(f"  {'─'*10} " + "  ".join("─────────" for _ in range(out_features)))
        print(f"  {'mean':<10} {mean_row}")

    # ── spread ────────────────────────────────────────────────────────────────
    section("SPREAD  (max − min across alternatives per logit)")
    print(f"\n  Spread = how different the alternatives are.")
    print(f"  HIGH spread → uncertain.   LOW spread → consistent.\n")
    print(f"  {'':5} " + "  ".join(f"logit_{i:>2}" for i in range(out_features)))
    print(f"  {'':5} " + "  ".join("─────────" for _ in range(out_features)))
    for b in range(batch_size):
        row   = "  ".join(f"{output.spread[b, i].item():>9.4f}"
                          for i in range(out_features))
        score = output.uncertainty[b].item()
        flag  = "← uncertain" if score > 0.5 else "← consistent"
        print(f"  s{b:<4} {row}    uncertainty={score:.4f}  {flag}")

    # ── uncertainty summary ───────────────────────────────────────────────────
    section("UNCERTAINTY SUMMARY  (mean spread across all logits)")
    print(f"\n  One number per sample — summarises overall model certainty.\n")
    for b in range(batch_size):
        score = output.uncertainty[b].item()
        bar   = "█" * min(40, int(score * 20))
        print(f"  Sample {b}:  {score:.4f}   {bar}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    separator("█")
    print("  LogitMaskingLayer — Interactive Demo")
    print("  Alternatives Generator for ANN")
    print("  RCSE Master Research Project, TU Ilmenau")
    separator("█")

    # ── step 1: get user configuration ────────────────────────────────────────
    section("CONFIGURATION")
    print("\n  Press Enter to accept the default value shown in brackets.\n")

    in_features  = ask_int("Number of input neurons (in_features)", default=8)
    out_features = ask_int("Number of output logits (out_features)", default=3)
    num_masks    = ask_int("Number of masks per logit (num_masks)", default=3)
    ratio        = ask_float("Connection ratio per mask (0.1 – 0.9)", default=0.5)
    batch_size   = ask_int("Number of input samples (batch_size)", default=4)
    seed         = ask_int("Random seed (for reproducibility)", default=42,
                           min_val=0, max_val=99999)

    # ── step 2: build model ────────────────────────────────────────────────────
    section("BUILDING MODEL")

    # Base layer — a standard PyTorch Linear layer
    # This represents the output layer of any trained neural network
    torch.manual_seed(seed)
    base_linear = nn.Linear(in_features, out_features)

    print(f"\n  Base layer:   nn.Linear({in_features}, {out_features})")
    print(f"  Weight shape: {list(base_linear.weight.shape)}")
    print(f"    (rows = logits, cols = input neurons)")

    # Wrap it with the masking layer
    masking_layer = LogitMaskingLayer(
        base_layer       = base_linear,
        num_masks        = num_masks,
        connection_ratio = ratio,
        seed             = seed,
    )

    print(f"\n  LogitMaskingLayer:")
    print(f"    {masking_layer}")
    print(f"\n  Connections per mask: {masking_layer.n_connections} "
          f"out of {in_features}  "
          f"({ratio*100:.0f}%)")

    # ── step 3: show binary masks ──────────────────────────────────────────────
    section("BINARY MASKS  (which connections each mask keeps)")
    print(f"\n  Shape: {list(masking_layer.binary_masks.shape)}")
    print(f"  Interpretation: binary_masks[mask_index, logit_index, input_index]")
    print(f"  1 = connection kept,   0 = connection removed\n")

    for m in range(num_masks):
        print(f"  Mask {m}:")
        print(f"  {'':10} " +
              " ".join(f"in{i:<3}" for i in range(in_features)))
        for i in range(out_features):
            bits = "  ".join(
                f"  {'1' if masking_layer.binary_masks[m, i, j].item() == 1 else '·'}"
                for j in range(in_features)
            )
            print(f"  logit_{i:<4} {bits}")
        print()

    # ── step 4: create sample input and run forward pass ──────────────────────
    section("FORWARD PASS")

    torch.manual_seed(seed + 1)
    x = torch.randn(batch_size, in_features)
    print(f"\n  Input x — shape: {list(x.shape)}")
    print(f"  ({batch_size} samples, each with {in_features} features)\n")
    for b in range(batch_size):
        vals = "  ".join(f"{v.item():>+6.3f}" for v in x[b])
        print(f"  sample_{b}: [ {vals} ]")

    # Run the masking layer
    with torch.no_grad():
        output = masking_layer(x)

    # ── step 5: print all results ──────────────────────────────────────────────
    print_results(output, in_features, out_features, num_masks, batch_size)

    # ── step 6: key insight ────────────────────────────────────────────────────
    separator()
    print("\n  KEY INSIGHT")
    separator("─")
    print("\n  The SPREAD value is the foundation of uncertainty estimation.")
    print("  When spread is HIGH across the alternatives of a logit,")
    print("  the model is sensitive to which inputs it sees → UNCERTAIN.")
    print("  When spread is LOW, the model is stable → CERTAIN.")
    print()
    print("  Next steps in the project:")
    print("   Step 2 → Weighted Probability: score each alternative by")
    print("            how close it is to the group mean")
    print("   Step 3 → Behavior Function: measure if each alternative")
    print("            improves alignment with the training distribution")
    print("   Step 4 → Prospect Certainty: combine scores into final Ω")
    print()
    separator("█")
    print()


if __name__ == "__main__":
    main()
