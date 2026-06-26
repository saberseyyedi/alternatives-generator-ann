"""
run_masking.py
==============
Demonstration script for LogitMaskingLayer.

Fixed internally:
    base_layer = nn.Linear(8, 3)
    x          = torch.randn(4, 8)

User controls only:
    - num_masks
    - connection_ratio

Run:
    python run_masking.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from alternatives_generator import LogitMaskingLayer

# ══════════════════════════════════════════════════════════════════════════════
# Fixed demo configuration
# ══════════════════════════════════════════════════════════════════════════════

IN_FEATURES  = 8
OUT_FEATURES = 3
BATCH_SIZE   = 4
MODEL_SEED   = 42
INPUT_SEED   = 7

MASK_COLOURS = [
    "#e05c5c", "#5cb85c", "#5c7de0", "#e0c45c",
    "#a05ce0", "#e08c5c", "#5ccce0", "#e05ca0",
    "#8ce05c", "#5ce0a0",
]

# ══════════════════════════════════════════════════════════════════════════════
# Print helpers
# ══════════════════════════════════════════════════════════════════════════════

W = 65

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

def ask_num_masks():
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

def ask_connection_ratio():
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
# Terminal output
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
          f"(= round({IN_FEATURES} x {connection_ratio}) per mask per logit)")

def print_original(output):
    section("ORIGINAL LOGITS  —  fully connected, no masking")
    print(f"\n  Shape: {list(output.original.shape)}"
          f"  ->  {BATCH_SIZE} samples x {OUT_FEATURES} logits\n")
    header = "  ".join(f"  logit_{i}" for i in range(OUT_FEATURES))
    print(f"  {'Sample':<10}  {header}")
    print(f"  {'------':<10}  " + "  ".join("--------" for _ in range(OUT_FEATURES)))
    for b in range(BATCH_SIZE):
        vals = "  ".join(f"{output.original[b,i].item():>+8.4f}" for i in range(OUT_FEATURES))
        print(f"  sample_{b:<3}   {vals}")

def print_masked(output, num_masks):
    section("MASKED LOGITS  —  partial connections per mask")
    print(f"\n  Shape: {list(output.masked.shape)}"
          f"  ->  {BATCH_SIZE} samples x {OUT_FEATURES} logits x {num_masks} masks\n")
    for b in range(BATCH_SIZE):
        print(f"  -- Sample {b} " + "-" * 50)
        print()
        header = "  ".join(f"  logit_{i}" for i in range(OUT_FEATURES))
        print(f"  {'':12}  {header}")
        print(f"  {'':12}  " + "  ".join("--------" for _ in range(OUT_FEATURES)))
        orig = "  ".join(f"{output.original[b,i].item():>+8.4f}" for i in range(OUT_FEATURES))
        print(f"  {'original':<12}  {orig}")
        for m in range(num_masks):
            row = "  ".join(f"{output.masked[b,i,m].item():>+8.4f}" for i in range(OUT_FEATURES))
            print(f"  {f'mask_{m}':<12}  {row}")
        print()

def print_mean(output):
    section("MEAN  —  average across original + all masks")
    print(f"\n  Shape: {list(output.mean.shape)}\n")
    header = "  ".join(f"  logit_{i}" for i in range(OUT_FEATURES))
    print(f"  {'Sample':<10}  {header}")
    print(f"  {'------':<10}  " + "  ".join("--------" for _ in range(OUT_FEATURES)))
    for b in range(BATCH_SIZE):
        vals = "  ".join(f"{output.mean[b,i].item():>+8.4f}" for i in range(OUT_FEATURES))
        print(f"  sample_{b:<3}   {vals}")

def print_spread(output):
    section("SPREAD  —  max - min across alternatives per logit")
    print()
    print("  HIGH spread  ->  masks disagree  ->  UNCERTAIN output")
    print("  LOW  spread  ->  masks agree     ->  CONSISTENT output")
    print()
    header = "  ".join(f"  logit_{i}" for i in range(OUT_FEATURES))
    print(f"  {'Sample':<10}  {header}    {'Uncertainty':>12}  Interpretation")
    print(f"  {'------':<10}  " + "  ".join("--------" for _ in range(OUT_FEATURES)) +
          "    " + "-" * 12 + "  " + "-" * 14)
    for b in range(BATCH_SIZE):
        spread_vals = "  ".join(f"{output.spread[b,i].item():>8.4f}" for i in range(OUT_FEATURES))
        u    = output.uncertainty[b].item()
        flag = "UNCERTAIN    " if u > 0.5 else "consistent   "
        print(f"  sample_{b:<3}   {spread_vals}    {u:>12.4f}  {flag}")
    print()
    line("-")
    print("  Uncertainty per sample (visual):")
    print()
    for b in range(BATCH_SIZE):
        u   = output.uncertainty[b].item()
        bar = "#" * min(40, int(u * 15))
        print(f"  sample_{b}: {u:.4f}  {bar}")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# Network diagram
# ══════════════════════════════════════════════════════════════════════════════

def plot_network_diagram(layer, output, sample_index=0):
    """
    Draw a network diagram showing:
      - Left  column : input neurons (grey circles)
      - Middle column: original logits (larger grey circles)
      - Right column : mask nodes (small coloured circles)
      - Grey lines   : full connections (input → logit)
      - Coloured lines: masked connections (input → mask, only where mask=1)
      - Node values from sample_index
      - μ_i (mean) and d_i (spread) brackets per logit group

    Saves to: demo/logit_masking_network.png
    """

    in_f   = layer.in_features
    out_f  = layer.out_features
    n_mask = layer.num_masks

    # ── figure ────────────────────────────────────────────────────────────────
    fig_h = max(10, in_f * 1.2 + 1)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.5, in_f + 0.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── x positions of each column ────────────────────────────────────────────
    X_LEFT  = 1.8    # input neurons
    X_LOGIT = 6.8    # original logits
    X_MASK  = 10.2   # mask nodes
    X_BRACE = 12.4   # bracket for d/mu

    # ── y positions ───────────────────────────────────────────────────────────
    # Input neurons: evenly spaced top to bottom
    left_ys = np.linspace(in_f - 0.6, 0.6, in_f)

    # Original logits: evenly spread in the middle third
    logit_ys = np.linspace(in_f * 0.80, in_f * 0.20, out_f)

    # Mask nodes: cluster around each logit
    mask_spacing = min(0.70, (in_f * 0.45) / max(out_f * n_mask, 1))
    mask_ys = []
    for i in range(out_f):
        c    = logit_ys[i]
        span = mask_spacing * (n_mask - 1)
        ys   = [c + span / 2 - m * mask_spacing for m in range(n_mask)]
        mask_ys.append(ys)

    # ── node sizes ────────────────────────────────────────────────────────────
    R_LEFT  = 0.36
    R_LOGIT = 0.32
    R_MASK  = 0.20

    # ── drawing helpers ───────────────────────────────────────────────────────

    def draw_circle(cx, cy, r, face, edge="#555555", lw=1.4, z=4):
        ax.add_patch(plt.Circle(
            (cx, cy), r, facecolor=face,
            edgecolor=edge, linewidth=lw, zorder=z
        ))

    def draw_line(x1, y1, x2, y2, colour, lw=1.0, alpha=0.55, z=2):
        ax.plot([x1, x2], [y1, y2],
                color=colour, linewidth=lw,
                alpha=alpha, zorder=z,
                solid_capstyle="round")

    def draw_arrow_label(x_start, y, colour, label, size=8.5,
                         bold=False, offset=0.52):
        """Small horizontal arrow then text — used for node value labels."""
        ax.annotate("",
            xy=(x_start + offset - 0.05, y),
            xytext=(x_start, y),
            arrowprops=dict(
                arrowstyle="-|>", color=colour,
                lw=1.0, mutation_scale=7
            ),
            zorder=5,
        )
        ax.text(x_start + offset + 0.02, y, label,
                fontsize=size, color=colour,
                ha="left", va="center",
                fontweight="bold" if bold else "normal",
                zorder=6)

    def label(x, y, s, size=8, colour="#333333",
              ha="left", va="center"):
        ax.text(x, y, s, fontsize=size, color=colour,
                ha=ha, va=va, zorder=6)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — connections (draw before nodes so nodes appear on top)
    # ══════════════════════════════════════════════════════════════════════════

    # Grey — full connections: every input neuron → every original logit
    for ni in range(in_f):
        for li in range(out_f):
            draw_line(
                X_LEFT + R_LEFT,   left_ys[ni],
                X_LOGIT - R_LOGIT, logit_ys[li],
                colour="#c8c8c8", lw=0.85, alpha=0.45
            )

    # Coloured — masked connections: input neuron → mask node (only where mask==1)
    for li in range(out_f):
        for m in range(n_mask):
            col = MASK_COLOURS[m % len(MASK_COLOURS)]
            for ni in range(in_f):
                if layer.binary_masks[m, li, ni].item() == 1:
                    draw_line(
                        X_LEFT + R_LEFT,  left_ys[ni],
                        X_MASK - R_MASK,  mask_ys[li][m],
                        colour=col, lw=1.6, alpha=0.82
                    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — nodes
    # ══════════════════════════════════════════════════════════════════════════

    # Input neurons — light grey
    for ni in range(in_f):
        draw_circle(X_LEFT, left_ys[ni], R_LEFT,
                    face="#dcdcdc", edge="#888888")

    # Original logits — medium grey, slightly larger
    for li in range(out_f):
        draw_circle(X_LOGIT, logit_ys[li], R_LOGIT,
                    face="#aaaaaa", edge="#333333", lw=2.0)

    # Mask nodes — each mask has its own colour
    for li in range(out_f):
        for m in range(n_mask):
            col = MASK_COLOURS[m % len(MASK_COLOURS)]
            draw_circle(X_MASK, mask_ys[li][m], R_MASK,
                        face=col, edge="#222222", lw=1.2)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — labels and values
    # ══════════════════════════════════════════════════════════════════════════

    # Input neuron labels (to the left of each circle)
    for ni in range(in_f):
        label(X_LEFT - R_LEFT - 0.10, left_ys[ni],
              f"neuron_{ni}", size=7.5,
              colour="#555555", ha="right")

    # Original logit labels with value
    for li in range(out_f):
        val = output.original[sample_index, li].item()
        draw_arrow_label(
            X_LOGIT + R_LOGIT + 0.10,
            logit_ys[li],
            colour="#111111",
            label=f"u\u0302_{li} = {val:+.3f}",
            size=9.5, bold=True, offset=0.50
        )

    # Mask node labels with value
    for li in range(out_f):
        for m in range(n_mask):
            val = output.masked[sample_index, li, m].item()
            col = MASK_COLOURS[m % len(MASK_COLOURS)]
            draw_arrow_label(
                X_MASK + R_MASK + 0.10,
                mask_ys[li][m],
                colour=col,
                label=f"u\u0302_{li},{m+1} = {val:+.3f}",
                size=8.0, bold=False, offset=0.48
            )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — μ and d brackets per logit group
    # ══════════════════════════════════════════════════════════════════════════

    for li in range(out_f):
        mu_val = output.mean[sample_index, li].item()
        d_val  = output.spread[sample_index, li].item()

        # y range: covers original logit + all its masks
        all_ys = [logit_ys[li]] + mask_ys[li]
        y_top  = max(all_ys) + R_MASK + 0.12
        y_bot  = min(all_ys) - R_MASK - 0.12
        y_mid  = (y_top + y_bot) / 2

        bx = X_BRACE

        # Bracket shape: top horizontal, vertical spine, bottom horizontal, mid tick
        bracket_col = "#333333"
        ax.plot([bx, bx + 0.14], [y_top, y_top], color=bracket_col, lw=1.8, zorder=4)
        ax.plot([bx + 0.14, bx + 0.14], [y_top, y_bot], color=bracket_col, lw=1.8, zorder=4)
        ax.plot([bx, bx + 0.14], [y_bot, y_bot], color=bracket_col, lw=1.8, zorder=4)
        ax.plot([bx + 0.14, bx + 0.28], [y_mid, y_mid], color=bracket_col, lw=1.8, zorder=4)

        # d_i label — spread value
        ax.text(bx + 0.35, y_mid,
                f"d\u2080{li+1} = {d_val:.3f}",
                fontsize=9.5, color="#111111",
                fontweight="bold", va="center", ha="left", zorder=6)

        # μ_i label — mean value, placed just below the mask cluster
        mu_y = min(mask_ys[li]) - mask_spacing * 0.75
        ax.text(X_MASK + R_MASK + 0.12, mu_y,
                f"\u03bc_{li+1} = {mu_val:+.3f}",
                fontsize=8, color="#666666",
                va="center", ha="left",
                style="italic", zorder=6)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — title, caption, legend
    # ══════════════════════════════════════════════════════════════════════════

    ax.set_title(
        "Alternatives Generator — Logit Masking Network Diagram",
        fontsize=13, fontweight="bold", pad=16, color="#111111"
    )

    caption = (
        f"Fig. 1   "
        f"N\u1d39 = {layer.num_masks} masks per logit,   "
        f"connection ratio = {layer.connection_ratio}   "
        f"({layer.n_connections} of {in_f} connections per mask)   "
        f"[sample {sample_index}]"
    )
    fig.text(0.5, 0.008, caption,
             ha="center", fontsize=8.5,
             color="#666666", style="italic")

    # Build legend
    handles = [
        mpatches.Patch(facecolor="#dcdcdc", edgecolor="#888888",
                       label=f"Input neurons  ({in_f} nodes)"),
        mpatches.Patch(facecolor="#aaaaaa", edgecolor="#333333",
                       label=f"Original logits  ({out_f} nodes, fully connected)"),
        plt.Line2D([0], [0], color="#c8c8c8", lw=2,
                   label="Grey lines = full connections (all neurons)"),
    ]
    for m in range(n_mask):
        col = MASK_COLOURS[m % len(MASK_COLOURS)]
        handles.append(mpatches.Patch(
            facecolor=col, edgecolor="#222222",
            label=f"Mask {m}  "
                  f"(neurons: {[ni for ni in range(in_f) if layer.binary_masks[m, 0, ni].item()==1]})"
        ))

    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=7.5,
        framealpha=0.93,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.0, 0.0)
    )

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs("demo", exist_ok=True)
    save_path = os.path.join("demo", "logit_masking_network.png")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Diagram saved -> {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    line("#")
    print("  Alternatives Generator for ANN")
    print("  LogitMaskingLayer -- Demonstration")
    print("  RCSE Master Research Project, TU Ilmenau")
    line("#")

    section("USER INPUT")
    print()
    num_masks        = ask_num_masks()
    connection_ratio = ask_connection_ratio()

    torch.manual_seed(MODEL_SEED)
    base_layer = nn.Linear(IN_FEATURES, OUT_FEATURES)

    layer = LogitMaskingLayer(
        base_layer       = base_layer,
        num_masks        = num_masks,
        connection_ratio = connection_ratio,
        seed             = MODEL_SEED,
    )

    torch.manual_seed(INPUT_SEED)
    x = torch.randn(BATCH_SIZE, IN_FEATURES)

    with torch.no_grad():
        output = layer(x)

    print_input_parameters(num_masks, connection_ratio, layer.n_connections)
    print_original(output)
    print_masked(output, num_masks)
    print_mean(output)
    print_spread(output)

    line("#")
    print("  The SPREAD values above are the foundation of uncertainty")
    print("  estimation. They feed into the Weighted Probability (Step 3)")
    print("  and Prospect Certainty (Step 4) computations.")
    line("#")
    print()

    section("GENERATING NETWORK DIAGRAM")
    print()
    plot_network_diagram(layer, output, sample_index=0)
    print()


if __name__ == "__main__":
    main()
