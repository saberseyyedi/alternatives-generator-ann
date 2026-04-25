"""
Step 1 – Alternatives Generator: Logit Masking Demo
=====================================================
Architecture:
  - 4 neurons on the left side (previous layer)
  - 2 logits on the right side (output layer), fully connected
  - 3 masks per logit, each randomly connected to 2 of the 4 left neurons

Outputs:
  1. Terminal: weights, outputs, connection table
  2. Graphical: network diagram matching Fig.1 from the paper

Run:
    python demo/step1_masks_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── reproducibility ────────────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

N_LEFT  = 4   # neurons in the previous layer
N_RIGHT = 2   # logits in the output layer
N_MASKS = 3   # masks per logit
N_CONN  = 2   # how many left neurons each mask connects to (half of N_LEFT)

# colours per mask group — matching the paper's Fig.1 style
# logit_0 group: red, green, blue masks
# logit_1 group: purple, yellow, pink masks
MASK_COLORS = [
    ["#e05c5c", "#5cb85c", "#5c7de0"],   # masks for logit_0
    ["#a05ce0", "#e0c45c", "#e08c5c"],   # masks for logit_1
]
LOGIT_COLOR  = "#aaaaaa"   # grey — original logits (like paper)
LEFT_COLOR   = "#cccccc"   # light grey — left side neurons


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

class Logit:
    def __init__(self, index: int):
        self.index    = index
        self.weights  = RNG.uniform(-1.0, 1.0, size=N_LEFT)
        self.bias     = RNG.uniform(-0.5, 0.5)
        self.masks    = []

    def forward(self, x: np.ndarray) -> float:
        return float(np.dot(self.weights, x) + self.bias)


class Mask:
    def __init__(self, logit_index: int, mask_index: int):
        self.logit_index  = logit_index
        self.mask_index   = mask_index
        self.connected_to = sorted(
            RNG.choice(N_LEFT, size=N_CONN, replace=False).tolist()
        )
        self.weights = RNG.uniform(-1.0, 1.0, size=N_CONN)
        self.bias    = RNG.uniform(-0.5, 0.5)

    def forward(self, x: np.ndarray) -> float:
        x_partial = x[self.connected_to]
        return float(np.dot(self.weights, x_partial) + self.bias)


# ══════════════════════════════════════════════════════════════════════════════
# Build network
# ══════════════════════════════════════════════════════════════════════════════

def build_network() -> list:
    logits = []
    for i in range(N_RIGHT):
        logit = Logit(index=i)
        for j in range(1, N_MASKS + 1):
            logit.masks.append(Mask(logit_index=i, mask_index=j))
        logits.append(logit)
    return logits


# ══════════════════════════════════════════════════════════════════════════════
# Terminal output helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_section(title):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")


def show_input(x):
    print_section("1. LEFT SIDE — 4 Input Neuron Values")
    print()
    for i, val in enumerate(x):
        bar  = "█" * int(abs(val) * 20)
        sign = "+" if val >= 0 else "-"
        print(f"    neuron_{i}  =  {val:+.4f}   {sign}{bar}")


def show_weights(logits):
    print_section("2. WEIGHTS — Logits and Their Masks")
    for logit in logits:
        print(f"\n  {'─'*61}")
        print(f"  logit_{logit.index}  (fully connected to all {N_LEFT} neurons)")
        print(f"  {'─'*61}")
        print(f"\n    [logit_{logit.index}]  bias = {logit.bias:+.4f}")
        print(f"    {'Neuron':<12} {'Weight':>10}  {'Connection'}")
        print(f"    {'──────':<12} {'──────':>10}  {'──────────'}")
        for i, w in enumerate(logit.weights):
            print(f"    neuron_{i:<5}  {w:>+10.4f}  ✓ connected")
        for mask in logit.masks:
            print(f"\n    [mask_{logit.index}_{mask.mask_index}]  "
                  f"bias = {mask.bias:+.4f}  "
                  f"→ connected to neurons {mask.connected_to}")
            print(f"    {'Neuron':<12} {'Weight':>10}  {'Connection'}")
            print(f"    {'──────':<12} {'──────':>10}  {'──────────'}")
            for i in range(N_LEFT):
                if i in mask.connected_to:
                    w = mask.weights[mask.connected_to.index(i)]
                    print(f"    neuron_{i:<5}  {w:>+10.4f}  ✓ connected")
                else:
                    print(f"    neuron_{i:<5}  {'——':>10}   ✗ not connected")


def show_outputs(logits, x):
    print_section("3. OUTPUTS — Logits and Their Masks")
    for logit in logits:
        print(f"\n  logit_{logit.index} group:")
        print(f"  {'Node':<14} {'Output Value':>14}  {'Type'}")
        print(f"  {'────':<14} {'────────────':>14}  {'────'}")
        val = logit.forward(x)
        print(f"  {'logit_'+str(logit.index):<14} {val:>+14.6f}  original (fully connected)")
        mask_vals = []
        for mask in logit.masks:
            mval = mask.forward(x)
            mask_vals.append(mval)
            print(f"  {'mask_'+str(logit.index)+'_'+str(mask.mask_index):<14} "
                  f"{mval:>+14.6f}  mask → neurons {mask.connected_to}")
        all_vals = [val] + mask_vals
        spread   = max(all_vals) - min(all_vals)
        mean     = np.mean(all_vals)
        print(f"\n  {'mean of group:':<24} {mean:>+10.6f}")
        print(f"  {'spread (max - min):':<24} {spread:>10.6f}  "
              f"{'← low spread = consistent' if spread < 0.5 else '← high spread = uncertain'}")


def show_summary(logits, x):
    print_section("SUMMARY")
    print(f"\n  Network structure:")
    print(f"    Left side  : {N_LEFT} neurons")
    print(f"    Right side : {N_RIGHT} logits  (fully connected to all {N_LEFT})")
    print(f"    Masks      : {N_MASKS} per logit  ({N_CONN} of {N_LEFT} connections each)")
    print(f"    Total nodes: {N_RIGHT} logits + {N_RIGHT*N_MASKS} masks "
          f"= {N_RIGHT + N_RIGHT*N_MASKS} output nodes\n")
    for logit in logits:
        val = logit.forward(x)
        print(f"    logit_{logit.index} = {val:+.6f}")
        for mask in logit.masks:
            mval = mask.forward(x)
            diff = mval - val
            print(f"      mask_{logit.index}_{mask.mask_index} = {mval:+.6f}  "
                  f"(diff from logit: {diff:+.6f})")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# Graphical output — matches Fig.1 style from the paper
# ══════════════════════════════════════════════════════════════════════════════

def draw_network(logits: list, x: np.ndarray) -> None:
    """
    Draw the network diagram:
      - Left column  : 4 large grey circles (left neurons)
      - Middle column: 2 large grey circles (original logits)
                       6 small coloured circles (masks, 3 per logit)
      - Arrows       : grey for logit connections, coloured for mask connections
      - Labels       : u_hat notation matching the paper
      - Right side   : d1/d2 distance braces and mu1/mu2 mean labels
    """

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── node positions ─────────────────────────────────────────────────────────
    # left neurons — evenly spaced vertically
    left_x = 2.5
    left_y = [8.5, 6.5, 4.5, 2.5]   # neuron_0 at top

    # right side — two groups separated by a gap
    right_logit_x = 5.8   # original logit x position
    mask_x        = 7.5   # mask x position

    # group 0 (top): logit_0 at y=8.5, masks at 9.0, 8.0, 7.0
    # group 1 (bot): logit_1 at y=4.0, masks at 4.5, 3.5, 2.5
    logit_y = [8.5, 4.0]
    mask_y  = [
        [9.2, 8.2, 7.2],   # masks for logit_0
        [4.7, 3.7, 2.7],   # masks for logit_1
    ]

    # ── node sizes ─────────────────────────────────────────────────────────────
    r_left  = 0.42   # left neuron radius
    r_logit = 0.38   # original logit radius
    r_mask  = 0.24   # mask radius

    # ── helper: draw a circle node ─────────────────────────────────────────────
    def draw_node(cx, cy, radius, facecolor, edgecolor="#555555", lw=1.5, zorder=3):
        circle = plt.Circle((cx, cy), radius,
                             facecolor=facecolor, edgecolor=edgecolor,
                             linewidth=lw, zorder=zorder)
        ax.add_patch(circle)

    # ── helper: draw an arrow ──────────────────────────────────────────────────
    def draw_arrow(x1, y1, x2, y2, color, lw=1.2, alpha=0.7, zorder=2):
        ax.annotate("",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                alpha=alpha,
                mutation_scale=10,
            ),
            zorder=zorder,
        )

    # ── helper: LaTeX-style label ──────────────────────────────────────────────
    def label(text, cx, cy, fontsize=10, color="black", ha="left", va="center"):
        ax.text(cx, cy, text, fontsize=fontsize, color=color,
                ha=ha, va=va, zorder=5,
                fontfamily="DejaVu Sans")

    # ══════════════════════════════════════════════════════════════════════════
    # Draw connections FIRST (so nodes appear on top)
    # ══════════════════════════════════════════════════════════════════════════

    for li, logit in enumerate(logits):
        ly = logit_y[li]

        # -- fully connected arrows (left → logit), light grey
        for ni in range(N_LEFT):
            ny = left_y[ni]
            draw_arrow(left_x + r_left, ny,
                       right_logit_x - r_logit, ly,
                       color="#bbbbbb", lw=1.0, alpha=0.5)

        # -- mask arrows (left → mask), coloured per mask
        for mi, mask in enumerate(logit.masks):
            my = mask_y[li][mi]
            mc = MASK_COLORS[li][mi]
            for ni in mask.connected_to:
                ny = left_y[ni]
                draw_arrow(left_x + r_left, ny,
                           mask_x - r_mask, my,
                           color=mc, lw=1.6, alpha=0.85)

    # ══════════════════════════════════════════════════════════════════════════
    # Draw nodes
    # ══════════════════════════════════════════════════════════════════════════

    # left neurons
    for ni in range(N_LEFT):
        draw_node(left_x, left_y[ni], r_left,
                  facecolor=LEFT_COLOR, edgecolor="#888888")

    # logits and masks
    for li, logit in enumerate(logits):
        # original logit — grey, larger
        draw_node(right_logit_x, logit_y[li], r_logit,
                  facecolor=LOGIT_COLOR, edgecolor="#555555")

        # masks — coloured, smaller
        for mi, mask in enumerate(logit.masks):
            draw_node(mask_x, mask_y[li][mi], r_mask,
                      facecolor=MASK_COLORS[li][mi],
                      edgecolor="#333333", lw=1.2)

    # ══════════════════════════════════════════════════════════════════════════
    # Labels — u_hat notation matching the paper
    # ══════════════════════════════════════════════════════════════════════════

    arrow_label_x = mask_x + r_mask + 0.15

    for li, logit in enumerate(logits):
        val = logit.forward(x)

        # original logit label with arrow
        lx = right_logit_x + r_logit + 0.12
        ly = logit_y[li]
        ax.annotate("",
            xy=(lx + 0.55, ly),
            xytext=(lx, ly),
            arrowprops=dict(arrowstyle="-|>", color="#333333",
                            lw=1.2, mutation_scale=9),
            zorder=4)
        ax.text(lx + 0.65, ly,
                f"û$_{li}$ = {val:+.3f}",
                fontsize=9.5, va="center", color="#222222", zorder=5)

        # mask labels
        for mi, mask in enumerate(logit.masks):
            mval = mask.forward(x)
            mc   = MASK_COLORS[li][mi]
            my   = mask_y[li][mi]
            mx   = mask_x + r_mask + 0.12
            ax.annotate("",
                xy=(mx + 0.55, my),
                xytext=(mx, my),
                arrowprops=dict(arrowstyle="-|>", color=mc,
                                lw=1.2, mutation_scale=9),
                zorder=4)
            ax.text(mx + 0.65, my,
                    f"û$_{{{li},{mi+1}}}$ = {mval:+.3f}",
                    fontsize=8.5, va="center", color=mc, zorder=5)

    # ══════════════════════════════════════════════════════════════════════════
    # Braces for d1, d2 and mu labels — matching paper style
    # ══════════════════════════════════════════════════════════════════════════

    brace_x = 9.35

    for li, logit in enumerate(logits):
        val       = logit.forward(x)
        mask_vals = [m.forward(x) for m in logit.masks]
        mu_val    = float(np.mean(mask_vals))

        # y range for brace: from lowest mask to original logit
        y_top = max(logit_y[li], max(mask_y[li]))
        y_bot = min(logit_y[li], min(mask_y[li]))
        y_mid = (y_top + y_bot) / 2

        # vertical brace using a simple bracket drawn with lines
        bx = brace_x
        ax.plot([bx, bx + 0.08, bx + 0.08, bx],
                [y_top, y_top, y_bot, y_bot],
                color="#333333", lw=1.5, zorder=4,
                solid_capstyle="round")
        ax.plot([bx + 0.08, bx + 0.18],
                [(y_top + y_bot) / 2, (y_top + y_bot) / 2],
                color="#333333", lw=1.5, zorder=4)

        # d label
        ax.text(bx + 0.25, y_mid,
                f"$d_{li+1}$",
                fontsize=11, va="center", color="#222222", zorder=5)

        # mu label — between masks only
        mu_y = float(np.mean(mask_y[li]))
        ax.text(mask_x + r_mask + 0.12, mu_y - 0.55,
                f"$\\mu_{li+1}$",
                fontsize=10, va="center", color="#555555",
                style="italic", zorder=5)

    # ══════════════════════════════════════════════════════════════════════════
    # Title and caption
    # ══════════════════════════════════════════════════════════════════════════

    ax.set_title("Logit Masking — Alternatives Generator",
                 fontsize=13, fontweight="bold", pad=14, color="#111111")

    caption = (
        f"Fig. 1  Method of masking and ratio:  "
        f"$N_1^M = N_2^M = {N_MASKS}$,  "
        f"$R_i = {{{N_CONN}/{N_LEFT}}}$ connections per mask  "
        f"(random, seed=42)"
    )
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9,
             color="#444444", style="italic")

    # ══════════════════════════════════════════════════════════════════════════
    # Legend
    # ══════════════════════════════════════════════════════════════════════════

    legend_elements = [
        mpatches.Patch(facecolor=LEFT_COLOR,  edgecolor="#888888", label="Left neurons (previous layer)"),
        mpatches.Patch(facecolor=LOGIT_COLOR, edgecolor="#555555", label="Original logits (fully connected)"),
    ]
    for li in range(N_RIGHT):
        for mi in range(N_MASKS):
            legend_elements.append(
                mpatches.Patch(
                    facecolor=MASK_COLORS[li][mi],
                    edgecolor="#333333",
                    label=f"mask_{li}_{mi+1}  → neurons {logits[li].masks[mi].connected_to}"
                )
            )

    ax.legend(handles=legend_elements,
              loc="lower left",
              fontsize=7.5,
              framealpha=0.9,
              edgecolor="#cccccc",
              bbox_to_anchor=(0.0, 0.03))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("demo/step1_network_diagram.png", dpi=150, bbox_inches="tight")
    print("\n  ✓ Diagram saved → demo/step1_network_diagram.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "█"*65)
    print("  STEP 1 – Alternatives Generator: Logit Masking Demo")
    print(f"  {N_LEFT} left neurons │ {N_RIGHT} logits │ "
          f"{N_MASKS} masks per logit │ {N_CONN}/{N_LEFT} connections per mask")
    print("█"*65)

    logits = build_network()
    x      = np.array([0.8, 0.3, 0.6, 0.1])

    show_input(x)
    show_weights(logits)
    show_outputs(logits, x)
    show_summary(logits, x)

    print("\n  Generating network diagram...")
    draw_network(logits, x)


if __name__ == "__main__":
    main()
