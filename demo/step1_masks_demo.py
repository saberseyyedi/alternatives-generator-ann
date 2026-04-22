"""
Step 1 – Alternatives Generator: Logit Masking Demo
=====================================================
Architecture:
  - 4 neurons on the left side (previous layer)
  - 2 logits on the right side (output layer), fully connected
  - 3 masks per logit, each randomly connected to 2 of the 4 left neurons

What this file shows:
  1. The 4 left-side neuron values (input)
  2. Weight connections for each logit and each mask
  3. Output value for each logit and each mask
  4. Visual summary of connections

Run:
    python demo/step1_masks_demo.py
"""

import numpy as np

# ── reproducibility ────────────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

N_LEFT   = 4   # neurons in the previous layer
N_RIGHT  = 2   # logits in the output layer
N_MASKS  = 3   # masks per logit
N_CONN   = 2   # how many left neurons each mask connects to (half of 4)


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

class Logit:
    """
    One original output logit — fully connected to all left neurons.

    Attributes
    ----------
    index       : which logit this is (0 or 1)
    weights     : shape (N_LEFT,)  — one weight per left neuron
    bias        : scalar
    masks       : list of Mask objects
    """
    def __init__(self, index: int):
        self.index   = index
        self.weights = RNG.uniform(-1.0, 1.0, size=N_LEFT)
        self.bias    = RNG.uniform(-0.5, 0.5)
        self.masks   = []

    def forward(self, x: np.ndarray) -> float:
        """z = w · x + b   (no activation here, raw logit value)"""
        return float(np.dot(self.weights, x) + self.bias)


class Mask:
    """
    One mask for a logit — partially connected to N_CONN left neurons.

    Attributes
    ----------
    logit_index  : which logit this mask belongs to
    mask_index   : which mask this is (1, 2, or 3)
    connected_to : list of left-neuron indices this mask connects to
    weights      : shape (N_CONN,)  — one weight per connected neuron
    bias         : scalar
    """
    def __init__(self, logit_index: int, mask_index: int):
        self.logit_index  = logit_index
        self.mask_index   = mask_index

        # randomly pick N_CONN neurons from the left side (no replacement)
        self.connected_to = sorted(
            RNG.choice(N_LEFT, size=N_CONN, replace=False).tolist()
        )

        # weights only for the connected neurons
        self.weights = RNG.uniform(-1.0, 1.0, size=N_CONN)
        self.bias    = RNG.uniform(-0.5, 0.5)

    def forward(self, x: np.ndarray) -> float:
        """z = w · x_partial + b   (only uses connected neurons)"""
        x_partial = x[self.connected_to]
        return float(np.dot(self.weights, x_partial) + self.bias)


# ══════════════════════════════════════════════════════════════════════════════
# Build the network
# ══════════════════════════════════════════════════════════════════════════════

def build_network() -> list[Logit]:
    logits = []
    for i in range(N_RIGHT):
        logit = Logit(index=i)
        for j in range(1, N_MASKS + 1):
            logit.masks.append(Mask(logit_index=i, mask_index=j))
        logits.append(logit)
    return logits


# ══════════════════════════════════════════════════════════════════════════════
# Print helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_section(title: str) -> None:
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")


def print_subsection(title: str) -> None:
    print(f"\n  {'─'*61}")
    print(f"  {title}")
    print(f"  {'─'*61}")


# ══════════════════════════════════════════════════════════════════════════════
# Display functions
# ══════════════════════════════════════════════════════════════════════════════

def show_input(x: np.ndarray) -> None:
    print_section("1. LEFT SIDE — 4 Input Neuron Values")
    print()
    for i, val in enumerate(x):
        bar = "█" * int(abs(val) * 20)
        sign = "+" if val >= 0 else "-"
        print(f"    neuron_{i}  =  {val:+.4f}   {sign}{bar}")
    print()


def show_weights(logits: list[Logit]) -> None:
    print_section("2. WEIGHTS — Logits and Their Masks")

    for logit in logits:
        print_subsection(f"logit_{logit.index}  (fully connected to all {N_LEFT} neurons)")

        # original logit weights
        print(f"\n    [logit_{logit.index}]  bias = {logit.bias:+.4f}")
        print(f"    {'Neuron':<12} {'Weight':>10}  {'Connection'}")
        print(f"    {'──────':<12} {'──────':>10}  {'──────────'}")
        for i, w in enumerate(logit.weights):
            print(f"    neuron_{i:<5}  {w:>+10.4f}  ✓ connected")

        # each mask
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


def show_outputs(logits: list[Logit], x: np.ndarray) -> None:
    print_section("3. OUTPUTS — Logits and Their Masks")

    for logit in logits:
        print(f"\n  logit_{logit.index} group:")
        print(f"  {'Node':<14} {'Output Value':>14}  {'Type'}")
        print(f"  {'────':<14} {'────────────':>14}  {'────'}")

        # original logit
        val = logit.forward(x)
        print(f"  {'logit_'+str(logit.index):<14} {val:>+14.6f}  original (fully connected)")

        # each mask
        mask_vals = []
        for mask in logit.masks:
            mval = mask.forward(x)
            mask_vals.append(mval)
            connected_str = f"neurons {mask.connected_to}"
            print(f"  {'mask_'+str(logit.index)+'_'+str(mask.mask_index):<14} "
                  f"{mval:>+14.6f}  mask (uses {connected_str})")

        # show spread
        all_vals = [logit.forward(x)] + mask_vals
        spread = max(all_vals) - min(all_vals)
        mean   = np.mean(all_vals)
        print(f"\n  {'mean of group:':<24} {mean:>+10.6f}")
        print(f"  {'spread (max - min):':<24} {spread:>10.6f}  "
              f"{'← low spread = consistent' if spread < 0.5 else '← high spread = uncertain'}")


def show_visual(logits: list[Logit], x: np.ndarray) -> None:
    print_section("4. VISUAL — Connection Map")

    # column headers
    node_names = []
    for logit in logits:
        node_names.append(f"lgt_{logit.index}")
        for mask in logit.masks:
            node_names.append(f"m{logit.index}_{mask.mask_index}")

    header = "              " + "  ".join(f"{n:^7}" for n in node_names)
    print(f"\n{header}")
    print(f"              " + "  ".join("───────" for _ in node_names))

    # one row per left neuron
    for i in range(N_LEFT):
        val = x[i]
        row = f"  neuron_{i} ({val:+.2f})  "
        for logit in logits:
            # original logit — always connected
            row += f"  {'✓':^7}"
            # masks
            for mask in logit.masks:
                if i in mask.connected_to:
                    w = mask.weights[mask.connected_to.index(i)]
                    row += f"  {'✓':^7}"
                else:
                    row += f"  {'✗':^7}"
        print(row)

    # legend
    print(f"\n  ✓ = connected    ✗ = not connected")
    print(f"  lgt = original logit (fully connected)")
    print(f"  m   = mask (partially connected, {N_CONN} of {N_LEFT} neurons)")

    # weight map
    print(f"\n  Weight values at each connection:")
    print(f"\n{header}")
    print(f"              " + "  ".join("───────" for _ in node_names))

    for i in range(N_LEFT):
        val = x[i]
        row = f"  neuron_{i} ({val:+.2f})  "
        for logit in logits:
            w = logit.weights[i]
            row += f"  {w:^+7.3f}"
            for mask in logit.masks:
                if i in mask.connected_to:
                    w = mask.weights[mask.connected_to.index(i)]
                    row += f"  {w:^+7.3f}"
                else:
                    row += f"  {'──':^7}"
        print(row)

    # output row
    print(f"              " + "  ".join("───────" for _ in node_names))
    out_row = f"  {'output':^14}  "
    for logit in logits:
        v = logit.forward(x)
        out_row += f"  {v:^+7.3f}"
        for mask in logit.masks:
            mv = mask.forward(x)
            out_row += f"  {mv:^+7.3f}"
    print(out_row)


def show_summary(logits: list[Logit], x: np.ndarray) -> None:
    print_section("SUMMARY")
    print(f"\n  Network structure:")
    print(f"    Left side  : {N_LEFT} neurons")
    print(f"    Right side : {N_RIGHT} logits  (fully connected to all {N_LEFT} neurons)")
    print(f"    Masks      : {N_MASKS} per logit  "
          f"(each connected to {N_CONN} of {N_LEFT} neurons randomly)")
    print(f"    Total nodes: {N_RIGHT} logits + {N_RIGHT * N_MASKS} masks "
          f"= {N_RIGHT + N_RIGHT * N_MASKS} output nodes")

    print(f"\n  Output values:")
    for logit in logits:
        val = logit.forward(x)
        print(f"\n    logit_{logit.index} = {val:+.6f}")
        for mask in logit.masks:
            mval = mask.forward(x)
            diff = mval - val
            print(f"      mask_{logit.index}_{mask.mask_index} = {mval:+.6f}  "
                  f"(diff from logit: {diff:+.6f})")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "█"*65)
    print("  STEP 1 – Alternatives Generator: Logit Masking Demo")
    print(f"  {N_LEFT} left neurons │ {N_RIGHT} logits │ "
          f"{N_MASKS} masks per logit │ {N_CONN}/{N_LEFT} connections per mask")
    print("█"*65)

    # ── build network ──────────────────────────────────────────────────────────
    logits = build_network()

    # ── sample input (4 left neuron values) ───────────────────────────────────
    x = np.array([0.8, 0.3, 0.6, 0.1])

    # ── show everything ────────────────────────────────────────────────────────
    show_input(x)
    show_weights(logits)
    show_outputs(logits, x)
    show_visual(logits, x)
    show_summary(logits, x)

    print("\n" + "█"*65 + "\n")


if __name__ == "__main__":
    main()
