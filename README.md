# Alternatives Generator for ANN

A lightweight PyTorch module that generates masked alternative outputs for
each logit in a neural network, providing an interpretable uncertainty signal
without retraining or modifying the base model.

> **Research Project** — RCSE Master Programme, TU Ilmenau, Summer Semester 2026  
> Based on: Yousef & Li, *Prospect certainty for data-driven models*, Scientific Reports 15:8278 (2025)

---

## Key Idea

Standard neural networks always produce a confident prediction, even for
unfamiliar inputs. This module attaches to any linear output layer and
generates **masked alternatives** for each logit — partial copies that reuse
the original weights but connect to only a random subset of input neurons.

The **spread** (max − min) across alternatives signals uncertainty:

- **Low spread** → alternatives agree → prediction is reliable
- **High spread** → alternatives disagree → prediction is uncertain

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/alternatives-generator-ann.git
cd alternatives-generator-ann

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch numpy matplotlib pytest
```

---

## How to Run

```bash
python run_masking.py
```

You will be asked two questions:

```
Number of masks per logit    [default: 3]:
Connection ratio (0.1–0.9)   [default: 0.5]:
```

The script prints original logits, masked alternatives, mean, and spread for
each logit group, then saves a network diagram to
`demo/logit_masking_network.png`.

---

## Module Usage

```python
import torch
import torch.nn as nn
from src.alternatives_generator import LogitMaskingLayer

# wrap any existing linear layer
base = nn.Linear(128, 10)
layer = LogitMaskingLayer(
    base_layer       = base,
    num_masks        = 3,
    connection_ratio = 0.5,
    seed             = 42,
)

x = torch.randn(4, 128)
output = layer(x)

print(output.original)     # (4, 10)  — original logits
print(output.masked)       # (4, 10, 3) — masked alternatives
print(output.spread)       # (4, 10)  — uncertainty per logit
print(output.uncertainty)  # (4,)     — scalar score per sample
```

The module infers `in_features` and `out_features` automatically.
The user controls only `num_masks` and `connection_ratio`.

### Output fields

| Field | Shape | Description |
|---|---|---|
| `original` | `(batch, out)` | Unmasked logit values |
| `masked` | `(batch, out, num_masks)` | All masked alternatives |
| `mean` | `(batch, out)` | Mean across all alternatives |
| `spread` | `(batch, out)` | Max − min across alternatives |
| `uncertainty` | `(batch,)` | Mean spread across all logits |

---

## Project Structure

```
alternatives-generator-ann/
├── src/
│   └── alternatives_generator/
│       ├── __init__.py
│       └── logit_masking.py      # LogitMaskingLayer module
├── tests/
│   └── test_logit_masking.py     # 15 automated unit tests
├── run_masking.py                 # interactive demo script
└── README.md
```

---

## Run Tests

```bash
pytest tests/test_logit_masking.py -v
```

---

## Limitations

- Wraps `nn.Linear` output layers only (no Conv layers yet)
- Masks are fixed after initialisation — not adaptive during training
- Spread is a raw uncertainty signal; calibrated scoring (Steps 3–4 of the
  full pipeline) is left for future work

---

## Reference

Yousef, Q. & Li, P. (2025). Prospect certainty for data-driven models.
*Scientific Reports*, 15, 8278.
https://doi.org/10.1038/s41598-025-89679-6

Authors' original code: https://doi.org/10.5281/zenodo.14541878

---

## 11. Graphical Output

Running `python run_masking.py` automatically generates and saves a network
diagram to `demo/logit_masking_network.png`.

### What the diagram shows

The diagram has three columns:

**Left column — Input neurons** (grey circles)
One circle per input neuron. Labels show `neuron_0` through `neuron_N`.

**Middle column — Original logits** (larger grey circles)
One circle per output logit. Fully connected to all input neurons.
Labels show the computed value: `û_0 = +0.324`

**Right column — Mask nodes** (small coloured circles)
Each logit has `num_masks` mask alternatives.
Each mask has a unique colour shared between its circle, its connections,
and its label.
Labels show: `û_0,1 = +0.182`

### Connection lines

| Line colour | Meaning |
|---|---|
| Grey | Full connection — this input connects to the original logit |
| Coloured | Masked connection — this input connects to this specific mask |
| No line | This input was not selected for this mask |

The coloured lines make it immediately visible which neurons each mask
can "see" and which are hidden from it.

### Annotations

**μ_i (mean)** — the average value across the original logit and all its
masks. Placed below each mask cluster.

**d_i (spread)** — the range (max − min) across all alternatives for
logit i. Shown in the bracket on the right side.
A large d_i means the alternatives disagree — the model is uncertain.
A small d_i means they agree — the model is consistent.

### Example diagram

![Network diagram](demo/logit_masking_network.png)
