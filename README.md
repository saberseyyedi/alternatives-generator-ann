# Alternatives Generator for ANN
**Logit Masking for Uncertainty Estimation in Neural Networks**

Master Research Project · RCSE · TU Ilmenau · 2026

---

## Overview

This project implements a reusable PyTorch module that extends the output
layer of any existing neural network by generating **masked alternative
outputs** for each logit.

By comparing these alternatives, the module produces an interpretable
uncertainty signal — without modifying the base model or retraining.

---

## Key Idea

A standard output logit is **fully connected** to all neurons in the previous
layer. A **masked alternative** connects to only a random subset of those
neurons, using the original trained weights with some connections zeroed out:

```
masked_weight = original_weight * binary_mask
```

Multiple alternatives are generated per logit. Their **spread** (max − min)
measures how much they disagree — high spread means uncertain output, low
spread means consistent output.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/alternatives-generator-ann.git
cd alternatives-generator-ann

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install torch numpy matplotlib pytest
```

---

## How to Run

```bash
python run_masking.py
```

The script asks two questions:

```
Number of masks per logit [default: 3]:
Connection ratio (0.1 – 0.9) [default: 0.5]:
```

It then prints the original logits, masked alternatives, mean, spread, and
uncertainty score per sample, and saves a network diagram to
`demo/logit_masking_network.png`.

---

## Module Usage

```python
import torch
import torch.nn as nn
from src.alternatives_generator import LogitMaskingLayer

# Any existing linear layer
base = nn.Linear(128, 10)

# Wrap it
layer = LogitMaskingLayer(
    base_layer       = base,
    num_masks        = 3,
    connection_ratio = 0.5,
    seed             = 42,
)

# Forward pass
x      = torch.randn(4, 128)
output = layer(x)

print(output.original)      # (4, 10)   — original logits
print(output.masked)        # (4, 10, 3) — masked alternatives
print(output.spread)        # (4, 10)   — uncertainty per logit
print(output.uncertainty)   # (4,)      — scalar score per sample
```

The module infers `in_features` and `out_features` automatically.
You only control `num_masks` and `connection_ratio`.

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
│
├── src/
│   └── alternatives_generator/
│       ├── __init__.py
│       └── logit_masking.py      ← core reusable module
│
├── tests/
│   └── test_logit_masking.py     ← 15 automated unit tests
│
├── run_masking.py                ← interactive demo script
└── README.md
```

---

## Run Tests

```bash
pytest tests/test_logit_masking.py -v
```

---

## Limitations

- Wraps `nn.Linear` output layers only (no Conv support yet)
- Masks are fixed after initialisation — not adaptive during training
- Spread is a raw uncertainty signal; calibrated scoring is future work

---

## Reference

Yousef, Q. & Li, P. (2025). *Prospect certainty for data-driven models.*
Scientific Reports, 15, 8278.
[https://doi.org/10.1038/s41598-025-89679-6](https://doi.org/10.1038/s41598-025-89679-6)
