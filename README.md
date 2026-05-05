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
<https://doi.org/10.1038/s41598-025-89679-6>
