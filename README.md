# Alternatives Generator for ANN
### Master Research Project – RCSE, TU Ilmenau  
*15 credits · Research basis for the following 30-credit thesis*

---

## Project Overview

This module implements the **logit masking** concept introduced in:

> Yousef, Q. & Li, P. *Prospect certainty for data-driven models.*  
> Scientific Reports 15, 8278 (2025). https://doi.org/10.1038/s41598-025-89679-6

The core idea: a trained neural network is deterministic — every input always
maps to exactly one output. When the deployment data comes from a different
distribution than the training data, this determinism becomes a liability.
The paper addresses this by generating **mask alternatives** for each output
logit, each with a partial (configurable) random connection to the previous
layer. These alternatives are then scored by a **prospect certainty** function.

This project delivers a **reusable Python module** that can attach the
Alternatives Generator to *any* existing ANN without touching its internals.

---

## Deliverables

| # | Deliverable | File / Folder |
|---|-------------|---------------|
| 1 | Python library | `src/alternatives_generator/` |
| 2 | Demo examples | `demo/` |
| 3 | Documentation | `docs/` |
| 4 | Technical report | `docs/report.md` |
| 5 | Tests | `tests/` |

---

## Development Roadmap

```
Step 1  ──► DONE – Logit Masking demo
Step 2  ──► WeightedProbability function
Step 3  ──► BehaviorFunction
Step 4  ──► ProspectCertainty integration
Step 5  ──► Full demo + evaluation
Step 6  ──► Packaging, docs, GitHub release
```

---

## Step-by-Step Plan (Detailed)

### STEP 1  –  Logit Masking Demo  
**Status:** Done. 
**Goal:** Demonstrate that mask nodes can be attached to any output logit,
each with a random partial connection to the previous layer.

**What the demo shows:**
- 4 neurons on the left side (previous layer)
- 2 logits on the right side (output layer), fully connected to all 4
- 3 masks per logit, each randomly connected to 2 of the 4 left neurons
- Weight values for every connection
- Output value for every logit and every mask
- A visual connection map showing which neurons each mask uses

**Key concept:** Each mask sees only part of the input, so it produces
a slightly different output value than the original logit. The spread
between these values is the first signal of uncertainty.

**Files:** `demo/step1_masks_demo.py`

**Configuration (easy to change at the top of the file):**
```python
N_LEFT  = 4   # neurons in the previous layer
N_RIGHT = 2   # logits in the output layer
N_MASKS = 3   # masks per logit
N_CONN  = 2   # connections per mask (half of N_LEFT)
```

---

### STEP 2  –  WeightedProbability Function  *(next sprint)*
**Goal:** Assign each node (original + masks) a weighted probability Pr^w.

**Concept (from paper Section "Weighted probability"):**

The probability reflects three things:
1. **Occurrence count** — how often does this node's value appear among all siblings?
2. **Disparity level** — how far is it from the mean of all siblings?
3. **Logit priority** — original logit gets a slight edge over masks (bias constant s)

Equations from the paper:
```
Pr_t(u_t) = count(u_t, {masks, logit}) / (N^M + 1)

w_i,t   = 1 / ln(|u_i,t − µ| / d + e + ε)        # logit weight
w_i,j,t = 1 / ln(|u_i,j,t − µ| / d + e + ε + s)  # mask weight

Pr^w_t(u_t, w_t) = w_t · Pr_t(u_t)   [then normalized across siblings]
```

**Files to create:**
- `src/alternatives_generator/weighted_probability.py`
- `demo/step2_weighted_probability.py`
- `tests/test_step2.py`

---

### STEP 3  –  Behavior Function  *(third sprint)*
**Goal:** Measure how much each node helps the model output distribution
align with the training label distribution.

**Concept (from paper Section "Behavior function"):**
```
b(u_t) = W2(Y^s, Y_{t-1}) − W2(Y^s, Y_t(g(u_t)))
```
- Positive b → node improves alignment with training distribution
- Negative b → node moves output distribution further away
- Zero        → node has no effect

**Files to create:**
- `src/alternatives_generator/behavior.py`
- `demo/step3_behavior_function.py`
- `tests/test_step3.py`

---

### STEP 4  –  ProspectCertainty Integration  *(fourth sprint)*
**Goal:** Combine Pr^w and b() through Kahneman-Tversky prospect theory
to produce a scalar certainty score Omega for each node.

**Concept (from paper Section "Prospect certainty"):**
```
Omega(u_t) = Omega^b(b(u_t)) * Omega^w(Pr^w_t(u_t))

Omega^b(b) = b^e1                 if b >= epsilon  (gain)
           = -gamma_b * (-b)^e2   if b <  epsilon  (loss)

Omega^w(Pr^w) = exp(-(- ln Pr^w)^gamma_w)   [Prelec function]

Parameters: e1 = e2 = 0.88,  gamma_b = 2.25,  gamma_w = 0.61
```

**Files to create:**
- `src/alternatives_generator/prospect.py`
- `demo/step4_prospect_certainty.py`
- `tests/test_step4.py`

---

### STEP 5  –  Full Demo and Evaluation  *(fifth sprint)*
**Goal:** Reproduce the paper's benchmark experiments as a demonstration.

**Tasks:**
- Regression demo with synthetic data: f(x) = x*sin(x) + h1*x + h2
- Plot the certainty index Omega over the input range
- Compare output accuracy with and without the module

**Files:**
- `demo/step5_regression_benchmark.py`
- `demo/step5_classification_demo.py`

---

### STEP 6  –  Packaging, Documentation and GitHub  *(final sprint)*
**Goal:** A clean, installable package suitable for the thesis appendix.

**Tasks:**
- `pyproject.toml` for `pip install -e .`
- `docs/report.md` — technical report
- GitHub repository with README, usage examples
- CI: GitHub Action running `pytest tests/`

---

## Repository Structure (target)

```
alternatives_generator/
│
├── src/
│   └── alternatives_generator/
│       ├── __init__.py               public API
│       ├── core.py                   AlternativesGenerator class (Step 1)
│       ├── weighted_probability.py   Pr^w functions (Step 2)
│       ├── behavior.py               behavior function b() (Step 3)
│       └── prospect.py               Omega certainty function (Step 4)
│
├── demo/
│   ├── step0_simple_ann_numpy.py     DONE – archived baseline (pure NumPy)
│   ├── step0_simple_ann.py           DONE – archived baseline (PyTorch)
│   ├── step1_masks_demo.py           DONE – logit masking demo
│   ├── step2_weighted_probability.py
│   ├── step3_behavior_function.py
│   ├── step4_prospect_certainty.py
│   ├── step5_regression_benchmark.py
│   └── step5_classification_demo.py
│
├── tests/
│   ├── test_step1.py
│   ├── test_step2.py
│   ├── test_step3.py
│   └── test_step4.py
│
├── docs/
│   └── report.md
│
├── pyproject.toml
└── README.md
```

---

## Key Design Principles

1. **Non-invasive** — the module wraps any model without changing its weights or training loop
2. **Framework-agnostic core** — all math is implemented in NumPy; PyTorch is used as the model interface layer
3. **Configurable** — all hyperparameters (N^M, R, epsilon) are exposed as constructor arguments
4. **Observable** — every intermediate value (z, Pr, b, Omega) can be inspected for research purposes
5. **Reproducible** — all random operations accept a seed

---

## Dependencies

| Package | Purpose | Step |
|---------|---------|------|
| `numpy` | All core math | 0-6 |
| `scipy` | wasserstein_distance | 3 |
| `torch` | Model definition, training loop, tensor ops | 0-6 |
| `matplotlib` | Plots and visualisation | 5 |
| `pytest` | Unit tests | all |

Install: `pip install torch numpy scipy matplotlib pytest`

---

## Reference

Yousef, Q. & Li, P. (2025). Prospect certainty for data-driven models.  
*Scientific Reports*, 15, 8278. https://doi.org/10.1038/s41598-025-89679-6

Code (authors' original): https://doi.org/10.5281/zenodo.14541878
