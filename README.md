# Alternatives Generator for ANN
### Master Research Project – RCSE, TU Ilmenau  

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
Step 0  ──► Baseline ANN (this meeting)
Step 1  ──► AlternativesGenerator module
Step 2  ──► WeightedProbability function
Step 3  ──► BehaviorFunction
Step 4  ──► ProspectCertainty integration
Step 5  ──► Full demo + evaluation
Step 6  ──► Packaging, docs, GitHub release
```

---

## Step-by-Step Plan (Detailed)

### ✅ STEP 0  –  Baseline ANN  *(current meeting)*
**Goal:** A working, transparent ANN to review with the supervisor.  
**Files:** `demo/step0_simple_ann_numpy.py` (pure NumPy, no framework needed)  
         `demo/step0_simple_ann.py`       (PyTorch version)  
**Architecture:** `Input(4) → Dense(5, ReLU) → Dense(5, ReLU) → Dense(1, linear)`  
**What it shows:**
- Weight matrix W and bias b for every layer, before and after training  
- Pre-activation values z = Wx + b  
- Post-activation values a = activation(z)  
- Which neurons are "active" (a > 0 for ReLU)  

**Review checkpoint:** Run step0 with supervisor, confirm output format is useful.

---

### STEP 1  –  AlternativesGenerator Module  *(first module sprint)*
**Goal:** Attach N_mask extra output nodes to any existing model's output layer.

**Concept (from paper §"Logit masking"):**
- For each original output logit u_i, generate N^M_i mask nodes
- Each mask connects to only a *fraction* R_i,j of the previous layer's neurons
- The fraction and which neurons are chosen randomly
- The original logit remains unchanged (fully connected)

**API design:**
```python
from src.alternatives_generator import AlternativesGenerator

ag = AlternativesGenerator(
    base_model    = my_trained_model,   # any model with a Dense output layer
    n_masks       = 3,                  # N^M: masks per logit
    ratio_range   = (0.4, 0.7),         # R: min/max random connection fraction
    seed          = 42,
)

# Forward pass returns original logits + all mask outputs
original, alternatives = ag.forward(x)
# original      shape: (batch, n_logits)
# alternatives  shape: (batch, n_logits, n_masks)
```

**Key design decisions to discuss:**
- How to freeze base model weights vs. allow fine-tuning  
- Whether masks share the previous-layer weight matrix or get their own weights  
  → Paper implies **own weight subset** (sparse copy of W)  
- How to handle Conv layers before the output (flatten, pool, or restrict to Dense)

**Files to create:**
- `src/alternatives_generator/core.py`     — AlternativesGenerator class  
- `src/alternatives_generator/__init__.py` — public API  
- `demo/step1_attach_module.py`            — demo  
- `tests/test_step1.py`                    — unit tests  

---

### STEP 2  –  WeightedProbability Function  *(second sprint)*
**Goal:** Assign each node (original + masks) a weighted probability Pr^w.

**Concept (from paper §"Weighted probability"):**

The probability reflects three things:
1. **Occurrence count** — how often does this node's value appear among all siblings?
2. **Disparity level** — how far is it from the mean of all siblings?
3. **Logit priority** — original logit gets a slight edge over masks (bias constant s)

Equations from the paper:

```
Pr_t(u_t) = count(u_t, {masks, logit}) / (N^M + 1)

w_i,t   = 1 / ln(|û_i,t − µ| / d + e + ε)          # logit weight
w_i,j,t = 1 / ln(|û_i,j,t − µ| / d + e + ε + s)    # mask weight

Pr^w_t(u_t, w_t) = w_t · Pr_t(u_t)   [then normalized across siblings]
```

**What to implement:**
- `src/alternatives_generator/weighted_probability.py`
  - `compute_probability(values)`
  - `compute_weights(values, is_logit=False)`
  - `compute_weighted_probability(values)`

**Demo:** Visualise the circle-size diagram from Fig. 2 of the paper.

---

### STEP 3  –  Behavior Function  *(third sprint)*
**Goal:** Measure how much each node "helps" the model's output distribution
         align with the training label distribution.

**Concept (from paper §"Behavior function"):**

Uses the Wasserstein-2 distance:
```
b(u_t) = W₂(Y^s, Ŷ_{t-1}) − W₂(Y^s, Ŷ_t(g(u_t)))
```

- Positive b → node improves alignment with training distribution  
- Negative b → node moves output distribution further away  
- Zero        → node has no effect  

**What to implement:**
- `src/alternatives_generator/behavior.py`
  - `wasserstein2(dist_a, dist_b)` (can use `scipy.stats.wasserstein_distance`)
  - `BehaviorTracker` — maintains a rolling output history `Ŷ`
  - `compute_behavior(node_value, y_source, y_history)`

**Important note:** This function requires **access to the source label
distribution Y^s**. The module will accept this as a parameter during
initialisation or as a running buffer updated during deployment.

---

### STEP 4  –  ProspectCertainty Integration  *(fourth sprint)*
**Goal:** Combine Pr^w and b() through Kahneman–Tversky prospect theory
         to produce a scalar certainty score Ω for each node.

**Concept (from paper §"Prospect certainty"):**

```
Ω(u_t) = Ω^b(b(u_t)) · Ω^w(Pr^w_t(u_t))

Ω^b(b) = b^ε₁                  if b ≥ ε   (gain)
        = −γ_b · (−b)^ε₂       if b < ε   (loss)

Ω^w(Pr^w) = exp(−(−ln Pr^w)^γ_w)          # Prelec function

Parameters (Kahneman & Tversky, 1992):
  ε₁ = ε₂ = 0.88,  γ_b = 2.25,  γ_w = 0.61
```

**What to implement:**
- `src/alternatives_generator/prospect.py`
  - `value_function(b, eps1=0.88, eps2=0.88, gamma_b=2.25, ref=0.0)`
  - `prelec_weighting(prob, gamma_w=0.61)`
  - `prospect_certainty(b, prob)`  → Ω

**Selection logic:**
- **Regression:** for each logit, pick the node (logit or mask) with max Ω  
- **Classification:** for each class, pick refined logit; then pick class with max Ω  

---

### STEP 5  –  Full Demo & Evaluation  *(fifth sprint)*
**Goal:** Reproduce the paper's benchmark experiments as a demonstration.

**Tasks:**
- Regression demo with synthetic data (eq. 15 from paper):  `f(x) = x·sin(x) + h₁x + h₂`
- Classification demo (optional): a small image dataset subset  
- Plot the certainty index Ω over the input range (reproduce Fig. 6 style)
- Compare output accuracy with and without the module

**Files:**
- `demo/step5_regression_benchmark.py`
- `demo/step5_classification_demo.py`

---

### STEP 6  –  Packaging, Documentation & GitHub  *(final sprint)*
**Goal:** A clean, installable package suitable for the thesis appendix.

**Tasks:**
- `pyproject.toml` / `setup.py` for `pip install -e .`  
- `docs/report.md` — technical report (concept, API, results)
- GitHub repository with a proper `README.md`, badges, and usage examples  
- CI: a simple GitHub Action running `pytest tests/`

---

## Repository Structure (target)

```
alternatives_generator/
│
├── src/
│   └── alternatives_generator/
│       ├── __init__.py            ← public API
│       ├── core.py                ← AlternativesGenerator class (Step 1)
│       ├── weighted_probability.py ← Pr^w functions (Step 2)
│       ├── behavior.py            ← behavior function b() (Step 3)
│       └── prospect.py            ← Ω certainty function (Step 4)
│
├── demo/
│   ├── step0_simple_ann_numpy.py  ← ✅ DONE – baseline ANN (pure NumPy)
│   ├── step0_simple_ann.py        ← ✅ DONE – baseline ANN (PyTorch)
│   ├── step1_attach_module.py
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
└── README.md                      ← this file
```

---

## Key Design Principles

1. **Non-invasive** — the module wraps any model without changing its weights or training loop
2. **Framework-agnostic core** — all math is implemented in NumPy; PyTorch is used as the model interface layer
3. **Configurable** — all hyperparameters (N^M, R, ε) are exposed as constructor arguments
4. **Observable** — every intermediate value (z, Pr, b, Ω) can be inspected for research purposes
5. **Reproducible** — all random operations accept a seed

---

## Dependencies

| Package | Purpose | Step |
|---------|---------|------|
| `numpy` | All core math | 0–6 |
| `scipy` | `wasserstein_distance` | 3 |
| `torch` | Model definition, training loop, tensor ops | 0–6 |
| `matplotlib` | Plots and visualisation | 5 |
| `pytest` | Unit tests | all |

Install: `pip install torch numpy scipy matplotlib pytest`

---

## Reference

Yousef, Q. & Li, P. (2025). Prospect certainty for data-driven models.  
*Scientific Reports*, 15, 8278. https://doi.org/10.1038/s41598-025-89679-6

Code (authors' original): https://doi.org/10.5281/zenodo.14541878
