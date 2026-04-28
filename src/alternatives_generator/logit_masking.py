"""
src/alternatives_generator/logit_masking.py
============================================
LogitMaskingLayer — a reusable PyTorch module.

This module wraps any existing torch.nn.Linear layer and generates
masked alternative outputs for each logit.

Key design decision:
    Masks do NOT use random independent weights.
    Instead, each mask REUSES the original layer's weights,
    but zeros out some connections via a binary mask matrix.

    masked_weight = original_weight * binary_mask

This is more scientifically sound because:
    - The masks stay grounded in the trained model's knowledge
    - Differences between masks reflect genuine input sensitivity
    - Random weights would introduce noise unrelated to the model

Author: Master Research Project — RCSE, TU Ilmenau
Reference: Yousef & Li, Scientific Reports 15:8278 (2025)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════════════════
# Return type — clean, named, easy to access
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MaskingOutput:
    """
    Structured output from one forward pass of LogitMaskingLayer.

    Attributes
    ----------
    original  : shape (batch, out_features)
                The output of the original unmasked layer.

    masked    : shape (batch, out_features, num_masks)
                The output of every mask for every logit.
                masked[:, i, j] = output of mask j for logit i.

    mean      : shape (batch, out_features)
                Mean of all alternatives (original + masks) per logit.

    spread    : shape (batch, out_features)
                max - min across all alternatives per logit.
                HIGH spread → uncertain.  LOW spread → consistent.

    uncertainty : shape (batch,)
                Scalar per sample: mean spread across all logits.
                One number that summarises how uncertain the whole
                output is for each input sample.
    """
    original    : torch.Tensor
    masked      : torch.Tensor
    mean        : torch.Tensor
    spread      : torch.Tensor
    uncertainty : torch.Tensor


# ══════════════════════════════════════════════════════════════════════════════
# Main module
# ══════════════════════════════════════════════════════════════════════════════

class LogitMaskingLayer(nn.Module):
    """
    Wraps a torch.nn.Linear layer and generates masked alternatives
    for each output logit.

    Parameters
    ----------
    base_layer        : nn.Linear
        The trained (or untrained) linear layer to wrap.
        in_features and out_features are read automatically.

    num_masks         : int
        How many mask alternatives to generate per logit.
        Paper uses 3 as default. More masks = better uncertainty
        estimate but higher compute cost.

    connection_ratio  : float  (0 < ratio < 1)
        Fraction of input connections each mask keeps.
        0.5 means each mask uses 50% of the original connections.
        Lower ratio = more diversity between masks.
        Higher ratio = masks behave more like the original.

    seed              : int or None
        Optional random seed for reproducibility.
        Set this when you need the same masks every run.

    Example
    -------
    >>> linear = nn.Linear(128, 10)
    >>> layer  = LogitMaskingLayer(linear, num_masks=3, connection_ratio=0.5)
    >>> x      = torch.randn(5, 128)
    >>> output = layer(x)
    >>> print(output.spread.shape)   # (5, 10)
    """

    def __init__(
        self,
        base_layer       : nn.Linear,
        num_masks        : int   = 3,
        connection_ratio : float = 0.5,
        seed             : int   = None,
    ):
        super().__init__()

        # ── validate inputs ───────────────────────────────────────────────────
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be a torch.nn.Linear instance.")
        if not (0 < connection_ratio < 1):
            raise ValueError("connection_ratio must be between 0 and 1 (exclusive).")
        if num_masks < 1:
            raise ValueError("num_masks must be at least 1.")

        # ── store configuration ───────────────────────────────────────────────
        self.base_layer       = base_layer
        self.num_masks        = num_masks
        self.connection_ratio = connection_ratio
        self.seed             = seed

        # ── read dimensions automatically from base_layer ─────────────────────
        self.in_features  = base_layer.in_features    # e.g. 128
        self.out_features = base_layer.out_features   # e.g. 10

        # ── how many connections each mask keeps ──────────────────────────────
        # e.g. in_features=8, ratio=0.5 → each mask keeps 4 connections
        self.n_connections = max(1, round(self.in_features * connection_ratio))

        # ── build and register the binary mask matrices ───────────────────────
        # binary_masks shape: (num_masks, out_features, in_features)
        # binary_masks[m, i, :] = which inputs are active for mask m, logit i
        binary_masks = self._build_binary_masks()

        # Register as a buffer (not a trainable parameter — masks are fixed)
        # Buffers are saved with the model and moved to GPU automatically
        self.register_buffer("binary_masks", binary_masks)

    # ── mask construction ─────────────────────────────────────────────────────

    def _build_binary_masks(self) -> torch.Tensor:
        """
        Build all binary mask matrices.

        For every mask m and every logit i, randomly choose n_connections
        input indices to keep (set to 1). All others are 0.

        Returns
        -------
        binary_masks : shape (num_masks, out_features, in_features)
        """
        # Use a local generator so we don't disturb the global random state
        gen = torch.Generator()
        if self.seed is not None:
            gen.manual_seed(self.seed)
        else:
            gen.seed()   # random seed based on time

        masks = torch.zeros(
            self.num_masks,
            self.out_features,
            self.in_features,
        )

        for m in range(self.num_masks):
            for i in range(self.out_features):
                # Pick n_connections random indices from 0..in_features-1
                # torch.randperm gives a random permutation; take the first n
                perm    = torch.randperm(self.in_features, generator=gen)
                indices = perm[: self.n_connections]
                masks[m, i, indices] = 1.0

        return masks   # shape: (num_masks, out_features, in_features)

    # ── forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> MaskingOutput:
        """
        Run a forward pass and return original + masked outputs.

        Parameters
        ----------
        x : torch.Tensor  shape (batch_size, in_features)

        Returns
        -------
        MaskingOutput  (see class docstring for field descriptions)
        """
        batch_size = x.shape[0]

        # ── 1. original output (unmasked) ─────────────────────────────────────
        # Uses the full weight matrix and bias from base_layer
        # y_original shape: (batch_size, out_features)
        y_original = self.base_layer(x)

        # ── 2. masked outputs ─────────────────────────────────────────────────
        # For each mask m:
        #   masked_weight = original_weight * binary_mask[m]
        #   y_masked[m]   = x @ masked_weight.T + bias
        #
        # base_layer.weight shape: (out_features, in_features)
        # binary_masks[m]   shape: (out_features, in_features)
        # masked_weight     shape: (out_features, in_features)

        W    = self.base_layer.weight   # (out_features, in_features)
        bias = self.base_layer.bias     # (out_features,)  or None

        # y_masked will accumulate results: (batch, out_features, num_masks)
        y_masked = torch.zeros(
            batch_size, self.out_features, self.num_masks,
            device=x.device, dtype=x.dtype
        )

        for m in range(self.num_masks):
            # Apply binary mask to the weight matrix
            # masked_W shape: (out_features, in_features)
            masked_W = W * self.binary_masks[m]

            # Compute masked output: x @ masked_W.T
            # Result shape: (batch_size, out_features)
            y_m = x @ masked_W.t()

            # Add bias if it exists (bias is not masked — same for all)
            if bias is not None:
                y_m = y_m + bias

            y_masked[:, :, m] = y_m

        # ── 3. stack original with masked for statistics ───────────────────────
        # all_outputs shape: (batch, out_features, 1 + num_masks)
        # Axis -1: [original, mask_0, mask_1, mask_2, ...]
        all_outputs = torch.cat(
            [y_original.unsqueeze(-1), y_masked],
            dim=-1
        )

        # ── 4. compute mean and spread across alternatives ─────────────────────
        # mean shape:   (batch, out_features)
        # spread shape: (batch, out_features)
        mean   = all_outputs.mean(dim=-1)
        spread = all_outputs.max(dim=-1).values - all_outputs.min(dim=-1).values

        # ── 5. scalar uncertainty score per sample ────────────────────────────
        # Average spread across all logits → one number per sample
        # HIGH = uncertain,  LOW = consistent
        # uncertainty shape: (batch,)
        uncertainty = spread.mean(dim=-1)

        return MaskingOutput(
            original    = y_original,
            masked      = y_masked,
            mean        = mean,
            spread      = spread,
            uncertainty = uncertainty,
        )

    # ── readable summary ──────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        """Shows in print(layer) output."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_masks={self.num_masks}, "
            f"connection_ratio={self.connection_ratio}, "
            f"n_connections={self.n_connections}"
        )
