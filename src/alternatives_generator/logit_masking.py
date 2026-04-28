"""
src/alternatives_generator/logit_masking.py
============================================
LogitMaskingLayer — reusable PyTorch module.

Wraps any existing torch.nn.Linear layer and generates masked
alternative outputs for each logit. Designed to be general and
reusable across any model architecture.

Key design decision:
    masked_weight = original_weight * binary_mask

    Masks reuse the original layer's trained weights.
    Only some connections are kept (set by connection_ratio).
    This ensures spread reflects genuine model sensitivity,
    not random noise.

Reference:
    Yousef & Li, "Prospect certainty for data-driven models"
    Scientific Reports 15:8278 (2025)

Author: Master Research Project — RCSE, TU Ilmenau
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════════════════
# Output container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MaskingOutput:
    """
    All outputs from one forward pass of LogitMaskingLayer.

    Fields
    ------
    original    : Tensor (batch, out_features)
                  Output of the original unmasked layer.

    masked      : Tensor (batch, out_features, num_masks)
                  Output of every mask for every logit.
                  masked[:, i, j] = output of mask j for logit i.

    mean        : Tensor (batch, out_features)
                  Mean across all alternatives (original + all masks).

    spread      : Tensor (batch, out_features)
                  max − min across all alternatives per logit.
                  HIGH spread → uncertain.  LOW spread → consistent.

    uncertainty : Tensor (batch,)
                  Mean spread across all logits per sample.
                  One scalar that summarises overall output uncertainty.
    """
    original    : torch.Tensor
    masked      : torch.Tensor
    mean        : torch.Tensor
    spread      : torch.Tensor
    uncertainty : torch.Tensor


# ══════════════════════════════════════════════════════════════════════════════
# Module
# ══════════════════════════════════════════════════════════════════════════════

class LogitMaskingLayer(nn.Module):
    """
    Wraps a torch.nn.Linear layer and generates masked alternatives
    for each output logit.

    Parameters
    ----------
    base_layer        : nn.Linear
        Any trained or untrained linear layer.
        in_features and out_features are inferred automatically.

    num_masks         : int
        Number of mask alternatives to generate per logit.
        More masks → more stable uncertainty estimate.
        Recommended: 3 (default, based on paper experiments).

    connection_ratio  : float  (0 < ratio < 1)
        Fraction of input connections each mask keeps.
        0.5 → each mask uses 50% of the original connections.
        Lower → more diversity between masks.
        Higher → masks behave more like the original logit.

    seed              : int or None
        Random seed for reproducibility. Default: 42.

    Usage
    -----
    >>> base = nn.Linear(8, 3)
    >>> layer = LogitMaskingLayer(base, num_masks=3, connection_ratio=0.5)
    >>> x = torch.randn(4, 8)
    >>> out = layer(x)
    >>> print(out.spread)    # (4, 3)
    >>> print(out.uncertainty)  # (4,)
    """

    def __init__(
        self,
        base_layer       : nn.Linear,
        num_masks        : int   = 3,
        connection_ratio : float = 0.5,
        seed             : int   = 42,
    ):
        super().__init__()

        # ── validation ────────────────────────────────────────────────────────
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                f"base_layer must be nn.Linear, got {type(base_layer).__name__}"
            )
        if not (0.0 < connection_ratio < 1.0):
            raise ValueError(
                f"connection_ratio must be between 0 and 1 (exclusive), "
                f"got {connection_ratio}"
            )
        if num_masks < 1:
            raise ValueError(
                f"num_masks must be at least 1, got {num_masks}"
            )

        # ── store settings ────────────────────────────────────────────────────
        self.base_layer       = base_layer
        self.num_masks        = num_masks
        self.connection_ratio = connection_ratio
        self.seed             = seed

        # ── infer dimensions from base_layer automatically ────────────────────
        self.in_features  = base_layer.in_features
        self.out_features = base_layer.out_features

        # Number of connections each mask keeps
        # e.g. in_features=8, ratio=0.5 → n_connections=4
        self.n_connections = max(1, round(self.in_features * connection_ratio))

        # ── build binary mask matrices and register as buffer ─────────────────
        # Buffers are not trainable parameters.
        # They move to GPU automatically and are saved with the model.
        # Shape: (num_masks, out_features, in_features)
        self.register_buffer(
            "binary_masks",
            self._build_binary_masks()
        )

    # ── internal: build masks ─────────────────────────────────────────────────

    def _build_binary_masks(self) -> torch.Tensor:
        """
        Create all binary mask matrices.

        For each mask m and each logit i:
            - randomly choose n_connections input indices
            - set those positions to 1, all others to 0

        Returns
        -------
        Tensor of shape (num_masks, out_features, in_features)
        """
        gen = torch.Generator()
        gen.manual_seed(self.seed)

        # Start with all zeros
        masks = torch.zeros(
            self.num_masks,
            self.out_features,
            self.in_features,
        )

        for m in range(self.num_masks):
            for i in range(self.out_features):
                # Random permutation → take first n_connections indices
                chosen = torch.randperm(
                    self.in_features, generator=gen
                )[: self.n_connections]
                masks[m, i, chosen] = 1.0

        return masks

    # ── forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> MaskingOutput:
        """
        Forward pass: compute original and masked logits.

        Parameters
        ----------
        x : Tensor (batch_size, in_features)

        Returns
        -------
        MaskingOutput with fields:
            original, masked, mean, spread, uncertainty
        """
        batch_size = x.shape[0]

        # ── original logits ───────────────────────────────────────────────────
        # Full weight matrix, no masking
        # Shape: (batch, out_features)
        y_original = self.base_layer(x)

        # ── masked logits ─────────────────────────────────────────────────────
        W    = self.base_layer.weight   # (out_features, in_features)
        bias = self.base_layer.bias     # (out_features,)

        # Collect masked outputs: (batch, out_features, num_masks)
        y_masked = torch.zeros(
            batch_size,
            self.out_features,
            self.num_masks,
            device=x.device,
            dtype=x.dtype,
        )

        for m in range(self.num_masks):
            # Zero out connections not in this mask
            # masked_W shape: (out_features, in_features)
            masked_W = W * self.binary_masks[m]

            # Compute output with masked weights
            # Shape: (batch, out_features)
            y_m = x @ masked_W.t()
            if bias is not None:
                y_m = y_m + bias

            y_masked[:, :, m] = y_m

        # ── statistics across all alternatives ───────────────────────────────
        # Stack: (batch, out_features, 1 + num_masks)
        # Axis -1 order: [original, mask_0, mask_1, ..., mask_n]
        all_vals = torch.cat(
            [y_original.unsqueeze(-1), y_masked],
            dim=-1,
        )

        # Mean: (batch, out_features)
        mean = all_vals.mean(dim=-1)

        # Spread = max − min: (batch, out_features)
        spread = (
            all_vals.max(dim=-1).values
            - all_vals.min(dim=-1).values
        )

        # Uncertainty = mean spread across logits: (batch,)
        uncertainty = spread.mean(dim=-1)

        return MaskingOutput(
            original    = y_original,
            masked      = y_masked,
            mean        = mean,
            spread      = spread,
            uncertainty = uncertainty,
        )

    # ── string representation ─────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_masks={self.num_masks}, "
            f"connection_ratio={self.connection_ratio}, "
            f"n_connections={self.n_connections}"
        )
