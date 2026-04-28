"""
tests/test_logit_masking.py
============================
Unit tests for LogitMaskingLayer.

Run:
    pytest tests/test_logit_masking.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import pytest
from alternatives_generator import LogitMaskingLayer, MaskingOutput


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def small_layer():
    """A small Linear layer: 4 inputs, 2 outputs."""
    torch.manual_seed(0)
    return nn.Linear(4, 2)

@pytest.fixture
def masking_layer(small_layer):
    """LogitMaskingLayer wrapping the small layer."""
    return LogitMaskingLayer(small_layer, num_masks=3,
                             connection_ratio=0.5, seed=42)

@pytest.fixture
def sample_input():
    """A batch of 3 samples, 4 features."""
    torch.manual_seed(0)
    return torch.randn(3, 4)


# ══════════════════════════════════════════════════════════════════════════════
# Shape tests
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputShapes:

    def test_original_shape(self, masking_layer, sample_input):
        out = masking_layer(sample_input)
        # (batch=3, out_features=2)
        assert out.original.shape == (3, 2)

    def test_masked_shape(self, masking_layer, sample_input):
        out = masking_layer(sample_input)
        # (batch=3, out_features=2, num_masks=3)
        assert out.masked.shape == (3, 2, 3)

    def test_mean_shape(self, masking_layer, sample_input):
        out = masking_layer(sample_input)
        assert out.mean.shape == (3, 2)

    def test_spread_shape(self, masking_layer, sample_input):
        out = masking_layer(sample_input)
        assert out.spread.shape == (3, 2)

    def test_uncertainty_shape(self, masking_layer, sample_input):
        out = masking_layer(sample_input)
        # One scalar per sample
        assert out.uncertainty.shape == (3,)


# ══════════════════════════════════════════════════════════════════════════════
# Value tests
# ══════════════════════════════════════════════════════════════════════════════

class TestValues:

    def test_spread_is_nonnegative(self, masking_layer, sample_input):
        """Spread = max - min, so always >= 0."""
        out = masking_layer(sample_input)
        assert (out.spread >= 0).all()

    def test_uncertainty_is_nonnegative(self, masking_layer, sample_input):
        out = masking_layer(sample_input)
        assert (out.uncertainty >= 0).all()

    def test_original_matches_base_layer(self, small_layer, masking_layer,
                                          sample_input):
        """Original output must equal the base layer's direct output."""
        out = masking_layer(sample_input)
        expected = small_layer(sample_input)
        assert torch.allclose(out.original, expected, atol=1e-6)

    def test_mean_is_mean_of_all(self, masking_layer, sample_input):
        """Mean must equal the mean of [original, mask_0, mask_1, mask_2]."""
        out = masking_layer(sample_input)
        # Stack original and masked
        stacked = torch.cat(
            [out.original.unsqueeze(-1), out.masked], dim=-1
        )
        expected_mean = stacked.mean(dim=-1)
        assert torch.allclose(out.mean, expected_mean, atol=1e-6)

    def test_spread_is_max_minus_min(self, masking_layer, sample_input):
        out = masking_layer(sample_input)
        stacked = torch.cat(
            [out.original.unsqueeze(-1), out.masked], dim=-1
        )
        expected_spread = (stacked.max(dim=-1).values
                           - stacked.min(dim=-1).values)
        assert torch.allclose(out.spread, expected_spread, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# Mask structure tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMaskStructure:

    def test_binary_masks_are_binary(self, masking_layer):
        """All values in binary_masks must be 0 or 1."""
        bm = masking_layer.binary_masks
        assert ((bm == 0) | (bm == 1)).all()

    def test_binary_masks_shape(self, masking_layer):
        """Shape must be (num_masks, out_features, in_features)."""
        bm = masking_layer.binary_masks
        assert bm.shape == (3, 2, 4)

    def test_each_mask_has_correct_n_connections(self, masking_layer):
        """Each logit in each mask must have exactly n_connections active."""
        bm = masking_layer.binary_masks
        expected = masking_layer.n_connections
        for m in range(masking_layer.num_masks):
            for i in range(masking_layer.out_features):
                n_active = bm[m, i].sum().int().item()
                assert n_active == expected, (
                    f"mask {m}, logit {i}: expected {expected} connections, "
                    f"got {n_active}"
                )

    def test_reproducibility_with_seed(self, small_layer):
        """Same seed must produce identical masks."""
        l1 = LogitMaskingLayer(small_layer, num_masks=3,
                               connection_ratio=0.5, seed=99)
        l2 = LogitMaskingLayer(small_layer, num_masks=3,
                               connection_ratio=0.5, seed=99)
        assert torch.equal(l1.binary_masks, l2.binary_masks)

    def test_different_seeds_produce_different_masks(self, small_layer):
        """Different seeds should (almost certainly) produce different masks."""
        l1 = LogitMaskingLayer(small_layer, num_masks=3,
                               connection_ratio=0.5, seed=1)
        l2 = LogitMaskingLayer(small_layer, num_masks=3,
                               connection_ratio=0.5, seed=2)
        assert not torch.equal(l1.binary_masks, l2.binary_masks)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration tests
# ══════════════════════════════════════════════════════════════════════════════

class TestConfiguration:

    def test_auto_infers_in_features(self, small_layer):
        layer = LogitMaskingLayer(small_layer, num_masks=2,
                                  connection_ratio=0.5)
        assert layer.in_features == 4

    def test_auto_infers_out_features(self, small_layer):
        layer = LogitMaskingLayer(small_layer, num_masks=2,
                                  connection_ratio=0.5)
        assert layer.out_features == 2

    def test_n_connections_calculation(self, small_layer):
        # ratio=0.5, in_features=4 → n_connections = round(4*0.5) = 2
        layer = LogitMaskingLayer(small_layer, num_masks=2,
                                  connection_ratio=0.5)
        assert layer.n_connections == 2

    def test_invalid_ratio_raises(self, small_layer):
        with pytest.raises(ValueError):
            LogitMaskingLayer(small_layer, num_masks=2, connection_ratio=0.0)
        with pytest.raises(ValueError):
            LogitMaskingLayer(small_layer, num_masks=2, connection_ratio=1.0)

    def test_invalid_base_layer_raises(self):
        with pytest.raises(TypeError):
            LogitMaskingLayer(nn.ReLU(), num_masks=2, connection_ratio=0.5)

    def test_invalid_num_masks_raises(self, small_layer):
        with pytest.raises(ValueError):
            LogitMaskingLayer(small_layer, num_masks=0, connection_ratio=0.5)

    def test_large_network(self):
        """Module should work with a realistic-size layer."""
        big_layer = nn.Linear(512, 100)
        layer     = LogitMaskingLayer(big_layer, num_masks=3,
                                      connection_ratio=0.5, seed=0)
        x   = torch.randn(16, 512)
        out = layer(x)
        assert out.original.shape    == (16, 100)
        assert out.masked.shape      == (16, 100, 3)
        assert out.uncertainty.shape == (16,)
