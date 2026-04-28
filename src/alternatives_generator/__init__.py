"""
alternatives_generator
======================
A reusable PyTorch module for generating masked logit alternatives
to quantify output uncertainty in neural networks.

Reference: Yousef & Li, Scientific Reports 15:8278 (2025)
"""

from .logit_masking import LogitMaskingLayer, MaskingOutput

__all__ = ["LogitMaskingLayer", "MaskingOutput"]
__version__ = "0.1.0"
