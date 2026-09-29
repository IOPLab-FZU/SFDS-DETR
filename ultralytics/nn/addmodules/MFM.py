"""Backward-compatible import for checkpoints created before paper alignment."""

from .fafeb import CEM

MFM = CEM

__all__ = ("MFM",)
