"""Backward-compatible import for checkpoints created before paper alignment."""

from .fafeb import SE

SELayer = SE

__all__ = ("SELayer",)
